import os
import csv
import hashlib
import uuid
import joblib
import scipy.sparse as sp
import pandas as pd
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.utils import timezone
import json
import re
import requests as http_requests

from .models import DetectionLog, ReportedNumber, Feedback


# ── Model loading ─────────────────────────────────────────────────────────────

CURRENT_DIR     = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH      = os.path.join(CURRENT_DIR, 'ml', 'sms_phishing_model_rf.pkl')
VECTORIZER_PATH = os.path.join(CURRENT_DIR, 'ml', 'tfidf_vectorizer_rf.pkl')
THRESHOLD_PATH  = os.path.join(CURRENT_DIR, 'ml', 'threshold.txt')

try:
    model        = joblib.load(MODEL_PATH)
    vectorizer   = joblib.load(VECTORIZER_PATH)
    MODEL_LOADED = True
    print("✅ ML models loaded from:", CURRENT_DIR)
except Exception as e:
    print(f"❌ Error loading models: {e}")
    MODEL_LOADED = False
    model = vectorizer = None

try:
    with open(THRESHOLD_PATH) as f:
        SPAM_THRESHOLD = float(f.read().strip())
    print(f"✅ Loaded calibrated threshold: {SPAM_THRESHOLD}")
except Exception:
    SPAM_THRESHOLD = 0.50
    print(f"⚠  threshold.txt not found, using default: {SPAM_THRESHOLD}")

# ── YarnGPT config ────────────────────────────────────────────────────────────

YARNGPT_API_KEY = os.environ.get('YARNGPT_API_KEY', '')
YARNGPT_URL     = 'https://yarngpt.ai/api/v1/tts'

# Map language codes → best YarnGPT voice for that language/gender
LANG_VOICE_MAP = {
    'en':  'Idera',     # Melodic, gentle — general Nigerian English
    'pid': 'Tayo',      # Upbeat, energetic — fits Pidgin energy
    'yo':  'Wura',      # Young, sweet — Yoruba female voice
    'ha':  'Umar',      # Calm, smooth — Hausa male voice
    'ig':  'Chinenye',  # Engaging, warm — Igbo female voice
}

# TTS text templates per language
def build_tts_text(label: str, confidence: int, risk: str, recommendation: str, language: str) -> str:
    templates = {
        'en': {
            'spam': (
                f"Warning! This SMS has been detected as spam. "
                f"Confidence level: {confidence} percent. "
                f"Risk level: {risk}. "
                f"{recommendation}"
            ),
            'legitimate': (
                f"This message appears legitimate. "
                f"Confidence level: {confidence} percent. "
                f"{recommendation}"
            ),
        },
        'pid': {
            'spam': (
                f"Warning! This SMS na scam message. "
                f"We don confirm am {confidence} percent. "
                f"Risk level na {risk}. "
                f"{recommendation}"
            ),
            'legitimate': (
                f"This message dey clean, e no be scam. "
                f"We confirm am {confidence} percent. "
                f"{recommendation}"
            ),
        },
        'yo': {
            'spam': (
                f"Ìkìlọ̀! A ti ṣe awari pe SMS yii jẹ spam. "
                f"Igbẹkẹle wa jẹ ogorun {confidence}. "
                f"Ipele ewu: {risk}. "
                f"{recommendation}"
            ),
            'legitimate': (
                f"Ifiranṣẹ yii dabi ẹnipe o jẹ gidi. "
                f"Igbẹkẹle wa jẹ ogorun {confidence}. "
                f"{recommendation}"
            ),
        },
        'ha': {
            'spam': (
                f"Gargadi! An gano cewa wannan SMS zamba ne. "
                f"Tabbas namu shine kashi {confidence} cikin dari. "
                f"Matakin haɗari: {risk}. "
                f"{recommendation}"
            ),
            'legitimate': (
                f"Wannan sakon yana da inganci. "
                f"Tabbas namu shine kashi {confidence} cikin dari. "
                f"{recommendation}"
            ),
        },
        'ig': {
            'spam': (
                f"Ọ dị njọ! Achọpụtara na ozi SMS a bụ aghụghọ. "
                f"Ntụkwasị obi anyị bụ {confidence} n'otu narị. "
                f"Ọkwa ihere: {risk}. "
                f"{recommendation}"
            ),
            'legitimate': (
                f"Ozi a yiri ka ọ dị mọọ. "
                f"Ntụkwasị obi anyị bụ {confidence} n'otu narị. "
                f"{recommendation}"
            ),
        },
    }
    lang_templates = templates.get(language, templates['en'])
    return lang_templates.get(label, lang_templates.get('spam', ''))


# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = ' '.join(text.split())
    text = re.sub(r'https?://\S+',  ' PHISHURL ',    text)
    text = re.sub(r'www\.\S+',      ' PHISHURL ',    text)
    text = re.sub(r'\b\w+\.(net|xyz|info|cc|tk|ml|ga|ng)\b', ' SUSPECTTLD ', text)
    text = re.sub(r'\b(0[789][01]\d{8})\b', ' NGNPHONE ', text)
    text = re.sub(r'\+234\d{10}',   ' NGNPHONE ',    text)
    text = re.sub(r'[₦n]\s*[\d,]+', ' NAIRAAMT ',    text)
    text = re.sub(r'\b\d[\d,]*\s*(naira|dollars|pounds)\b', ' MONEYAMT ', text,
                  flags=re.IGNORECASE)
    text = re.sub(r'!{2,}', ' MULTIEXCLAIM ',   text)
    text = re.sub(r'\?{2,}', ' MULTIQUESTION ', text)
    return text


def extract_fraud_features_single(text: str) -> list:
    if not isinstance(text, str):
        text = ""
    t = text.lower()

    urgency_words = ['urgent', 'immediately', 'now', 'today', 'expires',
                     'suspended', 'blocked', 'deactivated', 'act now',
                     'limited time', 'verify now', 'click here']
    prize_words   = ['congratulations', 'winner', 'won', 'prize', 'grant',
                     'lottery', 'selected', 'compensation', 'reward',
                     'million', 'n200', 'n500', 'n1000', 'n5000']
    banking_words = ['bvn', 'otp', 'pin', 'atm', 'cvv', 'account number',
                     'bank details', 'nin', 'verify', 'confirm',
                     'password', 'login', 'credential']
    impersonated  = ['gtb', 'zenith', 'access bank', 'first bank', 'uba',
                     'jamb', 'waec', 'nnpc', 'efcc', 'mtn', 'airtel']

    urgency_score       = sum(1 for w in urgency_words if w in t)
    prize_score         = sum(1 for w in prize_words   if w in t)
    banking_score       = sum(1 for w in banking_words if w in t)
    impersonation_score = sum(1 for w in impersonated  if w in t)

    has_url               = 1 if re.search(r'https?://|www\.|bit\.ly|short\.url', t) else 0
    has_suspicious_domain = 1 if re.search(r'\b\w+\.(net|xyz|info|cc|tk|ml|ga)\b', t) else 0
    has_phone             = 1 if re.search(r'\b0[789][01]\d{8}\b', text) else 0
    phone_plus_urgency    = has_phone * urgency_score

    letters    = [c for c in text if c.isalpha()]
    caps_ratio = sum(1 for c in letters if c.isupper()) / len(letters) if letters else 0

    exclamation_count  = min(text.count('!'), 5)
    total_fraud_signal = (urgency_score + prize_score + banking_score
                          + has_suspicious_domain * 2 + impersonation_score)

    return [
        urgency_score, prize_score, banking_score,
        has_url, has_suspicious_domain, has_phone, phone_plus_urgency,
        round(caps_ratio, 3), exclamation_count, impersonation_score,
        total_fraud_signal,
    ]


def analyze_indicators(message: str):
    keywords = {
        'bvn': 3, 'otp': 3, 'nin': 3, 'pin': 3, 'atm': 2,
        'suspended': 3, 'blocked': 2, 'deactivated': 2,
        'verify': 2, 'confirm': 2, 'urgent': 2, 'immediately': 2,
        'winner': 3, 'won': 3, 'lottery': 3, 'prize': 2,
        'congratulations': 2, 'grant': 2, 'selected': 2,
        'click here': 3, 'act now': 3, 'limited time': 2,
        'bank details': 3, 'account number': 3, 'password': 3,
    }
    msg_lower = message.lower()
    indicators, risk_score = [], 0
    for kw, weight in keywords.items():
        if kw in msg_lower:
            indicators.append(kw)
            risk_score += weight
    if re.search(r'https?://\S+|www\.\S+', message):
        indicators.append('contains URL')
        risk_score += 2
    if re.search(r'\b0[789][01]\d{8}\b|\+234\d{10}', message):
        indicators.append('contains Nigerian phone number')
        risk_score += 1
    return indicators, risk_score


def _classify(spam_prob: float):
    t = SPAM_THRESHOLD * 100
    if spam_prob >= t:
        return 'phishing', 'High'
    elif spam_prob >= t * 0.65:
        return 'suspicious', 'Medium'
    return 'safe', 'Low'


def _verdict(classification: str):
    if classification == 'phishing':
        return (
            "Phishing / Spam SMS",
            "Danger! This is a phishing attempt",
            "Do not click any links or respond. Delete this message immediately.",
        )
    elif classification == 'suspicious':
        return (
            "Suspicious SMS",
            "Warning! This content is suspicious",
            "Be cautious. Verify the sender before taking any action.",
        )
    return (
        "Legitimate SMS",
        "This content appears safe",
        "This message seems legitimate, but always stay vigilant.",
    )


def _rule_based_probs(risk_score: int):
    if risk_score >= 10: return 88.0, 12.0
    elif risk_score >= 6: return 72.0, 28.0
    elif risk_score >= 3: return 55.0, 45.0
    return 18.0, 82.0


# ── Main detection endpoint ───────────────────────────────────────────────────

@csrf_exempt
@require_http_methods(["POST"])
def check_message(request):
    try:
        data     = json.loads(request.body)
        message  = data.get('message', '').strip()
        language = data.get('language', 'en')

        if not message:
            return JsonResponse({'error': 'No message provided'}, status=400)

        indicators, risk_score = analyze_indicators(message)
        message_clean = preprocess_text(message)
        mode = 'ml'

        if MODEL_LOADED and model and vectorizer:
            try:
                text_vec      = vectorizer.transform([message_clean])
                manual_feats  = extract_fraud_features_single(message)
                full_vec      = sp.hstack([text_vec, sp.csr_matrix([manual_feats])])

                if hasattr(model, 'predict_proba'):
                    probs      = model.predict_proba(full_vec)[0]
                    spam_prob  = float(probs[1] * 100)
                    legit_prob = float(probs[0] * 100)
                else:
                    pred       = model.predict(full_vec)[0]
                    spam_prob  = 85.0 if pred == 1 else 15.0
                    legit_prob = 100.0 - spam_prob
            except Exception as e:
                print(f"ML prediction error: {e}")
                mode = 'rule_based'
                spam_prob, legit_prob = _rule_based_probs(risk_score)
        else:
            mode = 'rule_based'
            spam_prob, legit_prob = _rule_based_probs(risk_score)

        classification, risk_level          = _classify(spam_prob)
        pred_text, msg_text, recommendation = _verdict(classification)

        label      = 'legitimate' if classification == 'safe' else 'spam'
        confidence = round(max(spam_prob, legit_prob), 2)
        det_id     = str(uuid.uuid4())[:16]

        try:
            DetectionLog.objects.create(
                detection_id      = det_id,
                message_hash      = hashlib.sha256(message.encode()).hexdigest(),
                label             = label,
                confidence        = confidence,
                risk_level        = risk_level,
                language          = language,
                mode              = mode,
                indicators        = indicators,
                spam_probability  = round(spam_prob, 2),
                legit_probability = round(legit_prob, 2),
            )
        except Exception as e:
            print(f"DB save error (non-fatal): {e}")

        return JsonResponse({
            'prediction':        pred_text,
            'classification':    classification,
            'spam_probability':  round(spam_prob, 2),
            'legit_probability': round(legit_prob, 2),
            'confidence':        confidence,
            'message':           msg_text,
            'recommendation':    recommendation,
            'indicators':        indicators,
            'risk_score':        risk_score,
            'model_loaded':      MODEL_LOADED,
            'detection_id':      det_id,
        })

    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON'}, status=400)
    except Exception as e:
        return JsonResponse({'error': f'Server error: {str(e)}'}, status=500)


# ── Report endpoint ───────────────────────────────────────────────────────────

@csrf_exempt
@require_http_methods(["POST"])
def report_number(request):
    try:
        data            = json.loads(request.body)
        number          = re.sub(r'\s+', '', data.get('number', '').strip())
        message         = data.get('message', '')
        language        = data.get('language', 'en')
        predicted_label = data.get('predicted_label', '')

        if not number:
            return JsonResponse({'error': 'No phone number provided'}, status=400)

        threshold = ReportedNumber.AUTO_FLAG_THRESHOLD
        reported, created = ReportedNumber.objects.get_or_create(
            number=number,
            defaults={'language': language, 'predicted_label': predicted_label,
                      'sample_message': message[:500]},
        )

        if not created:
            reported.report_count += 1
            reported.last_reported = timezone.now()
            if message:
                reported.sample_message = message[:500]

        if reported.report_count >= threshold and not reported.auto_flagged:
            reported.auto_flagged = True
            reported.flagged_at   = timezone.now()

        reported.save()
        flagged = reported.auto_flagged

        return JsonResponse({
            'success':      True,
            'report_count': reported.report_count,
            'auto_flagged': flagged,
            'threshold':    threshold,
            'message': (
                f"Number {number} auto-flagged for telco review." if flagged
                else f"Report recorded. {threshold - reported.report_count} more needed."
            ),
        })

    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON'}, status=400)
    except Exception as e:
        return JsonResponse({'error': f'Server error: {str(e)}'}, status=500)


# ── Feedback endpoint ─────────────────────────────────────────────────────────

@csrf_exempt
@require_http_methods(["POST"])
def submit_feedback(request):
    try:
        data = json.loads(request.body)
        detection_id    = data.get('detection_id', '')
        original_label  = data.get('original_label', '')
        corrected_label = data.get('corrected_label', '')
        language        = data.get('language', 'en')

        if not detection_id or not corrected_label:
            return JsonResponse({'error': 'Missing required fields'}, status=400)

        Feedback.objects.create(
            detection_id=detection_id, original_label=original_label,
            corrected_label=corrected_label, language=language,
        )
        return JsonResponse({'success': True, 'message': 'Feedback recorded. Thank you!'})

    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON'}, status=400)
    except Exception as e:
        return JsonResponse({'error': f'Server error: {str(e)}'}, status=500)


# ── Audio endpoint (YarnGPT) ──────────────────────────────────────────────────

@csrf_exempt
@require_http_methods(["POST"])
def generate_audio(request):
    try:
        data           = json.loads(request.body)
        label          = data.get('label', 'spam')
        confidence     = data.get('confidence', 0)
        risk_level     = data.get('risk_level', 'High')
        language       = data.get('language', 'en')
        recommendation = data.get('recommendation', '')

        if not YARNGPT_API_KEY:
            return JsonResponse(
                {'error': 'YARNGPT_API_KEY not configured'},
                status=503
            )

        # Build the spoken text in the correct language
        c        = round(float(confidence))
        tts_text = build_tts_text(label, c, risk_level, recommendation, language)
        voice    = LANG_VOICE_MAP.get(language, 'Idera')

        # Call YarnGPT
        yarngpt_response = http_requests.post(
            YARNGPT_URL,
            headers={
                'Authorization': f'Bearer {YARNGPT_API_KEY}',
                'Content-Type':  'application/json',
            },
            json={
                'text':            tts_text,
                'voice':           voice,
                'response_format': 'mp3',
            },
            timeout=30,
            stream=True,
        )

        if yarngpt_response.status_code != 200:
            print(f"YarnGPT error {yarngpt_response.status_code}: {yarngpt_response.text}")
            return JsonResponse(
                {'error': f'YarnGPT error: {yarngpt_response.status_code}'},
                status=502
            )

        # Stream audio bytes back to frontend
        audio_bytes = b''.join(yarngpt_response.iter_content(chunk_size=8192))
        return HttpResponse(
            audio_bytes,
            content_type='audio/mpeg',
            status=200,
        )

    except http_requests.Timeout:
        return JsonResponse({'error': 'YarnGPT request timed out'}, status=504)
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON'}, status=400)
    except Exception as e:
        print(f"Audio generation error: {e}")
        return JsonResponse({'error': f'Server error: {str(e)}'}, status=500)


# ── Admin endpoints ───────────────────────────────────────────────────────────

@require_http_methods(["GET"])
def admin_stats(request):
    try:
        total = DetectionLog.objects.count()
        spam  = DetectionLog.objects.filter(label='spam').count()
        legit = DetectionLog.objects.filter(label='legitimate').count()
        return JsonResponse({
            'total_scanned':    total,
            'spam_detected':    spam,
            'legit_detected':   legit,
            'spam_rate':        round(spam / total * 100, 1) if total else 0,
            'reported_numbers': ReportedNumber.objects.count(),
            'flagged_telco':    ReportedNumber.objects.filter(auto_flagged=True).count(),
            'feedback_pending': Feedback.objects.filter(processed=False).count(),
            'spam_threshold':   SPAM_THRESHOLD,
            'model_loaded':     MODEL_LOADED,
        })
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


@require_http_methods(["GET"])
def admin_logs(request):
    try:
        limit = min(int(request.GET.get('limit', 50)), 500)
        logs  = DetectionLog.objects.all()[:limit]
        return JsonResponse([{
            'id': l.id, 'detection_id': l.detection_id, 'label': l.label,
            'confidence': l.confidence, 'risk_level': l.risk_level,
            'language': l.language, 'mode': l.mode,
            'indicators': l.indicators, 'timestamp': l.timestamp.isoformat(),
        } for l in logs], safe=False)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


@require_http_methods(["GET"])
def admin_numbers(request):
    try:
        reported = [{'number': r.number, 'report_count': r.report_count,
                     'language': r.language, 'predicted_label': r.predicted_label,
                     'first_reported': r.first_reported.isoformat(),
                     'last_reported': r.last_reported.isoformat()}
                    for r in ReportedNumber.objects.all()[:100]]
        flagged  = [{'number': f.number, 'report_count': f.report_count,
                     'flagged_by': 'community',
                     'flagged_at': f.flagged_at.isoformat() if f.flagged_at else '',
                     'telco_exported': False}
                    for f in ReportedNumber.objects.filter(auto_flagged=True)[:100]]
        return JsonResponse({'reported': reported, 'flagged': flagged})
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


@require_http_methods(["GET"])
def admin_feedback(request):
    try:
        return JsonResponse([{
            'id': fb.id, 'detection_id': fb.detection_id,
            'original_label': fb.original_label, 'corrected_label': fb.corrected_label,
            'language': fb.language, 'timestamp': fb.timestamp.isoformat(),
            'processed': fb.processed,
        } for fb in Feedback.objects.all()[:100]], safe=False)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


@require_http_methods(["GET"])
def admin_export(request):
    try:
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename="phishguard_detections.csv"'
        writer = csv.writer(response)
        writer.writerow(['ID', 'Detection ID', 'Label', 'Confidence', 'Risk Level',
                         'Language', 'Mode', 'Spam %', 'Legit %', 'Timestamp'])
        for log in DetectionLog.objects.all().iterator():
            writer.writerow([log.id, log.detection_id, log.label, log.confidence,
                             log.risk_level, log.language, log.mode,
                             log.spam_probability, log.legit_probability,
                             log.timestamp.isoformat()])
        return response
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


# ── Utility endpoints ─────────────────────────────────────────────────────────

@require_http_methods(["GET"])
def health_check(request):
    return JsonResponse({
        'status':       'healthy',
        'model_loaded': MODEL_LOADED,
        'threshold':    SPAM_THRESHOLD,
        'timestamp':    str(timezone.now()),
        'audio':        'YarnGPT' if YARNGPT_API_KEY else 'browser-fallback',
    })


@require_http_methods(["GET"])
def api_home(request):
    return JsonResponse({
        'message':      'PhishGuard NG API',
        'version':      '2.1',
        'model_loaded': MODEL_LOADED,
        'spam_threshold': SPAM_THRESHOLD,
        'endpoints': {
            'check_message': 'POST /api/check-message/',
            'report':        'POST /api/report/',
            'feedback':      'POST /api/feedback/',
            'audio':         'POST /api/audio/',
            'health':        'GET  /api/health/',
        },
    })


@csrf_exempt
@require_http_methods(["POST"])
def predict_sms(request):
    return check_message(request)