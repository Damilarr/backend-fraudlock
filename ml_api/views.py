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
TERMII_API_KEY = os.environ.get('TERMII_API_KEY', '')
TERMII_BASE    = 'https://v3.api.termii.com'

def normalise_nigerian_number(number: str) -> str:
    number = re.sub(r'\s+|-', '', number.strip())
    if number.startswith('+234'):
        return number[1:]
    if number.startswith('234'):
        return number
    if number.startswith('0') and len(number) == 11:
        return '234' + number[1:]
    return number

def get_termii_carrier(number: str) -> dict:
    if not TERMII_API_KEY:
        return {'network': None, 'status': None, 'dnd': None}
    try:
        resp = http_requests.get(
            f'{TERMII_BASE}/api/check/dnd',
            params={
                'api_key':      TERMII_API_KEY,
                'phone_number': normalise_nigerian_number(number),
                'type':         '2',
            },
            timeout=8,
        )
        if resp.status_code == 200:
            data = resp.json()
            return {
                'network':      data.get('network', None),
                'network_code': data.get('network_code', None),
                'status':       data.get('status', None),
                'dnd':          'dnd' in str(data.get('status', '')).lower(),
            }
        return {'network': None, 'status': None, 'dnd': None}
    except Exception as e:
        return {'network': None, 'status': None, 'dnd': None}
YARNGPT_URL     = 'https://yarngpt.ai/api/v1/tts'

LANG_VOICE_MAP = {
    'en':  'Idera',
    'pid': 'Tayo',
    'yo':  'Wura',
    'ha':  'Umar',
    'ig':  'Chinenye',
}

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
    return indicators, risk_score


def _classify(spam_prob: float):
    t = SPAM_THRESHOLD * 100
    if spam_prob >= t:
        return 'phishing', 'High'
    elif spam_prob >= t * 0.65:
        return 'suspicious', 'Medium'
    return 'safe', 'Low'


# ── UPDATED: multilingual _verdict ───────────────────────────────────────────

def _verdict(classification: str, language: str = 'en'):
    verdicts = {
        'phishing': {
            'en':  ("Phishing / Spam SMS",
                    "Danger! This is a phishing attempt",
                    "Do not click any links or respond. Delete this message immediately."),
            'pid': ("Phishing / Spam SMS",
                    "Danger! This na phishing attempt",
                    "No click any link or reply. Delete this message now now."),
            'yo':  ("Phishing / Spam SMS",
                    "Ewu! Eyi jẹ igbiyanju jibiti",
                    "Má tẹ ọna asopọ eyikeyi. Pa ifiranṣẹ yii rẹ lẹsẹkẹsẹ."),
            'ha':  ("Phishing / Spam SMS",
                    "Haɗari! Wannan yunƙurin zamba ne",
                    "Kada ka danna wani hanyar haɗi. Goge wannan saƙo yanzu."),
            'ig':  ("Phishing / Spam SMS",
                    "Ihe egwu! Nke a bụ nnọchitere aghụghọ",
                    "Ekwela iji njikọ ọ bụla. Hichapụ ozi a ozugbo."),
        },
        'suspicious': {
            'en':  ("Suspicious SMS",
                    "Warning! This content is suspicious",
                    "Be cautious. Verify the sender before taking any action."),
            'pid': ("Suspicious SMS",
                    "Warning! This content get issue",
                    "Take am easy. Verify who send am before you do anything."),
            'yo':  ("Suspicious SMS",
                    "Ìkìlọ̀! Akoonu yii jẹ ifura",
                    "Ṣọra. Ṣayẹwo ẹni ti o fi ranṣẹ ṣaaju ki o to ṣe iṣe eyikeyi."),
            'ha':  ("Suspicious SMS",
                    "Gargadi! Wannan abun ciki yana da shakka",
                    "Yi hankali. Tabbatar da mai aika kafin daukar wani mataki."),
            'ig':  ("Suspicious SMS",
                    "Ọ dị njọ! Ọdịnaya a na-atọ atụ",
                    "Dị careful. Nọchite onye zitere tupu ị mee ihe ọ bụla."),
        },
        'safe': {
            'en':  ("Legitimate SMS",
                    "This content appears safe",
                    "This message seems legitimate, but always stay vigilant."),
            'pid': ("Legitimate SMS",
                    "This content dey clean",
                    "This message look genuine, but always dey alert."),
            'yo':  ("Legitimate SMS",
                    "Akoonu yii dabi ẹnipe o jẹ ailewu",
                    "Ifiranṣẹ yii dabi ẹnipe o jẹ gidi, ṣugbọn jẹ ki o mọ nigbagbogbo."),
            'ha':  ("Legitimate SMS",
                    "Wannan abun ciki yana da aminci",
                    "Wannan sakon yana da inganci, amma koyaushe ka kasance a faɗake."),
            'ig':  ("Legitimate SMS",
                    "Ozi a yiri ka ọ dị mọọ",
                    "Ozi a yiri ka ọ dị ezigbo, mana nọgide na-ele anya mgbe niile."),
        },
    }
    lang_verdicts = verdicts.get(classification, verdicts['safe'])
    return lang_verdicts.get(language, lang_verdicts['en'])


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

        classification, risk_level = _classify(spam_prob)

        # UPDATED: pass language so recommendation is in the correct language
        pred_text, msg_text, recommendation = _verdict(classification, language)

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
        data       = json.loads(request.body)
        label      = data.get('label', 'spam')
        confidence = data.get('confidence', 0)
        risk_level = data.get('risk_level', 'High')
        language   = data.get('language', 'en')

        # Build recommendation fresh in correct language — never trust frontend value
        label_for_verdict = 'phishing' if label == 'spam' else 'safe'
        _, _, recommendation = _verdict(label_for_verdict, language)

        if not YARNGPT_API_KEY:
            return JsonResponse(
                {'error': 'YARNGPT_API_KEY not configured'},
                status=503
            )

        c        = round(float(confidence))
        tts_text = build_tts_text(label, c, risk_level, recommendation, language)
        voice    = LANG_VOICE_MAP.get(language, 'Idera')

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
        'message':      ' FRAUDLOCK NG API',
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
@require_http_methods(["GET"])
def lookup_number(request):
    try:
        number_clean = re.sub(r'\s+', '', request.GET.get('number', '').strip())
        if not number_clean:
            return JsonResponse({'error': 'No number provided'}, status=400)

        reported = None
        for fmt in [number_clean, normalise_nigerian_number(number_clean),
                    '0' + normalise_nigerian_number(number_clean)[3:]]:
            try:
                reported = ReportedNumber.objects.get(number=fmt)
                break
            except ReportedNumber.DoesNotExist:
                continue

        carrier = get_termii_carrier(number_clean)

        if reported:
            return JsonResponse({
                'found':           True,
                'number':          reported.number,
                'report_count':    reported.report_count,
                'auto_flagged':    reported.auto_flagged,
                'predicted_label': reported.predicted_label,
                'first_reported':  reported.first_reported.isoformat(),
                'last_reported':   reported.last_reported.isoformat(),
                'flagged_at':      reported.flagged_at.isoformat() if reported.flagged_at else None,
                'threshold':       ReportedNumber.AUTO_FLAG_THRESHOLD,
                'network':         carrier.get('network'),
                'network_code':    carrier.get('network_code'),
                'dnd':             carrier.get('dnd'),
                'carrier_status':  carrier.get('status'),
            })
        else:
            return JsonResponse({
                'found':        False,
                'number':       number_clean,
                'report_count': 0,
                'auto_flagged': False,
                'network':      carrier.get('network'),
                'network_code': carrier.get('network_code'),
                'dnd':          carrier.get('dnd'),
                'threshold':    ReportedNumber.AUTO_FLAG_THRESHOLD,
            })
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


@csrf_exempt
@require_http_methods(["GET"])
def number_directory(request):
    try:
        qs = ReportedNumber.objects.all().order_by('-report_count')[:200]
        return JsonResponse([{
            'number':         r.number,
            'report_count':   r.report_count,
            'auto_flagged':   r.auto_flagged,
            'flagged_at':     r.flagged_at.isoformat() if r.flagged_at else None,
            'first_reported': r.first_reported.isoformat(),
            'last_reported':  r.last_reported.isoformat(),
        } for r in qs], safe=False)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)
def get_termii_carrier(number: str) -> dict:
    if not TERMII_API_KEY:
        return {'network': None, 'status': None, 'dnd': None}
    try:
        resp = http_requests.get(
            f'{TERMII_BASE}/api/check/dnd',
            params={
                'api_key':      TERMII_API_KEY,
                'phone_number': normalise_nigerian_number(number),
                'type':         '2',
            },
            timeout=8,
        )
        if resp.status_code in (200, 404):
              data = resp.json()
              network = data.get('network', None)
        if network:
         return {
            'network':      network,
            'network_code': data.get('network_code', None),
            'status':       data.get('status', None),
            'dnd':          False,  # 404 means NOT on DND
        }
        return {'network': None, 'status': None, 'dnd': None}
    except Exception as e:
        print("TERMII ERROR:", e)
        return {'network': None, 'status': None, 'dnd': None}
def predict_sms(request):
    return check_message(request)