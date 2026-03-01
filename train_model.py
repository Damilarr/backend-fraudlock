import pandas as pd
import joblib
import re
from datetime import datetime

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline


# ─────────────────────────────────────────────
# 1. LOAD DATASET
#    Same file and column names you already use
# ─────────────────────────────────────────────
data = pd.read_csv("nigerian_sms.csv")

X_raw = data["Message"]
y     = data["Spam"]   # 1 = spam/phishing, 0 = legitimate

print(f"Dataset loaded: {len(data)} samples")
print(f"Class distribution:\n{y.value_counts().to_string()}\n")


# ─────────────────────────────────────────────
# 2. NIGERIAN-AWARE TEXT PREPROCESSING
# ─────────────────────────────────────────────
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r'http\S+|www\.\S+', ' URL ', text)               # URLs
    text = re.sub(r'\+?234[\d\s\-]{7,}|0[789][01]\d{8}', ' PHONE ', text)  # Nigerian numbers
    text = re.sub(r'₦[\d,]+|naira\s*[\d,]+', ' MONEY ', text)       # Naira amounts
    text = re.sub(r'\b\d{4,}\b', ' NUMBER ', text)                   # Long digit strings
    text = re.sub(r'[^\w\s]', ' ', text)                             # Punctuation
    text = re.sub(r'\s+', ' ', text).strip()
    return text

X = X_raw.apply(clean_text)


# ─────────────────────────────────────────────
# 3. TRAIN / TEST SPLIT
#    Same parameters as your original script
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# ─────────────────────────────────────────────
# 4. VECTORIZER
# ─────────────────────────────────────────────
vectorizer = TfidfVectorizer(
    max_features=8000,
    ngram_range=(1, 2),
    sublinear_tf=True,
    min_df=2,
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec  = vectorizer.transform(X_test)


# ─────────────────────────────────────────────
# 5a. TRAIN SVM  (your original model, kept intact)
# ─────────────────────────────────────────────
print("Training SVM (LinearSVC)...")
svm_model = LinearSVC(
    random_state=42,
    max_iter=2000,
    class_weight="balanced",   # added: handles class imbalance in your dataset
    C=1.0,
)
svm_model.fit(X_train_vec, y_train)

svm_preds  = svm_model.predict(X_test_vec)
svm_scores = svm_model.decision_function(X_test_vec)
svm_acc    = accuracy_score(y_test, svm_preds)
svm_auc    = roc_auc_score(y_test, svm_scores)

print(f"  Accuracy : {svm_acc:.4f}")
print(f"  AUC-ROC  : {svm_auc:.4f}")


# ─────────────────────────────────────────────
# 5b. TRAIN RANDOM FOREST  (new addition)
# ─────────────────────────────────────────────
print("\nTraining Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)
rf_model.fit(X_train_vec, y_train)

rf_preds = rf_model.predict(X_test_vec)
rf_proba = rf_model.predict_proba(X_test_vec)[:, 1]
rf_acc   = accuracy_score(y_test, rf_preds)
rf_auc   = roc_auc_score(y_test, rf_proba)

print(f"  Accuracy : {rf_acc:.4f}")
print(f"  AUC-ROC  : {rf_auc:.4f}")


# ─────────────────────────────────────────────
# 6. CROSS-VALIDATION (5-fold)
#    Gives a more honest score than single split
# ─────────────────────────────────────────────
print("\nRunning 5-fold cross-validation...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

svm_pipe = make_pipeline(
    TfidfVectorizer(max_features=8000, ngram_range=(1, 2), sublinear_tf=True, min_df=2),
    LinearSVC(random_state=42, max_iter=2000, class_weight="balanced"),
)
rf_pipe = make_pipeline(
    TfidfVectorizer(max_features=8000, ngram_range=(1, 2), sublinear_tf=True, min_df=2),
    RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1),
)

svm_cv = cross_val_score(svm_pipe, X, y, cv=cv, scoring="f1", n_jobs=-1)
rf_cv  = cross_val_score(rf_pipe,  X, y, cv=cv, scoring="f1", n_jobs=-1)

print(f"  SVM CV F1 : {svm_cv.mean():.4f} ± {svm_cv.std():.4f}")
print(f"  RF  CV F1 : {rf_cv.mean():.4f} ± {rf_cv.std():.4f}")


# ─────────────────────────────────────────────
# 7. FULL EVALUATION REPORT
# ─────────────────────────────────────────────
separator = "=" * 60

print(f"\n{separator}")
print("SVM — LinearSVC")
print(separator)
print(f"Accuracy : {svm_acc:.4f}  |  AUC-ROC : {svm_auc:.4f}")
print(f"CV F1    : {svm_cv.mean():.4f} ± {svm_cv.std():.4f}")
print(classification_report(y_test, svm_preds, target_names=["Legitimate", "Spam"]))
print("Confusion Matrix:")
print(confusion_matrix(y_test, svm_preds))

print(f"\n{separator}")
print("Random Forest")
print(separator)
print(f"Accuracy : {rf_acc:.4f}  |  AUC-ROC : {rf_auc:.4f}")
print(f"CV F1    : {rf_cv.mean():.4f} ± {rf_cv.std():.4f}")
print(classification_report(y_test, rf_preds, target_names=["Legitimate", "Spam"]))
print("Confusion Matrix:")
print(confusion_matrix(y_test, rf_preds))

# Decide winner
winner    = "SVM" if svm_auc >= rf_auc else "Random Forest"
winner_fn = "sms_phishing_model.pkl"   # always the "active" model your app loads

print(f"\n{'─'*60}")
print(f"Winner by AUC-ROC : {winner}")
print(f"Active model saved as: {winner_fn}")
print(f"{'─'*60}\n")



best_model = svm_model if svm_auc >= rf_auc else rf_model

joblib.dump(best_model,  "sms_phishing_model.pkl")   # ← your existing load path
joblib.dump(vectorizer,  "tfidf_vectorizer.pkl")      # ← your existing load path
joblib.dump(svm_model,   "svm_model.pkl")             # individual save
joblib.dump(rf_model,    "rf_model.pkl")              # individual save

print("Saved:")
print("  sms_phishing_model.pkl  ← best model (loaded by your Django app)")
print("  tfidf_vectorizer.pkl    ← vectorizer  (loaded by your Django app)")
print("  svm_model.pkl           ← SVM standalone")
print("  rf_model.pkl            ← RF standalone")


# ─────────────────────────────────────────────
# 9. SAVE TEXT REPORT
# ─────────────────────────────────────────────
report_lines = [
    "SMS Phishing Detection — Training Report",
    f"Generated  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    f"Dataset    : nigerian_sms.csv  ({len(data)} samples)",
    f"Train/Test : {len(X_train)} / {len(X_test)}",
    separator,
    "",
    "[SVM — LinearSVC]",
    f"Accuracy  : {svm_acc:.4f}",
    f"AUC-ROC   : {svm_auc:.4f}",
    f"CV F1     : {svm_cv.mean():.4f} ± {svm_cv.std():.4f}",
    "",
    classification_report(y_test, svm_preds, target_names=["Legitimate", "Spam"]),
    f"Confusion Matrix:\n{confusion_matrix(y_test, svm_preds)}",
    "",
    separator,
    "",
    "[Random Forest]",
    f"Accuracy  : {rf_acc:.4f}",
    f"AUC-ROC   : {rf_auc:.4f}",
    f"CV F1     : {rf_cv.mean():.4f} ± {rf_cv.std():.4f}",
    "",
    classification_report(y_test, rf_preds, target_names=["Legitimate", "Spam"]),
    f"Confusion Matrix:\n{confusion_matrix(y_test, rf_preds)}",
    "",
    separator,
    f"Winner by AUC-ROC : {winner}",
    f"Active model      : {winner_fn}",
]

with open("training_report.txt", "w") as f:
    f.write("\n".join(report_lines))

print("\ntraining_report.txt saved.")
print("\nDone! Your Django app will automatically use the best model.")