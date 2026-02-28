# ml_classifier.py
# Drop this into: smart-inbox-backend/backend/app/services/
#
# What this does:
#   - Trains a Random Forest classifier on your real synced emails
#   - Labels: HIGH / MEDIUM / LOW
#   - Features: sentence-transformer embeddings (384-dim)
#   - Replaces rule-based calculate_importance() entirely
#   - Saves model to disk so it persists across restarts

import os
import pickle
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")

# ─── Paths ────────────────────────────────────────────────────────────────────
MODEL_DIR = Path(__file__).parent / "saved_models"
MODEL_DIR.mkdir(exist_ok=True)
CLASSIFIER_PATH = MODEL_DIR / "priority_classifier.pkl"
ENCODER_PATH    = MODEL_DIR / "label_encoder.pkl"

# ─── Embedding model (shared with embedding_service.py) ──────────────────────
_embedder = None

def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


# ─── Synthetic training data ──────────────────────────────────────────────────
# These seed the model before you have real labeled data.
# Once you have real emails, call train_on_real_emails() to retrain.

TRAINING_DATA = [
    # HIGH priority
    ("URGENT: Action required before EOD - budget approval needed", "HIGH"),
    ("Invoice overdue - payment required immediately", "HIGH"),
    ("Critical security vulnerability found in production", "HIGH"),
    ("Board meeting tomorrow - please confirm attendance", "HIGH"),
    ("Final warning: contract expires Friday", "HIGH"),
    ("Server down - immediate response needed", "HIGH"),
    ("Deadline today: submit quarterly report by 5pm", "HIGH"),
    ("Emergency: client escalation requires your attention now", "HIGH"),
    ("Legal notice: respond within 24 hours", "HIGH"),
    ("Your account will be suspended unless you act today", "HIGH"),
    ("Interview scheduled for tomorrow morning - please confirm", "HIGH"),
    ("Project blocked - need your decision to proceed", "HIGH"),
    ("Compliance audit due this week - documents required", "HIGH"),
    ("Final reminder: renewal deadline approaching", "HIGH"),
    ("CEO needs this report before the investor call", "HIGH"),
    ("Production outage alert - all hands needed", "HIGH"),
    ("Urgent: visa application requires documents by Thursday", "HIGH"),
    ("Last chance to respond to partnership offer", "HIGH"),
    ("Tax filing deadline this week", "HIGH"),
    ("Critical bug in release - hotfix needed ASAP", "HIGH"),

    # MEDIUM priority
    ("Team meeting next week to discuss roadmap", "MEDIUM"),
    ("Please review the attached proposal when you get a chance", "MEDIUM"),
    ("Monthly performance report attached for your review", "MEDIUM"),
    ("New feature request from client - needs evaluation", "MEDIUM"),
    ("Design review scheduled for next Tuesday", "MEDIUM"),
    ("Following up on our discussion last week", "MEDIUM"),
    ("Project update: milestone 2 completed", "MEDIUM"),
    ("Can we schedule a call this week to discuss?", "MEDIUM"),
    ("Feedback requested on the new onboarding flow", "MEDIUM"),
    ("Q3 retrospective notes - please add your comments", "MEDIUM"),
    ("Partnership proposal - interested in your thoughts", "MEDIUM"),
    ("HR policy update - please read before Friday", "MEDIUM"),
    ("Sprint planning next Monday - agenda attached", "MEDIUM"),
    ("Code review requested for PR #142", "MEDIUM"),
    ("Budget planning spreadsheet needs your input", "MEDIUM"),
    ("Vendor contract renewal coming up next month", "MEDIUM"),
    ("Training session next Wednesday - optional but recommended", "MEDIUM"),
    ("Customer feedback summary for this quarter", "MEDIUM"),
    ("Reminder: performance review next week", "MEDIUM"),
    ("Documentation update needed for API changes", "MEDIUM"),

    # LOW priority
    ("Company newsletter - highlights from this month", "LOW"),
    ("Your weekly digest is ready", "LOW"),
    ("Product update: new features shipped this week", "LOW"),
    ("Thank you for your recent purchase", "LOW"),
    ("Webinar invitation: AI trends in 2025", "LOW"),
    ("Your monthly account statement is available", "LOW"),
    ("Check out what's new in our platform", "LOW"),
    ("Friday fun: office trivia at 4pm", "LOW"),
    ("Blog post: 10 tips for better productivity", "LOW"),
    ("Community update from the team", "LOW"),
    ("Your subscription has been renewed", "LOW"),
    ("Podcast recommendation from a colleague", "LOW"),
    ("Office closed on Monday for public holiday", "LOW"),
    ("Congratulations on your work anniversary!", "LOW"),
    ("New article you might enjoy reading", "LOW"),
    ("Monthly team social - details inside", "LOW"),
    ("Slack tip of the week", "LOW"),
    ("Your profile was viewed on LinkedIn", "LOW"),
    ("Changelog: minor bug fixes and improvements", "LOW"),
    ("Invitation to join our beta program", "LOW"),
]


# ─── Feature engineering ──────────────────────────────────────────────────────

def build_features(texts: list[str]) -> np.ndarray:
    """
    Converts a list of email texts into feature vectors.
    Features = embeddings (384) + handcrafted (6) = 390 dims total.
    The handcrafted features help the model catch obvious signals
    even when embeddings are uncertain.
    """
    embedder = get_embedder()
    embeddings = embedder.encode(texts, show_progress_bar=False)

    extras = []
    urgent_words   = ["urgent", "asap", "immediately", "critical", "emergency",
                      "deadline", "today", "now", "overdue", "final", "last chance"]
    action_words   = ["required", "needed", "must", "approve", "confirm",
                      "respond", "action", "submit", "complete", "review"]
    low_words      = ["newsletter", "digest", "update", "unsubscribe",
                      "monthly", "weekly", "tip", "blog", "invite"]

    for text in texts:
        t = text.lower()
        extras.append([
            sum(w in t for w in urgent_words),          # urgent signal count
            sum(w in t for w in action_words),          # action signal count
            sum(w in t for w in low_words),             # low-priority signal
            len(text.split()),                          # word count
            int("!" in text),                          # exclamation mark
            int(any(c.isupper() for c in text[:20])),  # all-caps opening
        ])

    extras = np.array(extras, dtype=float)
    # Normalize word count to 0–1 range
    if extras[:, 3].max() > 0:
        extras[:, 3] /= extras[:, 3].max()

    return np.hstack([embeddings, extras])


# ─── Training ─────────────────────────────────────────────────────────────────

def train_classifier(data: list[tuple] = None, save: bool = True):
    """
    Trains the priority classifier.

    Args:
        data: list of (text, label) tuples. Defaults to TRAINING_DATA.
        save: whether to save model to disk.

    Returns:
        (classifier, label_encoder, accuracy, report)
    """
    if data is None:
        data = TRAINING_DATA

    texts  = [d[0] for d in data]
    labels = [d[1] for d in data]

    print(f"[Classifier] Training on {len(texts)} samples...")

    # Encode labels
    le = LabelEncoder()
    y  = le.fit_transform(labels)

    # Build features
    X = build_features(texts)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Random Forest — robust, interpretable, great for interviews
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        class_weight="balanced",   # handles imbalanced labels
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred   = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report   = classification_report(y_test, y_pred,
                                     target_names=le.classes_,
                                     zero_division=0)

    # Cross-validation score
    cv_scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy")

    print(f"[Classifier] Test accuracy:  {accuracy:.2%}")
    print(f"[Classifier] CV accuracy:    {cv_scores.mean():.2%} ± {cv_scores.std():.2%}")
    print(f"[Classifier] Report:\n{report}")

    if save:
        with open(CLASSIFIER_PATH, "wb") as f:
            pickle.dump(clf, f)
        with open(ENCODER_PATH, "wb") as f:
            pickle.dump(le, f)
        print(f"[Classifier] Saved to {MODEL_DIR}")

    return clf, le, accuracy, report


def load_classifier():
    """Loads saved classifier, trains if not found."""
    if CLASSIFIER_PATH.exists() and ENCODER_PATH.exists():
        with open(CLASSIFIER_PATH, "rb") as f:
            clf = pickle.load(f)
        with open(ENCODER_PATH, "rb") as f:
            le = pickle.load(f)
        print("[Classifier] Loaded from disk.")
        return clf, le
    else:
        print("[Classifier] No saved model found. Training now...")
        clf, le, _, _ = train_classifier()
        return clf, le


# ─── Prediction ───────────────────────────────────────────────────────────────

_clf_cache = None
_le_cache  = None

def predict_priority(text: str) -> dict:
    """
    Predicts priority for a single email text.

    Returns:
        {
            "priority": "HIGH" | "MEDIUM" | "LOW",
            "confidence": 0.0–1.0,
            "probabilities": {"HIGH": x, "MEDIUM": y, "LOW": z}
        }
    """
    global _clf_cache, _le_cache
    if _clf_cache is None:
        _clf_cache, _le_cache = load_classifier()

    features = build_features([text])
    pred_idx  = _clf_cache.predict(features)[0]
    proba     = _clf_cache.predict_proba(features)[0]

    priority  = _le_cache.inverse_transform([pred_idx])[0]
    confidence = float(proba.max())

    prob_dict = {
        label: float(prob)
        for label, prob in zip(_le_cache.classes_, proba)
    }

    return {
        "priority":      priority,
        "confidence":    confidence,
        "probabilities": prob_dict
    }


def predict_priority_batch(texts: list[str]) -> list[dict]:
    """Batch prediction — much faster than calling predict_priority() in a loop."""
    global _clf_cache, _le_cache
    if _clf_cache is None:
        _clf_cache, _le_cache = load_classifier()

    features  = build_features(texts)
    pred_idxs = _clf_cache.predict(features)
    probas    = _clf_cache.predict_proba(features)

    results = []
    for pred_idx, proba in zip(pred_idxs, probas):
        priority   = _le_cache.inverse_transform([pred_idx])[0]
        confidence = float(proba.max())
        prob_dict  = {
            label: float(p)
            for label, p in zip(_le_cache.classes_, proba)
        }
        results.append({
            "priority":      priority,
            "confidence":    confidence,
            "probabilities": prob_dict
        })
    return results


# ─── Retrain on real emails ────────────────────────────────────────────────────

def train_on_real_emails(labeled_emails: list[dict]):
    """
    Retrain classifier on your real labeled emails from Supabase.

    Args:
        labeled_emails: list of {"subject": str, "summary": str, "priority": str}

    Example:
        from services.ml_classifier import train_on_real_emails
        train_on_real_emails(emails_from_db)
    """
    data = []
    for email in labeled_emails:
        text  = f"{email.get('subject', '')} {email.get('summary', '')}"
        label = email.get("priority", "LOW")
        if label in ("HIGH", "MEDIUM", "LOW"):
            data.append((text, label))

    # Combine with seed data so model doesn't overfit to small real dataset
    combined = TRAINING_DATA + data
    print(f"[Classifier] Retraining on {len(combined)} samples "
          f"({len(data)} real + {len(TRAINING_DATA)} seed)...")
    train_classifier(combined)


# ─── Entry point for standalone training ─────────────────────────────────────
if __name__ == "__main__":
    clf, le, acc, report = train_classifier()
    print(f"\nDone. Classes: {list(le.classes_)}")
    print(f"\nTest prediction:")
    result = predict_priority("URGENT: Please approve the invoice before EOD today")
    print(result)
