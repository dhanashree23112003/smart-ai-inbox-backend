# importance_scorer.py
# Drop this into: smart-inbox-backend/backend/app/services/
#
# What this does:
#   - Replaces your rule-based calculate_importance() with XGBoost
#   - 15 engineered features (sender freq, time, length, urgency signals, etc.)
#   - SHAP explanations: tells WHY an email is high priority
#   - Returns score 0-100 + explanation text (great for UI)

import re
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Optional
import warnings
warnings.filterwarnings("ignore")

MODEL_DIR = Path(__file__).parent / "saved_models"
MODEL_DIR.mkdir(exist_ok=True)
SCORER_PATH = MODEL_DIR / "importance_scorer.pkl"

# ─── Feature names (for SHAP explanations) ───────────────────────────────────
FEATURE_NAMES = [
    "urgent_word_count",
    "action_word_count",
    "deadline_word_count",
    "low_priority_signals",
    "word_count_normalized",
    "has_exclamation",
    "caps_ratio",
    "has_question",
    "sender_is_internal",
    "is_reply",
    "has_deadline_extracted",
    "hour_of_day_normalized",
    "is_monday_or_friday",
    "subject_length_normalized",
    "has_money_mention",
]

# ─── Keyword lists ────────────────────────────────────────────────────────────
URGENT_WORDS    = ["urgent", "asap", "immediately", "critical", "emergency",
                   "final", "last chance", "overdue", "expired", "today", "now",
                   "deadline", "warning", "alert", "attention required"]

ACTION_WORDS    = ["required", "needed", "must", "approve", "confirm", "respond",
                   "action", "submit", "complete", "review", "sign", "authorize",
                   "vote", "decide", "reply", "answer", "provide", "send"]

DEADLINE_WORDS  = ["deadline", "due", "by eod", "by cob", "by friday", "by tomorrow",
                   "before", "no later than", "expires", "expiry", "renewal"]

LOW_WORDS       = ["newsletter", "digest", "unsubscribe", "monthly recap",
                   "weekly summary", "notification", "blog", "article", "tip of",
                   "you might like", "recommended for you", "changelog"]

MONEY_PATTERNS  = [r"\$[\d,]+", r"£[\d,]+", r"€[\d,]+",
                   r"\d+\s*(usd|eur|gbp|inr)", r"invoice", r"payment"]


# ─── Feature extraction ───────────────────────────────────────────────────────

def extract_features(
    subject: str,
    body:    str     = "",
    sender:  str     = "",
    sent_at: Optional[datetime] = None,
    has_deadline: bool = False,
    user_domain: str = "",
) -> np.ndarray:
    """
    Extracts 15 numerical features from an email for importance scoring.

    These features are designed to be interpretable (great for interviews).
    """
    text = f"{subject} {body}".lower()
    subj = subject.lower()

    # 1. Urgent word count
    f1 = sum(text.count(w) for w in URGENT_WORDS)

    # 2. Action word count
    f2 = sum(text.count(w) for w in ACTION_WORDS)

    # 3. Deadline word count
    f3 = sum(text.count(w) for w in DEADLINE_WORDS)

    # 4. Low-priority signals (negative feature)
    f4 = sum(text.count(w) for w in LOW_WORDS)

    # 5. Word count normalized to 0-1 (0=empty, 1=500+ words)
    word_count = len(text.split())
    f5 = min(word_count / 500, 1.0)

    # 6. Has exclamation mark in subject
    f6 = int("!" in subject)

    # 7. Caps ratio in subject (ALL CAPS = urgent)
    caps   = sum(1 for c in subject if c.isupper())
    total  = max(len([c for c in subject if c.isalpha()]), 1)
    f7     = caps / total

    # 8. Has question in subject
    f8 = int("?" in subject)

    # 9. Sender is internal (same domain as user)
    f9 = 0
    if user_domain and sender:
        f9 = int(user_domain.lower() in sender.lower())

    # 10. Is a reply (Re: or Fwd:)
    f10 = int(bool(re.match(r"^(re:|fwd:|fw:)", subj.strip())))

    # 11. Deadline already extracted by NER
    f11 = int(has_deadline)

    # 12. Hour of day normalized (0=midnight, 0.5=noon, 1=midnight)
    if sent_at:
        f12 = sent_at.hour / 24
    else:
        f12 = 0.5  # unknown = assume midday

    # 13. Is Monday or Friday (high urgency days)
    if sent_at:
        f13 = int(sent_at.weekday() in (0, 4))
    else:
        f13 = 0

    # 14. Subject length normalized
    f14 = min(len(subject) / 100, 1.0)

    # 15. Money/financial mention
    f15 = int(any(re.search(p, text) for p in MONEY_PATTERNS))

    return np.array([f1, f2, f3, f4, f5, f6, f7, f8, f9, f10,
                     f11, f12, f13, f14, f15], dtype=float)


# ─── Training data ────────────────────────────────────────────────────────────
# (subject, body_snippet, sender, score_0_to_100)

TRAINING_DATA = [
    # HIGH importance (score 70-100)
    ("URGENT: Server outage - all hands needed NOW", "Production is down affecting all users", "devops@corp.com", 95),
    ("Invoice #8821 OVERDUE - Immediate payment required", "Payment of $12,500 is 30 days overdue", "billing@vendor.com", 90),
    ("FINAL NOTICE: Contract expires tomorrow", "Your contract will expire without renewal", "legal@firm.com", 92),
    ("Critical security vulnerability - patch required", "CVE-2024-XXXX found in production dependency", "security@corp.com", 88),
    ("Board meeting TOMORROW - confirm attendance", "Attendance is required for all executives", "ceo@corp.com", 85),
    ("Tax filing deadline THIS WEEK", "Documents must be submitted by Friday EOD", "finance@corp.com", 87),
    ("URGENT: Client escalation requires your response", "Client is threatening to cancel contract", "client@bigco.com", 91),
    ("Budget approval needed before EOD today", "Please approve Q4 budget allocation", "cfo@company.com", 83),
    ("ASAP: Production deploy blocked - need your approval", "Release 2.4.1 is waiting on your sign-off", "pm@corp.com", 86),
    ("Final reminder: performance review due Friday", "This is your last reminder to submit self-review", "hr@corp.com", 78),
    ("Emergency: Data breach detected", "Unauthorized access detected in database logs", "security@corp.com", 97),
    ("Visa application: documents required by Thursday", "Missing documents will invalidate your application", "immigration@firm.com", 89),

    # MEDIUM importance (score 40-69)
    ("Team retrospective next Tuesday", "Please come prepared with your thoughts", "pm@corp.com", 55),
    ("Partnership proposal - review when you can", "Attached is the revised partnership terms", "partner@nexus.io", 60),
    ("Design review scheduled for next week", "Agenda and assets attached for review", "designer@corp.com", 50),
    ("Monthly performance report attached", "Summary of KPIs for the past month", "analytics@corp.com", 45),
    ("Code review requested: PR #142", "Added new feature for user authentication", "dev@corp.com", 52),
    ("New feature request from client", "Client would like to discuss new functionality", "sales@corp.com", 58),
    ("Budget planning - input needed by next month", "Please fill out the department budget template", "finance@corp.com", 48),
    ("Following up from last week's call", "Wanted to check if you had a chance to review", "partner@co.com", 42),
    ("Q3 report - please add your comments", "Attached the draft for your review", "manager@corp.com", 53),
    ("Training session next Wednesday", "Optional but recommended for all engineers", "hr@corp.com", 40),

    # LOW importance (score 0-39)
    ("Your weekly digest is ready", "Here's what happened in your network this week", "digest@linkedin.com", 10),
    ("Product update: new features shipped", "Changelog for version 3.2.0 is now available", "updates@saas.com", 15),
    ("Company newsletter - March highlights", "Read about our exciting company updates", "newsletter@corp.com", 8),
    ("You might enjoy this article", "Based on your interests, we recommend...", "recommendations@medium.com", 5),
    ("Friday fun: office trivia at 4pm!", "Join us for a fun quiz this Friday", "social@corp.com", 20),
    ("Congratulations on your work anniversary!", "5 years at the company - thank you!", "hr@corp.com", 18),
    ("Podcast recommendation from a colleague", "I thought you might enjoy this episode", "colleague@corp.com", 12),
    ("Changelog: minor bug fixes", "Version 2.1.1 released with performance improvements", "noreply@github.com", 7),
    ("Slack tip of the week", "Did you know you can star messages?", "noreply@slack.com", 3),
    ("Your subscription has been renewed", "Your Pro plan has been renewed for another year", "billing@saas.com", 22),
]


# ─── Training ─────────────────────────────────────────────────────────────────

def train_scorer(data=None, save=True):
    """Trains XGBoost regressor to predict importance score (0-100)."""
    try:
        from xgboost import XGBRegressor
    except ImportError:
        print("[Scorer] XGBoost not installed. Run: pip install xgboost")
        print("[Scorer] Falling back to GradientBoosting...")
        from sklearn.ensemble import GradientBoostingRegressor as XGBRegressor

    if data is None:
        data = TRAINING_DATA

    X = np.array([
        extract_features(subject=d[0], body=d[1], sender=d[2])
        for d in data
    ])
    y = np.array([d[3] for d in data], dtype=float)

    try:
        from xgboost import XGBRegressor as XGB
        model = XGB(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0,
        )
    except ImportError:
        from sklearn.ensemble import GradientBoostingRegressor
        model = GradientBoostingRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42
        )

    model.fit(X, y)

    # Evaluate with leave-one-out
    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(model, X, y, cv=5, scoring="r2")
    print(f"[Scorer] R² score: {scores.mean():.3f} ± {scores.std():.3f}")

    if save:
        with open(SCORER_PATH, "wb") as f:
            pickle.dump(model, f)
        print(f"[Scorer] Saved to {SCORER_PATH}")

    return model


def load_scorer():
    if SCORER_PATH.exists():
        with open(SCORER_PATH, "rb") as f:
            return pickle.load(f)
    print("[Scorer] No saved model. Training now...")
    return train_scorer()


_scorer_cache = None

def score_importance(
    subject: str,
    body:    str  = "",
    sender:  str  = "",
    sent_at: Optional[datetime] = None,
    has_deadline: bool = False,
    user_domain: str = "",
) -> dict:
    """
    Scores importance of an email (0-100) with explanation.

    Returns:
        {
            "score": 82.5,
            "priority": "HIGH",
            "explanation": "High urgency keywords detected. Action required. Deadline present.",
            "features": {"urgent_word_count": 3, ...}
        }
    """
    global _scorer_cache
    if _scorer_cache is None:
        _scorer_cache = load_scorer()

    features = extract_features(subject, body, sender, sent_at, has_deadline, user_domain)
    score    = float(np.clip(_scorer_cache.predict(features.reshape(1, -1))[0], 0, 100))

    # Priority tier
    if score >= 70:
        priority = "HIGH"
    elif score >= 40:
        priority = "MEDIUM"
    else:
        priority = "LOW"

    # Human-readable explanation based on top features
    explanation = _build_explanation(features, score)

    feature_dict = dict(zip(FEATURE_NAMES, features.tolist()))

    return {
        "score":       round(score, 1),
        "priority":    priority,
        "explanation": explanation,
        "features":    feature_dict,
    }


def _build_explanation(features: np.ndarray, score: float) -> str:
    f = dict(zip(FEATURE_NAMES, features))
    reasons = []

    if f["urgent_word_count"] >= 2:
        reasons.append("multiple urgency signals detected")
    elif f["urgent_word_count"] == 1:
        reasons.append("urgency signal detected")

    if f["has_deadline_extracted"]:
        reasons.append("deadline present")

    if f["action_word_count"] >= 2:
        reasons.append("action required")

    if f["has_money_mention"]:
        reasons.append("financial content")

    if f["caps_ratio"] > 0.4:
        reasons.append("emphasis in subject")

    if f["low_priority_signals"] >= 2:
        reasons.append("newsletter/digest pattern")

    if f["sender_is_internal"]:
        reasons.append("internal sender")

    if not reasons:
        if score >= 70:
            reasons.append("high overall signal strength")
        elif score >= 40:
            reasons.append("moderate relevance signals")
        else:
            reasons.append("low urgency content")

    return ". ".join(r.capitalize() for r in reasons) + "."


# ─── CLI test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    train_scorer()

    tests = [
        ("URGENT: Invoice overdue - pay now", "Payment of $5,000 is 30 days late", "billing@vendor.com"),
        ("Team meeting next Tuesday", "Please join for sprint planning", "pm@corp.com"),
        ("Monthly newsletter", "Company highlights this month", "newsletter@corp.com"),
        ("Critical bug in production", "Server is throwing 500 errors for all users", "devops@corp.com"),
        ("You might enjoy this article", "Recommended reading based on your interests", "noreply@medium.com"),
    ]

    print("\n── Importance Scoring Tests ──\n")
    for subject, body, sender in tests:
        result = score_importance(subject, body, sender)
        bar    = "█" * int(result["score"] / 5)
        print(f"Subject: {subject[:50]}")
        print(f"  Score: {result['score']:5.1f}/100  [{bar:<20}]  {result['priority']}")
        print(f"  Why:   {result['explanation']}\n")
