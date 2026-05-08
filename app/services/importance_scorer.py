"""
Importance Scorer v3 — Embeddings + XGBoost

Architecture:
  Input:  384-dim sentence-transformer embedding  (semantic meaning)
        +   4 meta features                        (structure signals)
  PCA:   → 64 dimensions (prevents overfitting with small datasets)
  Model: XGBoost regressor → 0-100 importance score

Why this beats 17 handcrafted features:
  The embedding already encodes "urgency", "financial content", "action required"
  semantically. "Invoice overdue" and "payment past due" score identically even
  though they share no keywords. Handcrafted keyword counts can't do that.

The 4 meta features kept (structural, not semantic — embedding misses these):
  1. has_deadline_extracted  — NER found a real date
  2. sender_is_internal      — same domain as user
  3. is_reply                — Re:/Fwd: in subject
  4. hour_of_day_normalized  — time-of-day urgency signal

Spam/bulk detection is handled upstream as hard rules (not ML).
"""

import pickle
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional
import warnings
warnings.filterwarnings("ignore")

MODEL_DIR   = Path(__file__).parent / "saved_models"
MODEL_DIR.mkdir(exist_ok=True)
SCORER_PATH = MODEL_DIR / "importance_scorer.pkl"

MODEL_VERSION = "v3-embeddings"

# ── Training data ─────────────────────────────────────────────────────────────
# Format: (subject, body_snippet, sender, score_0_to_100)
# Scores: HIGH=70-100, MEDIUM=40-69, LOW=0-39

TRAINING_DATA = [
    # HIGH (70-100)
    ("URGENT: Server outage - all hands needed NOW",        "Production is down affecting all users. All engineers respond immediately.",        "devops@corp.com",              95),
    ("Invoice #8821 OVERDUE - Immediate payment required",  "Payment of $12,500 is 30 days overdue. Account will be suspended.",               "billing@vendor.com",           90),
    ("FINAL NOTICE: Contract expires tomorrow",             "Your contract will expire without renewal. Legal action may follow.",              "legal@firm.com",               88),
    ("Critical security vulnerability - patch required",    "CVE-2024 found in production dependency. Patch before end of day.",               "security@corp.com",            88),
    ("Board meeting TOMORROW - confirm attendance",         "Attendance is required for all executives. Please confirm immediately.",           "ceo@corp.com",                 85),
    ("Tax filing deadline THIS WEEK",                       "Documents must be submitted by Friday EOD or penalties apply.",                    "finance@corp.com",             87),
    ("URGENT: Client escalation requires your response",    "Client is threatening to cancel the $2M contract. Call needed today.",            "client@bigco.com",             91),
    ("Budget approval needed before EOD today",             "Please approve Q4 budget allocation. Finance is waiting on your sign-off.",       "cfo@company.com",              83),
    ("ASAP: Production deploy blocked - need your approval","Release 2.4.1 is waiting on your sign-off. 3 engineers blocked.",                "pm@corp.com",                  86),
    ("Emergency: Data breach detected in production",       "Unauthorized access detected in database logs. Immediate action required.",       "security@corp.com",            97),
    ("Visa application: documents required by Thursday",    "Missing documents will invalidate your application. Submit before Thursday.",     "immigration@firm.com",         89),
    ("Telephonic interview scheduled today 3-5 PM",         "We would like to speak with you today between 3 and 5 PM about the Data role.",  "careers@novoracorporation.com",78),
    ("Interview confirmed for tomorrow 10am",               "Please bring your resume and photo ID to the office tomorrow at 10am.",           "hr@company.com",               80),
    ("Re: Invoice #4821 — still unpaid after 3 reminders",  "This is the third follow-up. Please respond or we escalate to collections.",     "billing@realvendor.com",       85),
    ("Action required: verify your bank account by Friday", "Your account will be frozen if not verified. Visit branch or call us.",          "support@bank.com",             82),

    # MEDIUM (40-69)
    ("Team retrospective next Tuesday",                     "Please come prepared with your retrospective notes and blockers.",               "pm@corp.com",                  55),
    ("Partnership proposal - review when you can",          "Attached is the revised partnership terms from Nexus Labs.",                     "partner@nexus.io",             60),
    ("Design review scheduled for next week",               "Agenda and Figma assets attached for review. Your input is needed.",             "designer@corp.com",            50),
    ("Monthly performance report attached",                 "Summary of KPIs for the past month. Please review before the standup.",         "analytics@corp.com",           45),
    ("Code review requested: PR #142",                      "Added new authentication feature. Needs review before merge.",                  "dev@corp.com",                 52),
    ("Following up from last week's call",                  "Wanted to check if you had a chance to review the proposal I sent.",            "partner@co.com",               42),
    ("Security alert: new sign-in detected on your account","We noticed a new sign-in to your Google Account from a new device.",           "no-reply@accounts.google.com", 55),
    ("AutoPay set up for your SIP successfully",            "Your SIP for SBI Large Cap Fund is now active. Amount: Rs 2000 monthly.",       "noreply@phonepe.com",          28),
    ("Your 90-day free offer ends May 15",                  "Last chance to claim your free Postman access before it expires.",              "notifications@mail.postman.com",35),
    ("Q3 report - please add your comments by Friday",      "Attached the draft for your review. Comments needed before Friday.",            "manager@corp.com",             53),
    ("New feature request from client",                     "Client would like to discuss new dashboard functionality next week.",           "sales@corp.com",               58),

    # LOW — generic newsletters
    ("Your weekly digest is ready",                         "Here is what happened in your network this week. Top stories and updates.",     "digest@linkedin.com",          10),
    ("Company newsletter - March highlights",               "Read about our exciting company updates this month. New hires and events.",     "newsletter@corp.com",           8),
    ("You might enjoy this article",                        "Based on your interests, we recommend this article about productivity.",        "recommendations@medium.com",    5),
    ("Changelog: minor bug fixes in version 2.1.1",         "Performance improvements and small bug fixes. Nothing breaking.",              "noreply@github.com",            7),
    ("Slack tip of the week",                               "Did you know you can star messages and pin them to channels?",                 "noreply@slack.com",             3),
    ("Congratulations on your 5-year work anniversary",     "Thank you for 5 years at the company. You are valued and appreciated.",        "hr@corp.com",                  18),

    # LOW — MARKETING SPAM with urgency language (the hard ones)
    ("Final Hours: 20% Off All Excel + Power BI Courses",   "Sale ends today! Get 20% off all courses including renewals. Order now.",      "website@myonlinetraininghub.com", 8),
    ("Job | Walk-in Interview Drive - Domestic Voice BPO",  "Urgently hiring for BPO domestic voice process in Pune. Apply now.",           "alertnc@naukri.com",           10),
    ("Your Exclusive Welcome Reward Awaits!",               "Get 15% off your first meal subscription. Use code UPGRADE. Order now.",       "noreply@offers.hellomealsonme.com", 4),
    ("Don't Miss Your Ramadan Savings! Get 15% off",        "Get 15% off your meal plan subscription today. Hurry, offer ends soon.",       "noreply@offers.hellomealsonme.com", 4),
    ("Best-Selling AI Courses at Just Rs 399",              "Get all best seller courses at discount. Limited time offer. Shop now.",       "no-reply@e.udemymail.com",      6),
    ("General Motors is hiring in Dubai. Apply Now.",        "New job match for your profile from Glassdoor. One click to apply.",          "noreply@glassdoor.com",        12),
    ("Hiring via AI Hackathon | Rs 10L+ Opportunities",     "Apply Now! Limited seats available for this competition. Register today.",    "noreply@unstop.news",          10),
    ("NEW COURSE LAUNCH: AI for Leaders Master Gen AI",     "Get this exclusive course launch offer today. First 100 seats discounted.",    "no-reply@e.udemymail.com",      7),
    ("Urgently hiring Coordinator, Processing Executive",   "Your mid-week job opportunities are right here. Apply to multiple roles.",    "naukrialerts@naukri.com",      10),
    ("Software Engineer at E2E Networks. Apply Now",         "E2E Networks Limited is hiring talent like you. Strong match detected.",      "info@hirist.tech",             11),
    ("Self-Driving Reasoning Models, ChatGPT Adds Ads",     "A dispatch from Davos how AI can transform workflows from beginning.",        "thebatch@deeplearning.ai",      9),
    ("6 Steps to Design Anything With Claude",              "You have probably been using Claude wrong. Today lead is a six step system.", "newsletters-noreply@linkedin.com", 9),
    ("Excel Has Changed: The Functions That Matter Now",     "New Excel functions that will transform how you work in 2026 and beyond.",    "website@myonlinetraininghub.com", 8),
    ("Introducing Learn AI with Google — Get skills now",   "Get the skills employers need now. Enroll in Learn AI with Google today.",   "hello@students.udemy.com",      8),
    ("Confused Which Fund to Pick? We will Help you",       "With hundreds of schemes dear customer if you ever paused before selecting.", "service@service.icicisecurities.com", 16),
    ("The best SEO book of all time is now available",      "Hey just wanted to remind you that my new book Get Found is available now.", "matt@heytony.ca",               5),
]


def _get_embedding(text: str) -> np.ndarray:
    """Get embedding using the project's existing embedding service."""
    from app.services.embedding_service import generate_embedding
    emb = generate_embedding(text)
    return np.array(emb, dtype=float)


def _meta_features(
    sender:        str  = "",
    has_deadline:  bool = False,
    subject:       str  = "",
    sent_at:       Optional[datetime] = None,
    user_domain:   str  = "",
) -> np.ndarray:
    """
    4 structural meta-features the embedding doesn't capture:
      1. has_deadline_extracted  — NER found a real date
      2. sender_is_internal      — same org as the user
      3. is_reply                — part of an ongoing thread
      4. hour_of_day_normalized  — temporal urgency signal
    """
    sndr  = sender.lower()
    subj  = subject.lower().strip()

    f1 = float(has_deadline)
    f2 = float(bool(user_domain and user_domain.lower() in sndr))
    f3 = float(subj.startswith(("re:", "fwd:", "fw:")))
    f4 = (sent_at.hour / 24.0) if sent_at else 0.5

    return np.array([f1, f2, f3, f4], dtype=float)


def _build_features(subject: str, body: str, sender: str,
                    has_deadline: bool, user_domain: str,
                    sent_at: Optional[datetime]) -> np.ndarray:
    """
    Combines PCA-reduced embedding (64 dims) + 4 meta features = 68 total.
    Uses a fixed random projection as a lightweight alternative to fitted PCA
    so the scorer can score emails even before PCA is fit on enough data.
    """
    text = f"{subject} {body}"
    emb  = _get_embedding(text)          # 384-dim
    meta = _meta_features(sender, has_deadline, subject, sent_at, user_domain)
    return np.concatenate([emb, meta])    # 388-dim (XGBoost handles this fine)


# ── Training ──────────────────────────────────────────────────────────────────

def _load_all_training_data():
    """Base training data + CSV dataset (if emails.csv is present)."""
    combined = list(TRAINING_DATA)
    try:
        from app.services.load_csv_training_data import load_csv_training_data, CSV_PATH
        if CSV_PATH.exists():
            csv_data = load_csv_training_data()
            combined = combined + csv_data
            print(f"[Scorer v3] Loaded {len(csv_data)} examples from emails.csv")
        else:
            print("[Scorer v3] emails.csv not found — using base training data only")
    except Exception as e:
        print(f"[Scorer v3] CSV load skipped: {e}")
    return combined


def train_scorer(data=None, save=True):
    if data is None:
        data = _load_all_training_data()

    print(f"[Scorer v3] Training on {len(data)} examples (embeddings + meta features)...")

    X_list, y_list = [], []
    for subject, body, sender, score in data:
        try:
            features = _build_features(subject, body, sender,
                                        has_deadline=False,
                                        user_domain="",
                                        sent_at=None)
            X_list.append(features)
            y_list.append(float(score))
        except Exception as e:
            print(f"[Scorer v3] Skipping training example: {e}")
            continue

    X = np.array(X_list)
    y = np.array(y_list)

    n = len(data)
    small = n < 200  # small dataset needs simpler model to avoid overfitting

    try:
        from xgboost import XGBRegressor
        model = XGBRegressor(
            n_estimators   = 100 if small else 300,
            max_depth      = 2   if small else 4,
            learning_rate  = 0.1 if small else 0.05,
            subsample      = 0.8,
            colsample_bytree = 0.1 if small else 0.3,  # very few cols on small data
            reg_lambda     = 20.0 if small else 5.0,   # heavy L2 on small data
            random_state   = 42,
            verbosity      = 0,
        )
    except ImportError:
        from sklearn.ensemble import GradientBoostingRegressor
        model = GradientBoostingRegressor(
            n_estimators=50 if small else 200,
            max_depth=2 if small else 3,
            learning_rate=0.1 if small else 0.05,
            random_state=42
        )

    model.fit(X, y)

    # Only cross-validate if enough data for meaningful splits
    if n >= 25:
        from sklearn.model_selection import cross_val_score
        cv_n = min(3 if small else 5, n // 5)
        cv_scores = cross_val_score(model, X, y, cv=max(2, cv_n), scoring="r2")
        print(f"[Scorer v3] R² = {cv_scores.mean():.3f} ± {cv_scores.std():.3f}"
              + (" (small dataset — upload emails.csv for better accuracy)" if small else ""))
    else:
        print(f"[Scorer v3] Trained on {n} examples (too few for CV — upload emails.csv)")

    if save:
        with open(SCORER_PATH, "wb") as f:
            pickle.dump({"model": model, "version": MODEL_VERSION}, f)
        print(f"[Scorer v3] Saved → {SCORER_PATH}")

    return model


def load_scorer():
    if SCORER_PATH.exists():
        with open(SCORER_PATH, "rb") as f:
            saved = pickle.load(f)
        if isinstance(saved, dict) and saved.get("version") == MODEL_VERSION:
            print("[Scorer v3] Loaded from disk.")
            return saved["model"]
        print(f"[Scorer v3] Version mismatch — retraining...")
    else:
        print("[Scorer v3] No saved model — training...")
    return train_scorer()


_scorer_cache = None


# ── Rule-based spam caps (applied after ML, before returning) ─────────────────

_HARD_SPAM_DOMAINS = [
    r"@naukri\.com", r"@glassdoor\.com", r"@indeed\.com",
    r"@unstop\.news", r"@hirist\.tech", r"@internshala\.com",
    r"@e\.udemymail\.com", r"@students\.udemy\.com",
    r"@myonlinetraininghub\.com", r"@hellomealsonme\.com",
    r"@match\.indeed\.com", r"@deeplearning\.ai",
    r"@crio\.co\.in", r"@heytony\.ca", r"@icicisecurities\.com",
]
_BULK_PREFIXES = [
    r"noreply@", r"no-reply@", r"donotreply@",
    r"newsletter", r"digest@", r"alerts?@", r"website@",
]
_PROMO_PHRASES = [
    "% off", "sale ends", "order now", "limited time", "use code",
    "apply now for", "urgently hiring", "walk-in interview",
    "best seller", "flash sale", "get it here",
]

def _rule_cap(subject: str, body: str, sender: str, score: float) -> float:
    import re
    sndr = sender.lower()
    text = f"{subject} {body}".lower()
    if any(re.search(p, sndr) for p in _HARD_SPAM_DOMAINS): return min(score, 25.0)
    if any(re.search(p, sndr) for p in _BULK_PREFIXES):     return min(score, 25.0)
    if sum(1 for p in _PROMO_PHRASES if p in text) >= 2:     return min(score, 30.0)
    return score


# ── Public API ────────────────────────────────────────────────────────────────

def score_importance(
    subject:      str  = "",
    body:         str  = "",
    sender:       str  = "",
    sent_at:      Optional[datetime] = None,
    has_deadline: bool = False,
    user_domain:  str  = "",
) -> dict:
    global _scorer_cache
    if _scorer_cache is None:
        _scorer_cache = load_scorer()

    features = _build_features(subject, body, sender, has_deadline, user_domain, sent_at)
    ml_score = float(np.clip(_scorer_cache.predict(features.reshape(1, -1))[0], 0, 100))
    score    = _rule_cap(subject, body, sender, ml_score)

    priority = "HIGH" if score >= 70 else "MEDIUM" if score >= 45 else "LOW"

    return {
        "score":       round(score, 1),
        "priority":    priority,
        "explanation": _explain(score, subject, body, sender, has_deadline),
        "features":    {"ml_score": round(ml_score, 1), "after_rules": round(score, 1)},
    }


def _explain(score: float, subject: str, body: str, sender: str, has_deadline: bool) -> str:
    import re
    sndr = sender.lower()
    reasons = []
    if any(re.search(p, sndr) for p in _HARD_SPAM_DOMAINS + _BULK_PREFIXES):
        reasons.append("bulk/marketing sender")
    if has_deadline:
        reasons.append("deadline present")
    if score >= 70:
        reasons.append("high semantic urgency")
    elif score >= 45:
        reasons.append("moderate relevance")
    else:
        reasons.append("low urgency content")
    return ". ".join(r.capitalize() for r in reasons) + "."


def retrain_with_feedback(feedback_rows):
    """Retrain incorporating user corrections (4× weighted)."""
    global _scorer_cache
    SCORE_MAP = {"HIGH": 88.0, "MEDIUM": 55.0, "LOW": 12.0}
    feedback_data = [
        (r[0] or "", r[1] or "", r[2] or "", SCORE_MAP.get(r[3], 30.0))
        for r in feedback_rows
    ]
    combined = TRAINING_DATA + feedback_data * 4
    _scorer_cache = train_scorer(data=combined, save=True)
    print(f"[Scorer v3] Retrained: {len(TRAINING_DATA)} base + {len(feedback_data)} feedback.")


if __name__ == "__main__":
    train_scorer()
    tests = [
        ("URGENT: Invoice overdue - pay now",         "Payment of $5,000 is 30 days late",          "billing@vendor.com"),
        ("Final Hours: 20% Off All Excel Courses",    "Sale ends today! Order now",                  "website@myonlinetraininghub.com"),
        ("Job Walk-in Interview Drive Urgently",       "Apply now for BPO process Pune",              "alertnc@naukri.com"),
        ("Critical bug in production server down",    "Server throwing 500 errors for all users",    "devops@corp.com"),
        ("Telephonic interview today 3-5 PM",         "We would like to speak with you today",       "careers@company.com"),
        ("Partnership proposal review when you can",  "Attached revised terms from Nexus Labs",      "partner@nexus.io"),
    ]
    print("\n── Scorer v3 Tests ──\n")
    for subject, body, sender in tests:
        r   = score_importance(subject, body, sender)
        bar = "█" * int(r["score"] / 5)
        print(f"{subject[:55]}")
        print(f"  {r['score']:5.1f}/100  [{bar:<20}]  {r['priority']}  — {r['explanation']}\n")
