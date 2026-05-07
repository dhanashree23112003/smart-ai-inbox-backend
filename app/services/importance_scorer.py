import re
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

# Bump this when features change — forces retrain instead of loading stale model
MODEL_VERSION = "v2"

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
    "marketing_language_count",   # NEW v2
    "spam_sender_score",          # NEW v2
]

# ── Keyword lists ─────────────────────────────────────────────────────────────

# Real urgency — specific, personal, professional
URGENT_WORDS   = ["overdue", "expired", "critical", "emergency", "immediately",
                  "asap", "breach", "outage", "blocked", "escalation",
                  "warning", "attention required", "action required"]

ACTION_WORDS   = ["required", "needed", "must", "approve", "confirm", "respond",
                  "submit", "complete", "review", "sign", "authorize",
                  "decide", "reply", "provide", "send", "verify", "validate"]

DEADLINE_WORDS = ["deadline", "due", "by eod", "by cob", "by friday",
                  "by tomorrow", "before", "no later than", "expires today",
                  "last day", "final deadline"]

LOW_WORDS      = ["newsletter", "digest", "unsubscribe", "monthly recap",
                  "weekly summary", "notification", "blog post", "tip of the week",
                  "you might like", "recommended for you", "changelog",
                  "highlights", "roundup", "curated", "top stories",
                  "your weekly", "your monthly", "stay tuned"]

# Promotional/marketing language — the main spam signal
MARKETING_WORDS = ["% off", "percent off", "discount", "sale ends", "sale now",
                   "shop now", "order now", "buy now", "limited time", "exclusive offer",
                   "special offer", "deal", "coupon", "promo", "best seller",
                   "free access", "save now", "upgrade now", "don't miss out",
                   "last chance to save", "offer ends", "apply now", "hiring now",
                   "urgently hiring", "walk-in", "walk in interview",
                   "opportunities for you", "jobs for you", "new job match",
                   "your subscription", "subscription renewal", "auto-renewal",
                   "your account has been", "welcome reward", "reward awaits",
                   "ramadan", "festive offer", "flash sale"]

# Bulk mailer sender patterns — email addresses that are almost never personal
BULK_SENDER_PATTERNS = [
    r"noreply@", r"no-reply@", r"donotreply@",
    r"alerts@", r"alertnc@", r"newsletters-noreply@",
    r"@naukri\.com", r"@glassdoor\.com",
    r"@unstop\.news", r"@hirist\.tech",
    r"@e\.udemymail\.com", r"@students\.udemy\.com",
    r"@mail\.internshala\.com",
    r"@myonlinetraininghub\.com",
    r"@hellomealsonme\.com",
    r"@match\.indeed\.com",
    r"@deeplearning\.ai",
    r"newsletter", r"digest@", r"promo@",
    r"marketing@", r"offers@", r"deals@",
]

MONEY_PATTERNS = [r"\$[\d,]+", r"£[\d,]+", r"€[\d,]+",
                  r"₹[\d,]+", r"\d+\s*(usd|eur|gbp|inr)",
                  r"\binvoice\b", r"\bpayment\b", r"\boverdue\b"]


# ── Feature extraction ────────────────────────────────────────────────────────

def extract_features(
    subject: str,
    body:    str     = "",
    sender:  str     = "",
    sent_at: Optional[datetime] = None,
    has_deadline: bool = False,
    user_domain: str = "",
) -> np.ndarray:

    text  = f"{subject} {body}".lower()
    subj  = subject.lower()
    sndr  = sender.lower()

    # 1. Urgent word count (real urgency only)
    f1 = sum(text.count(w) for w in URGENT_WORDS)

    # 2. Action word count
    f2 = sum(text.count(w) for w in ACTION_WORDS)

    # 3. Deadline word count
    f3 = sum(text.count(w) for w in DEADLINE_WORDS)

    # 4. Low-priority signals (negative)
    f4 = sum(text.count(w) for w in LOW_WORDS)

    # 5. Word count normalized
    f5 = min(len(text.split()) / 500, 1.0)

    # 6. Exclamation mark in subject
    f6 = int("!" in subject)

    # 7. Caps ratio in subject
    caps  = sum(1 for c in subject if c.isupper())
    total = max(len([c for c in subject if c.isalpha()]), 1)
    f7    = caps / total

    # 8. Question mark in subject
    f8 = int("?" in subject)

    # 9. Internal sender (same domain)
    f9 = 0
    if user_domain and sender:
        f9 = int(user_domain.lower() in sndr)

    # 10. Is a reply/forward
    f10 = int(bool(re.match(r"^(re:|fwd:|fw:)", subj.strip())))

    # 11. Deadline extracted by NER
    f11 = int(has_deadline)

    # 12. Hour of day normalized
    f12 = sent_at.hour / 24 if sent_at else 0.5

    # 13. Monday or Friday
    f13 = int(sent_at.weekday() in (0, 4)) if sent_at else 0

    # 14. Subject length normalized
    f14 = min(len(subject) / 100, 1.0)

    # 15. Financial mention (real invoices/payments, not discounts)
    # Only count if NOT also a marketing email (avoids "20% off ₹399" false positives)
    marketing_hit = sum(1 for w in MARKETING_WORDS if w in text)
    f15 = int(any(re.search(p, text) for p in MONEY_PATTERNS)) if marketing_hit == 0 else 0

    # 16. Marketing / promotional language count (strong spam signal)
    f16 = marketing_hit

    # 17. Spam sender score (0 = personal, 1-2 = bulk mailer, 3 = known spam domain)
    bulk_matches = sum(1 for p in BULK_SENDER_PATTERNS if re.search(p, sndr))
    f17 = min(bulk_matches, 3)

    return np.array([f1, f2, f3, f4, f5, f6, f7, f8, f9, f10,
                     f11, f12, f13, f14, f15, f16, f17], dtype=float)


# ── Training data ─────────────────────────────────────────────────────────────

TRAINING_DATA = [
    # ── HIGH importance (70-100) ──────────────────────────────────────────────
    ("URGENT: Server outage - all hands needed NOW",        "Production is down affecting all users",                       "devops@corp.com",                  95),
    ("Invoice #8821 OVERDUE - Immediate payment required",  "Payment of $12,500 is 30 days overdue",                       "billing@vendor.com",               90),
    ("FINAL NOTICE: Contract expires tomorrow",             "Your contract will expire without renewal action",             "legal@firm.com",                   88),
    ("Critical security vulnerability - patch required",    "CVE-2024-XXXX found in production dependency",                "security@corp.com",                88),
    ("Board meeting TOMORROW - confirm attendance",         "Attendance is required for all executives",                   "ceo@corp.com",                     85),
    ("Tax filing deadline THIS WEEK",                       "Documents must be submitted by Friday EOD",                   "finance@corp.com",                 87),
    ("URGENT: Client escalation requires your response",    "Client is threatening to cancel the contract",                "client@bigco.com",                 91),
    ("Budget approval needed before EOD today",             "Please approve Q4 budget allocation immediately",             "cfo@company.com",                  83),
    ("ASAP: Production deploy blocked - need your approval","Release 2.4.1 is waiting on your sign-off",                  "pm@corp.com",                      86),
    ("Emergency: Data breach detected in production",       "Unauthorized access detected in database logs",               "security@corp.com",                97),
    ("Visa application: documents required by Thursday",    "Missing documents will invalidate your application",          "immigration@firm.com",             89),
    ("Re: Invoice #4821 — still unpaid",                   "This is the third follow-up. Please respond.",                "billing@realvendor.com",           85),
    ("Interview confirmed for tomorrow 10am",               "Please bring your resume and ID",                             "hr@company.com",                   80),
    ("Telephonic interview scheduled today 3-5 PM",         "We would like to speak with you today between 3 and 5 PM",   "careers@novoracorporation.com",    78),

    # ── MEDIUM importance (40-69) ─────────────────────────────────────────────
    ("Team retrospective next Tuesday",                     "Please come prepared with your thoughts",                     "pm@corp.com",                      55),
    ("Partnership proposal - review when you can",          "Attached is the revised partnership terms",                   "partner@nexus.io",                 60),
    ("Design review scheduled for next week",               "Agenda and assets attached for review",                       "designer@corp.com",                50),
    ("Monthly performance report attached",                 "Summary of KPIs for the past month",                          "analytics@corp.com",               45),
    ("Code review requested: PR #142",                      "Added new feature for user authentication",                   "dev@corp.com",                     52),
    ("Following up from last week's call",                  "Wanted to check if you had a chance to review",               "partner@co.com",                   42),
    ("Security alert: new sign-in detected on your account","We noticed a new sign-in to your Google Account",            "no-reply@accounts.google.com",     55),
    ("Your 90-day free offer ends May 15",                  "Last chance to claim your free Postman access",               "notifications@mail.postman.com",   38),
    ("AutoPay set up for your SIP",                         "Your SIP for SBI Large Cap Fund is now active",               "noreply@phonepe.com",              28),
    ("Statement of account for Folio No. XXXXX694",        "Please find enclosed your statement of account",              "enq_sbimf@camsonline.com",         22),

    # ── LOW importance — generic newsletters ──────────────────────────────────
    ("Your weekly digest is ready",                         "Here's what happened in your network this week",              "digest@linkedin.com",              10),
    ("Company newsletter - March highlights",               "Read about our exciting company updates",                     "newsletter@corp.com",               8),
    ("You might enjoy this article",                        "Based on your interests, we recommend...",                    "recommendations@medium.com",        5),
    ("Changelog: minor bug fixes in v2.1.1",               "Performance improvements and small bug fixes",                "noreply@github.com",                7),
    ("Slack tip of the week",                               "Did you know you can star messages?",                         "noreply@slack.com",                 3),
    ("Congratulations on your work anniversary!",           "5 years at the company - thank you!",                         "hr@corp.com",                      18),

    # ── LOW importance — MARKETING SPAM with urgency language ─────────────────
    # These are the ones the old model got wrong — urgency words but still spam
    ("Final Hours: 20% Off All Excel + Power BI Courses",  "Sale ends today! Get 20% off all courses including renewals", "website@myonlinetraininghub.com",   8),
    ("Last Chance: Excel courses sale ends tonight",        "Don't miss out. Order now before the sale ends",              "newsletter@myonlinetraininghub.com", 7),
    ("Job | Walk-in Interview Drive - Domestic Voice",      "Urgently hiring for BPO domestic voice process in Pune",      "alertnc@naukri.com",               12),
    ("Csgo, Your Exclusive Welcome Reward Awaits!",         "Get 15% off your first meal subscription. Order Now",         "noreply@offers.hellomealsonme.com", 4),
    ("Don't Miss Your Ramadan Savings!",                    "Get 15% off your meal plan subscription today. Use UPGRADE",  "noreply@offers.hellomealsonme.com", 4),
    ("Best-Selling AI Courses at Just Rs 399",              "Get all best seller courses at discount now",                  "no-reply@e.udemymail.com",          6),
    ("General Motors is hiring in Dubai. Apply Now.",       "New job match for your profile from Glassdoor",               "noreply@glassdoor.com",            14),
    ("Hiring via AI Hackathon | Rs 10L+ Opportunities",     "Apply Now! Limited seats available for this competition",     "noreply@unstop.news",              10),
    ("NEW COURSE LAUNCH: AI for Leaders Master Gen AI",     "Get this exclusive course launch offer today",                 "no-reply@e.udemymail.com",          7),
    ("Dhanashree, Urgently hiring Coordinator, Intern",     "Your mid-week job opportunities are right here",              "naukrialerts@naukri.com",          12),
    ("Lab Assistants Science ICT role at GLOBAL INDIAN",    "It looks like your background could be a match",              "donotreply@match.indeed.com",      13),
    ("Urgently hiring for Engineering Intern role",         "Jobs outside your preference - AI ML Intern opportunities",   "recommendationnc@naukri.com",      11),
    ("Software Engineer at E2E Networks. Apply Now.",       "E2E Networks Limited is hiring talent like you",              "info@hirist.tech",                 12),
    ("Kabira, Urgent Requirement for Engineering Intern",   "Your mid-week opportunities - urgently hiring multiple",      "recommendationnc@naukri.com",      11),
    ("Self-Driving Reasoning Models ChatGPT Adds Ads",      "A dispatch from Davos how AI can transform workflows",        "thebatch@deeplearning.ai",         10),
    ("6 Steps to Design Anything With Claude",              "You have probably been using Claude wrong. Today's lead",     "newsletters-noreply@linkedin.com", 10),
    ("Why Your S2P Platform AI Architecture Matters",       "Every enterprise software vendor has an agentic AI story",   "newsletters-noreply@linkedin.com", 10),
    ("Excel Has Changed The Functions That Matter Now",     "New Excel functions that will transform how you work 2026",   "website@myonlinetraininghub.com",   8),
    ("Curly Braces The Secret to Smarter Excel Formulas",   "Unlock hidden formula functionality pros use to automate",   "website@myonlinetraininghub.com",   7),
    ("Introducing Learn AI with Google",                    "Get the skills employers need now",                           "hello@students.udemy.com",          8),
    ("Confused Which Fund to Pick? We will Help.",          "With hundreds of schemes Dear Customer if you ever paused",   "service@service.icicisecurities.com", 18),
    ("Your seat confirmation update is here",               "Tap to know more about your seat confirmation",               "student@mail.internshala.com",     15),
    ("We are Going Live - Save Your Spot Now",              "If Rs 12-20 LPA SDE roles are on your radar join session",   "team@crio.co.in",                  14),
    ("The best SEO book of all time.",                      "Hey Cago Just wanted to remind you that my new book",         "matt@heytony.ca",                   5),
]


# ── Training ──────────────────────────────────────────────────────────────────

def train_scorer(data=None, save=True):
    try:
        from xgboost import XGBRegressor as XGB
        model = XGB(n_estimators=300, max_depth=5, learning_rate=0.04,
                    subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0)
    except ImportError:
        from sklearn.ensemble import GradientBoostingRegressor
        model = GradientBoostingRegressor(n_estimators=300, max_depth=5, learning_rate=0.04, random_state=42)

    if data is None:
        data = TRAINING_DATA

    X = np.array([extract_features(subject=d[0], body=d[1], sender=d[2]) for d in data])
    y = np.array([d[3] for d in data], dtype=float)

    model.fit(X, y)

    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(model, X, y, cv=min(5, len(data)), scoring="r2")
    print(f"[Scorer] R² score: {scores.mean():.3f} ± {scores.std():.3f}")

    if save:
        with open(SCORER_PATH, "wb") as f:
            pickle.dump({"model": model, "version": MODEL_VERSION}, f)
        print(f"[Scorer] Saved to {SCORER_PATH} (version {MODEL_VERSION})")

    return model


def load_scorer():
    if SCORER_PATH.exists():
        with open(SCORER_PATH, "rb") as f:
            saved = pickle.load(f)
        # Support both old format (bare model) and new format (dict with version)
        if isinstance(saved, dict):
            if saved.get("version") == MODEL_VERSION:
                print("[Scorer] Loaded from disk.")
                return saved["model"]
            else:
                print(f"[Scorer] Version mismatch ({saved.get('version')} vs {MODEL_VERSION}). Retraining...")
        else:
            print("[Scorer] Old format detected. Retraining with v2 features...")
    else:
        print("[Scorer] No saved model. Training now...")
    return train_scorer()


_scorer_cache = None

# ── Rule-based hard caps (applied AFTER ML score) ─────────────────────────────
# These are deterministic and reliable regardless of training data quality.

# Senders that are NEVER personally urgent (job boards, promo mailers, newsletters)
HARD_SPAM_DOMAINS = [
    r"@naukri\.com", r"@glassdoor\.com", r"@indeed\.com", r"@linkedin\.com",
    r"@unstop\.news", r"@hirist\.tech", r"@internshala\.com",
    r"@e\.udemymail\.com", r"@students\.udemy\.com", r"@udemymail\.com",
    r"@myonlinetraininghub\.com", r"@hellomealsonme\.com",
    r"@match\.indeed\.com", r"@deeplearning\.ai",
    r"@crio\.co\.in", r"@heytony\.ca", r"@icicisecurities\.com",
]

# Sender address types that are always automated (not personal)
BULK_SENDER_PREFIXES = [
    r"^noreply@", r"^no-reply@", r"^donotreply@",
    r"^alerts?@", r"^newsletter", r"^digest@",
    r"^promo@", r"^marketing@", r"^offers@",
    r"^website@", r"^info@hirist", r"^payal\.",
    r"noreply@",  # substring match catches newsletters-noreply@, etc.
]

# Promotional language that definitively marks an email as non-urgent
PROMO_PHRASES = [
    "% off", "percent off", "sale ends", "sale now", "shop now",
    "order now", "buy now", "limited time offer", "exclusive offer",
    "don't miss out", "last chance to save", "offer ends", "use code",
    "subscribe now", "apply now for", "urgently hiring",
    "walk-in interview", "job alert", "new job match",
    "your subscription renewal", "best seller", "flash sale",
    "get it here", "sde roles", "lpa sde", "₹399", "$9.99",
]


def _rule_based_cap(subject: str, body: str, sender: str, ml_score: float) -> float:
    """
    Apply deterministic caps to the ML score.
    Returns a (possibly lower) score.
    """
    sndr = sender.lower()
    text = f"{subject} {body}".lower()

    is_hard_spam_domain  = any(re.search(p, sndr) for p in HARD_SPAM_DOMAINS)
    is_bulk_prefix       = any(re.search(p, sndr) for p in BULK_SENDER_PREFIXES)
    promo_count          = sum(1 for p in PROMO_PHRASES if p in text)

    # Hard cap 1: known job board / promo domain → max LOW
    if is_hard_spam_domain:
        return min(ml_score, 28.0)

    # Hard cap 2: bulk sender prefix → max LOW
    if is_bulk_prefix:
        return min(ml_score, 28.0)

    # Hard cap 3: 2+ promotional phrases → max LOW regardless of sender
    if promo_count >= 2:
        return min(ml_score, 32.0)

    # Soft cap: 1 promotional phrase + long subject (typical mass email) → max MEDIUM
    if promo_count >= 1 and len(subject) > 50:
        return min(ml_score, 45.0)

    return ml_score


def score_importance(
    subject: str,
    body:    str  = "",
    sender:  str  = "",
    sent_at: Optional[datetime] = None,
    has_deadline: bool = False,
    user_domain: str = "",
) -> dict:
    global _scorer_cache
    if _scorer_cache is None:
        _scorer_cache = load_scorer()

    features  = extract_features(subject, body, sender, sent_at, has_deadline, user_domain)
    ml_score  = float(np.clip(_scorer_cache.predict(features.reshape(1, -1))[0], 0, 100))
    score     = _rule_based_cap(subject, body, sender, ml_score)

    if score >= 70:
        priority = "HIGH"
    elif score >= 40:
        priority = "MEDIUM"
    else:
        priority = "LOW"

    explanation = _build_explanation(features, score)
    return {
        "score":       round(score, 1),
        "priority":    priority,
        "explanation": explanation,
        "features":    dict(zip(FEATURE_NAMES, features.tolist())),
    }


def _build_explanation(features: np.ndarray, score: float) -> str:
    f = dict(zip(FEATURE_NAMES, features))
    reasons = []

    if f["spam_sender_score"] >= 2:
        reasons.append("bulk/marketing sender")
    if f["marketing_language_count"] >= 2:
        reasons.append("promotional content")
    if f["urgent_word_count"] >= 2:
        reasons.append("multiple urgency signals")
    elif f["urgent_word_count"] == 1:
        reasons.append("urgency signal detected")
    if f["has_deadline_extracted"]:
        reasons.append("deadline present")
    if f["action_word_count"] >= 2:
        reasons.append("action required")
    if f["has_money_mention"]:
        reasons.append("financial content")
    if f["low_priority_signals"] >= 2:
        reasons.append("newsletter/digest pattern")
    if f["sender_is_internal"]:
        reasons.append("internal sender")
    if not reasons:
        reasons.append("high signal strength" if score >= 70 else "moderate relevance" if score >= 40 else "low urgency content")

    return ". ".join(r.capitalize() for r in reasons) + "."


def retrain_with_feedback(feedback_rows):
    """
    Retrain the scorer with accumulated user feedback.
    feedback_rows: list of (subject, snippet, sender, correct_priority)
    Called immediately after every /feedback submission.
    """
    global _scorer_cache

    SCORE_MAP = {"HIGH": 88.0, "MEDIUM": 55.0, "LOW": 12.0}

    # Convert DB rows to training format
    feedback_data = [
        (row[0] or "", row[1] or "", row[2] or "", SCORE_MAP.get(row[3], 30.0))
        for row in feedback_rows
    ]

    # User feedback gets 4× weight — their real inbox beats synthetic examples
    combined = TRAINING_DATA + feedback_data * 4

    _scorer_cache = train_scorer(data=combined, save=True)
    print(f"[Scorer] Retrained: {len(TRAINING_DATA)} base + {len(feedback_data)} feedback examples.")


if __name__ == "__main__":
    train_scorer()
    tests = [
        ("URGENT: Invoice overdue - pay now",          "Payment of $5,000 is 30 days late",        "billing@vendor.com"),
        ("Final Hours: 20% Off All Excel Courses",     "Sale ends today! Order now",               "website@myonlinetraininghub.com"),
        ("Job Walk-in Interview Drive Today Urgently", "Apply now for BPO process Pune",           "alertnc@naukri.com"),
        ("Critical bug in production server down",     "Server throwing 500 errors for all users", "devops@corp.com"),
        ("Telephonic interview today 3-5 PM",          "We would like to speak with you today",    "careers@company.com"),
    ]
    print("\n── Importance Scoring Tests ──\n")
    for subject, body, sender in tests:
        r   = score_importance(subject, body, sender)
        bar = "█" * int(r["score"] / 5)
        print(f"Subject: {subject[:55]}")
        print(f"  Score: {r['score']:5.1f}/100  [{bar:<20}]  {r['priority']}")
        print(f"  Why:   {r['explanation']}\n")
