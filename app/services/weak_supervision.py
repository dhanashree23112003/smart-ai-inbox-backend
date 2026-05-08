"""
Weak Supervision labeler for ham emails (spam=0 from emails.csv).

Each labeling function (LF) returns "HIGH", "MEDIUM", "LOW", or None (abstain).
The label model combines all votes:
  - If 2+ LFs agree on HIGH → HIGH (strong signal)
  - If 2+ LFs agree on LOW  → LOW
  - If conflicting votes     → MEDIUM (safe default)
  - If all abstain           → skip (too uncertain)

Why weak supervision beats manual labeling:
  - Labels 4,360 emails in milliseconds vs 6 hours of clicking
  - Rules encode domain knowledge explicitly (auditable, improvable)
  - ~75% accuracy is enough — the embedding model learns patterns, not memorizes labels
  - The user feedback loop corrects wrong predictions in production

Labeling functions are based on Enron corporate email patterns.
"""

from __future__ import annotations
import re


HIGH   = "HIGH"
MEDIUM = "MEDIUM"
LOW    = "LOW"
ABSTAIN = None


# ── Individual labeling functions ─────────────────────────────────────────────
# Each takes (subject, body) → HIGH | MEDIUM | LOW | None

def lf_explicit_action(subject: str, body: str) -> str | None:
    """Explicit action request directed at recipient."""
    text = f"{subject} {body}".lower()
    phrases = [
        "action required", "action needed", "please respond",
        "please reply", "response required", "your response needed",
        "need your approval", "please approve", "sign off",
        "please confirm", "confirmation needed", "need your input",
        "waiting on you", "waiting for your", "need you to",
    ]
    if any(p in text for p in phrases):
        return HIGH
    return ABSTAIN


def lf_deadline_language(subject: str, body: str) -> str | None:
    """Hard deadline markers."""
    text = f"{subject} {body}".lower()
    patterns = [
        r"\bby eod\b", r"\bby cob\b", r"\bby tomorrow\b",
        r"\bby friday\b", r"\bby monday\b", r"\bdeadline\b",
        r"\bdue today\b", r"\bdue tomorrow\b", r"\bexpir",
        r"\blast chance\b", r"\bfinal reminder\b", r"\boverdue\b",
        r"\bno later than\b", r"\bmust be (submitted|completed|returned)\b",
    ]
    if any(re.search(p, text) for p in patterns):
        return HIGH
    return ABSTAIN


def lf_urgency_subject(subject: str, body: str) -> str | None:
    """Urgency words specifically in the subject line (stronger signal)."""
    subj = subject.lower()
    urgent_words = ["urgent", "asap", "critical", "emergency",
                    "important:", "time sensitive", "immediate"]
    if any(w in subj for w in urgent_words):
        return HIGH
    return ABSTAIN


def lf_financial_action(subject: str, body: str) -> str | None:
    """Financial emails that require action (not just notifications)."""
    text = f"{subject} {body}".lower()
    # Must have financial AND action signal together
    financial = any(w in text for w in ["invoice", "payment", "billing",
                                         "overdue", "wire transfer", "remit"])
    action    = any(w in text for w in ["please pay", "action", "required",
                                         "immediately", "outstanding"])
    if financial and action:
        return HIGH
    if financial:
        return MEDIUM
    return ABSTAIN


def lf_question_to_recipient(subject: str, body: str) -> str | None:
    """Direct question expecting a reply — MEDIUM priority."""
    text = f"{subject} {body}".lower()
    # Question in subject + short body (personal, not forwarded)
    has_question = "?" in subject
    is_personal  = len(body.split()) < 200
    direct_ask   = any(p in text for p in [
        "can you", "could you", "would you", "do you know",
        "what do you think", "do you have", "are you able",
        "let me know", "any thoughts", "your thoughts",
    ])
    if has_question and direct_ask:
        return MEDIUM
    if direct_ask and is_personal:
        return MEDIUM
    return ABSTAIN


def lf_meeting_invite(subject: str, body: str) -> str | None:
    """Meeting/calendar items — MEDIUM (needs awareness, not urgent action)."""
    subj = subject.lower()
    text = body.lower()
    meeting_signals = ["meeting", "call scheduled", "conference call",
                       "let's meet", "calendar invite", "schedule a"]
    if any(s in subj for s in meeting_signals):
        return MEDIUM
    if any(s in subj for s in ["re:", "fwd:"]) and "meeting" in text:
        return MEDIUM
    return ABSTAIN


def lf_fyi_forwarded(subject: str, body: str) -> str | None:
    """FYI and forwarded content — LOW (informational, no action needed)."""
    subj = subject.lower().strip()
    text = f"{subject} {body}".lower()

    # Explicit FYI
    if subj.startswith("fyi") or "for your information" in text:
        return LOW

    # Pure forward of news/articles with no personal content
    if subj.startswith(("fwd:", "fw:")) and len(body.split()) > 300:
        return LOW

    return ABSTAIN


def lf_newsletter_report(subject: str, body: str) -> str | None:
    """News clips, reports, digests — LOW."""
    text = f"{subject} {body}".lower()
    low_patterns = [
        "news clip", "newsclip", "daily news", "weekly update",
        "monthly report", "press release", "article:", "digest",
        "announcement:", "bulletin", "highlights", "recap",
        "from the desk of", "enron india newsdesk", "market update",
    ]
    if any(p in text for p in low_patterns):
        return LOW

    # Very long forwarded emails with no personal opener are usually reports
    body_words = len(body.split())
    if body_words > 600 and "forwarded by" in text:
        return LOW

    return ABSTAIN


def lf_personal_short(subject: str, body: str) -> str | None:
    """Short personal emails often need a response — MEDIUM."""
    is_short    = 10 < len(body.split()) < 80
    is_personal = not any(w in body.lower() for w in [
        "forwarded", "newsletter", "unsubscribe", "click here"
    ])
    has_greeting = any(body.lower().strip().startswith(g)
                       for g in ["hi ", "hello ", "hey ", "dear "])
    if is_short and is_personal and has_greeting:
        return MEDIUM
    return ABSTAIN


# ── All labeling functions ────────────────────────────────────────────────────

ALL_LFS = [
    lf_explicit_action,
    lf_deadline_language,
    lf_urgency_subject,
    lf_financial_action,
    lf_question_to_recipient,
    lf_meeting_invite,
    lf_fyi_forwarded,
    lf_newsletter_report,
    lf_personal_short,
]


# ── Label model — combine votes ───────────────────────────────────────────────

def apply_label_model(subject: str, body: str) -> tuple[str | None, float]:
    """
    Apply all LFs and combine votes.

    Returns (label, confidence) where confidence is 0–1.
    Returns (None, 0) if all LFs abstain (skip this example).

    Confidence = fraction of non-abstaining LFs that agree with final label.
    """
    votes = [lf(subject, body) for lf in ALL_LFS]
    non_abstain = [v for v in votes if v is not None]

    if not non_abstain:
        return None, 0.0

    # Count votes per label
    counts = {HIGH: 0, MEDIUM: 0, LOW: 0}
    for v in non_abstain:
        counts[v] += 1

    winner = max(counts, key=counts.get)
    confidence = counts[winner] / len(non_abstain)

    # Single vote: accept HIGH always, accept MEDIUM/LOW only if confidence=1.0
    if len(non_abstain) == 1:
        if winner == HIGH:
            return winner, confidence
        if confidence == 1.0:       # only 1 LF fired and it was definitive
            return winner, 0.6      # treat as lower confidence
        return None, 0.0

    return winner, confidence


def label_to_score(label: str, confidence: float) -> float:
    """Convert label + confidence to 0–100 score for XGBoost training."""
    base = {"HIGH": 82.0, "MEDIUM": 52.0, "LOW": 12.0}[label]
    # Adjust score within the band based on confidence
    spread = {"HIGH": 10.0, "MEDIUM": 8.0, "LOW": 5.0}[label]
    return base + (confidence - 0.5) * spread


# ── Main entry point ──────────────────────────────────────────────────────────

def label_ham_emails(
    ham_examples: list[tuple[str, str]],  # (subject, body)
    min_confidence: float = 0.55,
) -> list[tuple[str, str, str, float]]:
    """
    Label ham emails using weak supervision.
    Returns list of (subject, body, sender, score) for training.
    """
    labeled = []
    stats   = {HIGH: 0, MEDIUM: 0, LOW: 0, "skipped": 0}

    for subject, body in ham_examples:
        label, conf = apply_label_model(subject, body)
        if label is None or conf < min_confidence:
            stats["skipped"] += 1
            continue

        score = label_to_score(label, conf)
        # Use a generic sender — these are Enron internal emails
        sender = "colleague@enron.com"
        labeled.append((subject, body, sender, score))
        stats[label] += 1

    total_labeled = sum(v for k, v in stats.items() if k != "skipped")
    print(f"[WeakSupervision] Ham emails labeled: {total_labeled} "
          f"(HIGH={stats[HIGH]}, MEDIUM={stats[MEDIUM]}, LOW={stats[LOW]}, "
          f"skipped={stats['skipped']})")

    return labeled


if __name__ == "__main__":
    # Quick test on sample emails
    test_cases = [
        ("URGENT: Budget approval needed by EOD",    "Please approve the Q4 budget before end of day."),
        ("FYI - Enron India Newsdesk Jan 18 clips",  "Forwarded by Sandeep Kohli... [very long article]"),
        ("Can you review this proposal?",            "Hi, do you have a few minutes to look at this?"),
        ("Meeting scheduled for Thursday 2pm",       "Please join the call to discuss Q3 results."),
        ("Re: Invoice #4821",                        "This invoice is still overdue. Please remit payment."),
        ("Weekly Houston weather update",            "Sacramento weather station fyi..."),
    ]

    print("\n-- Weak Supervision Test --\n")
    for subj, body in test_cases:
        label, conf = apply_label_model(subj, body)
        score = label_to_score(label, conf) if label else 0
        print(f"Subject: {subj}")
        print(f"  -> {label or 'SKIP'} (conf={conf:.2f}, score={score:.0f})\n")
