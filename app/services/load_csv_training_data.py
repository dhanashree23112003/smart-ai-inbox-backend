"""
Converts emails.csv (Enron + SpamAssassin dataset) into training examples
for the importance scorer.

Mapping:
  spam=1  → LOW  (score 5–15)   — 1,368 real spam emails
  spam=0  → auto-labeled via heuristics (HIGH/MEDIUM/LOW)

Usage:
  python -m app.services.load_csv_training_data
  → prints stats, saves enriched training data summary
"""

import re
import csv
import random
from pathlib import Path

CSV_PATH = Path(__file__).parent.parent.parent.parent / "emails.csv"


# ── Heuristic labeler for ham emails (spam=0) ────────────────────────────────

URGENT_SIGNALS  = ["urgent", "asap", "immediately", "critical", "emergency",
                   "overdue", "deadline", "action required", "please respond",
                   "by eod", "by cob", "by friday", "today", "expir"]

ACTION_SIGNALS  = ["please review", "please approve", "please confirm",
                   "action needed", "your response", "required", "must be",
                   "sign off", "authorize", "need your", "waiting on you"]

LOW_SIGNALS     = ["fyi", "for your information", "forwarded by", "newsletter",
                   "digest", "weekly", "monthly", "update:", "news clips",
                   "announcement", "reminder:", "no action needed"]


def parse_email(text: str) -> tuple[str, str]:
    """Split 'Subject: ... body...' into (subject, body)."""
    text = text.strip()
    if text.lower().startswith("subject:"):
        # Find end of subject line (first double space or newline)
        rest = text[8:].strip()
        # Subject ends at first occurrence of two or more spaces, or actual newline
        parts = re.split(r"\s{3,}|\n", rest, maxsplit=1)
        subject = parts[0].strip()
        body    = parts[1].strip() if len(parts) > 1 else ""
        return subject, body
    return "", text


def heuristic_score(subject: str, body: str) -> float:
    """
    Rough 0-100 score for ham emails where we have no label.
    Deliberately conservative — only assign HIGH/MEDIUM if signals are strong.
    """
    t = f"{subject} {body}".lower()

    urgent_hits  = sum(1 for w in URGENT_SIGNALS if w in t)
    action_hits  = sum(1 for w in ACTION_SIGNALS if w in t)
    low_hits     = sum(1 for w in LOW_SIGNALS    if w in t)

    # Very long emails are usually not urgent (reports, newsletters)
    length_penalty = min(len(t.split()) / 800, 1.0) * 15

    score = 35.0  # neutral baseline for ham
    score += urgent_hits * 12
    score += action_hits * 8
    score -= low_hits * 10
    score -= length_penalty

    return float(max(5.0, min(95.0, score)))


def load_csv_training_data(
    csv_path=CSV_PATH,
    max_spam: int = 1000,
    max_ham:  int = 800,
    random_seed: int = 42,
) -> list[tuple[str, str, str, float]]:
    """
    Returns (subject, body, sender, score) list for training.

    spam=1 → score=8 (LOW) — direct mapping, no guessing
    spam=0 → weak supervision via labeling functions — HIGH/MEDIUM/LOW
    """
    from app.services.weak_supervision import label_ham_emails

    random.seed(random_seed)

    spam_examples = []
    ham_raw       = []   # (subject, body) — unlabeled

    with open(csv_path, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text  = row.get("text", "").strip()
            label = row.get("spam", "0").strip()
            if not text:
                continue
            subject, body = parse_email(text)
            if not subject and not body:
                continue

            if label == "1":
                spam_examples.append((subject, body, "bulk@spam.com", 8.0))
            else:
                ham_raw.append((subject, body))

    # Apply weak supervision to get HIGH/MEDIUM/LOW labels on ham
    random.shuffle(ham_raw)
    ham_labeled = label_ham_emails(ham_raw, min_confidence=0.55)

    # Sample to prevent class imbalance
    random.shuffle(spam_examples)
    random.shuffle(ham_labeled)

    combined = spam_examples[:max_spam] + ham_labeled[:max_ham]
    random.shuffle(combined)

    high   = sum(1 for x in combined if x[3] >= 70)
    medium = sum(1 for x in combined if 40 <= x[3] < 70)
    low    = sum(1 for x in combined if x[3] < 40)
    print(f"[CSV] Final training set: {len(combined)} total "
          f"(HIGH={high}, MEDIUM={medium}, LOW={low})")

    return combined


def print_stats(data):
    high   = [x for x in data if x[3] >= 70]
    medium = [x for x in data if 40 <= x[3] < 70]
    low    = [x for x in data if x[3] < 40]
    print(f"\n── CSV Training Data Stats ──")
    print(f"  Total:  {len(data)}")
    print(f"  HIGH:   {len(high)}  (score ≥70)")
    print(f"  MEDIUM: {len(medium)}  (40–69)")
    print(f"  LOW:    {len(low)}  (<40)")
    print(f"\nSample LOW examples:")
    for s, b, _, sc in low[:3]:
        print(f"  [{sc:.0f}] {s[:80]}")
    print(f"\nSample HIGH examples:")
    for s, b, _, sc in high[:3]:
        print(f"  [{sc:.0f}] {s[:80]}")


if __name__ == "__main__":
    if not CSV_PATH.exists():
        print(f"CSV not found at {CSV_PATH}")
    else:
        data = load_csv_training_data()
        print_stats(data)
        print(f"\nReady to add {len(data)} examples to TRAINING_DATA in importance_scorer.py")
