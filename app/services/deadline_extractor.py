# deadline_extractor.py
# No spaCy — uses regex + dateutil only, but much smarter than your original.
# Works on Python 3.14. Drop into: backend/app/services/

import re
from datetime import datetime, timedelta
from dateutil import parser as dateutil_parser
from dateutil.relativedelta import relativedelta
from typing import Optional

DEADLINE_TRIGGERS = [
    "deadline", "due", "by eod", "by cob", "by end of day",
    "before", "no later than", "must be submitted", "required by",
    "needed by", "action by", "expires", "expiry", "respond by",
    "submit by", "complete by", "finish by", "send by", "due date",
    "closing date", "last date", "cutoff",
]

FALSE_POSITIVES = [
    r"\b(founded|established|since|as of|born|incorporated)\b",
    r"\b(version|v\d+\.\d+)\b",
    r"\b(invoice #|order #|ticket #)\s*\d+",
]

def _next_weekday(now, weekday, force_next=False):
    days_ahead = weekday - now.weekday()
    if days_ahead <= 0 or force_next:
        days_ahead += 7
    return (now + timedelta(days=days_ahead)).replace(hour=17, minute=0, second=0, microsecond=0)

RELATIVE_DATES = {
    "today":           lambda now: now.replace(hour=17, minute=0, second=0, microsecond=0),
    "eod":             lambda now: now.replace(hour=17, minute=0, second=0, microsecond=0),
    "end of day":      lambda now: now.replace(hour=17, minute=0, second=0, microsecond=0),
    "cob":             lambda now: now.replace(hour=17, minute=0, second=0, microsecond=0),
    "tomorrow":        lambda now: (now + timedelta(days=1)).replace(hour=17, minute=0, second=0, microsecond=0),
    "end of week":     lambda now: _next_weekday(now, 4),
    "eow":             lambda now: _next_weekday(now, 4),
    "end of month":    lambda now: (now + relativedelta(day=31)).replace(hour=17, minute=0, second=0, microsecond=0),
    "next week":       lambda now: now + timedelta(weeks=1),
    "next month":      lambda now: now + relativedelta(months=1),
    "this friday":     lambda now: _next_weekday(now, 4),
    "this thursday":   lambda now: _next_weekday(now, 3),
    "this wednesday":  lambda now: _next_weekday(now, 2),
    "this tuesday":    lambda now: _next_weekday(now, 1),
    "this monday":     lambda now: _next_weekday(now, 0),
    "next monday":     lambda now: _next_weekday(now, 0, force_next=True),
    "next tuesday":    lambda now: _next_weekday(now, 1, force_next=True),
    "next wednesday":  lambda now: _next_weekday(now, 2, force_next=True),
    "next thursday":   lambda now: _next_weekday(now, 3, force_next=True),
    "next friday":     lambda now: _next_weekday(now, 4, force_next=True),
    "monday":          lambda now: _next_weekday(now, 0),
    "tuesday":         lambda now: _next_weekday(now, 1),
    "wednesday":       lambda now: _next_weekday(now, 2),
    "thursday":        lambda now: _next_weekday(now, 3),
    "friday":          lambda now: _next_weekday(now, 4),
    "saturday":        lambda now: _next_weekday(now, 5),
    "sunday":          lambda now: _next_weekday(now, 6),
}

DATE_PATTERNS = [
    r"\b(\d{4}-\d{2}-\d{2})\b",
    r"\b(\d{1,2}/\d{1,2}/\d{2,4})\b",
    r"\b(\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{4})\b",
    r"\b((?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+\d{4})?)\b",
    r"\bby\s+(\d{1,2}(?::\d{2})?\s*(?:am|pm))\b",
]

def extract_deadline(text, reference_date=None):
    if not text or len(text.strip()) < 3:
        return None

    now      = reference_date or datetime.now()
    text_low = text.lower()
    candidates = []

    for phrase, resolver in RELATIVE_DATES.items():
        if phrase in text_low:
            idx     = text_low.find(phrase)
            context = text_low[max(0, idx - 100): idx + 100]
            trigger = any(t in context for t in DEADLINE_TRIGGERS)
            try:
                resolved   = resolver(now)
                days_ahead = (resolved - now).total_seconds() / 86400
                if days_ahead >= -0.1:
                    confidence = _score(phrase, days_ahead, trigger, context)
                    candidates.append({"deadline": resolved, "raw_text": phrase,
                                       "confidence": confidence, "context": context.strip()})
            except Exception:
                continue

    for pattern in DATE_PATTERNS:
        for match in re.finditer(pattern, text_low, re.IGNORECASE):
            raw         = match.group(1).strip()
            surrounding = text_low[max(0, match.start() - 50): match.end() + 50]
            if any(re.search(fp, surrounding) for fp in FALSE_POSITIVES):
                continue
            try:
                resolved = dateutil_parser.parse(raw, default=now, fuzzy=True, dayfirst=False)
                if resolved.year < now.year:
                    resolved = resolved.replace(year=now.year)
                if resolved.year > now.year + 2:
                    continue
                days_ahead = (resolved - now).total_seconds() / 86400
                if days_ahead < -1:
                    continue
                trigger    = any(t in surrounding for t in DEADLINE_TRIGGERS)
                confidence = _score(raw, days_ahead, trigger, surrounding)
                candidates.append({"deadline": resolved, "raw_text": match.group(0).strip(),
                                   "confidence": confidence, "context": surrounding.strip()})
            except Exception:
                continue

    if not candidates:
        return None

    best = max(candidates, key=lambda c: (c["confidence"], -(c["deadline"] - now).total_seconds()))
    return {"deadline": best["deadline"], "raw_text": best["raw_text"],
            "confidence": best["confidence"], "context": best["context"]}

def _score(raw, days_ahead, trigger_found, context):
    score = 0.35
    if trigger_found:       score += 0.35
    if days_ahead <= 0:     score += 0.15
    elif days_ahead <= 1:   score += 0.12
    elif days_ahead <= 3:   score += 0.08
    elif days_ahead <= 7:   score += 0.04
    if re.search(r"\d{1,2}(:\d{2})?\s*(am|pm)", raw): score += 0.08
    if re.search(r"\d{4}", raw):                       score += 0.05
    if any(w in context for w in ["urgent","asap","critical","final"]): score += 0.08
    return round(min(score, 1.0), 3)

def get_deadline_for_email(subject, body=""):
    result = extract_deadline(f"{subject} {body}")
    return result["deadline"] if result else None

if __name__ == "__main__":
    tests = [
        "URGENT: Budget approval due today EOD",
        "Please submit the report by next Friday",
        "Invoice overdue - must pay by March 5th",
        "Team meeting tomorrow at 2pm",
        "Company newsletter - no action needed",
        "Respond by end of week or we proceed without you",
    ]
    print("\n── Deadline Extraction Tests ──\n")
    for t in tests:
        result = extract_deadline(t)
        if result:
            print(f"OK  {t[:55]}")
            print(f"    -> {result['deadline'].strftime('%a %b %d %Y %I:%M%p')}  (conf: {result['confidence']:.0%})\n")
        else:
            print(f"--  {t[:55]}\n    -> No deadline\n")