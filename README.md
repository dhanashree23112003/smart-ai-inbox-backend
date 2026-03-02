<div align="center">

```
 _____ __  __    _    ____ _____      _    ___   ___ _   _ ____   _____  __
/ ____|  \/  |  / \  |  _ \_   _|   / \  |_ _| |_ _| \ | |  _ \ / _ \ \/ /
\___ \| |\/| | / _ \ | |_) || |    / _ \  | |   | ||  \| | |_) | | | \  /
 ___) | |  | |/ ___ \|  _ < | |   / ___ \ | |   | || |\  |  _ <| |_| /  \
|____/|_|  |_/_/   \_\_| \_\|_|  /_/   \_\___|___|_| \_|_| \_\\___//_/\_\
```

### The inbox assistant that actually works. Built in production. Deployed to the web. No shortcuts.

[![Live Demo](https://img.shields.io/badge/LIVE-smart--ai--inbox.vercel.app-7c3aed?style=for-the-badge&logo=vercel)](https://smart-ai-inbox.vercel.app)
[![Backend](https://img.shields.io/badge/API-HuggingFace_Spaces-ff9d00?style=for-the-badge&logo=huggingface)](https://dhanashree2311-smart-ai-inbox.hf.space)
[![ML Status](https://img.shields.io/badge/ML_PIPELINE-4_models_live-22c55e?style=for-the-badge)](https://dhanashree2311-smart-ai-inbox.hf.space/ml-status)
[![Made with](https://img.shields.io/badge/Made_with-Python_%2B_React-3b82f6?style=for-the-badge)](https://github.com/dhanashree23112003)

</div>

---

## The honest origin story

My Gmail had 4,000 unread emails. Important ones were buried under newsletters, LinkedIn notifications, and promotional spam. I kept missing deadlines.

So I did what any reasonable person does — I spent 3 weeks building an AI system instead of just unsubscribing from things.

No regrets.

---

## What it actually does

> Connects to your Gmail via OAuth. Reads your emails. Runs them through a 4-model ML pipeline. Tells you what's urgent, what can wait, and what should be in the trash. Lets you ask your inbox questions in plain English.

**The part most projects skip:** It's deployed. Real users. Real emails. Real Google OAuth. Not a localhost demo.

---

## The ML Pipeline — what's really running

### Model 1 — XGBoost Importance Scorer
The primary signal. Produces a 0–100 importance score using 15 hand-engineered features:

```
urgency keywords     → "ASAP", "deadline", "overdue", "by end of day"
action words         → "please review", "action required", "your response"  
financial signals    → "invoice", "payment", "billing", "$", "overdue"
sender reputation    → known domains, @gmail vs corporate domains
has_deadline         → boolean from deadline extractor
subject length       → longer subjects = more deliberate
time sensitivity     → "today", "tomorrow", "this week"
... and 8 more
```

Score >= 70 → HIGH. Score >= 45 → MEDIUM. Score < 45 → LOW.

### Model 2 — Random Forest Priority Classifier
384-dimensional sentence-transformer embeddings as features. Trained on **125 labeled emails** — 60 synthetic seed data + 65 manually labeled real emails from a real Gmail inbox.

The accuracy story people skip:
```
First run:  91.67% accuracy  ← looked great, was lying
After fix:  72.00% accuracy  ← real data, real patterns, honest number
```
The first model was overfit on 60 synthetic examples it had already seen. The second was tested on emails it had never encountered. 72% on real data beats 91% on fake data every time.

### The Ensemble — both models vote
```python
if clf_priority == scorer_priority:
    decision = "both_agree"           # confident
elif clf_confidence >= 0.80:
    decision = "classifier_override"  # RF very sure, overrides XGBoost
else:
    decision = "scorer_wins"          # default to XGBoost
```
Every email gets a `decision_method` logged. You can audit exactly why any email was classified the way it was.

### Model 3 — Deadline Extractor
Not ML — and that's intentional. Regex + dateutil with confidence scoring.

```
"by Friday"          → extracts date, confidence: 0.9
"end of week"        → extracts date, confidence: 0.7  
"5pm today"          → extracts date, confidence: 0.95
"sometime soon"      → no extraction, confidence too low
"in the year 2039"   → rejected by confidence filter
```
Rule-based is the right tool here. Interpretable, fast, reliable. No hallucinations.

### Model 4 — K-Means Email Clustering
Auto-groups emails by topic using PCA + K-Means. Silhouette score determines the optimal number of clusters — no hardcoded k.

```
Cluster 0 → Finance & Billing
Cluster 1 → Job Applications  
Cluster 2 → Team Communications
Cluster 3 → Newsletters & Digests
```
Meaningful at 50+ emails. That's why we sync 50, not 10.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        USER BROWSER                         │
│                   React + Tailwind (Vercel)                  │
└──────────────────────────┬──────────────────────────────────┘
                           │ HTTPS + x-session-token header
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    FASTAPI BACKEND                          │
│              Hugging Face Spaces (Docker)                    │
│                                                             │
│  /auth/login     → Google OAuth redirect                    │
│  /auth/callback  → token exchange, session creation         │
│  /gmail-sync     → fetch 50 emails, run ML pipeline         │
│  /important-emails → return scored, filtered emails         │
│  /ask            → semantic search + NLP routing            │
│  /trash-*        → Gmail API trash operations               │
│  /clusters       → K-Means topic grouping                   │
│  /ml-status      → live model health check                  │
└──────┬──────────────────────────────────────┬───────────────┘
       │                                      │
       ▼                                      ▼
┌─────────────────┐                ┌──────────────────────────┐
│   GMAIL API     │                │       SUPABASE           │
│                 │                │                          │
│ Read emails     │                │  emails table            │
│ Trash emails    │                │  users table             │
│ Get profile     │                │  sessions table          │
└─────────────────┘                │  pgvector embeddings     │
                                   └──────────────────────────┘
```

---

## Auth — multi-user, properly isolated

Every user signs in with their own Google account. Sessions are stored in Supabase — they survive server restarts. Google credentials stay in memory — re-populated on each login.

```
User clicks "Sign In"
    ↓
Backend redirects to Google OAuth (manual URL build, no PKCE)
    ↓
User picks THEIR Google account
    ↓
Backend exchanges code for access + refresh tokens
    ↓
Session token created → saved to Supabase sessions table
    ↓
Frontend receives ?session=uuid&email=user@gmail.com
    ↓
All API calls include x-session-token header
    ↓
Every endpoint queries sessions table to get user_id
    ↓
Every DB query filters by user_id — complete isolation
```

---

## The trash feature

One-click cleanup that actually works end to end:

```
User clicks "Trash LOW"
    ↓
Backend queries: SELECT gmail_message_id FROM emails 
                 WHERE priority = 'LOW' AND is_deleted = FALSE
    ↓
For each email: Gmail API → messages.trash(id)
    ↓
UPDATE emails SET is_deleted = TRUE
    ↓
Frontend removes emails from UI instantly
    ↓
Toast: "Moved 12 emails to Gmail Trash"
    ↓
Gmail Trash: recoverable for 30 days
```

Per-email trash also available. Both use the same pattern.

---

## Tech stack

| Layer | Technology | Why |
|---|---|---|
| Frontend | React + Tailwind | Fast iteration, glassmorphism UI |
| Backend | FastAPI (Python) | Async, typed, fast |
| Database | Supabase (Postgres) | pgvector for semantic search |
| Vector search | pgvector | Cosine similarity on embeddings |
| ML — scoring | XGBoost | Interpretable, fast, great on tabular |
| ML — classification | Random Forest | Ensemble with XGBoost |
| ML — embeddings | sentence-transformers | 384-dim semantic representations |
| ML — clustering | scikit-learn K-Means | Silhouette-score auto-k |
| Auth | Google OAuth2 | Real Gmail access |
| Deployment — backend | Hugging Face Spaces | Free GPU-ish containers |
| Deployment — frontend | Vercel | Zero config, fast CDN |

---

## Running locally

```bash
# Clone
git clone https://github.com/dhanashree23112003/smart-ai-inbox-backend
cd smart-ai-inbox-backend/backend

# Install
pip install -r requirements.txt

# Environment variables
cp .env.example .env
# Fill in: DATABASE_URL, GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET
#          FRONTEND_URL, BACKEND_URL, SECRET_KEY

# Run
uvicorn app.main:app --reload --port 8000

# Check ML models loaded
curl http://localhost:8000/ml-status
```

---

## What I'd do differently at scale

**Session storage:** Encrypt refresh tokens and store in Supabase. Currently re-auth required after server restart.

**Sync frequency:** Background job (APScheduler or Celery) to auto-sync every hour. Currently manual trigger.

**Classifier retraining:** Store user feedback (wrong priority labels) and retrain weekly. Human-in-the-loop ML.

**Embeddings:** Cache embeddings in Supabase. Currently regenerated on every sync. Expensive at scale.

**Rate limiting:** Gmail API has quotas. Need exponential backoff + queue for bulk operations.

**Clustering:** Switch to HDBSCAN for better handling of noise emails and variable cluster sizes.

---


<div align="center">

Built by **Dhanashree Bansode**

[LinkedIn](https://linkedin.com/in/dhanashree2311) · [Live Demo](https://smart-ai-inbox.vercel.app) · [API Status](https://dhanashree2311-smart-ai-inbox.hf.space/ml-status)

*If you actually read this whole README, you deserve to try the live demo.*

</div>
