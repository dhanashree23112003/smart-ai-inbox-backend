<div align="center">

```
███████╗███╗   ███╗ █████╗ ██████╗ ████████╗
██╔════╝████╗ ████║██╔══██╗██╔══██╗╚══██╔══╝
███████╗██╔████╔██║███████║██████╔╝   ██║   
╚════██║██║╚██╔╝██║██╔══██║██╔══██╗   ██║   
███████║██║ ╚═╝ ██║██║  ██║██║  ██║   ██║   
╚══════╝╚═╝     ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝   
        AI INBOX · 4-MODEL ML PIPELINE
```

**Your inbox has 2,847 unread emails. Four AI models just read all of them.**

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-smart--ai--inbox.vercel.app-6366f1?style=for-the-badge)](https://smart-ai-inbox.vercel.app/)
[![Backend](https://img.shields.io/badge/GitHub-Backend-181717?style=for-the-badge&logo=github)](https://github.com/dhanashree23112003/smart-ai-inbox-backend)
[![Frontend](https://img.shields.io/badge/GitHub-Frontend-181717?style=for-the-badge&logo=github)](https://github.com/dhanashree23112003/smart-ai-inbox-frontend)
[![Deployed on HuggingFace](https://img.shields.io/badge/Backend-HuggingFace_Spaces_(Docker)-yellow?style=for-the-badge)](https://huggingface.co/)
[![Deployed on Vercel](https://img.shields.io/badge/Frontend-Vercel-black?style=for-the-badge&logo=vercel)](https://smart-ai-inbox.vercel.app/)

---

> *Gmail doesn't know what's urgent. It doesn't know your deadlines.*
> *It doesn't learn from you. It just dumps everything in chronological order*
> *and calls it a day.*
>
> **Smart AI Inbox runs 4 ML models on every single email you receive.**

</div>

---

## 🧠 The Pipeline

Not one model. Not two. **Four — per email.**

```
Every incoming email
        │
        ▼
┌───────────────────────────────┐
│  MODEL 1: XGBoost Scorer      │  15 hand-crafted features
│  "How important is this?"     │  Output: 0–100 importance score
└──────────────┬────────────────┘
               │
               ▼
┌───────────────────────────────┐
│  MODEL 2: Random Forest       │  384-dim sentence-transformer embeddings
│  Priority Classifier          │  Output: HIGH / MEDIUM / LOW
└──────────────┬────────────────┘
               │
               ▼
┌───────────────────────────────┐
│  MODEL 3: Deadline Extractor  │  regex + dateutil + confidence scoring
│  "When is this due?"          │  Output: deadline date + confidence %
└──────────────┬────────────────┘
               │
               ▼
┌───────────────────────────────┐
│  MODEL 4: K-Means Clustering  │  silhouette-score auto-k selection
│  Topic Grouping               │  Output: topic cluster label
└──────────────┬────────────────┘
               │
               ▼
      📬 Organised. Prioritised.
         Deadline-aware. Clustered.
```

---

## ⚙️ Model Deep Dive

### Model 1 — XGBoost Importance Scorer

```python
# 15 hand-crafted features per email
features = [
    "sender_is_known",        # is sender in your contacts?
    "has_action_words",       # "please", "urgent", "required", "asap"
    "reply_chain_depth",      # how deep is this thread?
    "email_length_bucket",    # short = transactional, long = needs attention
    "hour_of_day",            # 3am email from boss hits different
    "has_attachment",
    "contains_deadline_language",
    "caps_ratio",             # ALL CAPS = someone is stressed
    "question_count",         # how many ?'s
    "cc_count",               # more CC'd = more visibility pressure
    # ... 5 more
]
# Output: 0–100 importance score
```

### Model 2 — Random Forest Priority Classifier

```python
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer("all-MiniLM-L6-v2")
embedding = embedder.encode(email_body)  # → 384-dim vector
priority = rf_classifier.predict([embedding])  # → HIGH / MEDIUM / LOW
```

### Model 3 — Deadline Extractor

```python
import dateutil.parser
import re

# Catches: "by Friday", "due 15th May", "submit before EOD", "deadline: 2026-06-01"
deadline, confidence = extract_deadline(email_text)
# confidence score included — so you know when to trust it
```

### Model 4 — K-Means Topic Clustering (Auto-K)

```python
from sklearn.metrics import silhouette_score

# Don't hardcode k. Find the best k automatically.
best_k, best_score = 2, -1
for k in range(2, max_k + 1):
    labels = KMeans(n_clusters=k).fit_predict(embeddings)
    score = silhouette_score(embeddings, labels)
    if score > best_score:
        best_k, best_score = k, score

# Clusters adapt to YOUR inbox, not a preset category list
```

---

## 🔁 The Feedback Loop That Actually Learns

Most "AI" tools don't learn from your corrections. This one does — **immediately.**

```
You mark email as wrong priority
              │
              ▼
   Correction stored in Supabase
   with 4× sample weight
              │
              ▼
   XGBoost retrained on-the-fly
   (your correction weighted 4×
    so it actually matters)
              │
              ▼
   CASE WHEN upsert — correction
   survives Gmail resyncs
              │
              ▼
   Next similar email → correct priority ✅
```

```sql
-- The DB-aware upsert that makes corrections survive resyncs
INSERT INTO email_scores (email_id, score, is_correction)
VALUES ($1, $2, true)
ON CONFLICT (email_id) DO UPDATE
  SET score = CASE
    WHEN email_scores.is_correction = true THEN email_scores.score  -- never overwrite a correction
    ELSE EXCLUDED.score
  END;
```

---

## 🐛 The Data Leakage Bug (Caught & Disclosed)

The model showed **92% accuracy**. Looked amazing. Published nothing.

Something felt off.

```python
# ❌ THE LEAKY VERSION
# Training features included email metadata that only exists AFTER classification
# The model was learning from the answer, not the question
X_train = df[["subject", "body", "label_derived_feature"]]  # ← leaked target info
# Result: 92% accuracy (fake) — would collapse on real unseen emails

# ✅ THE HONEST VERSION
# Strict feature isolation — only pre-classification signals allowed
X_train = df[["subject", "body", "sender", "timestamp_features"]]
# Result: 72% accuracy (real) — actually generalises
```

**92% (fake) → 72% (honest). Caught it. Fixed it. Disclosed it.**

A model that performs at 72% honestly is infinitely more useful than one that lies at 92%.

---

## 🗄️ Vector Search with pgvector

```python
# Supabase PostgreSQL + pgvector extension
# Cosine similarity search on 384-dim embeddings

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE emails (
  id UUID PRIMARY KEY,
  subject TEXT,
  embedding vector(384)  -- sentence-transformer output
);

-- Find similar emails to the current one
SELECT id, subject, 1 - (embedding <=> $1) AS similarity
FROM emails
ORDER BY embedding <=> $1
LIMIT 5;
```

No external vector DB. No Pinecone. Postgres does it all.

---

## 🔐 Multi-User Gmail OAuth2

```python
# Per-user session isolation — your emails stay yours
# Each user gets their own Gmail token, stored encrypted
# No cross-user data leakage, no shared model state

@router.get("/auth/gmail")
async def gmail_oauth(user_id: str):
    flow = google_auth_oauthlib.flow.Flow.from_client_secrets_file(
        CLIENT_SECRETS_FILE,
        scopes=["https://www.googleapis.com/auth/gmail.readonly"]
    )
    flow.redirect_uri = f"{BASE_URL}/auth/callback?user_id={user_id}"
    return {"auth_url": flow.authorization_url()[0]}
```

---

## 🏗️ Tech Stack

| Layer | Tech |
|-------|------|
| **Importance Scoring** | XGBoost (15 features, 0–100 score) |
| **Priority Classification** | Random Forest + Sentence-Transformers (384-dim) |
| **Deadline Extraction** | regex + dateutil + confidence scoring |
| **Topic Clustering** | K-Means + silhouette auto-k |
| **Vector Search** | Supabase pgvector (cosine similarity) |
| **Backend** | FastAPI (Python) |
| **Frontend** | React + Tailwind CSS |
| **Database** | Supabase (PostgreSQL + pgvector) |
| **Auth** | Gmail OAuth2 (per-user session isolation) |
| **Backend Deploy** | HuggingFace Spaces (Docker) |
| **Frontend Deploy** | Vercel |

---

## 🚀 Run It Locally

```bash
# Backend
git clone https://github.com/dhanashree23112003/smart-ai-inbox-backend
cd smart-ai-inbox-backend
pip install -r requirements.txt
cp .env.example .env
# Fill in: SUPABASE_URL, SUPABASE_KEY, GMAIL_CLIENT_ID, GMAIL_CLIENT_SECRET
uvicorn main:app --reload

# Frontend (new terminal)
git clone https://github.com/dhanashree23112003/smart-ai-inbox-frontend
cd smart-ai-inbox-frontend
npm install
npm run dev

# Visit http://localhost:5173
# Connect Gmail → watch 4 models process your inbox
```

---

## 🔑 Environment Variables

```env
SUPABASE_URL=your_supabase_project_url
SUPABASE_KEY=your_supabase_anon_key
GMAIL_CLIENT_ID=your_google_oauth_client_id
GMAIL_CLIENT_SECRET=your_google_oauth_client_secret
```

---

## 👩‍💻 Built By

**Dhanashree Bansode** — AI/ML Engineer · Ex ISRO Intern

- 🌐 [portfoliodhanashree.vercel.app](https://portfoliodhanashree.vercel.app)
- 💼 [linkedin.com/in/dhanashree2311](https://linkedin.com/in/dhanashree2311)
- 🐙 [github.com/dhanashree23112003](https://github.com/dhanashree23112003)

---

<div align="center">

*Your inbox doesn't have to be a dumpster fire.*

**Four models. Every email. Real-time learning. Zero excuses.**

</div>
