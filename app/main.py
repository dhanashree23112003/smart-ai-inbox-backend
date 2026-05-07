import os
import re
from datetime import datetime

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from app.config import FRONTEND_URL
from app.database import engine
from app.services.embedding_service import generate_embedding
from app.services.ml_classifier import predict_priority_batch, load_classifier
from app.services.deadline_extractor import extract_deadline as ner_extract_deadline
from app.services.importance_scorer import score_importance, load_scorer, retrain_with_feedback
from app.services.email_clustering import cluster_emails, get_cluster_summary_for_dashboard
from app.services.gmail_service import fetch_recent_emails, get_gmail_service
from app.routes.auth import router as auth_router, get_session

load_classifier()
load_scorer()

app = FastAPI()
app.include_router(auth_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL, "http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Emoji detector ────────────────────────────────────────────────────────────
_EMOJI_RE = re.compile(
    "[\U0001F300-\U0001FFFF\U00002600-\U000027BF\U0001F900-\U0001F9FF\U00002700-\U000027BF]",
    flags=re.UNICODE,
)
def _has_emoji(s: str) -> bool:
    return bool(_EMOJI_RE.search(s))

# ── Auth helper — every protected endpoint calls this ────────────────────────
def _require_user(request: Request) -> tuple:
    """Returns (user_id, creds). Raises 401 if not signed in."""
    session = get_session(request.headers.get("x-session-token", ""))
    if not session:
        raise HTTPException(status_code=401, detail="Not signed in. Visit /auth/login first.")
    return session["email"], session["creds"]

# ── Startup: add user_id column + indexes if not present ─────────────────────
def _run(conn, sql, label=""):
    """Execute one DDL statement, log but never crash on failure."""
    try:
        conn.execute(text(sql))
    except Exception as e:
        print(f"[schema] {label or sql[:60]}: {e}")

@app.on_event("startup")
def ensure_schema():
    # Each statement runs in its own transaction so one failure never blocks the rest.
    stmts = [
        ("CREATE sessions table", """
            CREATE TABLE IF NOT EXISTS sessions (
                token      TEXT PRIMARY KEY,
                email      TEXT NOT NULL,
                creds_b64  TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT NOW()
            )
        """),
        ("ADD user_id to emails",
            "ALTER TABLE emails ADD COLUMN IF NOT EXISTS user_id TEXT DEFAULT ''"),
        ("ALTER user_id type to TEXT",
            "ALTER TABLE emails ALTER COLUMN user_id TYPE TEXT USING user_id::TEXT"),
        ("INDEX on emails.user_id",
            "CREATE INDEX IF NOT EXISTS idx_emails_user_id ON emails(user_id)"),
        ("CREATE email_feedback table", """
            CREATE TABLE IF NOT EXISTS email_feedback (
                id               SERIAL PRIMARY KEY,
                user_id          TEXT NOT NULL DEFAULT '',
                gmail_message_id TEXT NOT NULL,
                correct_priority TEXT NOT NULL,
                subject          TEXT,
                snippet          TEXT,
                sender           TEXT,
                created_at       TIMESTAMP DEFAULT NOW()
            )
        """),
        ("ADD user_id to email_feedback",
            "ALTER TABLE email_feedback ADD COLUMN IF NOT EXISTS user_id TEXT NOT NULL DEFAULT ''"),
        ("ADD unique constraint to email_feedback", """
            DO $$ BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM pg_constraint
                    WHERE conname = 'email_feedback_user_id_gmail_message_id_key'
                ) THEN
                    ALTER TABLE email_feedback
                    ADD CONSTRAINT email_feedback_user_id_gmail_message_id_key
                    UNIQUE (user_id, gmail_message_id);
                END IF;
            END $$
        """),
        ("DROP old primary key on email_feedback if gmail_message_id only", """
            DO $$ BEGIN
                IF EXISTS (
                    SELECT 1 FROM pg_constraint
                    WHERE conname = 'email_feedback_pkey'
                    AND contype = 'p'
                ) THEN
                    ALTER TABLE email_feedback DROP CONSTRAINT IF EXISTS email_feedback_gmail_message_id_key;
                END IF;
            END $$
        """),
    ]
    for label, sql in stmts:
        try:
            with engine.begin() as conn:
                conn.execute(text(sql))
        except Exception as e:
            print(f"[schema] {label}: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# /gmail-sync
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/gmail-sync")
def gmail_sync(request: Request):
    user_id, creds = _require_user(request)
    try:
        emails = fetch_recent_emails(50, credentials=creds)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Gmail fetch failed: {str(e)}")

    texts = [e["subject"] + " " + e["snippet"] for e in emails]
    try:
        priority_results = predict_priority_batch(texts)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ML classification failed: {str(e)}")

    # Clean up ghost emails (no subject = old sent mail / drafts synced before INBOX fix)
    with engine.begin() as conn:
        conn.execute(text(
            "DELETE FROM emails WHERE user_id = :uid AND (subject IS NULL OR subject = '')"
        ), {"uid": user_id})

    stored = []
    with engine.begin() as conn:
        for i, email in enumerate(emails):
            try:
                # Skip emails with no subject — these are drafts or sent mail leaking in
                if not email.get("subject", "").strip():
                    continue

                text_content = email["subject"] + " " + email["snippet"]

                is_bulk = email.get("is_promo") or email.get("has_unsubscribe")

                deadline_result = ner_extract_deadline(text_content)
                deadline = deadline_result["deadline"] if deadline_result else None

                importance  = score_importance(subject=email["subject"], body=email["snippet"],
                                               sender=email["sender"], has_deadline=deadline is not None)
                clf_result  = priority_results[i]
                confidence  = clf_result["confidence"]
                score       = importance["score"]

                if is_bulk or _has_emoji(email["subject"]):
                    score      = min(score, 25.0)
                    confidence = 0.99

                priority     = "HIGH" if score >= 70 else "MEDIUM" if score >= 45 else "LOW"
                is_important = priority in ("HIGH", "MEDIUM")
                embedding    = generate_embedding(text_content)

                conn.execute(text("""
                    INSERT INTO emails
                        (gmail_message_id, user_id, sender, subject, summary, embedding,
                         importance_score, is_important, deadline, priority,
                         priority_confidence, importance_reason)
                    VALUES
                        (:gmail_id, :user_id, :sender, :subject, :summary, :embedding,
                         :score, :important, :deadline, :priority, :confidence, :reason)
                    ON CONFLICT (gmail_message_id) DO UPDATE SET
                        user_id             = EXCLUDED.user_id,
                        deadline            = EXCLUDED.deadline,
                        -- Never overwrite a human correction with ML scores
                        importance_score    = CASE WHEN emails.importance_reason = 'User feedback'
                                                   THEN emails.importance_score
                                                   ELSE EXCLUDED.importance_score END,
                        is_important        = CASE WHEN emails.importance_reason = 'User feedback'
                                                   THEN emails.is_important
                                                   ELSE EXCLUDED.is_important END,
                        priority            = CASE WHEN emails.importance_reason = 'User feedback'
                                                   THEN emails.priority
                                                   ELSE EXCLUDED.priority END,
                        priority_confidence = CASE WHEN emails.importance_reason = 'User feedback'
                                                   THEN emails.priority_confidence
                                                   ELSE EXCLUDED.priority_confidence END,
                        importance_reason   = CASE WHEN emails.importance_reason = 'User feedback'
                                                   THEN 'User feedback'
                                                   ELSE EXCLUDED.importance_reason END
                """), {
                    "gmail_id":  email["id"],   "user_id":   user_id,
                    "sender":    email["sender"], "subject":  email["subject"],
                    "summary":   text_content,   "embedding": str(embedding),
                    "score":     score,           "important": is_important,
                    "deadline":  deadline,        "priority":  priority,
                    "confidence": confidence,     "reason":   importance["explanation"],
                })

                stored.append({"subject": email["subject"], "priority": priority,
                                "score": round(score, 1)})
            except Exception:
                continue

    return {"synced_emails": stored}


# ══════════════════════════════════════════════════════════════════════════════
# /important-emails
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/important-emails")
def get_important_emails(request: Request):
    user_id, _ = _require_user(request)
    with engine.begin() as conn:
        rows = conn.execute(text("""
            SELECT sender, subject, summary, importance_score,
                   priority, priority_confidence, importance_reason, deadline, gmail_message_id
            FROM   emails
            WHERE  user_id = :uid
              AND  (is_deleted = FALSE OR is_deleted IS NULL)
            ORDER  BY importance_score DESC
        """), {"uid": user_id}).fetchall()

    return [{
        "sender": r[0], "subject": r[1], "summary": r[2],
        "importance_score": r[3], "priority": r[4], "confidence": r[5],
        "reason": r[6], "deadline": str(r[7]) if r[7] else None,
        "gmail_message_id": r[8],
    } for r in rows]


# ══════════════════════════════════════════════════════════════════════════════
# /ask
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/ask")
def ask_inbox(question: str, request: Request):
    if len(question) > 500:
        raise HTTPException(status_code=400, detail="Question too long (max 500 characters)")
    user_id, _ = _require_user(request)
    q = question.lower()

    with engine.begin() as conn:
        # Always load full inbox state so we can answer any question
        all_rows = conn.execute(text("""
            SELECT subject, summary, priority, deadline, importance_score, sender
            FROM   emails
            WHERE  user_id = :uid
              AND  (is_deleted = FALSE OR is_deleted IS NULL)
            ORDER  BY importance_score DESC
        """), {"uid": user_id}).fetchall()

    if not all_rows:
        return {"answer": "Your inbox is empty. Try syncing first."}

    emails_data = [{"subject": r[0], "summary": r[1], "priority": r[2],
                    "deadline": r[3], "score": r[4], "sender": r[5]}
                   for r in all_rows]

    high   = [e for e in emails_data if e["priority"] == "HIGH"]
    medium = [e for e in emails_data if e["priority"] == "MEDIUM"]
    low    = [e for e in emails_data if e["priority"] == "LOW"]

    # ── Deadline questions ────────────────────────────────────────────────────
    if any(w in q for w in ("deadline", "due", "when", "by when")):
        tasks = []
        for e in emails_data:
            dl = e["deadline"]
            if not dl:
                nr = ner_extract_deadline(e["summary"] or "")
                if nr and nr.get("confidence", 0) > 0.5:
                    dl = nr["deadline"]
            if dl:
                tasks.append({"subject": e["subject"], "priority": e["priority"], "deadline": str(dl)})
        if not tasks:
            return {"answer": "No deadlines found in your inbox."}
        lines = [f"[{t['priority']}] {t['subject']} — due {t['deadline']}" for t in tasks[:8]]
        return {"answer": f"You have **{len(tasks)} deadline(s)**:\n\n" + "\n".join(f"• {l}" for l in lines)}

    # ── Urgent / high priority questions ─────────────────────────────────────
    if any(w in q for w in ("urgent", "high", "important", "critical", "action")):
        if not high:
            return {"answer": "No urgent emails right now. " + (f"You have {len(medium)} medium priority emails." if medium else "Inbox looks clear!")}
        lines = [f"• **{e['subject']}** — from {e['sender']}" for e in high[:8]]
        return {"answer": f"You have **{len(high)} urgent email(s)**:\n\n" + "\n".join(lines)}

    # ── Summary questions ─────────────────────────────────────────────────────
    if any(w in q for w in ("summary", "summarize", "overview", "what's in", "inbox")):
        total = len(emails_data)
        lines = []
        if high:
            lines.append(f"🔴 **{len(high)} urgent** — needs attention now")
            for e in high[:3]:
                lines.append(f"   • {e['subject']}")
        if medium:
            lines.append(f"🟡 **{len(medium)} medium** — review when you can")
        if low:
            lines.append(f"⚪ **{len(low)} low** — newsletters, promotions, job alerts")
        return {"answer": f"**Inbox summary — {total} emails:**\n\n" + "\n".join(lines)}

    # ── Semantic fallback — embedding similarity ──────────────────────────────
    query_embedding = generate_embedding(question)
    with engine.begin() as conn:
        rows = conn.execute(text("""
            SELECT subject, priority, deadline, summary
            FROM   emails
            WHERE  user_id = :uid
              AND  (is_deleted = FALSE OR is_deleted IS NULL)
            ORDER  BY embedding <-> :qe
            LIMIT  5
        """), {"uid": user_id, "qe": str(query_embedding)}).fetchall()

    if not rows:
        return {"answer": "Nothing found matching your question."}
    lines = [f"• [{r[1]}] **{r[0]}**" + (f" — due {r[2]}" if r[2] else "") for r in rows]
    return {"answer": "Most relevant emails:\n\n" + "\n".join(lines)}


# ══════════════════════════════════════════════════════════════════════════════
# /daily-brief
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/daily-brief")
def daily_brief(request: Request):
    user_id, _ = _require_user(request)
    cutoff = datetime(2030, 1, 1)
    with engine.begin() as conn:
        rows = conn.execute(text("""
            SELECT summary, subject, priority, deadline
            FROM   emails
            WHERE  user_id = :uid
              AND  is_important = TRUE
              AND  (is_deleted = FALSE OR is_deleted IS NULL)
            ORDER  BY importance_score DESC
        """), {"uid": user_id}).fetchall()

    tasks = []
    for r in rows:
        stored = r[3]
        deadline = None
        if stored:
            try:
                dl = stored if isinstance(stored, datetime) else datetime.fromisoformat(str(stored))
                if dl < cutoff:
                    deadline = dl
            except Exception:
                pass
        if not deadline:
            nr = ner_extract_deadline(str(r[0]))
            if nr and nr.get("confidence", 0) > 0.5:
                deadline = nr["deadline"]
        tasks.append({"task": r[0], "subject": r[1], "priority": r[2],
                       "deadline": str(deadline) if deadline else None})

    return {"total_important": len(tasks), "urgent_tasks": tasks}


# ══════════════════════════════════════════════════════════════════════════════
# /clusters
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/clusters")
def get_clusters(request: Request):
    user_id, _ = _require_user(request)
    with engine.begin() as conn:
        rows = conn.execute(text("""
            SELECT id, subject, sender, summary, embedding, priority
            FROM   emails
            WHERE  user_id = :uid
              AND  (is_deleted = FALSE OR is_deleted IS NULL)
        """), {"uid": user_id}).fetchall()

    if not rows:
        return {"clusters": [], "message": "No emails to cluster"}

    email_list = [
        {"id": r[0], "subject": r[1], "sender": r[2], "summary": r[3],
         "embedding": [float(x) for x in r[4].strip("[]").split(",")], "priority": r[5]}
        for r in rows if r[4]
    ]
    result    = cluster_emails(email_list)
    dashboard = get_cluster_summary_for_dashboard(result["clusters"])
    return {"clusters": dashboard, "n_clusters": result["n_clusters"],
            "silhouette_score": result["silhouette_score"], "total_emails": len(email_list)}


# ══════════════════════════════════════════════════════════════════════════════
# /trash-low-priority
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/trash-low-priority")
def trash_low_priority(request: Request):
    user_id, creds = _require_user(request)
    try:
        service = get_gmail_service(credentials=creds)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Gmail connection failed: {str(e)}")

    with engine.begin() as conn:
        rows = conn.execute(text("""
            SELECT gmail_message_id FROM emails
            WHERE  user_id = :uid
              AND  priority = 'LOW'
              AND  (is_deleted = FALSE OR is_deleted IS NULL)
              AND  gmail_message_id IS NOT NULL
        """), {"uid": user_id}).fetchall()

    trashed = 0
    with engine.begin() as conn:
        for row in rows:
            gid = row[0]
            try:
                service.users().messages().trash(userId="me", id=gid).execute()
                conn.execute(text(
                    "UPDATE emails SET is_deleted = TRUE WHERE gmail_message_id = :id AND user_id = :uid"
                ), {"id": gid, "uid": user_id})
                trashed += 1
            except Exception:
                continue

    return {"trashed": trashed, "message": f"Moved {trashed} emails to Gmail Trash"}


# ══════════════════════════════════════════════════════════════════════════════
# /trash-email/{message_id}
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/trash-email/{message_id}")
def trash_single_email(message_id: str, request: Request):
    if not message_id or len(message_id) > 200:
        raise HTTPException(status_code=400, detail="Invalid message_id")
    user_id, creds = _require_user(request)
    try:
        service = get_gmail_service(credentials=creds)
        service.users().messages().trash(userId="me", id=message_id).execute()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Gmail trash failed: {str(e)}")

    with engine.begin() as conn:
        conn.execute(text(
            "UPDATE emails SET is_deleted = TRUE WHERE gmail_message_id = :id AND user_id = :uid"
        ), {"id": message_id, "uid": user_id})

    return {"trashed": 1, "message_id": message_id}


# ══════════════════════════════════════════════════════════════════════════════
# /feedback
# ══════════════════════════════════════════════════════════════════════════════

PRIORITY_SCORES = {"HIGH": 88.0, "MEDIUM": 55.0, "LOW": 12.0}

@app.post("/feedback")
def submit_feedback(gmail_message_id: str, correct_priority: str, request: Request):
    if correct_priority not in PRIORITY_SCORES:
        raise HTTPException(status_code=400, detail="priority must be HIGH, MEDIUM, or LOW")
    user_id, _ = _require_user(request)

    with engine.begin() as conn:
        row = conn.execute(text(
            "SELECT subject, summary, sender FROM emails WHERE gmail_message_id = :id AND user_id = :uid"
        ), {"id": gmail_message_id, "uid": user_id}).fetchone()

        if not row:
            raise HTTPException(status_code=404, detail="Email not found for this user")

        subject, snippet, sender = row[0], row[1], row[2]

        conn.execute(text("""
            INSERT INTO email_feedback (user_id, gmail_message_id, correct_priority, subject, snippet, sender)
            VALUES (:uid, :id, :priority, :subject, :snippet, :sender)
            ON CONFLICT (user_id, gmail_message_id) DO UPDATE SET
                correct_priority = EXCLUDED.correct_priority,
                created_at       = NOW()
        """), {"uid": user_id, "id": gmail_message_id, "priority": correct_priority,
               "subject": subject, "snippet": snippet, "sender": sender})

        new_score    = PRIORITY_SCORES[correct_priority]
        is_important = correct_priority in ("HIGH", "MEDIUM")
        conn.execute(text("""
            UPDATE emails
            SET priority = :priority, importance_score = :score,
                is_important = :important, importance_reason = 'User feedback'
            WHERE gmail_message_id = :id AND user_id = :uid
        """), {"priority": correct_priority, "score": new_score,
               "important": is_important, "id": gmail_message_id, "uid": user_id})

        # Only retrain on THIS user's feedback
        feedback_rows = conn.execute(text(
            "SELECT subject, snippet, sender, correct_priority FROM email_feedback WHERE user_id = :uid"
        ), {"uid": user_id}).fetchall()

    # Retrain — non-fatal if it fails
    try:
        retrain_with_feedback(feedback_rows)
        retrain_msg = f"Got it. Model retrained with {len(feedback_rows)} feedback examples."
    except Exception as e:
        print(f"[feedback] retrain failed (non-fatal): {e}")
        retrain_msg = "Feedback saved."

    return {"status": "ok", "new_priority": correct_priority, "new_score": new_score,
            "feedback_total": len(feedback_rows), "message": retrain_msg}


# ══════════════════════════════════════════════════════════════════════════════
# /search-test
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/search-test")
def search_test(query: str, request: Request):
    user_id, _ = _require_user(request)
    qe = generate_embedding(query)
    with engine.begin() as conn:
        rows = conn.execute(text("""
            SELECT sender, subject, summary, embedding <-> :qe AS distance
            FROM   emails
            WHERE  user_id = :uid
            ORDER  BY embedding <-> :qe
            LIMIT  3
        """), {"uid": user_id, "qe": str(qe)}).fetchall()
    return [{"sender": r[0], "subject": r[1], "summary": r[2], "distance": float(r[3])} for r in rows]


# ══════════════════════════════════════════════════════════════════════════════
# /ml-status  /health
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/ml-status")
def ml_status():
    from pathlib import Path
    d = Path(__file__).parent / "services" / "saved_models"
    chk = lambda f: "ready" if (d / f).exists() else "will train on first use"
    return {
        "models": {
            "priority_classifier": {"status": chk("priority_classifier.pkl"), "type": "Random Forest + embeddings"},
            "importance_scorer":   {"status": chk("importance_scorer.pkl"),   "type": "XGBoost + rule-based caps"},
            "deadline_extractor":  {"status": "ready",                         "type": "Regex + dateutil"},
            "email_clustering":    {"status": chk("kmeans_cluster.pkl"),       "type": "K-Means + silhouette auto-k"},
        },
        "pipeline": "Gmail → labels/headers → Embed → Score → Deadline → Cluster → Store",
    }

@app.get("/health")
def health():
    return {"status": "ok"}
