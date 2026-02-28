from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from app.services.embedding_service import generate_embedding
from app.database import engine

# ── NEW: Import ML services ────────────────────────────────────────────────────
from app.services.ml_classifier import predict_priority, predict_priority_batch
from app.services.deadline_extractor import extract_deadline as ner_extract_deadline
from app.services.importance_scorer import score_importance
from app.services.email_clustering import cluster_emails, get_cluster_summary_for_dashboard

from app.services.gmail_service import fetch_recent_emails
# Pre-train ML models on startup so they're ready immediately
from app.services.ml_classifier import load_classifier
from app.services.importance_scorer import load_scorer
load_classifier()
load_scorer()


from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ══════════════════════════════════════════════════════════════════════════════
# /gmail-sync  — now uses ML classifier + NER deadline + XGBoost scorer
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/gmail-sync")
def gmail_sync():
    emails = fetch_recent_emails(5)

    stored = []

    # Batch priority prediction (faster than one-by-one)
    texts = [e["subject"] + " " + e["snippet"] for e in emails]
    priority_results = predict_priority_batch(texts)

    with engine.begin() as connection:
        for i, email in enumerate(emails):
            text_content = email["subject"] + " " + email["snippet"]

            # ── ML 2: NER deadline extraction (replaces old regex) ─────────
            deadline_result = ner_extract_deadline(text_content)
            deadline        = deadline_result["deadline"] if deadline_result else None

            # ── ML 3: XGBoost importance scoring (replaces calculate_importance) ──
            importance = score_importance(
                subject      = email["subject"],
                body         = email["snippet"],
                sender       = email["sender"],
                has_deadline = deadline is not None,
            )

            # ── ML 1: Classifier priority (replaces calculate_priority) ────
            clf_result = priority_results[i]
            confidence = clf_result["confidence"]

            score = importance["score"]
            if score >= 70:
                priority = "HIGH"
            elif score >= 45:
                priority = "MEDIUM"
            else:
                priority = "LOW"

            is_important = priority in ["HIGH", "MEDIUM"]

            embedding = generate_embedding(text_content)

            connection.execute(
                text("""
                    INSERT INTO emails 
                    (gmail_message_id, sender, subject, summary, embedding,
                     importance_score, is_important, deadline, priority,
                     priority_confidence, importance_reason)
                    VALUES (:gmail_id, :sender, :subject, :summary, :embedding,
                            :score, :important, :deadline, :priority,
                            :confidence, :reason)
                    ON CONFLICT (gmail_message_id) DO NOTHING
                """),
                {
                    "gmail_id":   email["id"],
                    "sender":     email["sender"],
                    "subject":    email["subject"],
                    "summary":    text_content,
                    "embedding":  str(embedding),
                    "score":      importance["score"],
                    "important":  is_important,
                    "deadline":   deadline,
                    "priority":   priority,
                    "confidence": confidence,
                    "reason":     importance["explanation"],
                }
            )

            stored.append({
                "subject":          email["subject"],
                "importance_score": importance["score"],
                "priority":         priority,
                "confidence":       round(confidence, 2),
                "deadline":         str(deadline) if deadline else None,
                "reason":           importance["explanation"],
            })

    return {"synced_emails": stored}


# ══════════════════════════════════════════════════════════════════════════════
# /search-test  — unchanged, your original logic works fine here
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/search-test")
def search_test(query: str):
    query_embedding = generate_embedding(query)

    with engine.begin() as connection:
        result = connection.execute(
            text("""
                SELECT sender, subject, summary,
                       embedding <-> :query_embedding AS distance
                FROM emails
                ORDER BY embedding <-> :query_embedding
                LIMIT 3
            """),
            {"query_embedding": str(query_embedding)}
        )
        rows = result.fetchall()

    return [
        {
            "sender":   row[0],
            "subject":  row[1],
            "summary":  row[2],
            "distance": float(row[3])
        }
        for row in rows
    ]


# ══════════════════════════════════════════════════════════════════════════════
# /important-emails  — now returns ML confidence + reason
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/important-emails")
def get_important_emails():
    with engine.begin() as connection:
        result = connection.execute(
            text("""
                SELECT sender, subject, summary, importance_score,
                       priority, priority_confidence, importance_reason, deadline
                FROM emails
                WHERE is_important = TRUE
                ORDER BY importance_score DESC
            """)
        )
        rows = result.fetchall()

    return [
        {
            "sender":             row[0],
            "subject":            row[1],
            "summary":            row[2],
            "importance_score":   row[3],
            "priority":           row[4],
            "confidence":         row[5],
            "reason":             row[6],
            "deadline":           str(row[7]) if row[7] else None,
        }
        for row in rows
    ]


# ══════════════════════════════════════════════════════════════════════════════
# /daily-brief  — now uses NER deadline extractor
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/daily-brief")
def daily_brief():
    with engine.begin() as connection:
        result = connection.execute(
            text("""
                SELECT summary, subject, priority, deadline
                FROM emails
                WHERE is_important = TRUE
                ORDER BY importance_score DESC
            """)
        )
        rows = result.fetchall()

    tasks = []

    for row in rows:
        # Use stored deadline first, fall back to NER on summary
        
        stored_deadline = row[3]
        # Reject far-future dates (2039 bug)
        if stored_deadline and str(stored_deadline) < '2030-01-01':
            deadline = stored_deadline
        else:
            ner_result = ner_extract_deadline(str(row[0]))
            if ner_result and ner_result["confidence"] > 0.5:
                deadline = ner_result["deadline"]
            else:
                deadline = None

        tasks.append({
            "task":     row[0],
            "subject":  row[1],
            "priority": row[2],
            "deadline": str(deadline) if deadline else None,
        })

    return {
        "total_important": len(tasks),
        "urgent_tasks":    tasks
    }


# ══════════════════════════════════════════════════════════════════════════════
# /ask  — unchanged logic, works with ML-enriched data automatically
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/ask")
def ask_inbox(question: str):
    query_embedding = generate_embedding(question)

    with engine.begin() as connection:
        result = connection.execute(
            text("""
                SELECT summary, subject, priority, deadline
                FROM emails
                ORDER BY embedding <-> :query_embedding
                LIMIT 5
            """),
            {"query_embedding": str(query_embedding)}
        )
        rows = result.fetchall()

    tasks = []

    for row in rows:
        summary  = row[0]
        subject  = row[1]
        priority = row[2]
        deadline = row[3]

        # Try NER if no stored deadline
        if not deadline:
            ner_result = ner_extract_deadline(summary)
            if ner_result and ner_result["confidence"] > 0.5:
                deadline = ner_result["deadline"]

        if deadline:
            tasks.append({
                "task":     summary,
                "subject":  subject,
                "priority": priority,
                "deadline": str(deadline),
            })

    if not tasks:
        return {"answer": "No urgent deadlines found in your inbox."}

    response_lines = [
        f"[{t['priority']}] {t['subject']} | Deadline: {t['deadline']}"
        for t in tasks
    ]

    return {
        "answer": f"You have {len(tasks)} urgent task(s):\n" + "\n".join(response_lines)
    }


# ══════════════════════════════════════════════════════════════════════════════
# /clusters  — NEW: ML 4 — K-Means email topic clustering
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/clusters")
def get_clusters():
    with engine.begin() as connection:
        result = connection.execute(
            text("""
                SELECT id, subject, sender, summary, embedding, priority
                FROM emails
            """)
        )
        rows = result.fetchall()

    if not rows:
        return {"clusters": [], "message": "No emails to cluster"}

    emails = [
        {
            "id":        row[0],
            "subject":   row[1],
            "sender":    row[2],
            "summary":   row[3],
            "embedding": [float(x) for x in row[4].strip("[]").split(",")],
            "priority":  row[5],
        }
        for row in rows
        if row[4]  # skip emails without embeddings
    ]

    result    = cluster_emails(emails)
    dashboard = get_cluster_summary_for_dashboard(result["clusters"])

    return {
        "clusters":         dashboard,
        "n_clusters":       result["n_clusters"],
        "silhouette_score": result["silhouette_score"],
        "total_emails":     len(emails),
    }


# ══════════════════════════════════════════════════════════════════════════════
# /ml-status  — NEW: shows all 4 models and their status
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/ml-status")
def ml_status():
    from pathlib import Path
    services_dir = Path(__file__).parent / "services" / "saved_models"

    def model_status(filename):
        return "ready" if (services_dir / filename).exists() else "will train on first use"

    return {
        "models": {
            "priority_classifier": {
                "status":   model_status("priority_classifier.pkl"),
                "type":     "Random Forest on 384-dim embeddings + 6 handcrafted features",
            },
            "importance_scorer": {
                "status":   model_status("importance_scorer.pkl"),
                "type":     "XGBoost regressor with 15 engineered features",
            },
            "deadline_extractor": {
                "status":   "ready",
                "type":     "spaCy NER (en_core_web_sm) + context scoring",
            },
            "email_clustering": {
                "status":   model_status("kmeans_cluster.pkl"),
                "type":     "K-Means with silhouette-score auto-k + PCA",
            },
        },
        "pipeline": "Gmail → Embed → Classify → Score → Extract Deadline → Cluster → Store"
    }