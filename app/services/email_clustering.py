# email_clustering.py
# Drop this into: smart-inbox-backend/backend/app/services/
#
# What this does:
#   - Fetches all email embeddings from Supabase
#   - Runs K-Means clustering to group by topic
#   - Auto-labels each cluster (Finance, Work, Social, etc.)
#   - Saves cluster assignments back to Supabase
#   - Exposes an API-ready function to get clusters for dashboard

import numpy as np
import pickle
from pathlib import Path
from collections import Counter, defaultdict
from typing import Optional
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")

MODEL_DIR = Path(__file__).parent / "saved_models"
MODEL_DIR.mkdir(exist_ok=True)
CLUSTER_MODEL_PATH = MODEL_DIR / "kmeans_cluster.pkl"

# ─── Topic label inference ────────────────────────────────────────────────────
# Each cluster gets auto-labeled by checking which keywords dominate it.

TOPIC_KEYWORDS = {
    "Finance":      ["invoice", "payment", "budget", "billing", "cost", "salary",
                     "expense", "revenue", "finance", "accounting", "tax", "refund"],
    "Work":         ["meeting", "project", "deadline", "task", "team", "sprint",
                     "review", "report", "presentation", "milestone", "standup"],
    "Legal":        ["contract", "agreement", "legal", "compliance", "terms",
                     "policy", "regulation", "liability", "clause", "notice"],
    "HR":           ["hr", "hire", "onboarding", "performance", "leave", "vacation",
                     "interview", "offer", "payroll", "benefits", "holiday"],
    "Tech":         ["bug", "deploy", "server", "api", "code", "pull request",
                     "release", "outage", "git", "database", "security", "patch"],
    "Marketing":    ["campaign", "launch", "brand", "content", "seo", "social",
                     "newsletter", "promo", "customer", "lead", "conversion"],
    "Social":       ["lunch", "party", "celebrate", "fun", "offsite", "event",
                     "invitation", "anniversary", "birthday", "congratulations"],
    "Partnership":  ["partnership", "collaboration", "proposal", "vendor",
                     "supplier", "client", "deal", "offer", "opportunity"],
    "Updates":      ["update", "digest", "changelog", "release notes", "summary",
                     "recap", "highlights", "weekly", "monthly"],
}


def _infer_topic_label(texts: list[str]) -> str:
    """Scores each topic against the cluster's email texts and returns best match."""
    combined = " ".join(texts).lower()
    scores   = {}
    for topic, keywords in TOPIC_KEYWORDS.items():
        scores[topic] = sum(combined.count(kw) for kw in keywords)

    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "General"


# ─── Optimal K selection ─────────────────────────────────────────────────────

def find_optimal_k(embeddings: np.ndarray, k_min: int = 3, k_max: int = 10) -> int:
    """
    Uses silhouette score to find best number of clusters.
    Silhouette score = how well-separated the clusters are (-1 to 1, higher = better).
    """
    if len(embeddings) < k_min * 2:
        return min(k_min, max(2, len(embeddings) // 3))

    best_k     = k_min
    best_score = -1

    for k in range(k_min, min(k_max + 1, len(embeddings))):
        km    = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(embeddings)
        score  = silhouette_score(embeddings, labels, sample_size=min(500, len(embeddings)))
        print(f"  k={k}  silhouette={score:.4f}")
        if score > best_score:
            best_score = score
            best_k     = k

    print(f"[Clustering] Optimal k={best_k} (silhouette={best_score:.4f})")
    return best_k


# ─── Main clustering function ─────────────────────────────────────────────────

def cluster_emails(emails: list[dict], n_clusters: int = None, auto_k: bool = True) -> dict:
    """
    Clusters emails by topic using K-Means on their embeddings.

    Args:
        emails: list of dicts with keys: id, subject, summary, embedding (list[float])
        n_clusters: number of clusters (auto-detected if None and auto_k=True)
        auto_k: whether to auto-select optimal k

    Returns:
        {
            "clusters": [
                {
                    "id": 0,
                    "label": "Finance",
                    "count": 12,
                    "emails": [{"id": ..., "subject": ..., "cluster_id": 0}, ...]
                },
                ...
            ],
            "n_clusters": 5,
            "silhouette_score": 0.42
        }
    """
    if not emails:
        return {"clusters": [], "n_clusters": 0, "silhouette_score": 0}

    # Extract embeddings
    embeddings = np.array([e["embedding"] for e in emails], dtype=float)

    # Reduce dimensions for faster clustering (PCA to 50 dims)
    if embeddings.shape[1] > 50 and len(emails) > 20:
        pca = PCA(n_components=min(50, len(emails) - 1), random_state=42)
        embeddings_reduced = pca.fit_transform(embeddings)
        print(f"[Clustering] PCA: {embeddings.shape[1]}d → {embeddings_reduced.shape[1]}d "
              f"({pca.explained_variance_ratio_.sum():.1%} variance retained)")
    else:
        embeddings_reduced = embeddings

    # Determine k
    if n_clusters is None:
        if auto_k and len(emails) >= 6:
            k_max = min(10, len(emails) // 2)
            n_clusters = find_optimal_k(embeddings_reduced, k_min=3, k_max=k_max)
        else:
            n_clusters = min(5, len(emails))

    # Fit K-Means
    print(f"[Clustering] Running K-Means with k={n_clusters} on {len(emails)} emails...")
    km = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(embeddings_reduced)

    # Compute silhouette (quality metric)
    if len(set(labels)) > 1:
        sil = silhouette_score(embeddings_reduced, labels, sample_size=min(500, len(emails)))
    else:
        sil = 0.0

    # Save model
    with open(CLUSTER_MODEL_PATH, "wb") as f:
        pickle.dump(km, f)

    # Group emails by cluster
    cluster_groups = defaultdict(list)
    for email, label in zip(emails, labels):
        cluster_groups[int(label)].append(email)

    # Build result
    clusters = []
    for cluster_id, cluster_emails in sorted(cluster_groups.items()):
        texts = [f"{e.get('subject', '')} {e.get('summary', '')}" for e in cluster_emails]
        label = _infer_topic_label(texts)

        clusters.append({
            "id":     cluster_id,
            "label":  label,
            "count":  len(cluster_emails),
            "emails": [
                {
                    "id":         e.get("id"),
                    "subject":    e.get("subject", ""),
                    "sender":     e.get("sender", ""),
                    "cluster_id": cluster_id,
                    "priority":   e.get("priority", "LOW"),
                }
                for e in cluster_emails
            ]
        })

    # Sort by count descending
    clusters.sort(key=lambda c: c["count"], reverse=True)

    print(f"[Clustering] Done. {n_clusters} clusters, silhouette={sil:.4f}")
    for c in clusters:
        print(f"  Cluster {c['id']}: {c['label']} ({c['count']} emails)")

    return {
        "clusters":         clusters,
        "n_clusters":       n_clusters,
        "silhouette_score": round(float(sil), 4),
    }


def predict_cluster(text: str) -> Optional[int]:
    """
    Predicts which cluster a new email belongs to (for real-time incoming emails).
    Requires cluster model to be trained first.
    """
    if not CLUSTER_MODEL_PATH.exists():
        return None

    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    with open(CLUSTER_MODEL_PATH, "rb") as f:
        km = pickle.load(f)

    embedding = embedder.encode([text])
    label     = km.predict(embedding)[0]
    return int(label)


# ─── Supabase integration helpers ─────────────────────────────────────────────

def get_cluster_summary_for_dashboard(clusters: list[dict]) -> list[dict]:
    """
    Formats cluster data for frontend dashboard display.
    Returns a simplified list suitable for the Cluster view component.
    """
    colors = ["#7c3aed", "#2563eb", "#059669", "#d97706", "#dc2626",
              "#7c3aed", "#0891b2", "#9333ea", "#65a30d", "#ea580c"]

    return [
        {
            "id":    c["id"],
            "label": c["label"],
            "count": c["count"],
            "color": colors[c["id"] % len(colors)],
            "topEmails": c["emails"][:3],
        }
        for c in clusters
    ]


# ─── CLI test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from sentence_transformers import SentenceTransformer

    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    sample_emails = [
        {"id": 1, "subject": "Invoice #4821 overdue", "summary": "Payment required for invoice", "sender": "billing@vendor.com"},
        {"id": 2, "subject": "Q4 Budget Review", "summary": "Finance team needs sign-off on budget", "sender": "cfo@corp.com"},
        {"id": 3, "subject": "Sprint planning Monday", "summary": "Engineering team sprint kickoff", "sender": "pm@team.co"},
        {"id": 4, "subject": "Critical bug in production", "summary": "Server error affecting users", "sender": "devops@co.com"},
        {"id": 5, "subject": "Team offsite planning", "summary": "Vote for offsite location", "sender": "hr@corp.com"},
        {"id": 6, "subject": "Monthly newsletter", "summary": "Company highlights this month", "sender": "updates@corp.com"},
        {"id": 7, "subject": "Partnership proposal", "summary": "New business opportunity from Nexus Labs", "sender": "biz@nexus.io"},
        {"id": 8, "subject": "Tax filing deadline", "summary": "Please submit tax documents", "sender": "finance@corp.com"},
        {"id": 9, "subject": "New hire onboarding", "summary": "Welcome to the team checklist", "sender": "hr@corp.com"},
        {"id": 10, "subject": "API deprecation notice", "summary": "Old API endpoints will be removed", "sender": "dev@platform.io"},
    ]

    # Add embeddings
    texts = [f"{e['subject']} {e['summary']}" for e in sample_emails]
    vecs  = embedder.encode(texts).tolist()
    for e, v in zip(sample_emails, vecs):
        e["embedding"] = v

    result = cluster_emails(sample_emails)
    print("\n── Cluster Results ──")
    for c in result["clusters"]:
        print(f"\n📁 {c['label']} ({c['count']} emails)")
        for email in c["emails"]:
            print(f"   • {email['subject']}")
