import os
import pickle
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request

# gmail.modify allows reading + trashing/modifying emails
SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CREDENTIALS_PATH = os.path.join(BASE_DIR, "credentials.json")
TOKEN_PATH = os.path.join(BASE_DIR, "token.pickle")


def get_gmail_service(credentials=None):
    """
    Return a Gmail API service.
    - If `credentials` are supplied (from the OAuth web flow session), use them directly.
    - Otherwise try token.pickle (single-user dev mode, only if the file already exists).
    - Never blocks the server waiting for a browser — raises ValueError if no creds available.
    """
    if credentials:
        return build("gmail", "v1", credentials=credentials, cache_discovery=False)

    # Fallback: token.pickle from a previous InstalledAppFlow run (dev only)
    if os.path.exists(TOKEN_PATH):
        with open(TOKEN_PATH, "rb") as f:
            creds = pickle.load(f)
        if creds and creds.valid:
            return build("gmail", "v1", credentials=creds, cache_discovery=False)
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
            with open(TOKEN_PATH, "wb") as f:
                pickle.dump(creds, f)
            return build("gmail", "v1", credentials=creds, cache_discovery=False)

    raise ValueError("No Gmail credentials. Sign in via /auth/login first.")


def fetch_recent_emails(max_results=50, credentials=None):
    service = get_gmail_service(credentials=credentials)

    results = service.users().messages().list(
        userId="me",
        maxResults=max_results,
        labelIds=["INBOX"]   # only inbox — excludes Sent, Drafts, Spam
    ).execute()

    messages = results.get("messages", [])
    emails = []

    # Gmail category label IDs that mean "not primary inbox"
    PROMO_LABELS = {
        "CATEGORY_PROMOTIONS",
        "CATEGORY_UPDATES",
        "CATEGORY_SOCIAL",
        "CATEGORY_FORUMS",
    }

    for msg in messages:
        msg_data = service.users().messages().get(
            userId="me",
            id=msg["id"]
        ).execute()

        headers = msg_data["payload"]["headers"]
        subject = ""
        sender  = ""
        has_unsubscribe = False

        for header in headers:
            name = header["name"]
            if name == "Subject":
                subject = header["value"]
            elif name == "From":
                sender = header["value"]
            elif name in ("List-Unsubscribe", "List-Unsubscribe-Post"):
                has_unsubscribe = True

        snippet    = msg_data.get("snippet", "")
        label_ids  = set(msg_data.get("labelIds", []))
        is_promo   = bool(label_ids & PROMO_LABELS)  # True if Gmail flagged as non-primary

        emails.append({
            "id":              msg["id"],
            "subject":         subject,
            "sender":          sender,
            "snippet":         snippet,
            "is_promo":        is_promo,
            "has_unsubscribe": has_unsubscribe,
            "gmail_labels":    list(label_ids),
        })

    return emails


def trash_email_message(message_id: str):
    service = get_gmail_service()
    return service.users().messages().trash(userId="me", id=message_id).execute()
