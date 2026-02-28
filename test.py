from app.services.embedding_service import generate_embedding

@app.get("/test-embedding")
def test_embedding():
    vector = generate_embedding("Submit report by friday")
    return {"length": len(vector)}