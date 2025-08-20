from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
from fastapi.middleware.cors import CORSMiddleware
import os
import uvicorn

# -------------------------
# Step 1: Load dataset
# -------------------------
with open("qa_dataset.json", "r", encoding="utf-8") as f:
    qa_pairs = json.load(f)

questions = [item['question'] for item in qa_pairs]

# -------------------------
# Step 2: Generate embeddings
# -------------------------
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(questions, convert_to_numpy=True)

# -------------------------
# Step 3: Build FAISS index
# -------------------------
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings.astype('float32'))

# -------------------------
# Step 4: Start FastAPI server
# -------------------------
app = FastAPI()

# -------------------------
# CORS Setup
# -------------------------
origins = [
    "https://resumefrontend-two.vercel.app",  # deployed frontend URL
    "http://localhost:5500",                  # local frontend testing
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # replace "*" if you want specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Request model
# -------------------------
class QueryRequest(BaseModel):
    message: str

# -------------------------
# API endpoint
# -------------------------
@app.post("/api/v1/resume")
async def get_answer(request: QueryRequest):
    query = request.message
    query_vec = model.encode([query], convert_to_numpy=True).astype('float32')
    
    k = 1  # top 1 match
    distances, indices = index.search(query_vec, k)
    best_match_index = indices[0][0]

    answer = qa_pairs[best_match_index]['answer']
    return {"reply": answer}

# -------------------------
# Optional root endpoint
# -------------------------
@app.get("/")
async def root():
    return {"message": "Vector search API running!"}

# -------------------------
# Run app with dynamic port for Render
# -------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # Render sets PORT automatically
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=True)
