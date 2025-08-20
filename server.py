from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
from fastapi.middleware.cors import CORSMiddleware

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

# Allow frontend from localhost
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request model
class QueryRequest(BaseModel):
    message: str

# API endpoint
@app.post("/api/v1/resume")
async def get_answer(request: QueryRequest):
    query = request.message
    query_vec = model.encode([query], convert_to_numpy=True).astype('float32')
    
    k = 1  # top 1 match
    distances, indices = index.search(query_vec, k)
    best_match_index = indices[0][0]

    answer = qa_pairs[best_match_index]['answer']
    return {"reply": answer}

# Optional: root endpoint
@app.get("/")
async def root():
    return {"message": "Vector search API running!"}
