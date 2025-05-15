from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import faiss

# تحميل البيانات
data = pd.read_csv("chatbot_dataset_Final.csv")
questions = data["السؤال"].tolist()
answers = data["الإجابة"].tolist()

# تحميل نموذج SBERT
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

# استخراج التمثيلات العددية للأسئلة
def normalize(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norms

question_embeddings = model.encode(questions, convert_to_numpy=True)
question_embeddings = normalize(question_embeddings.astype(np.float32))

# إنشاء فهرس FAISS
index = faiss.IndexFlatIP(question_embeddings.shape[1])
index.add(question_embeddings)

# FastAPI
app = FastAPI()

class Question(BaseModel):
    text: str

@app.post("/ask")
def ask_question(q: Question):
    user_embedding = normalize(model.encode([q.text], convert_to_numpy=True).astype(np.float32))
    similarities, best_index = index.search(user_embedding, 1)
    best_match = best_index[0][0]

    if similarities[0][0] > 0.6:
        return {"answer": answers[best_match]}
    else:
        return {"answer": "عذرًا، لا أملك إجابة لهذا السؤال."}