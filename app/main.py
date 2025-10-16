from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware
from sentence_transformers import SentenceTransformer
import numpy as np
import json
import logging
import os
import csv
from datetime import datetime

from app.models import SearchRequest, ScoredChunk, ScoredProject, SearchResponse


LOG = logging.getLogger('uvicorn.error')
SCORE_THRESHOLD = 0.4
K = 5

PROJECTS_FILE = "data/projects.json"
CHUNKS_FILE = "data/chunks.json"
EMBEDDED_CHUNKS_FILE = "data/embedded_chunks.npy"
LOG_FILE = "data/logs.csv"


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    with open(PROJECTS_FILE, "r") as f:
        app.state.projects = json.load(f)
    with open(CHUNKS_FILE, "r") as f:
        app.state.chunks = json.load(f)
    app.state.embedded_chunks = np.load(EMBEDDED_CHUNKS_FILE)
    LOG.info(f"{len(app.state.projects)} projects loaded and {len(app.state.chunks)} chunks...")

    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["date", "address", "query", "response"])

    yield


app = FastAPI(lifespan=lifespan)

origins = [
    "http://localhost:1313",
    "https://opetit.fr",
    "https://www.opetit.fr"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(ProxyHeadersMiddleware, trusted_hosts="*")


@app.post("/search")
async def ranker(input_query: SearchRequest, request: Request):
    embedded_query = app.state.embedding_model.encode_query(input_query.query, prompt="Demande du client : ")
    similarities = app.state.embedding_model.similarity(embedded_query, app.state.embedded_chunks)[0].numpy()
    ranked_ids = np.argsort(similarities)[::-1][:K] #  Top K

    top_k_chunks = []
    project_stats = {}
    for idx in ranked_ids:
        chunk = app.state.chunks[idx]
        score = float(similarities[idx])
        if (score > SCORE_THRESHOLD):
            project_id = chunk["project_id"]
            if project_id not in project_stats:
                project_stats[project_id] = []
            project_stats[project_id].append(score)
            top_k_chunks.append(ScoredChunk(project_id=chunk["project_id"],
                                            score=score,
                                            content=chunk["content"]))

    for k, v in project_stats.items():
        project_stats[k] = np.mean(v)
        
    top_projects = []
    for idx in sorted(project_stats, key=project_stats.get, reverse=True):
        project = app.state.projects[idx]
        top_projects.append(ScoredProject(id=project["id"],
                                          score=float(project_stats[idx]),
                                          name=project["name"],
                                          company=project["company"],
                                          description=project["full_desc"]))
        
    response = SearchResponse(chunks=top_k_chunks, projects=top_projects)
        
    row = {
        "date": datetime.now().isoformat(),
        "address": request.client.host,
        "query": input_query.query,
        "response": response.model_dump_json()
    }
    with open(LOG_FILE, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        writer.writerow(row)
        
    return response


@app.get("/health")
def health():
    return {"running": True}