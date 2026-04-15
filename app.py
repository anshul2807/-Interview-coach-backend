"""
🎯 RAG Interview Question Retrieval API
========================================
FastAPI server that loads the pre-built ChromaDB vector store from ./chroma_db
and serves interview questions based on user resume data.

Run:  uvicorn app:app --reload --port 8000
Docs: http://localhost:8000/docs
"""

import os
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
CHROMA_PERSIST_DIR = "./chroma_db"
COLLECTION_NAME    = "interview_questions"
EMBEDDING_DIM      = 384

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Global references (set during lifespan)
# ─────────────────────────────────────────────
embed_fn: ONNXMiniLM_L6_V2 = None  # type: ignore
collection = None


# ─────────────────────────────────────────────
# Lifespan
# ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ONNX embedding function + ChromaDB collection at startup."""
    global embed_fn, collection

    # ── Load ONNX embedding function (auto-downloads model on first run) ──
    logger.info("Initialising ONNXMiniLM_L6_V2 embedding function...")
    embed_fn = ONNXMiniLM_L6_V2()

    # Smoke-test
    test = embed_fn(["hello world"])
    assert len(test[0]) == EMBEDDING_DIM, f"Unexpected embedding dim: {len(test[0])}"
    logger.info(f"Embedding function ready — dim: {EMBEDDING_DIM}")

    # ── Connect to persisted ChromaDB ──
    if not os.path.exists(CHROMA_PERSIST_DIR):
        raise RuntimeError(
            f"ChromaDB persist directory not found: {CHROMA_PERSIST_DIR}\n"
            "Run the Jupyter notebook first to build the vector store."
        )

    chroma_client = chromadb.PersistentClient(
        path=CHROMA_PERSIST_DIR,
        settings=Settings(anonymized_telemetry=False),
    )

    existing = [c.name for c in chroma_client.list_collections()]
    if COLLECTION_NAME not in existing:
        raise RuntimeError(
            f"Collection '{COLLECTION_NAME}' not found in {CHROMA_PERSIST_DIR}.\n"
            f"Available collections: {existing}\n"
            "Run the Jupyter notebook first to build the vector store."
        )

    collection = chroma_client.get_collection(name=COLLECTION_NAME)
    logger.info(
        f"ChromaDB loaded — collection '{COLLECTION_NAME}' with {collection.count():,} documents"
    )

    yield  # ← app is running

    logger.info("Shutting down...")


# ─────────────────────────────────────────────
# FastAPI App
# ─────────────────────────────────────────────
app = FastAPI(
    title="RAG Interview Prep API",
    description=(
        "Retrieves interview questions from a 49K+ GFG dataset "
        "using semantic similarity. No LLM — pure vector retrieval."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# Pydantic Models
# ─────────────────────────────────────────────
class ProjectInfo(BaseModel):
    name: str = Field(..., example="E-commerce Platform")
    points: str = Field(
        ..., example="Built full-stack app with cart, payment, admin dashboard"
    )
    tech_used: list[str] = Field(..., example=["react", "node.js", "mongodb"])


class ExperienceInfo(BaseModel):
    company_name: str = Field(..., example="Infosys")
    tech_used: list[str]  = Field(..., example=["java", "spring boot", "aws"])
    points: str = Field(
        ..., example="Developed RESTful APIs for banking application"
    )


class ResumeInput(BaseModel):
    """User resume information for personalized question retrieval."""

    experience: str = Field(
        ...,
        description="Years of experience: '0', '0-1', '1-3', '3-5', '5+'",
        example="0-1",
    )
    company_looking: Optional[str] = Field(
        default="",
        description="Target company (optional). Leave empty to search all.",
        example="google",
    )
    tech_skills: list[str] = Field(
        ...,
        description="List of technical skills",
        example=["python", "java", "sql", "react"],
    )
    projects: list[ProjectInfo] = Field(
        default=[],
        description="List of projects (can be empty)",
    )
    user_experience: list[ExperienceInfo] = Field(
        default=[],
        description="List of work experiences (optional, can be empty)",
    )
    num_chunks: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of output chunks (each chunk has easy+medium+hard)",
    )
    results_per_difficulty: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of questions per difficulty level per chunk",
    )


class QuestionItem(BaseModel):
    question: str
    company: str
    topic: str
    difficulty: str
    exp_level: str
    similarity_score: float


class Chunk(BaseModel):
    chunk_number: int
    easy: list[QuestionItem]
    medium: list[QuestionItem]
    hard: list[QuestionItem]


class RetrievalResponse(BaseModel):
    query_text: str
    total_chunks: int
    total_questions: int
    chunks: list[Chunk]


# ─────────────────────────────────────────────
# Core Logic
# ─────────────────────────────────────────────
def build_query_from_resume(resume: ResumeInput) -> str:
    """
    Convert structured resume data into a single semantic query string
    that can be embedded and compared against the vector DB.
    """
    parts = []

    parts.append(f"experience level: {resume.experience} years")

    if resume.company_looking:
        parts.append(f"company: {resume.company_looking.lower()}")

    if resume.tech_skills:
        skills_str = ", ".join([s.lower() for s in resume.tech_skills])
        parts.append(f"technical skills: {skills_str}")

    for proj in resume.projects:
        proj_parts = []
        if proj.name:
            proj_parts.append(f"project {proj.name.lower()}")
        if proj.points:
            proj_parts.append(proj.points.lower())
        if proj.tech_used:
            tech = ", ".join([t.lower() for t in proj.tech_used])
            proj_parts.append(f"technologies used: {tech}")
        if proj_parts:
            parts.append(". ".join(proj_parts))

    for exp_item in resume.user_experience:
        exp_parts = []
        if exp_item.company_name:
            exp_parts.append(f"worked at {exp_item.company_name.lower()}")
        if exp_item.tech_used:
            tech = ", ".join([t.lower() for t in exp_item.tech_used])
            exp_parts.append(f"technologies: {tech}")
        if exp_item.points:
            exp_parts.append(exp_item.points.lower())
        if exp_parts:
            parts.append(". ".join(exp_parts))

    return ". ".join(parts)


def retrieve_questions(resume: ResumeInput) -> RetrievalResponse:
    """
    Retrieve interview questions from ChromaDB based on user resume.
    Returns chunks, each containing easy + medium + hard questions.
    """
    query_text = build_query_from_resume(resume)

    # Embed query using ONNXMiniLM_L6_V2 — returns list[list[float]]
    query_embedding = embed_fn([query_text])

    company = (resume.company_looking or "").lower().strip()
    all_results: dict[str, list[dict]] = {}

    for diff_level, diff_label in [(1, "easy"), (2, "medium"), (3, "hard")]:
        where_filter = (
            {"$and": [{"difficulty": diff_level}, {"company": company}]}
            if company
            else {"difficulty": diff_level}
        )

        try:
            results = collection.query(
                query_embeddings=query_embedding,
                n_results=resume.results_per_difficulty * resume.num_chunks,
                where=where_filter,
                include=["documents", "metadatas", "distances"],
            )
        except Exception:
            if company:
                logger.warning(
                    f"No {diff_label} questions for '{company}', broadening search..."
                )
                results = collection.query(
                    query_embeddings=query_embedding,
                    n_results=resume.results_per_difficulty * resume.num_chunks,
                    where={"difficulty": diff_level},
                    include=["documents", "metadatas", "distances"],
                )
            else:
                results = {"documents": [[]], "metadatas": [[]], "distances": [[]]}

        questions: list[dict] = []
        seen: set[str] = set()

        if results["documents"] and results["documents"][0]:
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                q_text = doc
                if "question:" in doc:
                    q_text = doc.split("question:")[-1].strip()

                q_key = q_text[:100].strip().lower()
                if q_key in seen:
                    continue
                seen.add(q_key)

                questions.append(
                    {
                        "question": q_text,
                        "company": meta.get("company", "unknown"),
                        "topic": meta.get("topic", "general"),
                        "difficulty": diff_label,
                        "exp_level": meta.get("exp_level", "unknown"),
                        "similarity_score": round(1 - dist, 4),
                    }
                )

        all_results[diff_label] = questions

    chunks = []
    total_questions = 0

    for chunk_idx in range(resume.num_chunks):
        chunk_data: dict = {
            "chunk_number": chunk_idx + 1,
            "easy": [],
            "medium": [],
            "hard": [],
        }
        for diff_label in ["easy", "medium", "hard"]:
            start = chunk_idx * resume.results_per_difficulty
            end   = start + resume.results_per_difficulty
            chunk_data[diff_label]  = all_results[diff_label][start:end]
            total_questions        += len(chunk_data[diff_label])

        chunks.append(chunk_data)

    return RetrievalResponse(
        query_text=query_text,
        total_chunks=len(chunks),
        total_questions=total_questions,
        chunks=chunks,
    )


# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────
@app.get("/")
async def root():
    """Health check & info."""
    doc_count = collection.count() if collection else 0
    return {
        "status": "running",
        "service": "RAG Interview Prep API",
        "embedding_fn": "ONNXMiniLM_L6_V2",
        "total_documents": doc_count,
        "docs_url": "/docs",
    }


@app.post("/retrieve", response_model=RetrievalResponse)
async def retrieve(resume: ResumeInput):
    """
    🎯 Retrieve interview questions based on user resume.

    Returns questions grouped in chunks — each chunk contains
    **easy**, **medium**, and **hard** questions.
    """
    if collection is None:
        raise HTTPException(status_code=503, detail="Vector store not loaded")

    try:
        return retrieve_questions(resume)
    except Exception as e:
        logger.error(f"Retrieval error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def stats():
    """Collection statistics."""
    if collection is None:
        raise HTTPException(status_code=503, detail="Vector store not loaded")

    return {
        "collection_name": COLLECTION_NAME,
        "total_documents": collection.count(),
        "persist_directory": CHROMA_PERSIST_DIR,
        "embedding_fn": "ONNXMiniLM_L6_V2",
        "embedding_dim": EMBEDDING_DIM,
    }