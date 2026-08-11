"""
FastAPI Server & Web Bridge for Multi-Agent Research System
============================================================
Exposes REST endpoints to trigger the LangGraph workflow and serves
the interactive front-end web dashboard.
"""

import os
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

# Import LangGraph research graph from main.py
from main import build_research_graph, ResearchState, get_embeddings

app = FastAPI(
    title="Multi-Agent Research & Analytics API",
    description="API bridge for LangGraph RAG & Web Search Fallback system",
    version="1.0.0"
)


@app.on_event("startup")
async def warm_embedding_model():
    """
    The HuggingFace embedding model (all-MiniLM-L6-v2) is downloaded/loaded
    lazily on first use. Loading it here at startup (rather than letting the
    first user request trigger it) avoids that request hanging long enough
    to hit Render's proxy timeout and return a 502.
    """
    try:
        print("[Startup] Warming embedding model...")
        get_embeddings()
        print("[Startup] Embedding model ready.")
    except Exception as e:
        print(f"[Startup] Embedding model warm-up failed (non-fatal): {e}")

# Enable CORS for all origins (no credentials/cookies needed by this app,
# so allow_credentials stays False — combining "*" with credentials=True
# is invalid per the CORS spec and browsers will silently block requests)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class APIKeysConfig(BaseModel):
    """User-supplied API keys sent from the frontend per request."""
    groq_api_key: Optional[str] = Field(None, description="Groq API Key")
    supabase_url: Optional[str] = Field(None, description="Supabase Project URL")
    supabase_service_key: Optional[str] = Field(None, description="Supabase Service Role Key")
    tavily_api_key: Optional[str] = Field(None, description="Tavily Search API Key")


class ResearchRequest(BaseModel):
    topic: str = Field(..., description="Broad research topic query", json_schema_extra={"example": "High-Performance RAG Architecture"})
    doc_threshold: Optional[int] = Field(2, description="Minimum retrieved doc threshold before fallback")
    api_keys: Optional[APIKeysConfig] = Field(None, description="User-provided API keys (overrides server .env keys)")


class ResearchResponse(BaseModel):
    topic: str
    search_queries: List[str]
    retrieved_docs_count: int
    retrieved_docs: List[Dict[str, Any]]
    web_search_results: List[Dict[str, Any]]
    need_web_search: bool
    final_report: str


# ── Health check endpoint ──────────────────────────────────────────────────────
@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "Multi-Agent Research API is running"}


# ── Main research endpoint ─────────────────────────────────────────────────────
@app.post("/api/research", response_model=ResearchResponse)
async def run_research(request: ResearchRequest):
    """
    Executes the multi-agent graph workflow for the given research topic.
    API keys provided in the request body are passed directly into the graph
    builder so each request uses fresh credentials (fixes the startup race condition).
    """
    if not request.topic.strip():
        raise HTTPException(status_code=400, detail="Research topic cannot be empty.")

    # ── Resolve effective API keys: prefer per-request keys, fall back to env vars ──
    effective_groq_key = (
        (request.api_keys.groq_api_key if request.api_keys else None)
        or os.getenv("GROQ_API_KEY", "")
    )
    effective_tavily_key = (
        (request.api_keys.tavily_api_key if request.api_keys else None)
        or os.getenv("TAVILY_API_KEY", "")
    )
    effective_supabase_url = (
        (request.api_keys.supabase_url if request.api_keys else None)
        or os.getenv("SUPABASE_URL", "")
    )
    effective_supabase_key = (
        (request.api_keys.supabase_service_key if request.api_keys else None)
        or os.getenv("SUPABASE_SERVICE_KEY", "")
    )

    doc_threshold = request.doc_threshold if request.doc_threshold is not None else 2

    initial_state: ResearchState = {
        "topic": request.topic.strip(),
        "search_queries": [],
        "retrieved_docs": [],
        "web_search_results": [],
        "need_web_search": False,
        "final_report": ""
    }

    try:
        print(f"\n[Server] Research request for topic: '{request.topic}'")
        print(f"[Server] Groq key present: {bool(effective_groq_key)}, Tavily key present: {bool(effective_tavily_key)}")

        # Build a fresh graph with the resolved API keys for this request
        research_graph = build_research_graph(
            groq_api_key=effective_groq_key,
            tavily_api_key=effective_tavily_key,
            supabase_url=effective_supabase_url,
            supabase_service_key=effective_supabase_key,
            doc_threshold=doc_threshold,
        )

        final_state = research_graph.invoke(initial_state)

        # Serialize retrieved docs to JSON-safe format
        formatted_docs = []
        for doc in final_state.get("retrieved_docs", []):
            formatted_docs.append({
                "page_content": doc.page_content,
                "source": doc.metadata.get("source", "Supabase Vector Store"),
                "metadata": doc.metadata
            })

        return ResearchResponse(
            topic=final_state["topic"],
            search_queries=final_state.get("search_queries", []),
            retrieved_docs_count=len(formatted_docs),
            retrieved_docs=formatted_docs,
            web_search_results=final_state.get("web_search_results", []),
            need_web_search=final_state.get("need_web_search", False),
            final_report=final_state.get("final_report", "")
        )

    except Exception as e:
        print(f"[Server Error] Research workflow failed: {e}")
        raise HTTPException(status_code=500, detail=f"Multi-Agent execution error: {str(e)}")


# ── Static files & frontend (optional) ─────────────────────────────────────────
# This backend is deployed API-only (frontend is hosted separately, e.g. on
# Vercel). If a static/ folder happens to exist alongside this file, it will
# still be served for convenience; otherwise this is skipped entirely so the
# app starts cleanly with no frontend files present.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", response_class=FileResponse)
    async def read_index():
        index_path = os.path.join(STATIC_DIR, "index.html")
        if os.path.exists(index_path):
            return FileResponse(index_path, media_type="text/html")
        raise HTTPException(status_code=404, detail=f"index.html not found at {index_path}")
else:
    @app.get("/")
    async def read_root():
        return {"status": "ok", "message": "Multi-Agent Research API is running. See /health and /api/research."}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    print("=" * 70)
    print("  Starting Multi-Agent Research Web Dashboard Server")
    print(f"  Access dashboard at: http://127.0.0.1:{port}")
    print("=" * 70)
    uvicorn.run(app, host="0.0.0.0", port=port)
