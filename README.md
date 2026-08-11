# Backend — Multi-Agent Research & Analytics API

FastAPI + LangGraph service that runs the multi-agent RAG workflow
(Orchestrator → Supabase pgvector Retrieval → Tavily Web Search Fallback → Synthesis).

## Files
- `main.py` — LangGraph agent graph (orchestrator, retriever, web search, synthesis nodes)
- `server.py` — FastAPI app exposing `POST /api/research` and `GET /health`
- `requirements.txt` — Python dependencies
- `runtime.txt` / `Procfile` / `render.yaml` — deployment configs for Render
- `vercel.json` — not needed for Render; only relevant if you ever host the
  backend on Vercel instead (kept here for reference)
- `.env.example` — required environment variables (Groq, Supabase, Tavily keys)

## Deploying to Render

1. **Push this `backend/` folder to a GitHub repo** (it can be its own repo,
   or a subfolder of a monorepo — see step 3).

2. **Create a new Web Service on Render**
   - Go to https://dashboard.render.com → New → Web Service
   - Connect your GitHub repo

3. **If `backend/` is a subfolder of a bigger repo**, set the
   **Root Directory** to `backend` in Render's service settings so Render
   only builds/deploys this folder.

4. **Render will detect `render.yaml`** and pre-fill the settings below
   automatically. If you're configuring manually instead, use:
   - **Environment**: Python 3
   - **Build Command**:
     ```
     pip install --upgrade pip && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.5.1 && pip install --no-cache-dir -r requirements.txt
     ```
   - **Start Command**:
     ```
     uvicorn server:app --host 0.0.0.0 --port $PORT
     ```
   - **Health Check Path**: `/health`

5. **Set environment variables** in Render's dashboard (Environment tab):
   | Key | Value |
   |---|---|
   | `GROQ_API_KEY` | your Groq API key |
   | `SUPABASE_URL` | your Supabase project URL |
   | `SUPABASE_SERVICE_KEY` | your Supabase service role key |
   | `TAVILY_API_KEY` | your Tavily API key |
   | `DOC_RELEVANCE_THRESHOLD` | `2` (or your preferred value) |

   These are all optional at the code level (the app degrades gracefully
   without them — falls back to mock data / offline reports), but you need
   real keys for full functionality.

6. **Deploy.** Render will build and start the service. Your backend will be
   live at a URL like:
   ```
   https://multi-agent-rag-studio.onrender.com
   ```

7. **Verify it's working**:
   ```bash
   curl https://multi-agent-rag-studio.onrender.com/health
   # {"status":"ok","message":"Multi-Agent Research API is running"}
   ```

8. **Note the exact URL** — you'll need it to configure the frontend (see
   `frontend/README.md`, `frontend/config.js`).

### Notes
- Render's free/starter tier spins down on inactivity; the first request
  after idling can take 30–60s to respond (cold start). The frontend already
  handles slow/502 responses gracefully during this window.
- CORS is already enabled for all origins (`allow_origins=["*"]`) in
  `server.py`, so no backend changes are needed to accept requests from your
  Vercel-hosted frontend domain. If you want to lock this down later, change
  `allow_origins=["*"]` to `allow_origins=["https://your-frontend.vercel.app"]`.
