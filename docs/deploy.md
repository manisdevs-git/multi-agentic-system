# Deploy: Vercel front end, Railway back end

The browser never runs LangGraph. Railway runs the graph, corpus, `out/`, and MCP Send. Vercel serves `frontend/`.

## Railway (API)

1. New project from this GitHub repo.
2. Start command is already in `Procfile` / `railway.toml`:
   `PYTHONPATH=src python -m uvicorn support_reply.api:app --host 0.0.0.0 --port $PORT`
3. Variables:
   - `DEEPINFRA_API_KEY` (optional; cue path works without it)
   - `FRONTEND_ORIGIN` = `https://<your-app>.vercel.app` (comma-separate `http://localhost:5173` if needed)
   - `MCP_SERVER_URL` if Send should work (must be reachable from Railway, not `127.0.0.1` on your laptop)
   - `MAIL_TO`, `MCP_AUTH_TOKEN` as needed
4. Public URL, e.g. `https://support-reply-api.up.railway.app`. Check `GET /health`.

Sessions are in memory. Restarting the service drops threads that were not approved to `out/`.

## Vercel (UI)

1. Import the same GitHub repo.
2. Root Directory: `frontend`
3. Build: `npm run build` · Output: `dist`
4. Env: `VITE_API_URL` = the Railway origin, **no trailing slash**

## Local

```text
PYTHONPATH=src python -m uvicorn support_reply.api:app --host 127.0.0.1 --port 8000
cd frontend && npm install && npm run dev
```

Open `http://localhost:5173`. Vite proxies `/v1` to `:8000` when `VITE_API_URL` is empty.

`streamlit run streamlit_app.py` still works for an in-process desk. It is not the Vercel app.

## Do not put on Vercel

Python, `data/`, `out/`, MCP, Streamlit.
