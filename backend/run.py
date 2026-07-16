"""
backend/run.py
──────────────
Entry point for the FastAPI development server.

Run from project root:
    PYTHONPATH=. python backend/run.py
"""

import uvicorn

if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
