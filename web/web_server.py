import sys
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import uvicorn

# Add the project root to the Python path to allow imports from the main directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from application import Application

# --- FastAPI App Initialization with Lifespan ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles application startup and shutdown events.
    This is the recommended way to manage resources in modern FastAPI.
    """
    # --- Startup ---
    print("Initializing RAG application for the web server...")
    # Create and setup the application instance
    rag_app = Application()
    try:
        rag_app.setup()
        app.state.rag_app = rag_app
        print("RAG application is ready to serve requests.")
    except Exception as e:
        print(f"FATAL: RAG application setup failed: {e}")
        # Store None to indicate failure, so endpoints can handle it gracefully
        app.state.rag_app = None
    
    yield # The application runs here

    # --- Shutdown ---
    print("Web server is shutting down.")
    app.state.rag_app = None

app = FastAPI(lifespan=lifespan)
templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))

# --- API Endpoints ---
@app.get("/", response_class=HTMLResponse)
async def get_main_page(request: Request):
    """Serves the main HTML page."""
    rag_app = request.app.state.rag_app
    setup_error = "RAG application failed to initialize. Please check server logs." if not (rag_app and rag_app.is_setup) else None
    return templates.TemplateResponse("index.html", {"request": request, "setup_error": setup_error})

@app.post("/api/chat", response_class=JSONResponse) # Changed to JSONResponse
async def chat_endpoint(request: Request):
    """Handles a chat question, returns the answer as JSON."""
    data = await request.json()
    question = data.get("question")

    if not question:
        return JSONResponse({"error": "No question provided"}, status_code=400)

    rag_app = request.app.state.rag_app
    answer = "Error: RAG application is not initialized. Please check server logs for details." # Default error message
    
    if rag_app and rag_app.is_setup:
        print(f"Received question: {question}")
        answer = rag_app.ask(question)
        print(f"Generated answer: {answer}")
    else:
        print("RAG application not set up, returning error for chat.")

    return JSONResponse({"answer": answer}) # Return JSON

@app.post("/api/summarize", response_class=JSONResponse)
async def summarize_chat(request: Request):
    """Summarizes the provided conversation history."""
    data = await request.json()
    conversation_history = data.get("conversation_history") # Expecting a string or list of strings

    if not conversation_history:
        return JSONResponse({"error": "No conversation history provided for summarization"}, status_code=400)

    # If conversation_history is a list, join it into a single string
    if isinstance(conversation_history, list):
        conversation_history = "\n".join(conversation_history)

    rag_app = request.app.state.rag_app
    summary = "Error: RAG application is not initialized or summarization failed."

    if rag_app and rag_app.is_setup:
        print("Received request to summarize conversation.")
        summary = rag_app.summarize_conversation(conversation_history)
        print(f"Generated summary: {summary[:100]}...") # Print first 100 chars
    else:
        print("RAG application not set up, returning error for summarization.")

    return JSONResponse({"summary": summary})

def start():
    """Starts the Uvicorn web server."""
    print("Starting web server...")
    # Note: `reload=True` will re-run the lifespan events on each code change.
    # This is useful for development but can be slow if your setup() is intensive.
    uvicorn.run("web.web_server:app", host="0.0.0.0", port=8000, reload=True, log_level="info")

if __name__ == "__main__":
    start()