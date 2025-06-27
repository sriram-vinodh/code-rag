import sys
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, Form, Request
import logging
from pydantic import BaseModel
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import uvicorn

# Add the project root to the Python path to allow imports from the main directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from application import Application

class ChatRequest(BaseModel):
    """Pydantic model for chat requests."""
    question: str

class SummarizeRequest(BaseModel):
    """Pydantic model for summarization requests."""
    conversation_history: list[str]

logger = logging.getLogger(__name__)

# --- FastAPI App Initialization with Lifespan ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles application startup and shutdown events.
    This is the recommended way to manage resources in modern FastAPI.
    """ # noqa
    # --- Startup ---
    logger.info("Initializing RAG application for the web server...")
    # Create and setup the application instance
    rag_app = Application()
    try:
        rag_app.setup()
        app.state.rag_app = rag_app
        logger.info("RAG application is ready to serve requests.")
    except Exception as e:
        logger.critical(f"RAG application setup failed: {e}", exc_info=True)
        # Store None to indicate failure, so endpoints can handle it gracefully
        app.state.rag_app = None
    
    yield # The application runs here

    # --- Shutdown ---
    logger.info("Web server is shutting down.")
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
async def chat_endpoint(chat_request: ChatRequest, request: Request):
    """Handles a chat question, returns the answer as JSON."""
    if not chat_request.question:
        return JSONResponse({"error": "No question provided"}, status_code=400)

    rag_app = request.app.state.rag_app
    answer = "Error: RAG application is not initialized. Please check server logs for details." # Default error message
    
    if rag_app and rag_app.is_setup: # noqa
        logger.info(f"Received question: {chat_request.question}")
        answer = rag_app.ask(chat_request.question)
        logger.info(f"Generated answer: {answer}")
    else:
        logger.warning("RAG application not set up, returning error for chat.")

    return JSONResponse({"answer": answer}) # Return JSON

@app.post("/api/summarize", response_class=JSONResponse)
async def summarize_chat(summarize_request: SummarizeRequest, request: Request):
    """Summarizes the provided conversation history."""
    if not summarize_request.conversation_history:
        return JSONResponse({"error": "No conversation history provided for summarization"}, status_code=400)

    # If conversation_history is a list, join it into a single string
    conversation_text = "\n".join(summarize_request.conversation_history)

    rag_app = request.app.state.rag_app
    summary = "Error: RAG application is not initialized or summarization failed." # noqa

    if rag_app and rag_app.is_setup: # noqa
        logger.info("Received request to summarize conversation.")
        summary = rag_app.summarize_conversation(conversation_text) # noqa
        logger.info(f"Generated summary: {summary[:100]}...") # Print first 100 chars
    else:
        logger.warning("RAG application not set up, returning error for summarization.")

    return JSONResponse({"summary": summary})