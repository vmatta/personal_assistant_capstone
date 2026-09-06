"""Persistent memory: vector-indexed notes/knowledge base (RAG) plus a
simple JSON-backed to-do/schedule store used by the Scheduler & To-Do agent.
"""

import json
import re
import threading
import time
import uuid
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.utils import embedding_functions

from config import SETTINGS

logger = logging.getLogger("personal_assistant")

_VECTOR_CFG = SETTINGS.get("vector_db", {})
_PERSIST_DIR = Path(_VECTOR_CFG.get("persist_directory", "data/chromadb"))
_PERSIST_DIR.mkdir(parents=True, exist_ok=True)

_TODO_STORE_PATH = _PERSIST_DIR.parent / "todos.json"

# Lazy-load ChromaDB client and embedding function to avoid slow startup.
# A lock guards initialization so a background warm-up thread and a live
# request racing to be "first" never both pay the ~10s cold-start cost.
_client = None
_embedding_fn = None
_init_lock = threading.Lock()

def _get_client():
    """Get ChromaDB client, lazily initializing on first use (thread-safe)."""
    global _client
    if _client is None:
        with _init_lock:
            if _client is None:
                logger.debug("Initializing ChromaDB client...")
                _client = chromadb.PersistentClient(path=str(_PERSIST_DIR))
    return _client

def _get_embedding_fn():
    """Get embedding function, lazily initializing on first use (thread-safe)."""
    global _embedding_fn
    if _embedding_fn is None:
        with _init_lock:
            if _embedding_fn is None:
                logger.debug("Initializing sentence-transformer embedding model...")
                _embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name=_VECTOR_CFG.get("embedding_model", "all-MiniLM-L6-v2")
                )
                logger.debug("✓ Embedding model loaded")
    return _embedding_fn


_warm_up_thread: Optional[threading.Thread] = None


def warm_up_async() -> None:
    """Kick off ChromaDB + embedding-model initialization on a background thread.

    Call this once as soon as the app starts (e.g. Streamlit page load) so the
    ~10s cold-start cost of loading the sentence-transformer model happens
    while the user is reading the page / typing their first message, instead
    of blocking their first scheduling/notes request. Safe to call multiple
    times; only the first call actually starts a thread. Thread-safe against
    a real request racing in via the locks in _get_client/_get_embedding_fn.
    """
    global _warm_up_thread
    if _client is not None and _embedding_fn is not None:
        return  # already warm
    if _warm_up_thread is not None and _warm_up_thread.is_alive():
        return  # already warming up

    def _run():
        try:
            logger.debug("🔥 Warming up ChromaDB + embedding model in background...")
            _get_embedding_fn()
            _get_client()
            logger.debug("✓ Background warm-up complete")
        except Exception as exc:  # never let warm-up crash the app
            logger.warning(f"Background warm-up failed (will retry lazily on demand): {exc}")

    _warm_up_thread = threading.Thread(target=_run, name="memory-warmup", daemon=True)
    _warm_up_thread.start()

_NOTES_COLLECTION = _VECTOR_CFG.get("collections", {}).get("notes", "user_personal_notes")
_KNOWLEDGE_COLLECTION = _VECTOR_CFG.get("collections", {}).get("knowledge_base", "document_knowledge")


def _get_collection(name: str):
    return _get_client().get_or_create_collection(name=name, embedding_function=_get_embedding_fn())


def _normalize_note_text(text: str) -> str:
    normalized = re.sub(r"[^\w\s]", " ", (text or "").casefold())
    return " ".join(normalized.split())


def save_note(text: str) -> Optional[str]:
    """Persist a note and index it for semantic retrieval.

    Returns the new note id if inserted, or None if the same note is already
    present in the collection.
    """
    normalized = _normalize_note_text(text)
    if normalized:
        try:
            existing = _get_collection(_NOTES_COLLECTION).get(include=["documents"])
            for document in existing.get("documents") or []:
                if _normalize_note_text(document) == normalized:
                    return None
        except Exception:
            # Fall through to a normal save even if the lookup itself failed. This
            # keeps the app resilient while preserving the duplicate guard in the
            # common case of a valid persisted collection.
            pass

    note_id = str(uuid.uuid4())
    _get_collection(_NOTES_COLLECTION).add(
        documents=[text], ids=[note_id], metadatas=[{"created_at": time.time()}]
    )
    return note_id


def search_notes(query: str, k: int = 3) -> list[str]:
    """Semantic search over saved notes; returns up to k matching note texts."""
    result = _get_collection(_NOTES_COLLECTION).query(query_texts=[query], n_results=k)
    return result.get("documents", [[]])[0]


def list_notes(limit: int = 10) -> list[str]:
    """Return recently stored note texts for observability/debugging."""
    collection = _get_collection(_NOTES_COLLECTION)
    result = collection.get(include=["documents", "metadatas"])
    documents = result.get("documents") or []
    metadatas = result.get("metadatas") or []
    rows = list(zip(documents, metadatas))
    rows.sort(key=lambda row: (row[1] or {}).get("created_at", 0), reverse=True)
    return [document for document, _metadata in rows[:limit]]


def search_knowledge_base(query: str, k: int = 3) -> list[str]:
    """Semantic search over the general knowledge base collection."""
    result = _get_collection(_KNOWLEDGE_COLLECTION).query(query_texts=[query], n_results=k)
    return result.get("documents", [[]])[0]

def delete_note(note_id: str) -> bool:
    """Delete a saved note by ID. Returns True if successful, False if not found."""
    try:
        _get_collection(_NOTES_COLLECTION).delete(ids=[note_id])
        return True
    except Exception:
        return False

def _load_todos() -> list[dict]:
    if not _TODO_STORE_PATH.exists():
        return []
    raw_text = _TODO_STORE_PATH.read_text(encoding="utf-8")
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        todos, end_index = decoder.raw_decode(raw_text)
        if raw_text[end_index:].strip():
            _save_todos(todos)
        return todos


def _save_todos(todos: list[dict]) -> None:
    _TODO_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temp_path = _TODO_STORE_PATH.with_suffix(f"{_TODO_STORE_PATH.suffix}.tmp")
    temp_path.write_text(json.dumps(todos, indent=2), encoding="utf-8")
    temp_path.replace(_TODO_STORE_PATH)


def _parse_due(due: Optional[str]) -> Optional[datetime]:
    if not due:
        return None
    try:
        return datetime.fromisoformat(due.replace("Z", "+00:00"))
    except ValueError:
        return None


def _same_due(left: Optional[str], right: Optional[str]) -> bool:
    if not left or not right:
        return False
    if left == right:
        return True
    left_dt = _parse_due(left)
    right_dt = _parse_due(right)
    if left_dt is None or right_dt is None:
        return False
    if left_dt.tzinfo and right_dt.tzinfo:
        return left_dt.astimezone(timezone.utc) == right_dt.astimezone(timezone.utc)
    return left_dt.replace(tzinfo=None) == right_dt.replace(tzinfo=None)


def find_open_todos_at_due(due: Optional[str]) -> list[dict]:
    """Return open to-dos/scheduled items that occupy the same due instant."""
    if not due:
        return []
    return [todo for todo in _load_todos() if not todo["done"] and _same_due(todo.get("due"), due)]


def add_todo(description: str, due: Optional[str] = None) -> dict:
    """Add a to-do/scheduled item and persist it."""
    todos = _load_todos()
    item = {"id": str(uuid.uuid4()), "description": description, "due": due, "done": False}
    todos.append(item)
    _save_todos(todos)
    return item


def list_todos(include_done: bool = False) -> list[dict]:
    """List stored to-do/scheduled items."""
    todos = _load_todos()
    return todos if include_done else [t for t in todos if not t["done"]]


def complete_todo(todo_id: str) -> bool:
    """Mark a to-do item as done. Returns True if found and updated."""
    todos = _load_todos()
    for t in todos:
        if t["id"] == todo_id:
            t["done"] = True
            _save_todos(todos)
            return True
    return False
