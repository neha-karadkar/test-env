import asyncio as _asyncio

import time as _time
from observability.observability_wrapper import (
    trace_agent, trace_step, trace_step_sync, trace_model_call, trace_tool_call,
)
from config import settings as _obs_settings

import logging as _obs_startup_log
from contextlib import asynccontextmanager
from observability.instrumentation import initialize_tracer

_obs_startup_logger = _obs_startup_log.getLogger(__name__)

from modules.guardrails.content_safety_decorator import with_content_safety

GUARDRAILS_CONFIG = {
    'content_safety_enabled': True,
    'runtime_enabled': True,
    'content_safety_severity_threshold': 3,
    'check_toxicity': True,
    'check_jailbreak': True,
    'check_pii_input': False,
    'check_credentials_output': True,
    'check_output': True,
    'check_toxic_code_output': True,
    'sanitize_pii': False
}

import logging
import json
from typing import List, Optional, Dict, Any
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, model_validator

from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.models import VectorizedQuery
import openai

from config import Config

# =========================
# Constants and Configuration
# =========================

SYSTEM_PROMPT = (
    "You are a professional assistant specializing in answering user questions using information from a curated collection of resumes. "
    "Your task is to interpret the user's question, retrieve relevant context from the provided resume documents, and deliver a clear, accurate, and concise answer based strictly on the retrieved content. "
    "Do not fabricate information or provide details not present in the knowledge base. If the answer cannot be found, politely inform the user that the requested information is not available."
)
OUTPUT_FORMAT = (
    "Provide a direct, well-structured answer to the user's question, referencing only information found in the retrieved resume documents. "
    "Use complete sentences and maintain a professional tone."
)
FALLBACK_RESPONSE = (
    "I'm sorry, but I could not find relevant information to answer your question based on the available resume documents."
)
FEW_SHOT_EXAMPLES = [
    "What programming languages does Jane Smith know? -> Jane Smith is proficient in JavaScript (ES6+), TypeScript, HTML5, CSS3/SASS, and Python, as indicated in her resume.",
    "Who has experience with Tableau dashboards? -> Bob Williams has experience designing and maintaining interactive Tableau dashboards for executive reporting, as detailed in his resume."
]
SELECTED_DOCUMENT_TITLES = [
    "resumes_collection1.pdf",
    "resumes_collection2.pdf",
    "resumes_collection3.pdf",
    "resumes_collection4.pdf"
]
ENRICHED_FIELDS = ["entities", "keyphrases", "relationships"]
VALIDATION_CONFIG_PATH = Config.VALIDATION_CONFIG_PATH or str(Path(__file__).parent / "validation_config.json")

_logger = logging.getLogger(__name__)

# =========================
# Observability Lifespan
# =========================

@asynccontextmanager
async def _obs_lifespan(application):
    """Initialise observability on startup, clean up on shutdown."""
    try:
        _obs_startup_logger.info('')
        _obs_startup_logger.info('========== Agent Configuration Summary ==========')
        _obs_startup_logger.info(f'Environment: {getattr(Config, "ENVIRONMENT", "N/A")}')
        _obs_startup_logger.info(f'Agent: {getattr(Config, "AGENT_NAME", "N/A")}')
        _obs_startup_logger.info(f'Project: {getattr(Config, "PROJECT_NAME", "N/A")}')
        _obs_startup_logger.info(f'LLM Provider: {getattr(Config, "MODEL_PROVIDER", "N/A")}')
        _obs_startup_logger.info(f'LLM Model: {getattr(Config, "LLM_MODEL", "N/A")}')
        _cs_endpoint = getattr(Config, 'AZURE_CONTENT_SAFETY_ENDPOINT', None)
        _cs_key = getattr(Config, 'AZURE_CONTENT_SAFETY_KEY', None)
        if _cs_endpoint and _cs_key:
            _obs_startup_logger.info('Content Safety: Enabled (Azure Content Safety)')
            _obs_startup_logger.info(f'Content Safety Endpoint: {_cs_endpoint}')
        else:
            _obs_startup_logger.info('Content Safety: Not Configured')
        _obs_startup_logger.info('Observability Database: Azure SQL')
        _obs_startup_logger.info(f'Database Server: {getattr(Config, "OBS_AZURE_SQL_SERVER", "N/A")}')
        _obs_startup_logger.info(f'Database Name: {getattr(Config, "OBS_AZURE_SQL_DATABASE", "N/A")}')
        _obs_startup_logger.info('===============================================')
        _obs_startup_logger.info('')
    except Exception as _e:
        _obs_startup_logger.warning('Config summary failed: %s', _e)

    _obs_startup_logger.info('')
    _obs_startup_logger.info('========== Content Safety & Guardrails ==========')
    if GUARDRAILS_CONFIG.get('content_safety_enabled'):
        _obs_startup_logger.info('Content Safety: Enabled')
        _obs_startup_logger.info(f'  - Severity Threshold: {GUARDRAILS_CONFIG.get("content_safety_severity_threshold", "N/A")}')
        _obs_startup_logger.info(f'  - Check Toxicity: {GUARDRAILS_CONFIG.get("check_toxicity", False)}')
        _obs_startup_logger.info(f'  - Check Jailbreak: {GUARDRAILS_CONFIG.get("check_jailbreak", False)}')
        _obs_startup_logger.info(f'  - Check PII Input: {GUARDRAILS_CONFIG.get("check_pii_input", False)}')
        _obs_startup_logger.info(f'  - Check Credentials Output: {GUARDRAILS_CONFIG.get("check_credentials_output", False)}')
    else:
        _obs_startup_logger.info('Content Safety: Disabled')
    _obs_startup_logger.info('===============================================')
    _obs_startup_logger.info('')

    _obs_startup_logger.info('========== Initializing Agent Services ==========')
    # 1. Observability DB schema (imports are inside function — only needed at startup)
    try:
        from observability.database.engine import create_obs_database_engine
        from observability.database.base import ObsBase
        import observability.database.models  # noqa: F401
        _obs_engine = create_obs_database_engine()
        ObsBase.metadata.create_all(bind=_obs_engine, checkfirst=True)
        _obs_startup_logger.info('✓ Observability database connected')
    except Exception as _e:
        _obs_startup_logger.warning('✗ Observability database connection failed (metrics will not be saved)')
    # 2. OpenTelemetry tracer (initialize_tracer is pre-injected at top level)
    try:
        _t = initialize_tracer()
        if _t is not None:
            _obs_startup_logger.info('✓ Telemetry monitoring enabled')
        else:
            _obs_startup_logger.warning('✗ Telemetry monitoring disabled')
    except Exception as _e:
        _obs_startup_logger.warning('✗ Telemetry monitoring failed to initialize')
    _obs_startup_logger.info('=================================================')
    _obs_startup_logger.info('')
    yield

app = FastAPI(lifespan=_obs_lifespan,

    title="Resume Knowledge Answer Agent",
    description="Professional assistant for answering questions using curated resume documents. Answers are strictly grounded in retrieved resume content with privacy and compliance controls.",
    version=Config.SERVICE_VERSION if hasattr(Config, "SERVICE_VERSION") else "1.0.0",
    # SYNTAX-FIX: lifespan=_obs_lifespan
)

# =========================
# Input/Output Models
# =========================

class QueryRequest(BaseModel):
    query: str = Field(..., description="User question about the resumes")

    @model_validator(mode="after")
    def validate_content(self):
        if not self.query or not self.query.strip():
            raise ValueError("Query must be non-empty.")
        if len(self.query.strip()) > 50000:
            raise ValueError("Query exceeds maximum length.")
        self.query = self.query.strip()
        return self

class QueryResponse(BaseModel):
    success: bool = Field(..., description="Whether the query was processed successfully")
    answer: Optional[str] = Field(None, description="Agent's answer to the user question")
    error: Optional[str] = Field(None, description="Error message if any")
    tool_calls_made: Optional[List[str]] = Field(None, description="List of tool calls made (if any)")

# =========================
# Utility: LLM Output Sanitizer
# =========================

import re as _re

_FENCE_RE = _re.compile(r"```(?:\w+)?\s*\n(.*?)```", _re.DOTALL)
_LONE_FENCE_START_RE = _re.compile(r"^```\w*$")
_WRAPPER_RE = _re.compile(
    r"^(?:"
    r"Here(?:'s| is)(?: the)? (?:the |your |a )?(?:code|solution|implementation|result|explanation|answer)[^:]*:\s*"
    r"|Sure[!,.]?\s*"
    r"|Certainly[!,.]?\s*"
    r"|Below is [^:]*:\s*"
    r")",
    _re.IGNORECASE,
)
_SIGNOFF_RE = _re.compile(
    r"^(?:Let me know|Feel free|Hope this|This code|Note:|Happy coding|If you)",
    _re.IGNORECASE,
)
_BLANK_COLLAPSE_RE = _re.compile(r"\n{3,}")

def _strip_fences(text: str, content_type: str) -> str:
    """Extract content from Markdown code fences."""
    fence_matches = _FENCE_RE.findall(text)
    if fence_matches:
        if content_type == "code":
            return "\n\n".join(block.strip() for block in fence_matches)
        for match in fence_matches:
            fenced_block = _FENCE_RE.search(text)
            if fenced_block:
                text = text[:fenced_block.start()] + match.strip() + text[fenced_block.end():]
        return text
    lines = text.splitlines()
    if lines and _LONE_FENCE_START_RE.match(lines[0].strip()):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()

def _strip_trailing_signoffs(text: str) -> str:
    """Remove conversational sign-off lines from the end of code output."""
    lines = text.splitlines()
    while lines and _SIGNOFF_RE.match(lines[-1].strip()):
        lines.pop()
    return "\n".join(lines).rstrip()

@with_content_safety(config=GUARDRAILS_CONFIG)
def sanitize_llm_output(raw: str, content_type: str = "code") -> str:
    """
    Generic post-processor that cleans common LLM output artefacts.
    Args:
        raw: Raw text returned by the LLM.
        content_type: 'code' | 'text' | 'markdown'.
    Returns:
        Cleaned string ready for validation, formatting, or direct return.
    """
    if not raw:
        return ""
    text = _strip_fences(raw.strip(), content_type)
    text = _WRAPPER_RE.sub("", text, count=1).strip()
    if content_type == "code":
        text = _strip_trailing_signoffs(text)
    return _BLANK_COLLAPSE_RE.sub("\n\n", text).strip()

# =========================
# Azure AI Search Client (with Enriched Field Fallback)
# =========================

_enriched_available = None  # None = not yet checked, True/False after first search

async def _search_with_fallback(client, query, embedding, selected_titles, top_k):
    """Try search with enriched fields; fall back to base fields if index lacks them."""
    global _enriched_available
    from azure.core.exceptions import HttpResponseError

    vector_query = VectorizedQuery(vector=embedding, k_nearest_neighbors=top_k, fields="vector")
    base_fields = ["chunk", "title"]

    # If we already know enriched fields are not available, skip them
    if _enriched_available is False:
        select_fields = base_fields
    else:
        select_fields = base_fields + ENRICHED_FIELDS

    search_kwargs = {
        "search_text": query,
        "vector_queries": [vector_query],
        "top": top_k,
        "select": select_fields,
    }
    if selected_titles:
        odata_parts = [f"title eq '{t}'" for t in selected_titles]
        search_kwargs["filter"] = " or ".join(odata_parts)

    try:
        _t0 = _time.time()
        results = list(client.search(**search_kwargs))
        try:
            trace_tool_call(
                tool_name="search_client.search",
                latency_ms=int((_time.time() - _t0) * 1000),
                output=str(results)[:200] if results is not None else None,
                status="success",
            )
        except Exception:
            pass
        if _enriched_available is None:
            _enriched_available = True
            _logger.info("Enriched index fields are AVAILABLE — using: %s", ENRICHED_FIELDS)
        return results
    except HttpResponseError as e:
        if "Could not find a property named" in str(e) and _enriched_available is not False:
            _enriched_available = False
            _logger.warning("Enriched index fields NOT available in this index — falling back to base fields: %s", base_fields)
            search_kwargs["select"] = base_fields
            _t0 = _time.time()
            results = list(client.search(**search_kwargs))
            try:
                trace_tool_call(
                    tool_name="search_client.search",
                    latency_ms=int((_time.time() - _t0) * 1000),
                    output=str(results)[:200] if results is not None else None,
                    status="success",
                )
            except Exception:
                pass
            return results
        raise

# =========================
# Chunk Retrieval Layer
# =========================

class ChunkRetriever:
    """Retrieves relevant chunks from Azure AI Search."""

    def __init__(self):
        self._client = None

    def _get_client(self):
        if self._client is None:
            endpoint = Config.AZURE_SEARCH_ENDPOINT
            api_key = Config.AZURE_SEARCH_API_KEY
            index_name = Config.AZURE_SEARCH_INDEX_NAME
            if not endpoint or not api_key or not index_name:
                raise ValueError("Azure Search credentials are not configured.")
            self._client = SearchClient(
                endpoint=endpoint,
                index_name=index_name,
                credential=AzureKeyCredential(api_key),
            )
        return self._client

    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def retrieve_chunks(self, query: str, selected_document_titles: List[str], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieves relevant chunks from Azure AI Search using vector + keyword search.
        Returns a list of dicts with chunk, title, and (if available) enriched fields.
        """
        client = self._get_client()
        # Get embedding for the query
        embedding = await self._get_embedding(query)
        results = await _search_with_fallback(
            client, query, embedding, selected_document_titles, top_k
        )
        # Format context parts with enriched fields if available
        context_parts = []
        for r in results:
            part = r.get("chunk", "")
            if _enriched_available:
                for field in ENRICHED_FIELDS:
                    value = r.get(field)
                    if value:
                        part += f"\n{field}: {json.dumps(value) if isinstance(value, (list, dict)) else value}"
            context_parts.append(part)
        return context_parts

    async def _get_embedding(self, text: str) -> List[float]:
        """
        Generates an embedding for the given text using Azure OpenAI.
        """
        api_key = Config.AZURE_OPENAI_API_KEY
        endpoint = Config.AZURE_OPENAI_ENDPOINT
        deployment = Config.AZURE_OPENAI_EMBEDDING_DEPLOYMENT or "text-embedding-ada-002"
        if not api_key or not endpoint:
            raise ValueError("Azure OpenAI credentials are not configured.")
        client = openai.AsyncAzureOpenAI(
            api_key=api_key,
            api_version="2024-02-01",
            azure_endpoint=endpoint,
        )
        _t0 = _time.time()
        resp = await client.embeddings.create(
            input=text, model=deployment
        )
        try:
            trace_tool_call(
                tool_name="openai_client.embeddings.create",
                latency_ms=int((_time.time() - _t0) * 1000),
                output=str(resp)[:200] if resp is not None else None,
                status="success",
            )
        except Exception:
            pass
        return resp.data[0].embedding

# =========================
# LLM Service Layer
# =========================

class LLMService:
    """Handles LLM calls to Azure OpenAI."""

    def __init__(self):
        self._client = None
        self.model = Config.LLM_MODEL or "gpt-4.1"

    def _get_client(self):
        if self._client is None:
            api_key = Config.AZURE_OPENAI_API_KEY
            endpoint = Config.AZURE_OPENAI_ENDPOINT
            if not api_key or not endpoint:
                raise ValueError("Azure OpenAI credentials are not configured.")
            self._client = openai.AsyncAzureOpenAI(
                api_key=api_key,
                api_version="2024-02-01",
                azure_endpoint=endpoint,
            )
        return self._client

    async def generate_answer(self, system_prompt: str, context_chunks: List[str], user_query: str) -> str:
        """
        Calls LLM with system prompt, user query, and retrieved chunks.
        Returns the generated answer.
        """
        client = self._get_client()
        # Build context for LLM
        context_text = "\n\n".join(context_chunks)
        system_message = system_prompt + "\n\nOutput Format: " + OUTPUT_FORMAT
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_query},
            {"role": "system", "content": f"Context:\n{context_text}"}
        ]
        # Add few-shot examples as assistant messages
        for example in FEW_SHOT_EXAMPLES:
            messages.append({"role": "assistant", "content": example})

        _llm_kwargs = Config.get_llm_kwargs()
        _t0 = _time.time()
        response = await client.chat.completions.create(
            model=self.model,
            messages=messages,
            **_llm_kwargs
        )
        content = response.choices[0].message.content
        try:
            trace_model_call(
                provider="azure",
                model_name=self.model,
                prompt_tokens=getattr(getattr(response, "usage", None), "prompt_tokens", 0) or 0,
                completion_tokens=getattr(getattr(response, "usage", None), "completion_tokens", 0) or 0,
                latency_ms=int((_time.time() - _t0) * 1000),
                response_summary=content[:200] if content else "",
            )
        except Exception:
            pass
        return sanitize_llm_output(content, content_type="text")

# =========================
# Audit Logger
# =========================

class AuditLogger:
    """Logs user interactions, privacy redactions, and system events."""

    def __init__(self):
        self.logger = logging.getLogger("agent.audit")

    def log_event(self, event_type: str, details: Dict[str, Any]):
        try:
            self.logger.info(f"event_type={event_type} details={json.dumps(details, default=str)}")
        except Exception as e:
            self.logger.error(f"Failed to log event: {e}")

# =========================
# Error Handler
# =========================

class ErrorHandler:
    """Handles errors, retries, fallback responses, and escalation."""

    def __init__(self, audit_logger: AuditLogger):
        self.audit_logger = audit_logger

    async def handle_error(self, error_code: str, context: Dict[str, Any]) -> str:
        self.audit_logger.log_event("error", {"error_code": error_code, "context": context})
        if error_code == "KB_NOT_FOUND":
            return FALLBACK_RESPONSE
        elif error_code == "PRIVACY_VIOLATION":
            return "A privacy violation was detected. The request cannot be fulfilled."
        elif error_code == "INVALID_QUERY":
            return "Your query could not be processed. Please check your input and try again."
        else:
            return "An unexpected error occurred. Please try again later."

# =========================
# Main Agent Orchestration Layer
# =========================

class ResumeKnowledgeAgent:
    """
    Orchestrates query processing, chunk retrieval, LLM interaction, privacy checks, and response formatting.
    """

    def __init__(self):
        self.chunk_retriever = ChunkRetriever()
        self.llm_service = LLMService()
        self.audit_logger = AuditLogger()
        self.error_handler = ErrorHandler(self.audit_logger)

    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def process_query(self, user_query: str) -> Dict[str, Any]:
        """
        Main entry point; processes user query, orchestrates retrieval and response generation.
        Returns dict for QueryResponse.
        """
        # Step 1: Validate input (already validated by Pydantic, but double-check)
        if not user_query or not user_query.strip():
            return {
                "success": False,
                "answer": None,
                "error": "Query must be non-empty.",
                "tool_calls_made": None
            }
        user_query = user_query.strip()
        # Step 2: Log user interaction
        self.audit_logger.log_event("user_query", {"query": user_query})
        # Step 3: Retrieve chunks
        async with trace_step(
            "retrieve_chunks",
            step_type="tool_call",
            decision_summary="Retrieve relevant resume chunks from Azure AI Search",
            output_fn=lambda r: f"chunks={len(r)}"
        ) as step:
            try:
                context_chunks = await self.chunk_retriever.retrieve_chunks(
                    query=user_query,
                    selected_document_titles=SELECTED_DOCUMENT_TITLES,
                    top_k=5
                )
                step.capture(context_chunks)
            except Exception as e:
                self.audit_logger.log_event("retrieval_error", {"error": str(e)})
                answer = await self.error_handler.handle_error("KB_NOT_FOUND", {"error": str(e)})
                return {
                    "success": False,
                    "answer": answer,
                    "error": str(e),
                    "tool_calls_made": None
                }
        # Step 4: If no chunks found, return fallback
        if not context_chunks or all(not c.strip() for c in context_chunks):
            self.audit_logger.log_event("no_chunks_found", {"query": user_query})
            answer = await self.error_handler.handle_error("KB_NOT_FOUND", {"query": user_query})
            return {
                "success": True,
                "answer": answer,
                "error": None,
                "tool_calls_made": None
            }
        # Step 5: LLM answer generation
        async with trace_step(
            "generate_answer",
            step_type="llm_call",
            decision_summary="Generate answer using LLM with retrieved context",
            output_fn=lambda r: f"answer={str(r)[:80]}"
        ) as step:
            try:
                answer = await self.llm_service.generate_answer(
                    system_prompt=SYSTEM_PROMPT,
                    context_chunks=context_chunks,
                    user_query=user_query
                )
                step.capture(answer)
            except Exception as e:
                self.audit_logger.log_event("llm_error", {"error": str(e)})
                answer = await self.error_handler.handle_error("KB_NOT_FOUND", {"error": str(e)})
                return {
                    "success": False,
                    "answer": answer,
                    "error": str(e),
                    "tool_calls_made": None
                }
        # Step 6: Privacy check (PII redaction is handled by guardrails/content safety)
        # Step 7: Log answer
        self.audit_logger.log_event("answer_generated", {"answer": answer})
        return {
            "success": True,
            "answer": answer,
            "error": None,
            "tool_calls_made": None
        }

# =========================
# FastAPI Endpoints
# =========================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}

@app.post("/query", response_model=QueryResponse)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def query_endpoint(req: QueryRequest):
    """
    Main query endpoint. Accepts a user query and returns an answer based on resume documents.
    """
    agent = ResumeKnowledgeAgent()
    try:
        async with trace_step(
            "process_query",
            step_type="process",
            decision_summary="Process user query through agent orchestrator",
            output_fn=lambda r: f"success={r.get('success', False)}"
        ) as step:
            result = await agent.process_query(user_query=req.query)
            step.capture(result)
        return QueryResponse(**result)
    except Exception as e:
        _logger.error(f"Unhandled error in /query: {e}", exc_info=True)
        return QueryResponse(
            success=False,
            answer=None,
            error=f"An unexpected error occurred: {e}",
            tool_calls_made=None
        )

@app.exception_handler(Exception)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def generic_exception_handler(request: Request, exc: Exception):
    """Handle uncaught exceptions and malformed JSON."""
    _logger.error(f"Exception on {request.url}: {exc}", exc_info=True)
    if isinstance(exc, ValueError):
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "answer": None,
                "error": f"Invalid input: {exc}",
                "tool_calls_made": None,
                "tips": [
                    "Ensure your JSON is well-formed.",
                    "Check for missing or extra commas, brackets, or quotes.",
                    "Limit input to 50,000 characters."
                ]
            }
        )
    return JSONResponse(
        status_code=422,
        content={
            "success": False,
            "answer": None,
            "error": f"Malformed request: {exc}",
            "tool_calls_made": None,
            "tips": [
                "Ensure your JSON is well-formed.",
                "Check for missing or extra commas, brackets, or quotes.",
                "Limit input to 50,000 characters."
            ]
        }
    )

# =========================
# Entrypoint
# =========================

async def _run_agent():
    """Entrypoint: runs the agent with observability (trace collection only)."""
    import uvicorn

    # Unified logging config — routes uvicorn, agent, and observability through
    # the same handler so all telemetry appears in a single consistent stream.
    _LOG_CONFIG = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "()": "uvicorn.logging.DefaultFormatter",
                "fmt": "%(levelprefix)s %(name)s: %(message)s",
                "use_colors": None,
            },
            "access": {
                "()": "uvicorn.logging.AccessFormatter",
                "fmt": '%(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s',
            },
        },
        "handlers": {
            "default": {
                "formatter": "default",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stderr",
            },
            "access": {
                "formatter": "access",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
            },
        },
        "loggers": {
            "uvicorn":        {"handlers": ["default"], "level": "INFO", "propagate": False},
            "uvicorn.error":  {"level": "INFO"},
            "uvicorn.access": {"handlers": ["access"], "level": "INFO", "propagate": False},
            "agent":          {"handlers": ["default"], "level": "INFO", "propagate": False},
            "__main__":       {"handlers": ["default"], "level": "INFO", "propagate": False},
            "observability": {"handlers": ["default"], "level": "INFO", "propagate": False},
            "config": {"handlers": ["default"], "level": "INFO", "propagate": False},
            "azure":   {"handlers": ["default"], "level": "WARNING", "propagate": False},
            "urllib3": {"handlers": ["default"], "level": "WARNING", "propagate": False},
        },
    }

    config = uvicorn.Config(
        "agent:app",
        host="0.0.0.0",
        port=8080,
        reload=False,
        log_level="info",
        log_config=_LOG_CONFIG,
    )
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    _asyncio.run(_run_agent())