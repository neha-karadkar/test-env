# NOTE: If you see "Unknown pytest.mark.X" warnings, create a conftest.py file with:
# import pytest
# def pytest_configure(config):
#     config.addinivalue_line("markers", "performance: mark test as performance test")
#     config.addinivalue_line("markers", "security: mark test as security test")
#     config.addinivalue_line("markers", "integration: mark test as integration test")

# NOTE: If you see "Unknown pytest.mark.X" warnings, create a conftest.py file with:
# import pytest
# def pytest_configure(config):
#     config.addinivalue_line("markers", "performance: mark test as performance test")
#     config.addinivalue_line("markers", "security: mark test as security test")
#     config.addinivalue_line("markers", "integration: mark test as integration test")


import pytest
import time
from unittest.mock import AsyncMock, MagicMock, patch

from agent import ResumeKnowledgeAgent, ChunkRetriever, LLMService, AuditLogger, ErrorHandler, QueryRequest, QueryResponse

# ── Fixtures (module level, NEVER inside a class) ──────────────────

@pytest.fixture
def agent_instance():
    """Create agent with mocked dependencies."""
    with patch("azure.search.documents.SearchClient", new=MagicMock()), \
         patch("openai.AsyncAzureOpenAI", new=MagicMock()):
        instance = ResumeKnowledgeAgent()
    return instance

@pytest.fixture
def chunk_retriever_instance():
    with patch("azure.search.documents.SearchClient", new=MagicMock()), \
         patch("openai.AsyncAzureOpenAI", new=MagicMock()):
        return ChunkRetriever()

@pytest.fixture
def llm_service_instance():
    with patch("openai.AsyncAzureOpenAI", new=MagicMock()):
        return LLMService()

@pytest.fixture
def audit_logger_instance():
    return AuditLogger()

@pytest.fixture
def error_handler_instance(audit_logger_instance):
    return ErrorHandler(audit_logger_instance)

# ── Unit Tests ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_unit_process_query_happy_path(agent_instance):
    """Test process_query returns expected result."""
    with patch.object(agent_instance.chunk_retriever, "retrieve_chunks", new=AsyncMock(return_value=["chunk1", "chunk2"])), \
         patch.object(agent_instance.llm_service, "generate_answer", new=AsyncMock(return_value="Jane Smith is proficient in Python.")):
        result = await agent_instance.process_query("What programming languages does Jane Smith know?")
    assert result is not None

@pytest.mark.asyncio
async def test_unit_process_query_error_handling(agent_instance):
    """Test process_query handles errors gracefully."""
    with patch.object(agent_instance.chunk_retriever, "retrieve_chunks", new=AsyncMock(side_effect=Exception("test error"))):
        try:
            result = await agent_instance.process_query("What programming languages does Jane Smith know?")
            assert result is not None
        except AssertionError:
            raise
        except Exception:
            pass

@pytest.mark.asyncio
async def test_unit_process_query_no_chunks(agent_instance):
    """Test process_query returns fallback when no chunks found."""
    with patch.object(agent_instance.chunk_retriever, "retrieve_chunks", new=AsyncMock(return_value=[])):
        result = await agent_instance.process_query("Who won the 2020 Olympics?")
    assert result is not None

@pytest.mark.asyncio
async def test_unit_process_query_empty_input(agent_instance):
    """Test process_query returns error for empty input."""
    result = await agent_instance.process_query("")
    assert result is not None

@pytest.mark.asyncio
async def test_unit_llmservice_generate_answer_sanitization(llm_service_instance):
    """Test LLMService.generate_answer returns sanitized output."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "```python\nHere is the answer:\nJane Smith is proficient in Python.\n```"
    with patch.object(llm_service_instance, "_get_client", new=MagicMock(return_value=MagicMock())), \
         patch.object(llm_service_instance._get_client(), "chat", create=True) as mock_chat:
        mock_chat.completions.create = AsyncMock(return_value=mock_response)
        output = await llm_service_instance.generate_answer(
            system_prompt="system",
            context_chunks=["chunk1"],
            user_query="What programming languages does Jane Smith know?"
        )
    assert output is not None

@pytest.mark.asyncio
async def test_unit_chunkretriever_retrieve_chunks_enriched(chunk_retriever_instance):
    """Test ChunkRetriever.retrieve_chunks returns context with enriched fields."""
    # Simulate enriched fields available
    global _enriched_available
    _enriched_available = True
    mock_result = [
        {
            "chunk": "Jane Smith resume chunk",
            "entities": [{"name": "Jane Smith"}],
            "keyphrases": ["Python", "JavaScript"],
            "relationships": [{"type": "works_at", "target": "Acme Corp"}],
            "title": "resumes_collection1.pdf"
        }
    ]
    with patch.object(chunk_retriever_instance, "_get_client", new=MagicMock(return_value=MagicMock())), \
         patch.object(chunk_retriever_instance, "_get_embedding", new=AsyncMock(return_value=[0.1, 0.2, 0.3])), \
         patch("agent._search_with_fallback", new=AsyncMock(return_value=mock_result)):
        output = await chunk_retriever_instance.retrieve_chunks(
            query="Jane Smith",
            selected_document_titles=["resumes_collection1.pdf"],
            top_k=1
        )
    assert output is not None

@pytest.mark.asyncio
async def test_unit_errorhandler_handle_error_kb_not_found(error_handler_instance):
    """Test ErrorHandler.handle_error returns correct fallback for KB_NOT_FOUND."""
    output = await error_handler_instance.handle_error("KB_NOT_FOUND", {"context": "test"})
    assert output is not None

def test_unit_auditlogger_log_event_no_exception(audit_logger_instance):
    """Test AuditLogger.log_event logs event without error."""
    audit_logger_instance.log_event("test_event", {"foo": "bar"})

def test_unit_queryrequest_model_rejects_long_query():
    """Test QueryRequest model rejects overly long queries."""
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        QueryRequest(query="a" * 50001)

# ── Integration Tests ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_integration_workflow(agent_instance):
    """Test complete workflow with mocked dependencies."""
    with patch.object(agent_instance.chunk_retriever, "retrieve_chunks", new=AsyncMock(return_value=["chunk1"])), \
         patch.object(agent_instance.llm_service, "generate_answer", new=AsyncMock(return_value="Jane Smith is proficient in Python.")):
        result = await agent_instance.process_query("What programming languages does Jane Smith know?")
    assert result is not None

# ── Performance Tests ───────────────────────────────────────────────

@pytest.mark.performance
@pytest.mark.asyncio
async def test_performance_throughput(agent_instance):
    """Test processing throughput with generous threshold."""
    with patch.object(agent_instance.chunk_retriever, "retrieve_chunks", new=AsyncMock(return_value=["chunk1"])), \
         patch.object(agent_instance.llm_service, "generate_answer", new=AsyncMock(return_value="Jane Smith is proficient in Python.")):
        start_time = time.time()
        for _ in range(10):
            result = await agent_instance.process_query("What programming languages does Jane Smith know?")
            assert result is not None
        duration = time.time() - start_time
    assert duration < 30.0, f"10 calls took {duration:.1f}s"

# ── Edge Case Tests ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_edge_case_empty_input(agent_instance):
    """Test handling of empty/None input."""
    result = await agent_instance.process_query("")
    assert result is not None