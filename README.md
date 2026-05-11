# Resume Knowledge Answer Agent

A professional assistant that answers questions using information from a curated collection of resumes. Answers are strictly grounded in retrieved resume content, with privacy and compliance controls, leveraging Azure AI Search and Azure OpenAI for retrieval-augmented generation.

---

## Quick Start

### 1. Create a virtual environment:
```
python -m venv .venv
```

### 2. Activate the virtual environment:

**Windows:**
```
.venv\Scripts\activate
```

**macOS/Linux:**
```
source .venv/bin/activate
```

### 3. Install dependencies:
```
pip install -r requirements.txt
```

### 4. Environment setup:
Copy the example environment file and fill in all required values.
```
cp .env.example .env
```

### 5. Running the agent

**Direct execution:**
```
python code/agent.py
```

**As a FastAPI server:**
```
uvicorn code.agent:app --reload --host 0.0.0.0 --port 8000
```

---

## Environment Variables

**Agent Identity**
- `AGENT_NAME`
- `AGENT_ID`
- `PROJECT_NAME`
- `PROJECT_ID`
- `USE_KEY_VAULT`
- `OBS_AZURE_SQL_TRUST_SERVER_CERTIFICATE`

**General**
- `ENVIRONMENT`

**Azure Key Vault**
- `KEY_VAULT_URI`
- `AZURE_USE_DEFAULT_CREDENTIAL`

**Azure Authentication**
- `AZURE_TENANT_ID`
- `AZURE_CLIENT_ID`
- `AZURE_CLIENT_SECRET`

**LLM Configuration**
- `MODEL_PROVIDER`
- `LLM_MODEL`
- `LLM_TEMPERATURE`
- `LLM_MAX_TOKENS`
- `LLM_MODELS`

**API Keys / Secrets**
- `OPENAI_API_KEY`
- `AZURE_OPENAI_API_KEY`
- `AZURE_OPENAI_ENDPOINT`
- `AZURE_OPENAI_EMBEDDING_DEPLOYMENT`
- `ANTHROPIC_API_KEY`
- `GOOGLE_API_KEY`
- `AZURE_CONTENT_SAFETY_KEY`

**Service Endpoints**
- `AZURE_CONTENT_SAFETY_ENDPOINT`
- `AZURE_SEARCH_ENDPOINT`
- `AZURE_SEARCH_API_KEY`

**Observability**
- `SERVICE_NAME`
- `SERVICE_VERSION`
- `OBS_DATABASE_TYPE`
- `OBS_AZURE_SQL_SERVER`
- `OBS_AZURE_SQL_DATABASE`
- `OBS_AZURE_SQL_PORT`
- `OBS_AZURE_SQL_USERNAME`
- `OBS_AZURE_SQL_PASSWORD`
- `OBS_AZURE_SQL_SCHEMA`

**Agent-Specific**
- `AZURE_SEARCH_INDEX_NAME`
- `VALIDATION_CONFIG_PATH`
- `VERSION`
- `CONTENT_SAFETY_ENABLED`
- `CONTENT_SAFETY_SEVERITY_THRESHOLD`

---

## API Endpoints

### **GET** `/health`
Health check endpoint.

**Response:**
```
{
  "status": "ok"
}
```

---

### **POST** `/query`
Accepts a user query and returns an answer based on resume documents.

**Request body:**
```
{
  "query": "string (required)"
}
```

**Response:**
```
{
  "success": true|false,
  "answer": "string|null",
  "error": "string|null",
  "tool_calls_made": ["string", ...] | null
}
```

---

## Running Tests

### 1. Install test dependencies (if not already installed):
```
pip install pytest pytest-asyncio
```

### 2. Run all tests:
```
pytest tests/
```

### 3. Run a specific test file:
```
pytest tests/test_<module_name>.py
```

### 4. Run tests with verbose output:
```
pytest tests/ -v
```

### 5. Run tests with coverage report:
```
pip install pytest-cov
pytest tests/ --cov=code --cov-report=term-missing
```

---

## Deployment with Docker

### 1. Prerequisites: Ensure Docker is installed and running.

### 2. Environment setup: Copy `.env.example` to `.env` and configure all required environment variables.

### 3. Build the Docker image:
```
docker build -t resume-knowledge-answer-agent -f deploy/Dockerfile .
```

### 4. Run the Docker container:
```
docker run -d --env-file .env -p 8000:8000 --name resume-knowledge-answer-agent resume-knowledge-answer-agent
```

### 5. Verify the container is running:
```
docker ps
```

### 6. View container logs:
```
docker logs resume-knowledge-answer-agent
```

### 7. Stop the container:
```
docker stop resume-knowledge-answer-agent
```

---

## Notes

- All run commands must use the `code/` prefix (e.g., `python code/agent.py`, `uvicorn code.agent:app ...`).
- See `.env.example` for all required and optional environment variables.
- The agent requires access to LLM API keys and (optionally) Azure SQL for observability.
- For production, configure Key Vault and secure credentials as needed.

---

**Resume Knowledge Answer Agent** — Accurate, privacy-safe answers from your resume knowledge base.