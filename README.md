# REQTRACE AI

## Code Observation & FRD Evidence Analysis

REQTRACE AI is a local, evidence-based engineering review-support
application for comparing fictional software observations against supplied
functional requirements, source code, test evidence, and review evidence.

## Safety and Usage Boundary

Only fictional, dummy, or synthetic engineering documents and source code
may be used during development and demonstration.

REQTRACE AI provides evidence-based review support using the retrieved
documents. The result is not a substitute for authorized engineering,
quality, safety, certification, or release approval.

REQTRACE AI must not be represented as providing:

- Final engineering approval
- Certification approval
- Safety approval
- Closure authorization
- Airworthiness approval
- Production release authorization


## Current Development Status

Phase 1:

- Python verified
- Ollama connectivity verified
- `llama3.2:1b` verified
- `nomic-embed-text:latest` verified
- Text generation verified
- Embedding generation verified

Phase 2:

- Project structure
- Virtual environment
- Central configuration
- Logging
- Requirements
- Startup diagnostics
- Configuration tests

No document ingestion, vector-store creation, retrieval, LLM assessment,
or Streamlit interface is implemented in Phase 2.

## Environment

- Operating system: Windows
- Command shell: Windows CMD
- Execution: CPU only
- Primary generation model: `llama3.2:1b`
- Embedding model: `nomic-embed-text:latest`

## Initial Setup

Open Windows CMD in the project directory.

Create the virtual environment:

    python -m venv .venv

Activate it:

    .venv\Scripts\activate

Upgrade packaging tools:

    python -m pip install --upgrade pip setuptools wheel

Install dependencies:

    python -m pip install -r requirements.txt

Create the local configuration:

    copy .env.example .env

Run startup diagnostics:

    python app.py

Run tests:

    python -m pytest tests\test_config.py -v

## Phase 2 Limitations

Phase 2 does not:

- Create ChromaDB collections
- Generate embeddings
- Ingest engineering documents
- Implement retrieval
- Implement deterministic decisions
- Call the LLM
- Start Streamlit

These capabilities are introduced only in their designated development
phases.
