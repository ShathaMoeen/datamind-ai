# DataMind AI

DataMind AI is a bilingual, multi-agent data-analysis platform. It combines
deterministic Python tools with a locally hosted or cloud LLM to inspect
datasets, calculate statistics, create interactive charts, retrieve evidence
from PDF documents, and produce grounded analytical reports.

![DataMind AI analysis dashboard](docs/assets/datamind-dashboard-v3.png)

## Highlights

- Arabic and English responsive interface with RTL/LTR support
- Secure CSV and Excel upload with deterministic quality profiling
- Multi-agent orchestration for data, analysis, visualization, RAG, and reports
- Typed, allowlisted Pandas tools instead of arbitrary model-generated code
- Interactive Plotly visualizations
- PDF ingestion, chunking, embeddings, ChromaDB retrieval, and citations
- Fact-grounded reports that separate findings, interpretations, and advice
- Structured LLM output validated with Pydantic
- Local, private inference with Ollama; OpenAI is an optional provider
- Automated tests and Ruff linting

## Architecture

```mermaid
flowchart TB

    subgraph REQUEST["1 - Request Layer"]
        direction LR
        UI["Bilingual Web UI"] --> API["FastAPI API"]
        API --> ORCH["Orchestrator"]
    end

    subgraph AGENTS["2 - Specialist Agents"]
        direction LR
        DATA["Data Agent"]
        ANALYSIS["Analysis Agent"]
        VIS["Visualization Agent"]
        RAG["RAG Agent"]
    end

    ORCH --> DATA
    ORCH --> ANALYSIS
    ORCH --> VIS
    ORCH --> RAG

    subgraph EXECUTION["3 - Controlled Execution"]
        direction LR
        PYTHON["Validated Python Tools"]
        PLOTLY["Plotly"]
        RETRIEVER["Document Retriever"]
    end

    DATA --> PYTHON
    ANALYSIS --> PYTHON
    VIS --> PLOTLY
    RAG --> RETRIEVER

    subgraph PIPELINE["PDF Knowledge Pipeline"]
        direction LR
        PDF["PDF"] --> LOADER["Loader"]
        LOADER --> CHUNKS["Chunking"]
        CHUNKS --> EMBEDDINGS["Embeddings"]
        EMBEDDINGS --> CHROMA[("ChromaDB")]
    end

    CHROMA --> RETRIEVER

    subgraph REPORTING["4 - Validation and Reporting"]
        direction LR
        VALIDATE["Fact and Evidence Validation"]
        FACTS["Trusted Facts"]
        REPORT["Report Agent"]
        RESPONSE["Bilingual Result"]

        VALIDATE --> FACTS
        FACTS --> REPORT
        REPORT --> RESPONSE
    end

    PYTHON --> VALIDATE
    PLOTLY --> VALIDATE
    RETRIEVER --> VALIDATE

    subgraph MODELS["LLM Providers"]
        direction LR
        OLLAMA["Ollama"]
        OPENAI["OpenAI"]
        CLIENT["LLM Client"]

        OLLAMA --> CLIENT
        OPENAI --> CLIENT
    end

    CLIENT -.-> ORCH
    CLIENT -.-> REPORT

    classDef interface fill:#172554,stroke:#60a5fa,color:#ffffff
    classDef agent fill:#312e81,stroke:#a78bfa,color:#ffffff
    classDef tool fill:#164e63,stroke:#22d3ee,color:#ffffff
    classDef knowledge fill:#3f3f46,stroke:#a1a1aa,color:#ffffff
    classDef trusted fill:#14532d,stroke:#4ade80,color:#ffffff
    classDef model fill:#581c87,stroke:#c084fc,color:#ffffff

    class UI,API,ORCH,RESPONSE interface
    class DATA,ANALYSIS,VIS,RAG agent
    class PYTHON,PLOTLY,RETRIEVER tool
    class PDF,LOADER,CHUNKS,EMBEDDINGS,CHROMA knowledge
    class VALIDATE,FACTS,REPORT trusted
    class OLLAMA,OPENAI,CLIENT model

    style REQUEST fill:transparent,stroke:#4b5563,stroke-width:1px
    style AGENTS fill:transparent,stroke:#4b5563,stroke-width:1px
    style EXECUTION fill:transparent,stroke:#4b5563,stroke-width:1px
    style PIPELINE fill:transparent,stroke:#4b5563,stroke-width:1px
    style REPORTING fill:transparent,stroke:#4b5563,stroke-width:1px
    style MODELS fill:transparent,stroke:#4b5563,stroke-width:1px
```

The request flows through specialized agents and controlled tools. Retrieved evidence and calculated results are validated before the bilingual report is generated.

## Technology stack

- Python 3.11+
- FastAPI and Pydantic
- Pandas and Plotly
- Ollama or OpenAI
- Sentence Transformers and ChromaDB
- HTML, CSS, and vanilla JavaScript
- Pytest and Ruff

## Local setup

### 1. Clone and create the environment

```powershell
git clone https://github.com/ShathaMoeen/datamind-ai.git
cd datamind-ai
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

### 2. Configure the application

```powershell
Copy-Item .env.example .env
```

The default configuration uses Ollama locally and does not require a paid API
key. Never commit the generated `.env` file.

### 3. Install and prepare Ollama

Install Ollama, then download the configured model:

```powershell
ollama pull qwen3:1.7b
```

### 4. Start DataMind AI

```powershell
fastapi dev app/main.py
```

Open <http://127.0.0.1:8000>. Interactive API documentation is available at
<http://127.0.0.1:8000/docs>.

## Example questions

```text
Analyze monthly_income, monthly_expenses, and monthly_savings. Show the key
figures, create an appropriate chart, and provide three concise recommendations.
```

```text
قارن متوسط monthly_savings حسب region، وحدد أفضل وأسوأ منطقة، ثم أنشئ رسمًا
عموديًا وقدّم ثلاث توصيات مختصرة.
```

## API endpoints

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/api/v1/health` | Service health check |
| `POST` | `/api/v1/datasets/upload` | Upload CSV/XLSX data |
| `GET` | `/api/v1/datasets/{id}/profile` | Inspect dataset quality |
| `POST` | `/api/v1/documents/upload` | Upload a supporting PDF |
| `POST` | `/api/v1/analysis/run` | Run the multi-agent workflow |

## Quality checks

```powershell
python -m pytest
python -m ruff check .
```

## Safety and privacy

- `.env`, uploaded datasets, documents, vector stores, and virtual environments
  are excluded from Git.
- Dataset computations run through typed, allowlisted functions.
- Retrieved document text is treated as untrusted input.
- Report references are validated against trusted workflow facts.
- Local Ollama inference keeps prompts on the user's machine.

## Current limitations

- The lightweight local model may be slower and less fluent than hosted models.
- The application is currently intended for local development and portfolio use.
- Generated recommendations are analytical assistance, not financial advice.

## Author

Built by **Shatha Jadalhaq** as a portfolio project demonstrating applied AI,
multi-agent orchestration, RAG, backend engineering, and data visualization.
