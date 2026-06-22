# KDIGO CKD Dual RAG System & Evaluation Dashboard

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python Version](https://img.shields.io/badge/python-3.12-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green.svg)
![Neo4j](https://img.shields.io/badge/Neo4j-AuraDB-blue.svg)

**🌐 Live Demo:** [https://ckd-rag-analysis.onrender.com/](https://ckd-rag-analysis.onrender.com/)

A comprehensive AI-powered conversational agent designed to answer clinical queries based on the **KDIGO 2024 Clinical Practice Guideline for the Evaluation and Management of Chronic Kidney Disease (CKD)**. 

This repository implements a **Dual-RAG Architecture**, allowing users to dynamically switch between and compare a Traditional Vector-based RAG and an advanced Agentic (LangGraph-powered) RAG. It also features a built-in **Evaluation Engine** to objectively score and compare the performance of both pipelines.

---

## 🌟 Key Features

### 1. Multi-Pipeline Architecture
Choose the retrieval strategy that best fits your query:
*   **Traditional RAG (ChromaDB)**: Lightning-fast vector similarity search using lightweight ONNX embeddings. Ideal for straightforward fact retrieval.
*   **Agentic RAG (LangGraph)**: An intelligent state machine that actively routes queries, grades retrieved documents for relevance, rewrites poor queries, and double-checks the final answer for hallucinations before responding.
*   **Graph RAG (Neo4j)**: A hybrid retrieval system that combines vector similarity with Knowledge Graph traversals, perfect for complex, multi-hop medical queries (e.g., tracking disease progression to treatments).

### 2. Live Evaluation Dashboard
*   **Side-by-Side Comparison**: Ask a single question and watch both Traditional and Agentic RAG generate responses simultaneously.
*   **7-Metric Scoring**: The system uses an LLM-as-a-judge (RAGAS-inspired) to grade each pipeline out of 10 based on:
    *   *Faithfulness* (No hallucinations)
    *   *Answer Relevancy*
    *   *Context Precision*
    *   *Context Recall*
    *   *Token Efficiency*
    *   *Response Time*
    *   *Source Coverage*
*   **Radar Charts**: Visualize the strengths and weaknesses of each pipeline on an interactive radar chart.
*   **Historical Aggregation**: The dashboard tracks your win/loss record and averages scores across all past queries.

### 3. Automated Ingestion Pipeline
*   Upload the KDIGO PDF once.
*   The system automatically parses text, chunks it, computes dense vector embeddings, extracts medical entities (Biomarkers, Medications, Diseases), and populates **both** the local ChromaDB and the remote Neo4j Graph Database.

---

## 🛠️ Technology Stack

*   **Backend Framework**: FastAPI (Python)
*   **LLM Orchestration**: LangChain & LangGraph
*   **Large Language Model**: Groq (`llama-3.3-70b-versatile`)
*   **Vector Database**: ChromaDB (Local, ONNX-accelerated)
*   **Graph Database**: Neo4j (AuraDB)
*   **Frontend**: Vanilla HTML/JS/CSS (No build step required, served dynamically by FastAPI)
*   **Embeddings**: `chromadb` default ONNX model (`all-MiniLM-L6-v2`) — *Optimized for 512MB RAM constraints.*

---

## 🚀 Local Setup & Installation

### Prerequisites
*   Python 3.12+
*   A Neo4j AuraDB instance (Free tier works perfectly)
*   A Groq API Key (Free tier)

### 1. Clone the Repository
```bash
git clone https://github.com/MuhammadHassan-10/CKD-RAG-ANALYSIS.git
cd CKD-RAG-ANALYSIS
```

### 2. Install Dependencies
```bash
# Create a virtual environment
python -m venv backend/venv

# Activate it (Windows)
backend\venv\Scripts\activate
# Activate it (Mac/Linux)
source backend/venv/bin/activate

# Install requirements
pip install -r backend/requirements.txt
```

### 3. Environment Variables
Create a `.env` file in the `backend/` directory by copying the example file:
```bash
cp backend/.env.example backend/.env
```
Open `backend/.env` and add your keys:
```env
GROQ_API_KEY=your_groq_api_key_here
NEO4J_URI=neo4j+s://your-db-id.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_neo4j_password_here
```

### 4. Run the Server
```bash
cd backend
uvicorn app.main:app --reload
```
The application will be available at: **http://localhost:8000**

### 5. Run Initial Data Ingestion
1. Open the UI at `http://localhost:8000`
2. Click the **"Ingest PDF"** button in the top right corner.
3. Wait for the pipeline to parse the KDIGO guidelines and populate ChromaDB and Neo4j. (This process takes a few minutes).

---

## ☁️ Free Deployment (Render.com)

This application is configured for easy, free deployment using Render.

1. Fork or push this repository to your GitHub account.
2. Sign up at [Render.com](https://render.com/).
3. Create a new **Web Service** and connect your GitHub repository.
4. **Important Settings:**
   *   **Root Directory**: *(Leave this entirely blank!)*
   *   **Runtime**: `Docker`
   *   **Instance Type**: `Free`
5. **Environment Variables**: Add your `GROQ_API_KEY`, `NEO4J_URI`, `NEO4J_USERNAME`, and `NEO4J_PASSWORD` in the Advanced settings.
6. Click **Deploy**.

> **Note on Free Tier Limits:** The application has been heavily optimized (swapping PyTorch for ONNX) to fit within Render's strict 512MB RAM limit. If the server goes to sleep due to inactivity, the first request will take ~50 seconds to wake it up.

---

## 📁 Repository Structure
```
├── Dockerfile                  # Production deployment configuration
├── .dockerignore
├── .gitignore
├── backend/
│   ├── app/
│   │   ├── chat/               # RAG pipelines (Agentic, Traditional, Chain)
│   │   ├── evaluation/         # LLM-as-a-judge metrics engine
│   │   ├── ingestion/          # PDF parsing, Graph building, Chunking
│   │   ├── retrieval/          # Neo4j Graph retrieval logic
│   │   ├── vectorstore/        # ChromaDB setup and operations
│   │   ├── main.py             # FastAPI entrypoint
│   │   ├── config.py           # Environment variables & constants
│   │   └── models.py           # Pydantic schemas
│   ├── requirements.txt
│   └── .env.example
└── frontend/
    ├── index.html              # 3-Tab User Interface
    ├── index.js                # Frontend logic & Canvas Radar Charts
    └── index.css               # Styling
```

---

## 🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

## 📝 License
This project is licensed under the MIT License.
