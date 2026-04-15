# 🎯 Interview coach backend

A FastAPI-based backend that retrieves interview questions using semantic similarity from a pre-built ChromaDB vector database.

This API processes user resume data and returns relevant interview questions categorized by difficulty (easy, medium, hard).

---

## 🚀 Features

* Semantic search using vector embeddings
* ChromaDB persistent vector store
* Resume-based query generation
* Difficulty-wise question grouping
* Chunked responses for better consumption
* No LLM required (pure retrieval system)

---

## 🧠 Tech Stack

* FastAPI
* ChromaDB
* ONNX MiniLM (embedding model)
* Uvicorn
* Docker

---

## 📁 Project Structure

```
.
├── app.py
├── requirements.txt
├── chroma_db/
├── Dockerfile
└── README.md
```

---

## ⚙️ Prerequisites

* Docker installed
* Python 3.10+ (for local DB build)

---

## 🧱 Step 1: Build Vector Database

Before running the app, generate the ChromaDB vector store:

```bash
python build_db.py
```

This creates the `chroma_db/` directory required by the API.

---

## 🐳 Step 2: Build Docker Image

```bash
docker build -t rag-interview-api .
```

---

## ▶️ Step 3: Run the Container

```bash
docker run -p 8080:8080 rag-interview-api
```

---

## 🌐 API Access

* Base URL:
  `http://localhost:8080`

* Interactive Docs (Swagger UI):
  `http://localhost:8080/docs`

---

## 📌 API Endpoints

### `GET /`

Health check and service info

---

### `POST /retrieve`

Retrieve interview questions based on resume

#### Request Body Example:

```json
{
  "experience": "0-1",
  "company_looking": "google",
  "tech_skills": ["python", "sql", "react"],
  "projects": [],
  "user_experience": [],
  "num_chunks": 2,
  "results_per_difficulty": 3
}
```

#### Response:

* Returns chunks of questions
* Each chunk contains:

  * Easy questions
  * Medium questions
  * Hard questions

---

### `GET /stats`

Returns database and embedding stats

---

## ⚡ How It Works

1. Resume data is converted into a semantic query
2. Query is embedded using ONNX MiniLM
3. ChromaDB retrieves similar questions
4. Results are grouped into chunks by difficulty
5. API returns structured JSON response

---

## ⚠️ Important Notes

* `chroma_db/` must exist before running the app
* Embedding model downloads automatically on first run
* Ensure correct collection exists (`interview_questions`)
* Default embedding dimension: 384

---

## 🛠️ Troubleshooting

* **Database not found**

  * Run `python build_db.py`

* **Collection missing**

  * Ensure correct dataset is built

* **Port already in use**

  * Change port mapping:

    ```bash
    docker run -p 9090:8080 rag-interview-api
    ```

---

## 📄 License

MIT License
