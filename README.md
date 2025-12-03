# Flipkart Product Recommender

A sophisticated AI-powered product recommendation system built with RAG (Retrieval-Augmented Generation) that helps users discover products based on natural language queries using product reviews and descriptions.

## 🎯 Overview

This application leverages Large Language Models (LLMs) and vector databases to provide intelligent product recommendations. Users can ask questions in natural language, and the system retrieves relevant product information from a database of product reviews to generate contextual responses.

### Key Features

- **Natural Language Processing**: Ask questions about products in plain English
- **RAG-based Recommendations**: Uses Retrieval-Augmented Generation for accurate, context-aware responses
- **Conversation History**: Maintains context across multiple interactions per user session
- **Vector Search**: Semantic search over product reviews using embeddings
- **Real-time Chat Interface**: Interactive web-based chat interface
- **Monitoring & Metrics**: Prometheus metrics for observability
- **Session Management**: Unique session tracking for each user

## 🏗️ Architecture

### System Components

```
┌─────────────────┐
│   Web Browser   │
│  (Frontend UI)  │
└────────┬────────┘
         │ HTTP/HTTPS
         ▼
┌─────────────────┐
│   Flask App     │
│   (app.py)      │
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌─────────┐ ┌──────────────┐
│  RAG    │ │  Prometheus  │
│  Chain  │ │   Metrics    │
└────┬────┘ └──────────────┘
     │
┌────┴─────────────────────┐
│  RAGChainBuilder         │
│  - History Management    │
│  - Query Rewriting       │
│  - Context Retrieval     │
└────┬─────────────────────┘
     │
┌────┴─────────────────────┐
│  Vector Store            │
│  (AstraDB)               │
│  - Product Reviews       │
│  - Embeddings            │
└────┬─────────────────────┘
     │
┌────┴─────────────────────┐
│  LLM (Groq)              │
│  - Query Understanding    │
│  - Response Generation    │
└──────────────────────────┘
```

### Technology Stack

- **Backend Framework**: Flask (Python)
- **LLM Provider**: Groq (Llama 3.1 8B Instant)
- **Embedding Model**: BAAI/bge-base-en-v1.5 (HuggingFace)
- **Vector Database**: DataStax Astra DB
- **RAG Framework**: LangChain
- **Monitoring**: Prometheus
- **Containerization**: Docker
- **Orchestration**: Kubernetes (optional)

## 📊 Application Workflow

### 1. Data Ingestion Pipeline

```
CSV File (Product Reviews)
    │
    ▼
DataConverter
    │
    ├─► Reads CSV file
    ├─► Validates columns (product_title, review)
    ├─► Converts to LangChain Documents
    └─► Returns list of Document objects
    │
    ▼
DataIngestor
    │
    ├─► Initializes embedding model
    ├─► Connects to AstraDB vector store
    ├─► Generates embeddings for each document
    └─► Stores in vector database
```

### 2. Query Processing Flow

```
User Query
    │
    ▼
Flask App (/chat endpoint)
    │
    ├─► Validates input
    ├─► Gets/creates session ID
    └─► Invokes RAG Chain
    │
    ▼
RAGChainBuilder
    │
    ├─► STEP 1: History-Aware Query Rewriting
    │   └─► Uses chat history to rewrite query
    │       Example: "What about its price?" 
    │       → "What about the price of the Samsung Galaxy phone?"
    │
    ├─► STEP 2: Vector Search
    │   └─► Searches vector store for top 3 relevant reviews
    │
    ├─► STEP 3: Context Assembly
    │   └─► Combines retrieved documents with query
    │
    ├─► STEP 4: LLM Generation
    │   └─► Generates answer using context and history
    │
    └─► STEP 5: Response & History Update
        └─► Returns answer and stores in session history
    │
    ▼
Response to User
```

### 3. RAG Chain Architecture

The RAG (Retrieval-Augmented Generation) chain consists of:

1. **History-Aware Retriever**: Rewrites queries using conversation context
2. **Vector Retriever**: Searches for relevant product reviews
3. **Document Chain**: Combines retrieved documents
4. **QA Chain**: Generates answers using LLM
5. **Message History**: Maintains conversation context per session

## 🚀 Getting Started

### Prerequisites

- Python 3.13+
- DataStax Astra DB account
- Groq API key
- HuggingFace token (optional, for private models)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Flipkart-Product-Recommender
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -e .
   # or
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   
   Create a `.env` file in the root directory:
   ```env
   # Required
   GROQ_API_KEY=your_groq_api_key
   ASTRA_DB_APPLICATION_TOKEN=your_astra_db_token
   ASTRA_DB_KEYSPACE=your_keyspace_name
   ASTRA_DB_API_ENDPOINT=https://your-endpoint.apps.astra.datastax.com
   
   # Optional
   HF_TOKEN=your_huggingface_token
   HUGGINGFACEHUB_API_TOKEN=your_hf_hub_token
   FLASK_SECRET_KEY=your_secret_key_for_sessions
   ```

5. **Prepare data**
   
   Place your product review CSV file in the `data/` directory:
   ```
   data/
   └── flipkart_product_review.csv
   ```
   
   Required columns: `product_title`, `review`

6. **Initialize vector store** (First time only)
   ```bash
   python -m flipkart.data_loader.data_ingestion
   ```
   This will:
   - Read the CSV file
   - Convert reviews to documents
   - Generate embeddings
   - Store in AstraDB

7. **Run the application**
   ```bash
   python app.py
   ```

8. **Access the application**
   
   Open your browser and navigate to:
   ```
   http://localhost:5000
   ```

## 📁 Project Structure

```
Flipkart-Product-Recommender/
├── app.py                          # Flask application entry point
├── main.py                         # CLI entry point
├── requirements.txt                # Python dependencies
├── pyproject.toml                  # Project configuration
├── setup.py                        # Package setup
├── Dockerfile                      # Docker configuration
├── .env                            # Environment variables (create this)
│
├── flipkart/                       # Main package
│   ├── __init__.py
│   ├── config.py                   # Configuration management
│   ├── rag_chain.py                # RAG chain builder
│   │
│   └── data_loader/                # Data processing
│       ├── __init__.py
│       ├── data_converter.py       # CSV to Document converter
│       └── data_ingestion.py       # Vector store ingestion
│
├── utils/                          # Utility modules
│   ├── __init__.py
│   ├── logger.py                   # Logging configuration
│   └── custom_exception.py        # Custom exceptions
│
├── templates/                      # HTML templates
│   └── index.html                 # Chat interface
│
├── static/                         # Static files
│   └── style.css                  # CSS styles
│
├── data/                           # Data files
│   └── flipkart_product_review.csv
│
├── logs/                           # Application logs
│   └── log_YYYY-MM-DD.log
│
├── prometheus/                     # Prometheus configs
│   ├── prometheus-configmap.yaml
│   └── prometheus-deployment.yaml
│
└── grafana/                        # Grafana configs
    └── grafana-deployment.yaml
```

## ⚙️ Configuration

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GROQ_API_KEY` | Yes | API key for Groq LLM service |
| `ASTRA_DB_APPLICATION_TOKEN` | Yes | Authentication token for AstraDB |
| `ASTRA_DB_KEYSPACE` | Yes | Keyspace name in AstraDB |
| `ASTRA_DB_API_ENDPOINT` | Yes | AstraDB API endpoint URL |
| `HF_TOKEN` | No | HuggingFace token for model access |
| `HUGGINGFACEHUB_API_TOKEN` | No | HuggingFace Hub API token |
| `FLASK_SECRET_KEY` | No | Secret key for Flask sessions (defaults to dev key) |

### Model Configuration

Edit `flipkart/config.py` to customize:

- **Embedding Model**: `EMBEDDING_MODEL` (default: "BAAI/bge-base-en-v1.5")
- **LLM Model**: `LLM_MODEL` (default: "llama-3.1-8b-instant")
- **Temperature**: `LLM_TEMPERATURE` (default: 0.0)
- **Collection Name**: `COLLECTION_NAME` (default: "flipkart_database")
- **Retrieval Count**: `k` parameter in `rag_chain.py` (default: 3)

## 🔌 API Documentation

### Endpoints

#### `GET /`
Renders the main chat interface.

**Response**: HTML page

---

#### `POST /chat`
Processes user queries and returns product recommendations.

**Request**:
```json
{
  "msg": "What are the best phones under 20000?"
}
```

**Success Response** (200):
```json
{
  "answer": "Based on the reviews, here are some great phones...",
  "status": "success"
}
```

**Error Response** (400/500):
```json
{
  "error": "Error message",
  "status": "error"
}
```

**Validation Rules**:
- `msg` field is required
- Input cannot be empty
- Maximum length: 1000 characters

**Session Management**:
- Each user gets a unique session ID
- Conversation history is maintained per session
- Session persists across requests

---

#### `GET /metrics`
Prometheus metrics endpoint.

**Response**: Prometheus metrics in text format

**Metrics Available**:
- `request_count_total`: Total requests by method, endpoint, status
- `request_duration_seconds`: Request processing time
- `chat_errors_total`: Chat errors by error type

---

#### Error Handlers

- `404`: Endpoint not found
- `500`: Internal server error

## 🐳 Docker Deployment

### Build Docker Image

```bash
docker build -t flipkart-recommender:latest .
```

### Run Container

```bash
docker run -d \
  -p 5000:5000 \
  --env-file .env \
  --name flipkart-recommender \
  flipkart-recommender:latest
```

### Docker Compose (Optional)

Create `docker-compose.yml`:

```yaml
version: '3.8'
services:
  app:
    build: .
    ports:
      - "5000:5000"
    env_file:
      - .env
    volumes:
      - ./logs:/app/logs
      - ./data:/app/data
```

Run with:
```bash
docker-compose up -d
```

## ☸️ Kubernetes Deployment

The project includes Kubernetes deployment files for production deployment.

### Prerequisites

- Kubernetes cluster
- kubectl configured
- Docker image pushed to registry

### Deploy

1. **Create secrets**
   ```bash
   kubectl create secret generic llmops-secrets \
     --from-literal=GROQ_API_KEY="" \
     --from-literal=ASTRA_DB_APPLICATION_TOKEN="" \
     --from-literal=ASTRA_DB_KEYSPACE="default_keyspace" \
     --from-literal=ASTRA_DB_API_ENDPOINT="" \
     --from-literal=HF_TOKEN="" \
     --from-literal=HUGGINGFACEHUB_API_TOKEN=""
   ```

2. **Deploy application**
   ```bash
   kubectl apply -f flask-deployment.yaml
   ```

3. **Deploy monitoring** (optional)
   ```bash
   kubectl create namespace monitoring
   kubectl apply -f prometheus/prometheus-configmap.yaml
   kubectl apply -f prometheus/prometheus-deployment.yaml
   kubectl apply -f grafana/grafana-deployment.yaml
   ```

4. **Port forward**
   ```bash
   kubectl port-forward svc/flask-service 5000:80
   ```

## 📊 Monitoring

### Prometheus Metrics

Access metrics at: `http://localhost:5000/metrics`

**Key Metrics**:
- Request count by endpoint and status
- Request duration
- Chat error counts

### Grafana Dashboard

If Grafana is deployed:
1. Access Grafana UI (default: port 3000)
2. Add Prometheus as data source
3. Create dashboards for visualization

### Logs

Application logs are stored in:
```
logs/log_YYYY-MM-DD.log
```

Log format:
```
YYYY-MM-DD HH:MM:SS - LEVEL - MESSAGE
```

## 🔧 Development

### Running in Development Mode

```bash
python app.py
```

The app runs with:
- Debug mode: `True`
- Port: `5000`
- Host: `0.0.0.0`

### Code Structure

- **app.py**: Flask application and routes
- **flipkart/rag_chain.py**: RAG chain implementation
- **flipkart/data_loader/**: Data processing modules
- **flipkart/config.py**: Configuration management
- **utils/logger.py**: Logging utilities

### Adding New Features

1. **New Routes**: Add to `app.py` in `create_app()` function
2. **RAG Modifications**: Edit `flipkart/rag_chain.py`
3. **Data Processing**: Extend `flipkart/data_loader/`
4. **Configuration**: Update `flipkart/config.py`

## 🧪 Testing

### Manual Testing

1. Start the application
2. Open `http://localhost:5000`
3. Test queries:
   - "What are the best phones?"
   - "Tell me about Samsung phones"
   - "What about the battery life?" (follow-up question)

### API Testing

```bash
# Test chat endpoint
curl -X POST http://localhost:5000/chat \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "msg=What are the best phones?"

# Test metrics
curl http://localhost:5000/metrics
```

## 🐛 Troubleshooting

### Common Issues

1. **Missing Environment Variables**
   ```
   Error: Missing required environment variables
   ```
   **Solution**: Ensure all required variables are set in `.env`

2. **Vector Store Connection Failed**
   ```
   Error: Failed to connect to AstraDB
   ```
   **Solution**: Verify AstraDB credentials and endpoint

3. **No Data in Vector Store**
   ```
   Error: No documents found
   ```
   **Solution**: Run data ingestion first

4. **Session Issues**
   ```
   Error: Session not found
   ```
   **Solution**: Ensure `FLASK_SECRET_KEY` is set

## 📝 License

[Add your license here]

## 👥 Contributors

- Taiwo Sokunbi

## 🙏 Acknowledgments

- LangChain for RAG framework
- Groq for LLM API
- DataStax for AstraDB
- HuggingFace for embedding models

## 📚 Additional Resources

- [LangChain Documentation](https://python.langchain.com/)
- [Groq API Documentation](https://console.groq.com/docs)
- [AstraDB Documentation](https://docs.datastax.com/en/astra/)
- [Flask Documentation](https://flask.palletsprojects.com/)

---

**Note**: This is a production-ready application with proper error handling, logging, monitoring, and security features. Ensure all environment variables are properly configured before deployment.

