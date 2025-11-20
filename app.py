"""
Flask Application for Flipkart Product Recommender

This module creates and configures the Flask application with:
- RAG chain for product recommendations
- Prometheus metrics endpoint
- Chat interface for user queries
"""

# import logging
import os
import uuid
from typing import Dict, Any
from flask import Flask, render_template, request, Response, jsonify, session
from flipkart.data_loader.data_ingestion import DataIngestor
from flipkart.rag_chain import RAGChainBuilder
from prometheus_client import Counter, Histogram, generate_latest
from utils.logger import get_logger
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


logger = get_logger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter(
    "request_count_total", "Total number of requests", ["method", "endpoint", "status"]
)
REQUEST_DURATION = Histogram(
    "request_duration_seconds", "Request duration in seconds", ["method", "endpoint"]
)
CHAT_ERROR_COUNT = Counter(
    "chat_errors_total", "Total number of chat errors", ["error_type"]
)


def create_app() -> Flask:
    """
    Create and configure the Flask application.

    This function:
    1. Initializes the Flask app
    2. Sets up the RAG chain with vector store
    3. Configures routes for the web interface
    4. Sets up error handlers
    5. Configures session management

    Returns:
        Flask: Configured Flask application instance

    Raises:
        Exception: If data ingestion or RAG chain setup fails
    """
    app = Flask(__name__)
    # Set a secret key for session management (use environment variable in production)
    app.secret_key = os.getenv(
        "FLASK_SECRET_KEY", "dev-secret-key-change-in-production"
    )

    # Initialize RAG components
    try:
        logger.info("Initializing data ingestor and vector store...")
        data_ingestor = DataIngestor()
        vector_store = data_ingestor.ingest_data(load_existing_data=True)

        logger.info("Building RAG chain...")
        rag_chain_builder = RAGChainBuilder(vector_store)
        rag_chain = rag_chain_builder.build_rag_chain()
        logger.info("Application initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize application: {e}")
        raise

    @app.before_request
    def before_request():
        """Initialize session ID if not present."""
        if "session_id" not in session:
            session["session_id"] = str(uuid.uuid4())

    @app.route("/")
    def index() -> str:
        """
        Render the main index page.

        Returns:
            str: Rendered HTML template
        """
        REQUEST_COUNT.labels(method="GET", endpoint="/", status="200").inc()
        return render_template("index.html")

    @app.route("/chat", methods=["POST"])
    def chat() -> Response:
        """
        Handle chat requests and return product recommendations.

        Expected form data:
            msg (str): User's question/query about products

        Returns:
            Response: JSON response with answer or error message

        HTTP Status Codes:
            200: Success
            400: Bad request (missing or invalid input)
            500: Internal server error
        """
        REQUEST_COUNT.labels(method="POST", endpoint="/chat", status="200").inc()

        try:
            # Validate input
            if "msg" not in request.form:
                REQUEST_COUNT.labels(
                    method="POST", endpoint="/chat", status="400"
                ).inc()
                CHAT_ERROR_COUNT.labels(error_type="missing_input").inc()
                return jsonify(
                    {"error": "Missing required field: msg", "status": "error"}
                ), 400

            user_input = request.form["msg"].strip()

            # Validate input is not empty
            if not user_input:
                REQUEST_COUNT.labels(
                    method="POST", endpoint="/chat", status="400"
                ).inc()
                CHAT_ERROR_COUNT.labels(error_type="empty_input").inc()
                return jsonify(
                    {"error": "Input cannot be empty", "status": "error"}
                ), 400

            # Validate input length (prevent extremely long inputs)
            if len(user_input) > 1000:
                REQUEST_COUNT.labels(
                    method="POST", endpoint="/chat", status="400"
                ).inc()
                CHAT_ERROR_COUNT.labels(error_type="input_too_long").inc()
                return jsonify(
                    {
                        "error": "Input too long. Maximum 1000 characters allowed.",
                        "status": "error",
                    }
                ), 400

            # Get session ID (unique per user session)
            session_id = session.get("session_id", str(uuid.uuid4()))

            logger.info(
                f"Processing chat request from session {session_id}: {user_input[:50]}..."
            )

            # Invoke RAG chain with user input and session context
            with REQUEST_DURATION.labels(method="POST", endpoint="/chat").time():
                result = rag_chain.invoke(
                    {"input": user_input},
                    config={"configurable": {"session_id": session_id}},
                )

            # Extract answer from result
            answer = result.get("answer", "I'm sorry, I couldn't generate a response.")

            logger.info(f"Successfully generated response for session {session_id}")

            return jsonify({"answer": answer, "status": "success"}), 200

        except KeyError as e:
            REQUEST_COUNT.labels(method="POST", endpoint="/chat", status="500").inc()
            CHAT_ERROR_COUNT.labels(error_type="key_error").inc()
            logger.error(f"KeyError in chat endpoint: {e}")
            return jsonify(
                {"error": "Error processing request: missing data", "status": "error"}
            ), 500

        except Exception as e:
            REQUEST_COUNT.labels(method="POST", endpoint="/chat", status="500").inc()
            CHAT_ERROR_COUNT.labels(error_type="unknown").inc()
            logger.error(f"Unexpected error in chat endpoint: {e}", exc_info=True)
            return jsonify(
                {
                    "error": "An internal error occurred. Please try again later.",
                    "status": "error",
                }
            ), 500

    @app.route("/metrics")
    def metrics() -> Response:
        """
        Prometheus metrics endpoint.

        Returns:
            Response: Prometheus metrics in text format
        """
        REQUEST_COUNT.labels(method="GET", endpoint="/metrics", status="200").inc()
        return Response(generate_latest(), mimetype="text/plain")

    @app.errorhandler(404)
    def not_found(error: Any) -> Response:
        """Handle 404 errors."""
        REQUEST_COUNT.labels(
            method=request.method, endpoint=request.path, status="404"
        ).inc()
        return jsonify({"error": "Endpoint not found", "status": "error"}), 404

    @app.errorhandler(500)
    def internal_error(error: Any) -> Response:
        """Handle 500 errors."""
        REQUEST_COUNT.labels(
            method=request.method, endpoint=request.path, status="500"
        ).inc()
        logger.error(f"Internal server error: {error}")
        return jsonify({"error": "Internal server error", "status": "error"}), 500

    return app


if __name__ == "__main__":
    # Note: In production, use a proper WSGI server like gunicorn
    # and set debug=False
    app = create_app()
    app.run(debug=True, port=8000, host="0.0.0.0")
