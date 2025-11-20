import os
from dotenv import load_dotenv
from typing import Optional

load_dotenv()


class Config:
    """Configuration class for Flipkart Product Recommender.
    
    Loads environment variables and provides configuration settings
    for the application including API keys, database settings, and model parameters.
    """
    
    # API Keys
    GROQ_API_KEY: Optional[str] = os.getenv("GROQ_API_KEY")
    ASTRA_DB_APPLICATION_TOKEN: Optional[str] = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
    ASTRA_DB_KEYSPACE: Optional[str] = os.getenv("ASTRA_DB_KEYSPACE")
    ASTRA_DB_API_ENDPOINT: Optional[str] = os.getenv("ASTRA_DB_API_ENDPOINT")
    HF_TOKEN: Optional[str] = os.getenv("HF_TOKEN")
    HUGGINGFACEHUB_API_TOKEN: Optional[str] = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    
    # Model Configuration
    EMBEDDING_MODEL: str = "BAAI/bge-base-en-v1.5"
    LLM_MODEL: str = "llama-3.1-8b-instant"
    LLM_TEMPERATURE: float = 0.0
    
    # Database Configuration
    COLLECTION_NAME: str = "flipkart_database"
    
    @classmethod
    def validate(cls) -> None:
        """Validate that all required environment variables are set.
        
        Raises:
            ValueError: If any required environment variable is missing.
        """
        required_vars = {
            "GROQ_API_KEY": cls.GROQ_API_KEY,
            "ASTRA_DB_APPLICATION_TOKEN": cls.ASTRA_DB_APPLICATION_TOKEN,
            "ASTRA_DB_KEYSPACE": cls.ASTRA_DB_KEYSPACE,
            "ASTRA_DB_API_ENDPOINT": cls.ASTRA_DB_API_ENDPOINT,
        }
        
        missing = [var for var, value in required_vars.items() if not value]
        if missing:
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing)}"
            )
