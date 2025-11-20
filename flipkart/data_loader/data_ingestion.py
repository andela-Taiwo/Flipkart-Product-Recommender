from pathlib import Path
import logging
from typing import Optional
from flipkart.data_loader.data_converter import DataConverter
from flipkart.config import Config
from langchain_astradb import AstraDBVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)


class DataIngestor:
    """Handles data ingestion into AstraDB vector store.
    
    This class manages the process of loading data from CSV files,
    converting them to documents, and storing them in a vector database.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize DataIngestor with configuration.
        
        Args:
            config: Optional Config instance. If not provided, creates a new one.
            
        Raises:
            ValueError: If required configuration is missing.
        """
        self.config = config or Config()
        self.config.validate()
        
        try:
            self.embedding_model = HuggingFaceEmbeddings(
                model_name=self.config.EMBEDDING_MODEL
            )
            self.vstore = AstraDBVectorStore(
                embedding=self.embedding_model,
                collection_name=self.config.COLLECTION_NAME,
                token=self.config.ASTRA_DB_APPLICATION_TOKEN,
                api_endpoint=self.config.ASTRA_DB_API_ENDPOINT,
                namespace=self.config.ASTRA_DB_KEYSPACE
            )
            logger.info("DataIngestor initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing DataIngestor: {e}")
            raise
        
    def ingest_data(self, load_existing_data: bool = True, file_path: Optional[Path] = None) -> AstraDBVectorStore:
        """Ingest data into the vector store.
        
        Args:
            load_existing_data: If True, returns existing vector store without loading new data.
            file_path: Optional path to CSV file. If not provided, uses default location.
            
        Returns:
            AstraDBVectorStore instance.
            
        Raises:
            FileNotFoundError: If the data file doesn't exist.
            ValueError: If data conversion fails.
        """
        if load_existing_data:
            logger.info("Loading existing vector store")
            return self.vstore
        
        # Determine file path
        if file_path is None:
            project_root = Path(__file__).parent.parent.parent
            file_path = project_root / "data" / "flipkart_product_review.csv"
        
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        try:
            logger.info(f"Converting data from {file_path}")
            data_converter = DataConverter(file_path)
            documents = data_converter.convert_to_documents()
            
            logger.info(f"Adding {len(documents)} documents to vector store")
            self.vstore.add_documents(documents)
            logger.info("Data ingestion completed successfully")
            print("Igot hereeeee>>>>>>>>>>>>>>>>>>>>>>>>>")
            return self.vstore
        except Exception as e:
            logger.error(f"Error during data ingestion: {e}")
            raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    data_ingestor = DataIngestor()
    data_ingestor.ingest_data(load_existing_data=False)