import pandas as pd
from pathlib import Path
from typing import List
from langchain_core.documents import Document
import logging

logger = logging.getLogger(__name__)


class DataConverter:
    """Converts CSV data to LangChain Document format.

    This class reads product review data from a CSV file and converts it
    into a list of LangChain Document objects suitable for vector storage.
    """

    def __init__(self, file_path: str):
        """Initialize DataConverter with file path.

        Args:
            file_path: Path to the CSV file containing product reviews.

        Raises:
            FileNotFoundError: If the file doesn't exist.
        """
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")

    def convert_to_documents(self) -> List[Document]:
        """Convert CSV data to LangChain Document objects.

        Reads the CSV file and converts each row into a Document with
        review text as content and product title as metadata.

        Returns:
            List of Document objects.

        Raises:
            ValueError: If required columns are missing or DataFrame is empty.
        """
        try:
            df = pd.read_csv(self.file_path)
        except Exception as e:
            logger.error(f"Error reading CSV file: {e}")
            raise

        # Validate required columns
        required_columns = ["product_title", "review"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        if df.empty:
            raise ValueError("CSV file is empty")

        docs = [
            Document(
                page_content=str(row.review) if pd.notna(row.review) else "",
                metadata={
                    "product_name": str(row.product_title)
                    if pd.notna(row.product_title)
                    else "Unknown"
                },
            )
            for row in df[required_columns].itertuples(index=False)
        ]

        logger.info(f"Converted {len(docs)} documents from {len(df)} rows")
        return docs


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent
    file_path = project_root / "data" / "flipkart_product_review.csv"
    data_converter = DataConverter(str(file_path))
    documents = data_converter.convert_to_documents()
    print(f"Converted {len(documents)} documents")
