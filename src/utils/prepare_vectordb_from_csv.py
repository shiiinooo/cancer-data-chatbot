import pandas as pd
import chromadb
from utils.load_config import LoadConfig
from pyprojroot import here

class PrepareVectorDBFromCSV:
    """
    This class prepares a vector database from a CSV file.
    It loads the data into a ChromaDB collection by reading the CSV file,
    generating embeddings for the content, and storing the data.
    """
    
    def __init__(self, file_path: str):
        """
        Initialize the class with the CSV file path.
        
        Args:
            file_path (str): Path to the CSV file
        """
        self.file_path = file_path
        self.config = LoadConfig()
        persist_dir = str(here(self.config.config["directories"]["persist_directory"]))
        self.chroma_client = chromadb.PersistentClient(path=persist_dir)
        
    def read_csv(self) -> pd.DataFrame:
        """Read the CSV file into a pandas DataFrame."""
        return pd.read_csv(self.file_path)
        
    def run_pipeline(self):
        # Read CSV
        df = self.read_csv()
        
        # Create or get collection
        collection_name = self.config.collection_name
        
        try:
            self.chroma_client.delete_collection(name=collection_name)
        except ValueError:
            pass
            
        collection = self.chroma_client.create_collection(name=collection_name)
        
        # Convert DataFrame to documents - store as individual records
        documents = []
        for _, row in df.iterrows():
            doc = row.to_dict()
            # Convert numpy types to Python native types for proper serialization
            doc = {k: float(v) if isinstance(v, (float, int)) else str(v) 
                for k, v in doc.items()}
            documents.append(str(doc))
        
        # Get embeddings from Azure OpenAI
        try:
            response = self.config.azure_openai_embedding_client.embeddings.create(
                input=documents,
                model=self.config.embedding_model_name
            )
            embeddings = [item.embedding for item in response.data]
            
            # Add documents to collection with embeddings
            collection.add(
                documents=documents,
                embeddings=embeddings,
                ids=[str(i) for i in range(len(documents))],
                metadatas=[{"source": "csv"} for _ in documents]
            )
            
            print(f"Successfully processed {len(documents)} rows into ChromaDB collection '{collection_name}'")
        except Exception as e:
            print(f"Error generating embeddings: {e}")