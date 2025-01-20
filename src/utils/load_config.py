import os
from dotenv import load_dotenv
import yaml
from pyprojroot import here
import chromadb
from openai import AzureOpenAI

# Load environment variables
load_dotenv()

class LoadConfig:
    """Load configuration from YAML file and set up directories."""
    
    def __init__(self):
        """Initialize configuration from YAML file and set up Azure OpenAI clients."""
        config_path = here("configs/app_config.yml")
        with open(config_path, 'r') as file:
            app_config = yaml.safe_load(file)
        
        self.config = app_config
        self.load_directories(app_config=app_config)
        
        # Set up Azure OpenAI clients
        self.azure_openai_model_client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_MODEL_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_MODEL_ENDPOINT")
        )
        
        self.azure_openai_embedding_client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_EMBEDDING_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_EMBEDDING_ENDPOINT")
        )

        # Print environment variables for debugging
        print("Model API Key:", os.getenv("AZURE_OPENAI_MODEL_API_KEY"))
        print("Model Endpoint:", os.getenv("AZURE_OPENAI_MODEL_ENDPOINT"))
        print("Embedding API Key:", os.getenv("AZURE_OPENAI_EMBEDDING_API_KEY"))
        print("Embedding Endpoint:", os.getenv("AZURE_OPENAI_EMBEDDING_ENDPOINT"))
        print("API Version:", os.getenv("AZURE_OPENAI_API_VERSION"))
        print("Model Deployment Name:", os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"))
        print("Embedding Deployment Name:", os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"))
        
        # Load model configurations from environment variables
        self.model_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        self.embedding_model_name = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")
     
        # Load RAG configurations from environment variables
        self.collection_name = app_config["rag_config"]["collection_name"]
        self.top_k = app_config["rag_config"]["top_k"]
        self.rag_llm_system_role = app_config["llm_config"]["rag_llm_system_role"]
        
        # Initialize ChromaDB client
        self.chroma_client = chromadb.PersistentClient(
            path=str(here(app_config["directories"]["persist_directory"]))
        )

    def load_directories(self, app_config):
        """Create directories if they don't exist."""
        # Only create persist_directory as that's all we need now
        os.makedirs(
            here(app_config["directories"]["persist_directory"]), 
            exist_ok=True
        )

    def load_rag_config(self, app_config):
        """Load RAG-specific configurations."""
        self.collection_name = app_config["rag_config"]["collection_name"]
        self.top_k = app_config["rag_config"]["top_k"]