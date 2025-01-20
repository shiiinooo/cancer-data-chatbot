from utils.load_config import LoadConfig
import chromadb
from pyprojroot import here

def query_vectordb():
    # Initialize config and client
    config = LoadConfig().config
    persist_dir = str(here(config["directories"]["persist_directory"]))
    client = chromadb.PersistentClient(path=persist_dir)
    
    # Get the collection
    collection = client.get_collection(config["rag_config"]["collection_name"])
    
    # Example query
    results = collection.query(
        query_texts=["Tell me about cancer patients"],
        n_results=config["rag_config"]["top_k"]
    )
    
    # Print results
    print("\nQuery Results:")
    for i, result in enumerate(results['documents'][0]):
        print(f"\nResult {i+1}:")
        print(result)

if __name__ == "__main__":
    query_vectordb() 