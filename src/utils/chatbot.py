import os
from typing import List, Tuple
import json
from utils.load_config import LoadConfig
from openai import AzureOpenAI

# Initialize configuration
APPCFG = LoadConfig()

class ChatBot:
    def __init__(self):
        self.model_client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_MODEL_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_MODEL_ENDPOINT")
        )
        
        self.embedding_client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_EMBEDDING_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_EMBEDDING_ENDPOINT")
        )

    def respond(self, chatbot: List, message: str, chat_type: str, app_functionality: str) -> Tuple:
        try:
            if not isinstance(chatbot, list):
                chatbot = []

            if app_functionality == "Chat" and chat_type == "RAG with stored CSV/XLSX ChromaDB":
                # Get embeddings from Azure
                response = self.embedding_client.embeddings.create(
                    input=message,
                    model=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")
                )
                query_embeddings = response.data[0].embedding
                
                # Query ChromaDB
                vectordb = APPCFG.chroma_client.get_collection(name=APPCFG.collection_name)
                results = vectordb.query(
                    query_embeddings=query_embeddings,
                    n_results=APPCFG.top_k
                )
                
                # Properly format the context from ChromaDB results
                # Convert string representations back to structured data
                try:
                    context_data = [eval(doc) for doc in results['documents'][0]]
                    # Extract relevant numerical data
                    context = "Available data points:\n"
                    for data_point in context_data:
                        context += json.dumps(data_point, indent=2) + "\n"
                except:
                    context = "\n".join([str(doc) for doc in results['documents'][0]])

                # Enhanced prompt with better guidance
                prompt = f"""
                <|system|>You are a medical AI assistant. The context provided contains limited data. 
                Use only the available numerical data for calculations and specify if more data is needed.
                You will receive the user's question along with the search results of that question over a database. 
                Give the user the proper answer Without details.

                Context: {context}

                <|user|>{message}
                <|assistant|>
                """
                
                # Generate response using Azure OpenAI model
                response = self.model_client.chat.completions.create(
                    model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0  # Set to 0 for more consistent, factual responses
                )
                
                response_text = response.choices[0].message.content
                chatbot.append((message, response_text))
                return "", chatbot
            else:
                chatbot.append((message, "Unsupported chat type or functionality."))
                return "", chatbot

        except Exception as e:
            error_message = f"An error occurred: {str(e)}"
            chatbot.append((message, error_message))
            return "", chatbot