from dotenv import load_dotenv
import os
from openai import AzureOpenAI
import json

class AzureOpenAITest:
    """Test class to verify Azure OpenAI connectivity and credentials"""
    
    def __init__(self):
        # Load environment variables
        load_dotenv()
        
        # Initialize Azure OpenAI client for the model
        self.model_client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_MODEL_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_MODEL_ENDPOINT")
        )
        
        # Initialize Azure OpenAI client for embeddings
        self.embedding_client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_EMBEDDING_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_EMBEDDING_ENDPOINT")
        )

    def test_chat_completion(self):
        """Test chat completion API"""
        try:
            print("\n=== Chat Completion Test ===")
            
            messages = [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": "Hello!"}
            ]
            
            response = self.model_client.chat.completions.create(
                model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
                messages=messages
            )
            print(f"Chat Response: {response.choices[0].message.content}")
            return True
        except Exception as e:
            print(f"Error: {str(e)}")
            return False

    def test_embeddings(self):
        """Test embeddings API"""
        try:
            embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")
            print(f"\n=== Embeddings Test (using deployment: {embedding_deployment}) ===")
            
            request = {
                "model": embedding_deployment,
                "input": ["Hello world"]
            }
            print("Making request with parameters:")
            print(json.dumps(request, indent=2))
            
            response = self.embedding_client.embeddings.create(**request)
            print(f"Success! Embedding vector length: {len(response.data[0].embedding)}")
            return True
        except Exception as e:
            print(f"Error: {str(e)}")
            return False

    def verify_environment(self):
        """Verify environment variables and print detailed information"""
        print("\n=== Environment Variables Verification ===")
        required_vars = {
            "AZURE_OPENAI_MODEL_API_KEY": "Model API Key",
            "AZURE_OPENAI_MODEL_ENDPOINT": "Model Endpoint",
            "AZURE_OPENAI_EMBEDDING_API_KEY": "Embedding API Key",
            "AZURE_OPENAI_EMBEDDING_ENDPOINT": "Embedding Endpoint",
            "AZURE_OPENAI_API_VERSION": "API Version",
            "AZURE_OPENAI_DEPLOYMENT_NAME": "GPT Deployment Name",
            "AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME": "Embedding Deployment Name"
        }
        
        all_set = True
        for var, description in required_vars.items():
            value = os.getenv(var)
            if not value:
                print(f"❌ {description} ({var}) is not set")
                all_set = False
            else:
                masked_value = value
                if "API_KEY" in var:
                    masked_value = f"{value[:5]}...{value[-5:]}"
                print(f"✓ {description}: {masked_value}")
        
        if not all_set:
            print("\n⚠️  Some required environment variables are missing!")
        return all_set

def main():
    print("Starting Azure OpenAI Connection Tests...")
    tester = AzureOpenAITest()
    
    # First verify environment
    env_ok = tester.verify_environment()
    if not env_ok:
        print("\n⚠️  Please set all required environment variables before continuing.")
        return
    
    # Run tests
    chat_success = tester.test_chat_completion()
    embeddings_success = tester.test_embeddings()
    
    # Print summary
    print("\n=== Test Summary ===")
    print(f"Environment Variables: {'✓ All Set' if env_ok else '✗ Missing Some'}")
    print(f"Chat Completion Test: {'✓ Passed' if chat_success else '✗ Failed'}")
    print(f"Embeddings Test: {'✓ Passed' if embeddings_success else '✗ Failed'}")

if __name__ == "__main__":
    main()
    