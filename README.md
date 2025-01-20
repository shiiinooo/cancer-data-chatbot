# Cancer Data Chatbot with ChromaDB

This project is a chatbot application designed to interact with cancer-related data using a Retrieval-Augmented Generation (RAG) approach. The application leverages ChromaDB to store and query data, and uses Azure OpenAI for generating responses.

## Features

- **Chatbot Interface**: Built with Gradio for a user-friendly interface.
- **Data Processing**: Supports CSV files converting them into a SQL database.
- **ChromaDB Integration**: Stores and queries data using ChromaDB.
- **Azure OpenAI**: Utilizes Azure OpenAI for generating embeddings and chat completions.

## Setup

### Prerequisites

- Python 3.10 or higher
- Azure OpenAI credentials
- Required Python packages (see `requirements.txt`)

### Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/shiinoo/cancer-data-chatbot.git
   cd cancer-data-chatbot
   ```

2. **Install Dependencies**

   Install the required Python packages using pip:

   ```bash
   pip install -r requirements.txt
   ```

3. **Set Up Environment Variables**

   Create a `.env` file in the root directory and add your Azure OpenAI credentials:

   ```plaintext
   AZURE_OPENAI_MODEL_API_KEY=your_model_api_key
   AZURE_OPENAI_MODEL_ENDPOINT=your_model_endpoint
   AZURE_OPENAI_EMBEDDING_API_KEY=your_embedding_api_key
   AZURE_OPENAI_EMBEDDING_ENDPOINT=your_embedding_endpoint
   AZURE_OPENAI_API_VERSION=your_api_version
   AZURE_OPENAI_DEPLOYMENT_NAME=your_deployment_name
   AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME=your_embedding_deployment_name
   ```

4. **Prepare the Vector Database**

   Run the script to prepare the vector database from a CSV file:

   ```bash
   python src/prepare_csv_vectordb.py
   ```

## Usage

1. **Launch the Application**

   Start the Gradio application:

   ```bash
   python src/app.py
   ```

2. **Interact with the Chatbot**

   - Ask questions about the cancer data using the chatbot interface.

## Code Structure

- **src/app.py**: Main application file for launching the Gradio interface.
- **src/utils/prepare_vectordb_from_csv.py**: Prepares the vector database from CSV files.
- **src/utils/chatbot.py**: Contains the chatbot logic and interaction with Azure OpenAI.
- **src/query_vectordb.py**: Script for querying the ChromaDB.

## Configuration

- **configs/app_config.yml**: Configuration file for directories and RAG settings.


## Acknowledgments

- [Gradio](https://gradio.app/)
- [Azure OpenAI](https://azure.microsoft.com/en-us/services/cognitive-services/openai-service/)
- [ChromaDB](https://chromadb.com/)
