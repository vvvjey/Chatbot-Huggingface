from flask import Flask, request, jsonify
import os
import dotenv  # Hide the API token
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from langchain_huggingface import HuggingFaceEndpoint
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings

app = Flask(__name__)

# Load environment variables from .env (this will keep your API key safe)
dotenv.load_dotenv()

# Load the document (you can load documents here or provide API to upload files later)
documents = SimpleDirectoryReader(input_files=["paul_graham_essay.txt"]).load_data()

# Setup LLM and Embeddings (using environment variables for security)
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_TOKEN"),
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,
    repetition_penalty=1.03,
)

# Setup Embeddings
embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key=os.getenv("HUGGINGFACE_API_TOKEN"),
    model_name="sentence-transformers/all-MiniLM-l6-v2"
)

# Set the settings for index
Settings.embed_model = embeddings
Settings.llm = llm

# Create the index
index = VectorStoreIndex.from_documents(documents)

@app.route('/query', methods=['GET'])
def query():
    # Get the user's query from the URL parameter
    user_query = request.args.get('query')

    if not user_query:
        return jsonify({"error": "No query provided!"}), 400

    # Use the index to get the response to the query
    query_engine = index.as_query_engine()
    response = query_engine.query(user_query)

    # Return the response as JSON
    return jsonify({"response": str(response)})


if __name__ == '__main__':
    app.run(debug=True)
