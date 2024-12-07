from langchain_huggingface import HuggingFaceEndpoint
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from app.utils.config import get_huggingface_api_token

class ChatbotModel:
    def __init__(self, document_file):
        self.documents = SimpleDirectoryReader(input_files=[document_file]).load_data()

        # Set up embeddings and LLM
        self.llm = HuggingFaceEndpoint(
            repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
            huggingfacehub_api_token=get_huggingface_api_token(),
            task="text-generation",
            max_new_tokens=512,
            do_sample=False,
            repetition_penalty=1.03,
        )

        self.embeddings = HuggingFaceInferenceAPIEmbeddings(
            api_key=get_huggingface_api_token(),
            model_name="sentence-transformers/all-MiniLM-l6-v2"
        )

        Settings.embed_model = self.embeddings
        Settings.llm = self.llm

        # Create the index
        self.index = VectorStoreIndex.from_documents(self.documents)

    def query(self, user_query):
        query_engine = self.index.as_query_engine()
        response = query_engine.query(user_query)
        return response
