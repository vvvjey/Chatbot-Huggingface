import os
import dotenv

dotenv.load_dotenv()  # Load environment variables from .env file

def get_huggingface_api_token():
    return os.getenv('HUGGINGFACE_API_TOKEN')
