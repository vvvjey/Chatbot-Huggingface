from flask import request, jsonify
from app.models.chatbot import ChatbotModel
import os

def query_chatbot():
    user_query = request.args.get('query')
    
    if not user_query:
        return jsonify({"error": "No query provided!"}), 400
    
    chatbot = ChatbotModel(document_file=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'paul_graham_essay.txt'))
    response = chatbot.query(user_query)
    
    return jsonify({"response": str(response)})
