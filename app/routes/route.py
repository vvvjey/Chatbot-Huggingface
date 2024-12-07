from flask import Blueprint
from app.controllers.chatbotController import query_chatbot
api_routes = Blueprint('api/v1/', __name__)
api_routes.add_url_rule('/query', view_func=query_chatbot, methods=['GET'])
