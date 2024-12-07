from flask import Flask
from .routes.route import api_routes
import os
def create_app():
    app = Flask(__name__)
    
    # Register routes
    app.register_blueprint(api_routes,url_prefix='/api/v1')

    return app
