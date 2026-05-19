import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'agriculture_pest_key'
    KNOWLEDGE_BASE_DIR = os.path.join(os.path.dirname(__file__), 'knowledge_base')
    HISTORY_DIR = os.path.join(os.path.dirname(__file__), 'history')
    
    @staticmethod
    def init_app(app):
        os.makedirs(Config.KNOWLEDGE_BASE_DIR, exist_ok=True)
        os.makedirs(Config.HISTORY_DIR, exist_ok=True)