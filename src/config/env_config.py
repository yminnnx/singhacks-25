"""
Environment Configuration for AML Monitoring System
Loads environment variables from .env file
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Application configuration from environment variables"""
    
    # Groq API Configuration
    GROQ_API_KEY = os.getenv('GROQ_API_KEY')
    GROQ_MODEL = os.getenv('GROQ_MODEL', 'llama-3.3-70b-versatile')  # Updated default model
    
    # Application Settings
    APP_DEBUG = os.getenv('APP_DEBUG', 'False').lower() == 'true'
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    
    @classmethod
    def validate_config(cls):
        """Validate that required environment variables are set"""
        if not cls.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY environment variable is required")
        
        return True

# Example usage:
if __name__ == "__main__":
    try:
        Config.validate_config()
        print("✅ Environment configuration loaded successfully")
        print(f"   Groq API Key: {'***' + Config.GROQ_API_KEY[-4:] if Config.GROQ_API_KEY else 'Not set'}")
        print(f"   Default Model: {Config.GROQ_MODEL}")
        print(f"   Debug Mode: {Config.APP_DEBUG}")
    except ValueError as e:
        print(f"❌ Configuration error: {e}")