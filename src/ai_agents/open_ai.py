import dotenv
import os
from openai import OpenAI

dotenv.load_dotenv()

class OpenAiAgent:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))