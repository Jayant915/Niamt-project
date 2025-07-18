import os
import requests
from dotenv import load_dotenv

load_dotenv()
TOKEN = os.getenv("HF_API_TOKEN")

API_URL = "https://api-inference.huggingface.co/models/Davlan/xlm-roberta-base-ner-hrl"
HEADERS = {"Authorization": f"Bearer {TOKEN}"}

def get_entities(text):
    response = requests.post(API_URL, headers=HEADERS, json={"inputs": text})
    return response.json()