import numpy as np
from tensorflow.keras.models import load_model
from util.prepare_data import load_data, preprocess_single_input
import os
from dotenv import load_dotenv
import requests

# Constants
MAX_LEN = 50
DATA_PATH = "data/ner.csv"
MODEL_PATH = "model/ner_bilstm1_tf.h5"

# Load data and model once
X, y, word2idx, tag2idx = load_data(DATA_PATH, max_len=MAX_LEN)
idx2tag = {v: k for k, v in tag2idx.items()}
model = load_model(MODEL_PATH)

# Hugging Face API setup
load_dotenv()
TOKEN = os.getenv("HF_API_TOKEN")
URL = "https://api-inference.huggingface.co/models/Davlan/xlm-roberta-base-ner-hrl"
HEADERS = {"Authorization": f"Bearer {TOKEN}"}

def get_entities(text):
    x = preprocess_single_input(text, word2idx, MAX_LEN)
    x = np.array([x])
    
    y_pred = model.predict(x)
    y_pred = np.argmax(y_pred, axis=-1)[0]

    entities = []
    for i, pred_idx in enumerate(y_pred):
        word = text.split()[i] if i < len(text.split()) else ""
        tag = idx2tag.get(pred_idx, "O")
        if tag != "O":
            entities.append({"word": word, "entity_group": tag})

    response = requests.post(URL, headers=HEADERS, json={"inputs": text})
    if response.status_code == 200:
        return response.json()
    else:
        return [{"error": response.json()}]