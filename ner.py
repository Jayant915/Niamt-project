import numpy as np
from sklearn.metrics import classification_report
from tensorflow.keras.models import load_model
from prepare_data import load_data
import requests
import os  
from dotenv import load_dotenv

# Parameters
MAX_LEN = 50
DATA_PATH = "data/ner.csv"
MODEL_PATH = "model/ner_bilstm1_tf.h5"

# Load data
X, y, word2idx, tag2idx = load_data(DATA_PATH, max_len=MAX_LEN)

# Inverse tag2idx
idx2tag = {v: k for k, v in tag2idx.items()}

# Load model
model = load_model(MODEL_PATH)

# Predict
y_pred = model.predict(X)
y_pred = np.argmax(y_pred, axis=-1)

# Collect true and predicted tags, skipping paddings
flat_true_tags = []
flat_pred_tags = []

for i in range(len(X)):
    for j in range(MAX_LEN):
        true_idx = y[i][j]
        pred_idx = y_pred[i][j]
        if true_idx != tag2idx.get("O", 0):  # skip padding tokens
            flat_true_tags.append(idx2tag[true_idx])
            flat_pred_tags.append(idx2tag[pred_idx])

load_dotenv()
TOKEN = os.getenv("HF_API_TOKEN")

URL = "https://api-inference.huggingface.co/models/Davlan/xlm-roberta-base-ner-hrl"
HEADERS = {
    "Authorization": f"Bearer {TOKEN}"
}


def get_entities(text):
    response = requests.post(URL, headers=HEADERS, json={"inputs": text})
    if response.status_code == 200:
        return response.json()
    else:
        return [{"error": response.json()}]

# Always ask for user input
while True:
    text_input = input("\nEnter text for NER (or type 'exit' to quit): ")
    if text_input.lower() == 'exit':
        print("Exiting...")
        break

    entities = get_entities(text_input)

    if "error" in entities[0]:
        print("Error:", entities[0]["error"].get("error", "Unknown error"))
    else:
        print("\nNamed Entities:\n")
        for ent in entities:
            print(f"{ent['word']} → {ent['entity_group']} (score: {ent['score']:.2f})")
