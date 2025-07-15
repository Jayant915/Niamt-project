from prepare_data import load_data
from ner_model import build_model
from tensorflow.keras.losses import SparseCategoricalCrossentropy

MAX_LEN = 50
BATCH_SIZE = 32
EPOCHS = 5

# Load and prepare data
X, y, word2idx, tag2idx = load_data("data/ner.csv", max_len=MAX_LEN)

# Build model
model = build_model(len(word2idx), len(tag2idx), MAX_LEN)

# Compile with correct loss function
model.compile(optimizer="adam", loss=SparseCategoricalCrossentropy(), metrics=["accuracy"])

# Show model summary
model.summary()

# Train model
model.fit(X, y, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.1)

# Save model
model.save("ner_bilstm1_tf.h5")
