import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Dense, TimeDistributed, Dropout
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score

# --- Configuration Parameters ---
MAX_LEN = 75  # Increased max length
BATCH_SIZE = 64 # Changed batch size
EPOCHS = 10   # Increased epochs
EMBEDDING_DIM = 100 # New parameter for embedding dimension
LSTM_UNITS = 128  # New parameter for LSTM units
DROPOUT_RATE = 0.3 # New parameter for dropout rate
VALIDATION_SPLIT_RATIO = 0.15 # Changed validation split ratio
RANDOM_SEED = 42 # For reproducibility

# --- 1. Data Preparation Module (Expanded from prepare_data.py) ---
def load_data(filepath, max_len=None):
    """
    Loads data from a CSV file, processes it, and converts words/tags to indices.
    Assumes CSV format: 'Word', 'Tag' columns.
    """
    sentences = []
    tags = []
    current_sentence = []
    current_tags = []

    with open(filepath, 'r', encoding='utf-8') as f:
        # Skip header if present (assuming first line is header)
        f.readline()
        for line in f:
            line = line.strip()
            if not line: # Empty line indicates end of a sentence
                if current_sentence:
                    sentences.append(current_sentence)
                    tags.append(current_tags)
                current_sentence = []
                current_tags = []
            else:
                parts = line.split(',') # Assuming comma as delimiter
                if len(parts) == 2: # Expecting 'Word', 'Tag'
                    word = parts[0].strip()
                    tag = parts[1].strip()
                    current_sentence.append(word)
                    current_tags.append(tag)
                # else: Handle malformed lines if necessary (e.g., skip or log warning)
    
    # Add the last sentence if the file doesn't end with an empty line
    if current_sentence:
        sentences.append(current_sentence)
        tags.append(current_tags)

    print(f"Loaded {len(sentences)} sentences from {filepath}")

    # Create word and tag dictionaries
    words = sorted(list(set([word for sentence in sentences for word in sentence])))
    tags_set = sorted(list(set([tag for tag_list in tags for tag_list in tag_list])))

    # Add special tokens
    word2idx = {word: i + 2 for i, word in enumerate(words)} # 0 for padding, 1 for unknown
    word2idx["PAD"] = 0
    word2idx["UNK"] = 1

    tag2idx = {tag: i + 1 for i, tag in enumerate(tags_set)} # 0 for padding
    tag2idx["PAD"] = 0

    idx2word = {i: word for word, i in word2idx.items()}
    idx2tag = {i: tag for tag, i in tag2idx.items()}

    # Convert sentences and tags to numerical sequences
    X = [[word2idx.get(w, word2idx["UNK"]) for w in s] for s in sentences]
    y = [[tag2idx.get(t, tag2idx["PAD"]) for t in t_list] for t_list in tags]

    # Pad sequences to MAX_LEN
    X = pad_sequences(maxlen=max_len, sequences=X, padding="post", value=word2idx["PAD"])
    y = pad_sequences(maxlen=max_len, sequences=y, padding="post", value=tag2idx["PAD"])

    print(f"Padded sequences to length: {max_len}")
    print(f"Vocabulary size: {len(word2idx)}")
    print(f"Number of tags: {len(tag2idx)}")

    return X, y, word2idx, tag2idx, idx2word, idx2tag

# --- 2. NER Model Module (Expanded from ner_model.py) ---
def build_model(vocab_size, num_tags, max_len, embedding_dim=EMBEDDING_DIM, lstm_units=LSTM_UNITS, dropout_rate=DROPOUT_RATE):
    """
    Builds a Bidirectional LSTM-based NER model.
    """
    input_word = Input(shape=(max_len,), name='input_word')

    # Embedding layer
    embedding_layer = Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        input_length=max_len,
        mask_zero=True # Masking zero values for padded sequences
    )(input_word)

    # Bidirectional LSTM layer
    bilstm = Bidirectional(
        LSTM(
            units=lstm_units,
            return_sequences=True, # Important for sequence labeling
            recurrent_dropout=dropout_rate # Dropout for recurrent connections
        )
    )(embedding_layer)

    # Dropout for regularization
    dropout_layer = Dropout(dropout_rate)(bilstm)

    # TimeDistributed Dense layer for output at each timestep
    output = TimeDistributed(Dense(num_tags, activation="softmax"))(dropout_layer)

    model = Model(inputs=input_word, outputs=output)
    print("NER BiLSTM model built successfully.")
    return model

# --- Main Script ---
if __name__ == "__main__":
    # Load and prepare data
    print("--- Loading and Preparing Data ---")
    X, y, word2idx, tag2idx, idx2word, idx2tag = load_data(
        "data/ner.csv", max_len=MAX_LEN
    )

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=VALIDATION_SPLIT_RATIO, random_state=RANDOM_SEED
    )
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    # Build model
    print("\n--- Building Model ---")
    model = build_model(
        len(word2idx),
        len(tag2idx),
        MAX_LEN,
        embedding_dim=EMBEDDING_DIM,
        lstm_units=LSTM_UNITS,
        dropout_rate=DROPOUT_RATE
    )

    # Compile with correct loss function
    print("\n--- Compiling Model ---")
    # Note: SparseCategoricalCrossentropy is suitable when target 'y' is integers (not one-hot)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), # Explicit learning rate
        loss=SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
    )

    # Show model summary
    print("\n--- Model Summary ---")
    model.summary()

    # Train model
    print("\n--- Training Model ---")
    history = model.fit(
        X_train,
        y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_test, y_test), # Use dedicated test set for validation
        verbose=1 # Show training progress
    )

    # Evaluate model on the test set
    print("\n--- Evaluating Model ---")
    loss, accuracy = model.evaluate(X_test, y_test, batch_size=BATCH_SIZE, verbose=0)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")

    # Detailed Classification Report
    print("\n--- Generating Classification Report ---")
    y_pred_probs = model.predict(X_test, batch_size=BATCH_SIZE)
    y_pred = np.argmax(y_pred_probs, axis=-1)

    # Flatten y_true and y_pred, excluding padding
    y_true_flat = []
    y_pred_flat = []

    for i in range(len(y_test)):
        for j in range(MAX_LEN):
            true_tag_idx = y_test[i][j]
            predicted_tag_idx = y_pred[i][j]

            # Only include non-padding tags for evaluation
            if true_tag_idx != tag2idx["PAD"]:
                y_true_flat.append(idx2tag[true_tag_idx])
                y_pred_flat.append(idx2tag[predicted_tag_idx])
    
    # Generate report, excluding 'PAD' tag
    target_names = [idx2tag[i] for i in sorted(tag2idx.keys(), key=lambda k: tag2idx[k]) if idx2tag[i] != "PAD"]
    print(classification_report(y_true_flat, y_pred_flat, labels=target_names, zero_division=0))

    # Save model
    print("\n--- Saving Model ---")
    model_save_path = "ner_bilstm_expanded_tf.h5"
    model.save(model_save_path)
    print(f"Model saved to: {model_save_path}")

    # Example prediction (optional, for demonstration)
    print("\n--- Example Prediction ---")
    sample_sentence = ["I", "am", "visiting", "New", "York", "City", "next", "week", "."]
    
    # Convert sentence to indices
    sample_sequence = [word2idx.get(w, word2idx["UNK"]) for w in sample_sentence]
    
    # Pad and reshape for model input
    padded_sample = pad_sequences(maxlen=MAX_LEN, sequences=[sample_sequence], padding="post", value=word2idx["PAD"])
    
    # Predict
    predicted_probs = model.predict(padded_sample)
    predicted_tag_indices = np.argmax(predicted_probs, axis=-1)[0] # Get the first (and only) sequence

    # Map indices back to tags
    predicted_tags = [idx2tag[idx] for idx in predicted_tag_indices[:len(sample_sentence)]]

    print("Sentence:", " ".join(sample_sentence))
    print("Predicted Tags:", predicted_tags)