import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

def load_data(file_path, max_len=50):
    # Read the CSV file
    df = pd.read_csv(file_path, encoding="ISO-8859-1", on_bad_lines='skip').fillna(method='ffill')

    # Keep only necessary columns
    df = df[['word', 'tag']]

    # Create artificial sentence IDs using full stop
    df['sentence_id'] = (df['word'] == '.').cumsum()

    # Group by sentence ID
    sentences, tags = [], []
    for _, group in df.groupby("sentence_id"):
        word_list = group["word"].tolist()
        tag_list = group["tag"].tolist()
        sentences.append(word_list)
        tags.append(tag_list)

    # Create vocabularies
    words = set(w for s in sentences for w in s)
    tags_set = set(t for tag_seq in tags for t in tag_seq)

    word2idx = {w: i + 2 for i, w in enumerate(sorted(words))}
    word2idx["PAD"] = 0
    word2idx["UNK"] = 1

    tag2idx = {t: i for i, t in enumerate(sorted(tags_set))}
    tag2idx["PAD"] = len(tag2idx)  # just in case "O" is missing

    # Encode sequences
    X = [[word2idx.get(w, word2idx["UNK"]) for w in s] for s in sentences]
    y = [[tag2idx[t] for t in ts] for ts in tags]

    # Fix: Ensure all values are within valid range for embedding
    max_word_idx = max(word2idx.values()) - 1
    X = [[min(i, max_word_idx) for i in seq] for seq in X]

    # Pad sequences
    X = pad_sequences(X, maxlen=max_len, padding='post', value=word2idx["PAD"])
    y = pad_sequences(y, maxlen=max_len, padding='post', value=tag2idx.get("O", 0))

    return X, y, word2idx, tag2idx
