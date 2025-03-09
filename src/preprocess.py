import re
import pandas as pd
import os
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download("punkt")
nltk.download("stopwords")

DATA_DIR = "data"
TRAIN_PATH = os.path.join(DATA_DIR, "Train.csv")
VALID_PATH = os.path.join(DATA_DIR, "Valid.csv")
TEST_PATH = os.path.join(DATA_DIR, "Test.csv")
OUTPUT_PATH = os.path.join(DATA_DIR, "processed_reviews.csv")

TEXT_COLUMN = "text"

os.makedirs(DATA_DIR, exist_ok=True)

def load_data():
    """Load train, valid, and test datasets."""
    print("Loading datasets...")
    df_train = pd.read_csv(TRAIN_PATH)
    df_valid = pd.read_csv(VALID_PATH)
    df_test = pd.read_csv(TEST_PATH)

    # Combine all datasets
    df = pd.concat([df_train, df_valid, df_test], ignore_index=True)
    print(f"Loaded {len(df)} total reviews!")
    return df

def preprocess_text(text):
    """Preprocess text: lowercasing, removing special characters, tokenizing, and removing stopwords."""
    try:
        text = text.lower() 
        text = re.sub(r"[^a-zA-Z\s]", " ", text) 
        tokens = word_tokenize(text) 
        stop_words = set(stopwords.words("english"))
        tokens = [word for word in tokens if word not in stop_words] 
        return " ".join(tokens)
    except Exception as e:
        print(f"Error processing text: {e}")
        return text 

if __name__ == "__main__":
    df = load_data()

    print("Preprocessing text...")
    df["cleaned_text"] = df[TEXT_COLUMN].astype(str).apply(preprocess_text) 

    df.to_csv(OUTPUT_PATH, index=False)
    
    print(f"Data Preprocessing Complete! Saved to {OUTPUT_PATH}")
