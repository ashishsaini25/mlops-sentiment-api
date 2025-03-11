import re
import pandas as pd
import os
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from loguru import logger

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

DATA_DIR = "data"
TRAIN_PATH = os.path.join(DATA_DIR, "Train.csv")
VALID_PATH = os.path.join(DATA_DIR, "Valid.csv")
TEST_PATH = os.path.join(DATA_DIR, "Test.csv")
OUTPUT_PATH = os.path.join(DATA_DIR, "processed_reviews.csv")

TEXT_COLUMN = "text"

os.makedirs(DATA_DIR, exist_ok=True)

logger.add("preprocess.log", rotation="500 MB")

def load_data():
    """
    Load train, validation, and test datasets, and combine them into a single DataFrame.
    """
    logger.info("Loading datasets...")
    try:
        df_train = pd.read_csv(TRAIN_PATH)
        df_valid = pd.read_csv(VALID_PATH)
        df_test = pd.read_csv(TEST_PATH)

        df = pd.concat([df_train, df_valid, df_test], ignore_index=True)
        logger.info(f"Loaded {len(df)} total reviews!")
        return df
    except Exception as e:
        logger.error(f"Error loading datasets: {e}")
        return pd.DataFrame() 

def preprocess_text(text):
    """
    Preprocess the input text by performing the following steps:
    1. Lowercasing
    2. Removing special characters and numbers
    3. Tokenizing
    4. Removing stopwords
    5. Lemmatization
    """
    try:
        text = text.lower()

        text = re.sub(r"[^a-zA-Z\s]", " ", text)

        tokens = word_tokenize(text)

        stop_words = set(stopwords.words("english"))
        tokens = [word for word in tokens if word not in stop_words]

        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]

        return " ".join(tokens)
    except Exception as e:
        logger.error(f"Error processing text: {e}")
        return text

if __name__ == "__main__":
    df = load_data()

    if not df.empty:
        logger.info("Preprocessing text...")
        df["cleaned_text"] = df[TEXT_COLUMN].astype(str).apply(preprocess_text)

        try:
            df.to_csv(OUTPUT_PATH, index=False)
            logger.info(f"Data Preprocessing Complete! Saved to {OUTPUT_PATH}")
        except Exception as e:
            logger.error(f"Error saving processed data: {e}")
    else:
        logger.error("No data to process.")
