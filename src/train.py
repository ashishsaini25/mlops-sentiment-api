import pandas as pd
import joblib
import logging
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

PROCESSED_DATA_PATH = "data/processed_reviews.csv"
VECTORIZER_PATH = "models/vectorizer.pkl"
MODEL_PATH = "models/sentiment_model.pkl"

TEXT_COLUMN = "cleaned_text"  
LABEL_COLUMN = "label"

os.makedirs("models", exist_ok=True)

logging.info("Loading preprocessed dataset...")
df = pd.read_csv(PROCESSED_DATA_PATH)
logging.info(f"Dataset loaded! Total samples: {len(df)}")

logging.info("Fixing label mapping (-1 → 0)...")
df[LABEL_COLUMN] = df[LABEL_COLUMN].replace({-1: 0, 1: 1})
logging.info("Label mapping fixed!")

logging.info("Splitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    df[TEXT_COLUMN], df[LABEL_COLUMN], test_size=0.2, random_state=42, stratify=df[LABEL_COLUMN]
)
logging.info(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

logging.info("Initializing TF-IDF vectorizer...")
vectorizer = TfidfVectorizer(max_features=15000, ngram_range=(1, 2))

logging.info("Fitting and transforming train data...")
X_train_tfidf = vectorizer.fit_transform(X_train)
logging.info(f"Train data shape: {X_train_tfidf.shape}")

logging.info("Transforming test data...")
X_test_tfidf = vectorizer.transform(X_test)
logging.info(f"Test data shape: {X_test_tfidf.shape}")

logging.info("Initializing SVD for dimensionality reduction...")
svd = TruncatedSVD(n_components=300, random_state=42)

logging.info("Fitting and transforming train data...")
X_train_svd = svd.fit_transform(X_train_tfidf)
logging.info(f"Train data shape after SVD: {X_train_svd.shape}")

logging.info("Transforming test data...")
X_test_svd = svd.transform(X_test_tfidf)
logging.info(f"Test data shape after SVD: {X_test_svd.shape}")

logging.info("Training the LinearSVC model...")
model = LinearSVC(random_state=42, max_iter=1000)
model.fit(X_train_svd, y_train)
logging.info("Model training completed!")

logging.info("Evaluating model on test data...")
y_pred = model.predict(X_test_svd)
test_acc = accuracy_score(y_test, y_pred)
logging.info(f"Test Accuracy: {test_acc:.4f}")

logging.info("Saving the trained model and vectorizer...")
joblib.dump(vectorizer, VECTORIZER_PATH)
joblib.dump(svd, "models/svd.pkl")
joblib.dump(model, MODEL_PATH)
logging.info("Model & Vectorizer saved successfully!")
