import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from loguru import logger
import wandb

PROCESSED_DATA_PATH = "data/processed_reviews.csv"
VECTORIZER_PATH = "models/vectorizer.pkl"
MODEL_PATH = "models/sentiment_model.pkl"
SVD_PATH = "models/svd.pkl"
TEXT_COLUMN = "cleaned_text"
LABEL_COLUMN = "label"

os.makedirs("models", exist_ok=True)

logger.add("logs/training.log", rotation="500 MB", level="INFO")

wandb.init(project="sentiment_analysis")

hyperparameters = {
    "vectorizer_max_features": 15000,
    "svd_components": 300,
    "svc_max_iter": 1000,
    "test_size": 0.2,
    "random_state": 42
}

# Log hyperparameters to W&B
wandb.config.update(hyperparameters)

# Load preprocessed dataset
logger.info("Loading preprocessed dataset...")
df = pd.read_csv(PROCESSED_DATA_PATH)
logger.info(f"Dataset loaded! Total samples: {len(df)}")

# Fix label mapping (-1 → 0)
logger.info("Fixing label mapping (-1 → 0)...")
df[LABEL_COLUMN] = df[LABEL_COLUMN].replace({-1: 0, 1: 1})
logger.info("Label mapping fixed!")

# Split data into train and test sets
logger.info("Splitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    df[TEXT_COLUMN], df[LABEL_COLUMN], test_size=hyperparameters["test_size"], random_state=hyperparameters["random_state"], stratify=df[LABEL_COLUMN]
)
logger.info(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

# Initialize TF-IDF vectorizer
logger.info("Initializing TF-IDF vectorizer...")
vectorizer = TfidfVectorizer(max_features=hyperparameters["vectorizer_max_features"], ngram_range=(1, 2))

# Fit and transform train data
logger.info("Fitting and transforming train data...")
X_train_tfidf = vectorizer.fit_transform(X_train)
logger.info(f"Train data shape: {X_train_tfidf.shape}")

# Transform test data
logger.info("Transforming test data...")
X_test_tfidf = vectorizer.transform(X_test)
logger.info(f"Test data shape: {X_test_tfidf.shape}")

# Initialize SVD for dimensionality reduction
logger.info("Initializing SVD for dimensionality reduction...")
svd = TruncatedSVD(n_components=hyperparameters["svd_components"], random_state=hyperparameters["random_state"])

# Fit and transform train data with SVD
logger.info("Fitting and transforming train data with SVD...")
X_train_svd = svd.fit_transform(X_train_tfidf)
logger.info(f"Train data shape after SVD: {X_train_svd.shape}")

# Transform test data with SVD
logger.info("Transforming test data with SVD...")
X_test_svd = svd.transform(X_test_tfidf)
logger.info(f"Test data shape after SVD: {X_test_svd.shape}")

# Train the LinearSVC model
logger.info("Training the LinearSVC model...")
model = LinearSVC(random_state=hyperparameters["random_state"], max_iter=hyperparameters["svc_max_iter"])
model.fit(X_train_svd, y_train)
logger.info("Model training completed!")

# Evaluate model on test data
logger.info("Evaluating model on test data...")
y_pred = model.predict(X_test_svd)
test_acc = accuracy_score(y_test, y_pred)
logger.info(f"Test Accuracy: {test_acc:.4f}")

# Log test accuracy to W&B
wandb.log({"test_accuracy": test_acc})

# Save the trained model, vectorizer, and SVD
logger.info("Saving the trained model, vectorizer, and SVD...")
joblib.dump(vectorizer, VECTORIZER_PATH)
joblib.dump(svd, SVD_PATH)
joblib.dump(model, MODEL_PATH)
logger.info("Model, Vectorizer, and SVD saved successfully!")
 