import joblib

model = joblib.load("models/svm_best_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

test_texts = ["I love this!", "This is the not worst but best.", "Absolutely amazing experience."]
text_vectorized = vectorizer.transform(test_texts)
predictions = model.predict(text_vectorized)

for text, pred in zip(test_texts, predictions):
    print(f"'{text}' -> Prediction: {pred}")
