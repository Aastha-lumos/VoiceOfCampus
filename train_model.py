
"""
VoiceOfCampus: ML-Based College Feedback Analyzer
Model: TF-IDF + Logistic Regression
"""

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Step 1: Load dataset
data = pd.read_csv("training_data.csv")

# Remove missing and duplicate feedback
data = data.dropna(subset=["feedback_text", "sentiment"])
data["feedback_text"] = data["feedback_text"].astype(str).str.strip()
data["sentiment"] = data["sentiment"].astype(str).str.strip()

data = data[data["feedback_text"] != ""]
data = data.drop_duplicates(subset=["feedback_text"])

X = data["feedback_text"]
y = data["sentiment"]

print("Dataset distribution:")
print(y.value_counts())

# Step 2: Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Step 3: TF-IDF
vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    max_features=20000,
    sublinear_tf=True
)

X_train_numbers = vectorizer.fit_transform(X_train)
X_test_numbers = vectorizer.transform(X_test)

# Step 4: Train model
model = LogisticRegression(
    C=2,
    max_iter=2000,
    class_weight="balanced"
)

model.fit(X_train_numbers, y_train)

# Step 5: Evaluate
predictions = model.predict(X_test_numbers)

print("\nAccuracy:")
print(round(accuracy_score(y_test, predictions) * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, predictions, zero_division=0))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions, labels=model.classes_))
print("Class order:", model.classes_)

# Step 6: Save model
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("\nModel and vectorizer saved successfully!")

# Step 7: Test examples
test_feedback = [
    "The teachers are good but the classrooms are not clean.",
    "The food is not bad, but it is not very tasty either.",
    "I expected better internet facilities from our college.",
    "There is nothing wrong with the library.",
    "The canteen food is okay, but the prices are high.",
    "The college is not perfect, but overall it provides good facilities.",
    "I don't think the teachers are helpful.",
    "The Wi-Fi is sometimes good and sometimes poor."
]

print("\nTesting examples:")

for feedback in test_feedback:
    numbers = vectorizer.transform([feedback])
    prediction = model.predict(numbers)[0]
    print(feedback, "---->", prediction)
