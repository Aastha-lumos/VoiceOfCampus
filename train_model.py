"""
VoiceOfCampus: ML-Based College Feedback Analyzer
File: train_model.py

Purpose:
    Train a sentiment classification model (Positive / Negative / Neutral)
    using TF-IDF and Logistic Regression, evaluate it, and save it for use
    in the Streamlit application.
"""

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Step 1: Load the labeled dataset
data = pd.read_csv("training_data.csv")

X = data["feedback_text"]   # Input: the feedback text written by students
y = data["sentiment"]       # Output: the correct label (Positive / Negative / Neutral)

# Step 2: Split the data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Step 3: Convert text into numerical features using TF-IDF
vectorizer = TfidfVectorizer(lowercase=True)
X_train_numbers = vectorizer.fit_transform(X_train)   # Learn vocabulary from training data
X_test_numbers = vectorizer.transform(X_test)         # Apply the same vocabulary to test data

# Step 4: Train the Logistic Regression model
model = LogisticRegression(C=10, max_iter=1000)
model.fit(X_train_numbers, y_train)

# Step 5: Evaluate the model on the unseen test data
predictions = model.predict(X_test_numbers)
accuracy = accuracy_score(y_test, predictions)

print("Number of training examples:", len(X_train))
print("Number of testing examples:", len(X_test))
print("Model accuracy on test data:", round(accuracy * 100, 2), "%")

# Step 6: Save the trained model and the vectorizer for use in the app
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
print("The trained model and vectorizer have been saved successfully.")

# Step 7: Test the model with a sample feedback
sample_feedback = ["The Wi-Fi is very slow"]
sample_numbers = vectorizer.transform(sample_feedback)
print("Sample feedback:", sample_feedback[0])
print("Predicted sentiment:", model.predict(sample_numbers)[0])