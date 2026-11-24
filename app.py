import os
from dotenv import load_dotenv

load_dotenv()

ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD")
import re
from datetime import datetime

import joblib
import pandas as pd
import streamlit as st

FEEDBACK_FILE = "feedback.csv"

# Load the trained model and vectorizer saved by train_model.py
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Keywords used to find the issue category (simple keyword matching, no ML)
ISSUE_KEYWORDS = {
    "Wi-Fi / Internet": ["wifi", "wi-fi", "internet", "network", "router", "signal"],
    "Mess / Food": ["mess", "food", "canteen", "meal", "meals", "lunch", "dinner",
                    "breakfast", "taste", "tasteless", "tasty"],
    "Cleanliness": ["clean", "cleaned", "cleanliness", "dirty", "washroom", "washrooms",
                    "toilet", "toilets", "garbage", "hygiene", "hygienic", "unhygienic",
                    "smell", "dust"],
    "Library": ["library", "book", "books", "librarian"],
    "Transport": ["bus", "buses", "transport", "transportation", "route", "routes",
                  "driver", "drivers", "parking"],
}


def get_issue_category(text):
    """Return the first issue category whose keyword appears in the text."""
    words = re.findall(r"[a-z\-]+", text.lower())
    for category, keywords in ISSUE_KEYWORDS.items():
        for keyword in keywords:
            if keyword in words:
                return category
    return "Other"


def save_feedback(text, sentiment, category):
    """Append one feedback row to feedback.csv (creates the file if needed)."""
    row = pd.DataFrame([{
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "feedback_text": text,
        "sentiment": sentiment,
        "issue_category": category,
    }])
    row.to_csv(FEEDBACK_FILE, mode="a", header=not os.path.exists(FEEDBACK_FILE), index=False)


def show_dashboard():
    """Display the admin dashboard (only called after a successful login)."""
    if st.button("Logout"):
        st.session_state["admin_logged_in"] = False
        st.rerun()

    if not os.path.exists(FEEDBACK_FILE):
        st.info("No feedback has been submitted yet.")
        return

    df = pd.read_csv(FEEDBACK_FILE)

    # 1. Summary counts
    total = len(df)
    positive = (df["sentiment"] == "Positive").sum()
    negative = (df["sentiment"] == "Negative").sum()
    neutral = (df["sentiment"] == "Neutral").sum()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Feedback", total)
    col2.metric("Positive", positive)
    col3.metric("Negative", negative)
    col4.metric("Neutral", neutral)

    # 2. Most reported issues (only from negative feedback)
    st.subheader("Most Reported Issues (Negative Feedback)")
    negative_df = df[df["sentiment"] == "Negative"]
    if negative_df.empty:
        st.write("No negative feedback has been submitted yet.")
    else:
        issue_counts = negative_df["issue_category"].value_counts()
        st.bar_chart(issue_counts)
        issue_table = issue_counts.rename_axis("Issue Category").reset_index(name="Number of Reports")
        st.dataframe(issue_table, hide_index=True)

    # 3. Recent feedback (latest 10, newest first)
    st.subheader("Recent Feedback")
    recent = df.tail(10).iloc[::-1]
    st.dataframe(
        recent[["timestamp", "feedback_text", "sentiment", "issue_category"]],
        hide_index=True,
    )


st.set_page_config(page_title="VoiceOfCampus", page_icon="🎓")
st.title("🎓 VoiceOfCampus")
st.caption("ML-Based College Feedback Analyzer")

page = st.sidebar.radio("Select a section", ["Student Feedback", "Admin Dashboard"])

if page == "Student Feedback":
    st.header("Submit Your Feedback")
    feedback_text = st.text_area("Write your feedback about any college-related issue or experience:")

    if st.button("Submit Feedback"):
        if feedback_text.strip() == "":
            st.warning("Please write some feedback before submitting.")
        else:
            numbers = vectorizer.transform([feedback_text])
            sentiment = model.predict(numbers)[0]
            category = get_issue_category(feedback_text)
            save_feedback(feedback_text.strip(), sentiment, category)

            st.success("Thank you! Your feedback has been submitted.")
            if sentiment == "Positive":
                st.success("Predicted sentiment: Positive")
            elif sentiment == "Negative":
                st.error("Predicted sentiment: Negative")
            else:
                st.info("Predicted sentiment: Neutral")
            st.write("Issue category:", category)

else:
    st.header("Admin Dashboard")

    # Remember whether the admin has logged in during this browser session
    if "admin_logged_in" not in st.session_state:
        st.session_state["admin_logged_in"] = False

    if st.session_state["admin_logged_in"]:
        show_dashboard()
    else:
        st.write("This section is only for the college administration.")
        with st.form("admin_login_form"):
            password = st.text_input("Enter admin password", type="password")
            login_clicked = st.form_submit_button("Login")

        if login_clicked:
            if password == ADMIN_PASSWORD:
                st.session_state["admin_logged_in"] = True
                st.rerun()
            else:
                st.error("Incorrect password. Access denied.")