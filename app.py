import streamlit as st
import requests
from transformers import pipeline

# Initialize the QA model
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

# Fetch top headlines (requires your NewsAPI key)
def get_news(api_key):
    url = f"https://newsapi.org/v2/top-headlines?language=en&category=general&pageSize=5&apiKey={d66a96ac55834aff916182230091ba73}"
    response = requests.get(url)
    articles = response.json().get("articles", [])
    return " ".join([article["title"] + ". " + article.get("description", "") for article in articles])

st.title("🗞️ World News Q&A Assistant")

news_api_key = st.text_input("🔑 Enter your NewsAPI Key", type="password")

if news_api_key:
    with st.spinner("Fetching latest news..."):
        context = get_news(news_api_key)

    st.success("News loaded successfully.")
    st.subheader("Ask a question about the latest news:")
    question = st.text_input("🧠 Your Question")

    if question:
        with st.spinner("Thinking..."):
            result = qa_pipeline(question=question, context=context)
        st.markdown(f"**Answer:** {result['answer']}")
else:
    st.info("Please enter your NewsAPI key to start.")
