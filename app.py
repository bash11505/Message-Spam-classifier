
import streamlit as st
import pickle

model = pickle.load(open("models/spam_model.pkl", "rb"))
tfidf = pickle.load(open("models/tfidf_vectorizer.pkl", "rb"))

st.title("SMS Spam Classifier")

st.write("Enter a message to check whether it is spam or not.")

message = st.text_area("Enter your SMS message")

if st.button("Predict"):
    if message.strip() == "":
        st.warning("Please enter a message.")
    else:
        message_vector = tfidf.transform([message])
        prediction = model.predict(message_vector)[0]

        if prediction == 1:
            st.error("This message is Spam.")
        else:
            st.success("This message is Not Spam.")
