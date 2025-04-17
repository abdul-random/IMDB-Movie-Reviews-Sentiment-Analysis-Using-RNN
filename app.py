# # Import Liberary
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import pad_sequences 
from tensorflow.keras.datasets import imdb 
import streamlit as st

# Load the model
model = load_model("rnn_model.h5")

# Custom Functions need to be run  
def pre_process_text(text):
    # Get the word and corresponding index
    word_index = imdb.get_word_index()
    # Get the index of each work  
    sent_list = text.lower().split()
    encoded_sent_list = [word_index.get(word,2) + 3 for word in sent_list]
    return pad_sequences([encoded_sent_list], maxlen=500)

def predict_sentiment(review):
    encoded_text = pre_process_text(review)
    predict_proba = model.predict(encoded_text)[0][0]
    sentiment ="Positive" if predict_proba > 0.5 else "Negative"  
    return sentiment, predict_proba  

# Streamlit app
st.title("IMDB Movie Review Sentiment Analysis")
st.write("Please enter the movie review to determine if it is positive or negative.")

user_input = st.text_area("Movie Review")

if st.button("Classify"):
    sentiment,predict_proba = predict_sentiment(user_input)
    st.write("Sentiment: ", sentiment)
    st.write("Predict Score: ", predict_proba)
else:
    st.write("Please enter the movie review")
