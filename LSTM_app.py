import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
import pickle 
from tensorflow.keras.utils import pad_sequences
#load file
model=load_model("LSTM.keras")
#load pickle file
with open("tokenizer.pkl", "rb") as f:
    tokenizer=pickle.load(f)
def predict_next_word(model, tokenizer, text, max_length):
    # Text to sequence
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list)>= max_lenght:

    # Truncate last max_length - 1 tokens
     token_list = token_list[-(max_length - 1):]

    # Pad the sequence
    token_list = pad_sequences([token_list], maxlen=max_length - 1, padding='pre')

    # Predict the probabilities of next word
    prediction = model.predict(token_list, verbose=0)

    # Get the index of highest probability
    predicted_index = np.argmax(prediction, axis=1)[0]

    # Map index back to word
    for word, index in tokenizer.word_index.items():
        if index == predicted_index:
            return word
    return None
st.title("Next Word Prediction")
user_input=st.text_input("Enter the Sequence of Word","Looke with what courteous ")
if st.button("predict Next Word"):
   max_lenght=model.input_shape[1] +1
   next_word=predict_next_word(model,tokenizer,user_input,max_lenght)
   st.write(f"user input {user_input}")
   st.write(f"Next word  {next_word}")

