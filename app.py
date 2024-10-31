import streamlit as st
from transformers import BartTokenizer, BartForConditionalGeneration

# Load fine-tuned model
tokenizer = BartTokenizer.from_pretrained("./fine_tuned_bart")
model = BartForConditionalGeneration.from_pretrained("./fine_tuned_bart")

# Streamlit App Interface
st.title("Dialogue Summarization with BART")

# Input box for user to enter dialogue
input_text = st.text_area("Enter the dialogue here:")

if st.button("Summarize"):
    # Tokenize input and generate summary
    inputs = tokenizer([input_text], max_length=512, return_tensors="pt", truncation=True)
    summary_ids = model.generate(inputs["input_ids"], max_length=128, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)

    # Decode the summary
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    st.write("Summary:")
    st.write(summary)
