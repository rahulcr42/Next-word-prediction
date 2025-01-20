import numpy as np
from tensorflow.keras.models import load_model
from nltk.tokenize import RegexpTokenizer

# Load the trained model
model = load_model("pride_text_gen_model.h5")

# Load the Pride and Prejudice text file (replace with your file path)
with open("data.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Tokenize the text
tokenizer = RegexpTokenizer(r"\w+")
tokens = tokenizer.tokenize(text.lower())

# Create a mapping of unique tokens to indices (same as training)
unique_tokens = np.unique(tokens)
unique_token_index = {token: index for index, token in enumerate(unique_tokens)}

# Generate next word function
def predict_next_word(input_text, n_words=10):
    input_text = input_text.lower()
    input_words = input_text.split()
    if len(input_words) < n_words:
        raise ValueError("Input text must have at least 10 words.")
    
    # Prepare the input sequence
    X = np.zeros((1, n_words), dtype=int)
    for i, word in enumerate(input_words[-n_words:]):
        X[0, i] = unique_token_index.get(word, 0)  # Default to 0 if word is not found
    
    # Predict the next word
    prediction = model.predict(X)
    predicted_index = np.argmax(prediction)
    predicted_word = unique_tokens[predicted_index]
    
    return predicted_word

# Example usage
input_text = "it is a truth universally acknowledged"
next_word = predict_next_word(input_text)
print(f"Next word prediction for '{input_text}': {next_word}")
