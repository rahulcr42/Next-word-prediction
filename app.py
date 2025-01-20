import numpy as np
import pickle
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from nltk.tokenize import RegexpTokenizer

# Initialize Flask app
app = Flask(__name__)

# Load necessary files
model = load_model('next_word_prediction_model.h5')
with open('unique_token_index.pkl', 'rb') as f:
    unique_token_index = pickle.load(f)
with open('index_unique_token.pkl', 'rb') as f:
    index_unique_token = pickle.load(f)

# Initialize tokenizer
tokenizer = RegexpTokenizer(r"\w+")
def predict_next_word(input_text):
    # Clean and tokenize input text
    input_text = input_text.lower()
    tokens = tokenizer.tokenize(input_text)

    # If the input is empty or less than 1 word, return a message
    if len(tokens) < 1:
        return ["Please enter at least one word."]

    # Prepare the input sequence (use the last n_words tokens for context)
    n_words = 10  # Number of words to consider for prediction
    if len(tokens) < n_words:
        n_words = len(tokens)  # If there are fewer than 10 words, use all available words
    
    input_sequence = tokens[-n_words:]
    X = np.zeros((1, n_words), dtype=int)
    for i, word in enumerate(input_sequence):
        if word in unique_token_index:
            X[0, i] = unique_token_index[word]

    # Predict the next word (top 3 predictions)
    prediction = model.predict(X)
    top_indices = prediction[0].argsort()[-3:][::-1]  # Get the top 3 predicted indices
    
    print("Top predicted indices:", top_indices)  # Debug print
    predicted_words = [index_unique_token[i] for i in top_indices]
    
    print("Predicted words:", predicted_words)  # Debug print
    
    return predicted_words


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    input_text = request.json["input_text"]
    predicted_words = predict_next_word(input_text)
    return jsonify(predicted_words=predicted_words)

if __name__ == "__main__":
    app.run(debug=True)
