from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
from nltk.tokenize import RegexpTokenizer
import re

# Load the model
model = load_model("next_word_prediction_model.h5")

# Load your token mappings (unique_token_index and reverse_token_index)
# These should be saved from your original code to map words to indices and vice versa
# Here we assume you saved 'unique_token_index' and 'reverse_token_index' in a separate pickle file

# You should load these from a saved file (pickled or similar). Below is just an example:
with open("unique_token_index.pkl", "rb") as f:
    unique_token_index = pickle.load(f)

reverse_token_index = {index: token for token, index in unique_token_index.items()}

# Initialize Flask app
app = Flask(__name__)

# Function to generate text
def generate_text(model, seed_text, num_words=10):
    tokenizer = RegexpTokenizer(r"\w+")
    input_sequence = tokenizer.tokenize(seed_text.lower())
    input_indices = [unique_token_index[word] for word in input_sequence if word in unique_token_index]

    for _ in range(num_words):
        padded_input = np.pad(input_indices, (n_words - len(input_indices), 0), 'constant')
        padded_input = padded_input.reshape(1, n_words)

        pred_probs = model.predict(padded_input)
        pred_index = np.argmax(pred_probs)
        next_word = reverse_token_index[pred_index]

        input_indices.append(pred_index)
        input_indices = input_indices[1:]

        seed_text += " " + next_word

    return seed_text

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    # Get the user input from the form
    seed_text = request.form['seed_text']

    # Generate the text using the model
    generated_text = generate_text(model, seed_text)

    return render_template('index.html', generated_text=generated_text, seed_text=seed_text)

if __name__ == '__main__':
    app.run(debug=True)
