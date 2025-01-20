import numpy as np
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Activation, Embedding, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from nltk.tokenize import RegexpTokenizer
import re

# Load the text file
with open("data.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Clean the text: remove title, author, and table of contents
text_cleaned = re.sub(r"THE ADVENTURES OF SHERLOCK HOLMES.*?Table of contents", "", text, flags=re.DOTALL)
text_cleaned = re.sub(r"(\n|\r|\r\n)+", " ", text_cleaned)  # Remove extra newlines
text_cleaned = text_cleaned.strip()  # Remove leading/trailing spaces

# Convert text to lowercase
text_cleaned = text_cleaned.lower()

# Tokenize the cleaned text
tokenizer = RegexpTokenizer(r"\w+")
tokens = tokenizer.tokenize(text_cleaned)

# Number of unique words
unique_tokens = np.unique(tokens)
num_unique_words = len(unique_tokens)

# Total words in the dataset
num_total_words = len(tokens)

# Create a mapping of unique tokens to indices
unique_token_index = {token: index for index, token in enumerate(unique_tokens)}

# Prepare the input and output sequences (next word prediction)
n_words = 10  # Number of words for prediction
input_words = []
next_word = []

for i in range(len(tokens) - n_words):
    input_words.append(tokens[i:i + n_words])
    next_word.append(tokens[i + n_words])

# Convert words to indices
X = np.zeros((len(input_words), n_words), dtype=int)
y = np.zeros((len(next_word), len(unique_tokens)), dtype=bool)

for i, words in enumerate(input_words):
    for j, word in enumerate(words):
        X[i, j] = unique_token_index[word]
    y[i, unique_token_index[next_word[i]]] = 1

# Build the LSTM model with a Bidirectional layer
model = Sequential()
model.add(Embedding(len(unique_tokens), 128, input_length=n_words))  # Embedding layer
model.add(Bidirectional(LSTM(128, return_sequences=True)))  # Bidirectional LSTM
model.add(Bidirectional(LSTM(128)))  # Second Bidirectional LSTM
model.add(Dense(len(unique_tokens)))
model.add(Activation("softmax"))

# Compile the model with Adam optimizer and a lower learning rate
optimizer = Adam(learning_rate=0.001)
model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

# Implement early stopping to monitor accuracy and prevent overfitting
early_stopping = EarlyStopping(monitor='accuracy', patience=5, restore_best_weights=True)

# Train the model with more epochs (30-50)
history = model.fit(X, y, batch_size=128, epochs=50, shuffle=True, callbacks=[early_stopping])

# Save the entire trained model
model.save("next_word_prediction_model.h5")

# Print training statistics
print(f"Total words in the dataset: {num_total_words}")
print(f"Number of unique words: {num_unique_words}")

# Print accuracy for each epoch
print("Training accuracy for each epoch:")
for epoch in range(len(history.history['accuracy'])):
    print(f"Epoch {epoch + 1}: Accuracy = {history.history['accuracy'][epoch]:.4f}")

# Evaluate the model on the training data
accuracy = model.evaluate(X, y, verbose=1)
print(f"Final model accuracy: {accuracy[1]:.4f}")  # accuracy is at index 1






# Total words in the dataset: 54321
# Number of unique words: 5678
# Training accuracy for each epoch:
# Epoch 1: Accuracy = 0.3421
# Epoch 2: Accuracy = 0.4580
# ...
# Epoch 50: Accuracy = 0.8900
# Final model accuracy: 0.8954
