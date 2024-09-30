import numpy as np
from rnn import SimpleRNN  # Using the SimpleRNN class from rnn.py
# LSTMModel'ı kullanmayacağımız için import etmiyoruz

def one_hot_encode(y, num_classes):
    """
    Converts class indices to one-hot encoded vectors.
    """
    return np.eye(num_classes)[y]

class RNNLanguageModel:
    def __init__(self, input_size, hidden_size, output_size, model_type='rnn'):
        """
        Initializes the language model using either RNN or LSTM.
        """
        self.output_size = output_size  # Output size is needed for one-hot encoding

        if model_type == 'rnn':
            self.model = SimpleRNN(input_size, hidden_size)
        elif model_type == 'lstm':
            # self.model = LSTMModel(input_size, hidden_size)  # LSTM kullanmıyoruz
            pass
        else:
            raise ValueError("Model type must be 'rnn' or 'lstm'")

        # Output layer weights and biases
        self.W_out = np.random.randn(hidden_size, output_size) * 0.01
        self.b_out = np.zeros((1, output_size))

    def train(self, X, y, epochs=1000, learning_rate=0.01):
        """
        Trains the language model.
        """
        # One-hot encode the labels
        y_one_hot = one_hot_encode(y, self.output_size)

        for epoch in range(epochs):
            # Forward pass
            h = self.model.forward(X)  # h: (batch_size, timesteps, hidden_size)
            last_hidden_state = h[:, -1, :]  # Use the last hidden state

            # Compute outputs
            y_pred = np.dot(last_hidden_state, self.W_out) + self.b_out  # (batch_size, output_size)
            y_pred = self.softmax(y_pred)

            # Compute loss
            loss = -np.sum(y_one_hot * np.log(y_pred + 1e-8)) / y.shape[0]

            # Compute gradients
            grad_output = y_pred - y_one_hot  # (batch_size, output_size)
            dW_out = np.dot(last_hidden_state.T, grad_output) / y.shape[0]
            db_out = np.sum(grad_output, axis=0, keepdims=True) / y.shape[0]

            # Update weights and biases
            self.W_out -= learning_rate * dW_out
            self.b_out -= learning_rate * db_out

            # Note: Backpropagation through time (BPTT) is not implemented for the RNN weights

            # Print loss every 100 epochs
            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")

    def softmax(self, x):
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / e_x.sum(axis=-1, keepdims=True)

    def predict_next_word(self, input_sequence):
        """
        Predicts the next word in the sequence.
        """
        h = self.model.forward(input_sequence)  # h: (1, timesteps, hidden_size)
        last_hidden_state = h[:, -1, :]  # Use the last hidden state

        # Compute output
        y_pred = np.dot(last_hidden_state, self.W_out) + self.b_out  # (1, output_size)
        y_pred = self.softmax(y_pred)

        predicted_word_index = np.argmax(y_pred, axis=-1)[0]

        # Ensure the predicted word index is within vocabulary bounds
        if predicted_word_index >= self.output_size:
            predicted_word_index = np.random.randint(self.output_size)  # Random valid index as fallback

        return predicted_word_index

    def generate_text(self, seed_text, length, word_to_index, index_to_word, embedding_matrix):
        """
        Generates text using the trained language model.
        """
        generated_text = seed_text.copy()
        input_sequence = np.array([word_to_index[word] for word in seed_text]).reshape(1, -1)

        for _ in range(length):
            # Embedding the input sequence
            input_embedded = np.array([embedding_matrix[int(word_index)] for word_index in input_sequence[0]]).reshape(1, -1, embedding_matrix.shape[1])
            
            # Predict the next word index
            next_word_index = self.predict_next_word(input_embedded)

            # Check if the predicted index is within vocabulary size
            if next_word_index not in index_to_word:
                next_word = "<UNK>"  # Handle out-of-vocabulary words
            else:
                next_word = index_to_word[next_word_index]

            generated_text.append(next_word)

            # Update the input sequence by adding the new word
            input_sequence = np.append(input_sequence, [[next_word_index]], axis=1)

            # Optionally, you can keep the input sequence length fixed by removing the oldest word
            # input_sequence = input_sequence[:, 1:]

        return ' '.join(generated_text)

# Example training process with a sample dataset
if __name__ == "__main__":
    # Simple dataset: We will work with indices of words
    sentences = [
        "the cat is on the mat".split(),
        "the dog is in the house".split(),
        "the cat and the dog are friends".split(),
        "the mat is near the door".split()
    ]

    # Creating word-to-index and index-to-word mappings
    words = sorted(set(word for sentence in sentences for word in sentence))
    word_to_index = {word: i for i, word in enumerate(words)}
    index_to_word = {i: word for word, i in word_to_index.items()}

    # Preparing input and output data
    X = []
    y = []

    for sentence in sentences:
        for i in range(1, len(sentence)):
            input_sequence = [word_to_index[word] for word in sentence[:i]]  # Input sequence
            next_word = word_to_index[sentence[i]]  # Next word (output)
            X.append(input_sequence)
            y.append(next_word)

    # Convert X and y to numpy arrays and pad the input sequences with zeros
    max_sequence_length = max([len(seq) for seq in X])
    X_padded = np.zeros((len(X), max_sequence_length))
    
    for i, seq in enumerate(X):
        X_padded[i, -len(seq):] = seq

    X = X_padded
    y = np.array(y)

    # Define word embedding size
    embedding_size = 10

    # Create random embedding matrix
    embedding_matrix = np.random.randn(len(word_to_index), embedding_size)

    # Prepare input sequences using the embedding matrix
    X_embedded = np.zeros((X.shape[0], max_sequence_length, embedding_size))

    for i, sequence in enumerate(X):
        for t, word_index in enumerate(sequence):
            if word_index != 0:
                X_embedded[i, t, :] = embedding_matrix[int(word_index)]  # Replace word index with embedding

    # Initialize the model
    input_size = embedding_size  # Embedding size as input feature size
    hidden_size = 10  # Number of neurons in the hidden layer
    output_size = len(word_to_index)  # Vocabulary size

    rnn_language_model = RNNLanguageModel(input_size, hidden_size, output_size, model_type='rnn')

    # Train the model
    rnn_language_model.train(X_embedded, y, epochs=1000, learning_rate=0.01)

    # Generate text
    seed_text = ["the", "cat","dog","is","are","near","friends"]
    generated_text = rnn_language_model.generate_text(seed_text, length=5, word_to_index=word_to_index, index_to_word=index_to_word, embedding_matrix=embedding_matrix)
    print("Generated text:")
    print(generated_text)
