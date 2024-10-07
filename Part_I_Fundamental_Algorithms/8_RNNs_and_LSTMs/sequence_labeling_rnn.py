import numpy as np
from rnn import SimpleRNN  # Import the RNN model from rnn.py
from sklearn.preprocessing import StandardScaler

# Function to one-hot encode labels
def one_hot_encode(labels, num_classes):
    """
    Converts class indices to one-hot encoded vectors.
    Sınıf indekslerini one-hot encoded vektörlere dönüştürür.
    """
    return np.eye(num_classes)[labels]

# SequenceLabelingRNN class using SimpleRNN
class SequenceLabelingRNN:
    def __init__(self, input_size, hidden_size, output_size):
        """
        Initializes the sequence labeling RNN using the SimpleRNN class.
        """
        # Use the SimpleRNN class for sequence modeling
        self.rnn = SimpleRNN(input_size, hidden_size)

        # Output layer weights and biases
        self.W_out = np.random.randn(hidden_size, output_size) * 0.01
        self.b_out = np.zeros((1, output_size))

    def forward(self, X):
        """
        Forward pass through the model.
        """
        h = self.rnn.forward(X)  # h: (batch_size, timesteps, hidden_size)
        batch_size, timesteps, _ = h.shape

        h_reshaped = h.reshape(batch_size * timesteps, -1)
        y_pred = np.dot(h_reshaped, self.W_out) + self.b_out  # Shape: (batch_size * timesteps, output_size)
        y_pred = y_pred.reshape(batch_size, timesteps, -1)

        return y_pred

    def compute_loss(self, y_true, y_pred):
        y_pred = self.softmax(y_pred)
        loss = -np.sum(y_true * np.log(y_pred + 1e-8)) / y_true.shape[0]
        return loss

    def softmax(self, x):
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / e_x.sum(axis=-1, keepdims=True)

    def backward(self, X, y_true, y_pred, learning_rate):
        # Gradient of loss w.r.t output
        y_pred = self.softmax(y_pred)
        grad_output = y_pred - y_true  # Shape: (batch_size, timesteps, output_size)
        batch_size, timesteps, output_size = grad_output.shape

        grad_output_reshaped = grad_output.reshape(batch_size * timesteps, output_size)

        # Get hidden states
        h = self.rnn.hidden_states  # Shape: (batch_size, timesteps, hidden_size)
        h_reshaped = h.reshape(batch_size * timesteps, -1)  # Shape: (batch_size * timesteps, hidden_size)

        # Compute gradients for output layer
        dW_out = np.dot(h_reshaped.T, grad_output_reshaped)  # Shape: (hidden_size, output_size)
        db_out = np.sum(grad_output_reshaped, axis=0, keepdims=True)  # Shape: (1, output_size)

        # Update weights and biases
        self.W_out -= learning_rate * dW_out
        self.b_out -= learning_rate * db_out

        # Backpropagation through time (BPTT) for the RNN part
        dh_next = grad_output_reshaped.dot(self.W_out.T).reshape(batch_size, timesteps, -1)
        self.rnn.backward(X, dh_next)

    def train(self, X, y, epochs=1000, learning_rate=0.1):
        for epoch in range(epochs):
            y_pred = self.forward(X)
            loss = self.compute_loss(y, y_pred)
            self.backward(X, y, y_pred, learning_rate)
            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")

    def predict(self, X):
        y_pred = self.forward(X)
        y_pred = self.softmax(y_pred)
        return np.argmax(y_pred, axis=-1)

# Örnek kullanım
if __name__ == "__main__":
    # Sample dataset for sequence labeling
    sentences = [
        ["the", "dog", "barks"],
        ["the", "cat", "meows"],
        ["the", "bird", "chirps"]
    ]

    labels = [
        [0, 1, 2],  # Labels for "the dog barks"
        [0, 1, 2],
        [0, 1, 2]
    ]

    word_to_index = {"the": 0, "dog": 1, "cat": 2, "bird": 3, "barks": 4, "meows": 5, "chirps": 6}
    index_to_label = {0: "DET", 1: "NOUN", 2: "VERB"}

    # Convert sentences and labels to indices
    X = np.array([[word_to_index[word] for word in sentence] for sentence in sentences])
    y = np.array(labels)

    # Convert y to one-hot encoding
    num_classes = len(index_to_label)
    y_one_hot = np.eye(num_classes)[y]  # Shape: (batch_size, timesteps, num_classes)

    # Reshape X to 3D (batch_size, timesteps, input_size)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # Normalize inputs using StandardScaler
    scaler = StandardScaler()
    X = X.reshape(-1, 1)  # Flatten for scaler
    X = scaler.fit_transform(X).reshape(-1, 3, 1)  # Reshape back to original shape

    # Define hyperparameters
    input_size = 1
    hidden_size = 10
    output_size = num_classes

    # Initialize the sequence labeling RNN
    rnn_model = SequenceLabelingRNN(input_size, hidden_size, output_size)

    # Train the model
    rnn_model.train(X, y_one_hot, epochs=1000, learning_rate=0.1)

    # Predict labels for a new sentence
    test_sentence = np.array([[word_to_index["the"], word_to_index["cat"], word_to_index["meows"]]])
    test_sentence = test_sentence.reshape((test_sentence.shape[0], test_sentence.shape[1], 1))
    test_sentence = scaler.transform(test_sentence.reshape(-1, 1)).reshape(-1, 3, 1)
    predicted_labels = rnn_model.predict(test_sentence)

    # Map predicted labels back to human-readable form
    predicted_label_names = [index_to_label[label] for label in predicted_labels[0]]
    print(f"Predicted labels for 'the cat meows': {predicted_label_names}")