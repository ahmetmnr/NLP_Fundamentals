import numpy as np
import pickle

EPSILON = 1e-8  # Numerical stability constant

# Optimizers
class SGDOptimizer:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def update(self, param, grad):
        return param - self.learning_rate * grad

class MomentumOptimizer:
    def __init__(self, learning_rate, momentum=0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity = {}

    def update(self, param, grad, param_name):
        if param_name not in self.velocity:
            self.velocity[param_name] = np.zeros_like(grad)
        self.velocity[param_name] = self.momentum * self.velocity[param_name] - self.learning_rate * grad
        return param + self.velocity[param_name]

class AdamOptimizer:
    def __init__(self, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}
        self.v = {}
        self.t = 0

    def update(self, param, grad, param_name):
        if param_name not in self.m:
            self.m[param_name] = np.zeros_like(grad)
            self.v[param_name] = np.zeros_like(grad)

        self.t += 1
        self.m[param_name] = self.beta1 * self.m[param_name] + (1 - self.beta1) * grad
        self.v[param_name] = self.beta2 * self.v[param_name] + (1 - self.beta2) * (grad ** 2)

        m_hat = self.m[param_name] / (1 - self.beta1 ** self.t)
        v_hat = self.v[param_name] / (1 - self.beta2 ** self.t)

        return param - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

# Softmax function
def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))  # Numeric stability
    return e_x / e_x.sum(axis=-1, keepdims=True)

# Simple RNN class
class SimpleRNN:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Initialize weights
        self.W_xh = np.random.randn(input_size, hidden_size) * 0.01  # Input to hidden
        self.W_hh = np.random.randn(hidden_size, hidden_size) * 0.01  # Hidden to hidden
        self.b_h = np.zeros((1, hidden_size))

        # Initialize gradients for backpropagation
        self.dW_xh = np.zeros_like(self.W_xh)
        self.dW_hh = np.zeros_like(self.W_hh)
        self.db_h = np.zeros_like(self.b_h)

    def forward(self, x):
        batch_size, timesteps, _ = x.shape
        h = np.zeros((batch_size, self.hidden_size))
        hidden_states = np.zeros((batch_size, timesteps, self.hidden_size))

        for t in range(timesteps):
            x_t = x[:, t, :]
            h = np.tanh(np.dot(x_t, self.W_xh) + np.dot(h, self.W_hh) + self.b_h)
            hidden_states[:, t, :] = h
            print(f"Forward step {t}, h: {h}")

        self.hidden_states = hidden_states  # Store hidden states for backward pass
        return hidden_states  # Shape: (batch_size, timesteps, hidden_size)

    def reset_gradients(self):
        self.dW_xh.fill(0)
        self.dW_hh.fill(0)
        self.db_h.fill(0)
        print("Gradients reset.")

    def backward(self, x, dh_next):
        batch_size, timesteps, _ = x.shape

        # Reset gradients
        self.reset_gradients()

        dh = dh_next  # Gradient from the output layer

        for t in reversed(range(timesteps)):
            x_t = x[:, t, :]
            h_t = self.hidden_states[:, t, :]
            h_prev = self.hidden_states[:, t - 1, :] if t > 0 else np.zeros((batch_size, self.hidden_size))

            # Derivative of tanh activation
            dtanh = (1 - h_t ** 2) * dh[:, 0, :]

            # Apply gradient clipping
            dtanh = np.clip(dtanh, -1, 1)

            # Gradients for weights and biases
            self.dW_xh += np.dot(x_t.T, dtanh) / batch_size
            self.dW_hh += np.dot(h_prev.T, dtanh) / batch_size
            self.db_h += np.sum(dtanh, axis=0, keepdims=True) / batch_size

            print(f"Backward step {t}, dW_xh: {self.dW_xh}, dW_hh: {self.dW_hh}, db_h: {self.db_h}")

            # Pass the gradient to the previous time step
            dh = np.dot(dtanh, self.W_hh.T).reshape(batch_size, 1, -1)
            if np.all(np.abs(dh) < 1e-6):
                print("Gradient has become too small, stopping backpropagation.")
                break

    def update_parameters(self, optimizer):
        self.W_xh = optimizer.update(self.W_xh, self.dW_xh, "W_xh")
        self.W_hh = optimizer.update(self.W_hh, self.dW_hh, "W_hh")
        self.b_h = optimizer.update(self.b_h, self.db_h, "b_h")
        print(f"Updated parameters, W_xh: {self.W_xh}, W_hh: {self.W_hh}, b_h: {self.b_h}")

    def save_model(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
        print(f"Model saved to {filename}")

    @staticmethod
    def load_model(filename):
        with open(filename, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded from {filename}")
        return model

# Extended SimpleRNN class with output layer
class SimpleRNNWithOutput(SimpleRNN):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__(input_size, hidden_size)
        # Output layer weights and biases
        self.W_out = np.random.randn(hidden_size, output_size) * 0.01
        self.b_out = np.zeros((1, output_size))

    def compute_output(self, h):
        return softmax(np.dot(h, self.W_out) + self.b_out)

    def backward_with_output(self, x, y, optimizer):
        batch_size, timesteps, _ = x.shape
        h = self.forward(x)  # Forward pass
        last_hidden_state = h[:, -1, :]

        # Compute output gradients
        y_pred = self.compute_output(last_hidden_state)
        grad_output = y_pred - y

        dW_out = np.dot(last_hidden_state.T, grad_output) / batch_size
        db_out = np.sum(grad_output, axis=0, keepdims=True) / batch_size

        dh_next = np.dot(grad_output, self.W_out.T).reshape(batch_size, 1, -1)
        self.backward(x, dh_next)  # Backpropagate through time

        # Update output layer weights and biases
        self.W_out = optimizer.update(self.W_out, dW_out, "W_out")
        self.b_out = optimizer.update(self.b_out, db_out, "b_out")

        # Update hidden layer weights and biases
        self.update_parameters(optimizer)

    def calculate_loss(self, y_pred, y):
        return -np.sum(y * np.log(y_pred + EPSILON)) / y.shape[0]

# Training and Testing functions
def train(rnn, X, y, epochs, optimizer):
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}")
        h = rnn.forward(X)
        last_hidden_state = h[:, -1, :]

        y_pred = rnn.compute_output(last_hidden_state)
        loss = rnn.calculate_loss(y_pred, y)
        print(f"Loss: {loss}")

        rnn.backward_with_output(X, y, optimizer)

        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")

def test(rnn, X):
    for input_data in X:
        h = rnn.forward(input_data.reshape(1, 1, -1))
        last_hidden_state = h[:, -1, :]
        output = rnn.compute_output(last_hidden_state)
        print(f"Input: {input_data.flatten()}, Predicted Output: {output}")

# Example usage
if __name__ == "__main__":
    X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])
    y = np.array([[1, 0],
                  [0, 1],
                  [0, 1],
                  [1, 0]])

    X = X.reshape(X.shape[0], 1, X.shape[1])

    input_size = 2
    hidden_size = 5
    output_size = 2

    rnn = SimpleRNNWithOutput(input_size, hidden_size, output_size)

    epochs = 1000
    learning_rate = 0.01
    optimizer = AdamOptimizer(learning_rate)

    train(rnn, X, y, epochs, optimizer)

    print("\nTesting the model:")
    test(rnn, X)