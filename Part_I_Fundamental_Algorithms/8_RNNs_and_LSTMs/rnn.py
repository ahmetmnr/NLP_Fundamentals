import numpy as np

class SimpleRNN:
    def __init__(self, input_size, hidden_size, output_size):
        """
        Initializes the Simple RNN with random weights for input, hidden, and output layers.
        Basit RNN'i giriş, gizli ve çıktı katmanları için rastgele ağırlıklarla başlatır.
        
        :param input_size: Number of input features (Girdi özelliklerinin sayısı)
        :param hidden_size: Number of hidden units (Gizli katmandaki nöronların sayısı)
        :param output_size: Number of output units (Çıktı birimlerinin sayısı)
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Initialize weights for input to hidden layer (Girdi katmanından gizli katmana ağırlıklar)
        self.W_xh = np.random.randn(input_size, hidden_size) * 0.01

        # Initialize weights for hidden to hidden layer (Gizli katmandan gizli katmana ağırlıklar)
        self.W_hh = np.random.randn(hidden_size, hidden_size) * 0.01

        # Initialize weights for hidden to output layer (Gizli katmandan çıktı katmanına ağırlıklar)
        self.W_hy = np.random.randn(hidden_size, output_size) * 0.01

        # Initialize biases (Bias'ları başlat)
        self.b_h = np.zeros((1, hidden_size))
        self.b_y = np.zeros((1, output_size))

        # Initialize hidden state to zeros (Başlangıç gizli durumu sıfırlarla başlat)
        self.h = np.zeros((1, hidden_size))
    
    def sigmoid(self, z):
        """
        Sigmoid activation function.
        Sigmoid aktivasyon fonksiyonu.
        
        :param z: Weighted sum of inputs (Girdi değerlerinin ağırlıklı toplamı)
        :return: Sigmoid activation output (Sigmoid fonksiyonunun çıktısı)
        """
        return 1 / (1 + np.exp(-z))
    
    def sigmoid_derivative(self, z):
        """
        Derivative of the sigmoid function.
        Sigmoid fonksiyonunun türevi.
        
        :param z: Output of sigmoid function (Sigmoid fonksiyonunun çıktısı)
        :return: Derivative of sigmoid (Sigmoid fonksiyonunun türevi)
        """
        return z * (1 - z)
    
    def softmax(self, z):
        """
        Softmax activation function for output layer.
        Çıktı katmanı için softmax aktivasyon fonksiyonu.
        
        :param z: Weighted sum of inputs (Girdi değerlerinin ağırlıklı toplamı)
        :return: Softmax output (Softmax fonksiyonunun çıktısı)
        """
        exp_z = np.exp(z - np.max(z))  # Stability trick for large numbers
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def forward(self, x):
        """
        Forward pass through the RNN.
        RNN üzerinden ileri besleme işlemi.
        
        :param x: Input data (Girdi verisi)
        :return: Output from the RNN (RNN'den çıkan çıktı)
        """
        # Compute the new hidden state (Yeni gizli durumu hesapla)
        self.h = np.tanh(np.dot(x, self.W_xh) + np.dot(self.h, self.W_hh) + self.b_h)

        # Compute the output (Çıktıyı hesapla)
        y = self.softmax(np.dot(self.h, self.W_hy) + self.b_y)
        
        return y

    def backward(self, x, y_true, y_pred, learning_rate=0.01):
        """
        Backpropagation through time (BPTT) for RNN.
        Zaman içerisinde geri yayılım (BPTT) işlemi.
        
        :param x: Input data (Girdi verisi)
        :param y_true: True labels (Gerçek etiketler)
        :param y_pred: Predicted output from the forward pass (İleri beslemedeki tahmin edilen çıktı)
        :param learning_rate: Learning rate for updating the weights (Ağırlıkları güncellemek için öğrenme oranı)
        """
        # Output layer error (Çıktı katmanı hatası)
        output_error = y_pred - y_true

        # Gradient for weights from hidden to output (Gizli katmandan çıktı katmanına ağırlıklar için gradyan)
        dW_hy = np.dot(self.h.T, output_error)
        db_y = np.sum(output_error, axis=0, keepdims=True)

        # Backpropagation through the hidden layer (Gizli katman üzerinden geri yayılım)
        hidden_error = np.dot(output_error, self.W_hy.T) * (1 - self.h ** 2)  # Tanh derivative

        # Gradient for weights from input to hidden and hidden to hidden (Girdi ve gizli katman için gradyanlar)
        dW_xh = np.dot(x.T, hidden_error)
        dW_hh = np.dot(self.h.T, hidden_error)
        db_h = np.sum(hidden_error, axis=0, keepdims=True)

        # Update weights and biases (Ağırlıklar ve bias değerlerini güncelle)
        self.W_hy -= learning_rate * dW_hy
        self.b_y -= learning_rate * db_y
        self.W_xh -= learning_rate * dW_xh
        self.W_hh -= learning_rate * dW_hh
        self.b_h -= learning_rate * db_h

    def train(self, X, y, epochs=1000, learning_rate=0.01):
        """
        Trains the RNN using forward and backward propagation.
        İleri ve geri yayılım kullanarak RNN'i eğitir.
        
        :param X: Input data (Girdi verisi)
        :param y: True labels (Gerçek etiketler)
        :param epochs: Number of training iterations (Eğitim iterasyonlarının sayısı)
        :param learning_rate: Learning rate for weight updates (Ağırlık güncellemeleri için öğrenme oranı)
        """
        for epoch in range(epochs):
            # Forward pass (İleri besleme)
            y_pred = self.forward(X)

            # Compute loss (Kayıp hesapla)
            loss = -np.sum(y * np.log(y_pred + 1e-8))  # Cross-entropy loss

            # Backward pass (Geri yayılım)
            self.backward(X, y, y_pred, learning_rate)

            # Print the loss every 100 epochs (Her 100 epoch'ta kaybı yazdır)
            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")

# Example usage (Örnek kullanım)
if __name__ == "__main__":
    # Sample input and output for XOR problem
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Input sequences (Girdi dizileri)
    y = np.array([[1, 0], [0, 1], [0, 1], [1, 0]])  # True labels for XOR (XOR için gerçek etiketler)

    # Initialize the RNN
    input_size = 2  # Girdi boyutu (örneğin XOR problemi için 2)
    hidden_size = 5  # Gizli katmandaki nöron sayısı
    output_size = 2  # Çıktı boyutu (örneğin iki sınıf)

    rnn = SimpleRNN(input_size, hidden_size, output_size)

    # Train the RNN
    rnn.train(X, y, epochs=1000, learning_rate=0.01)

    # Test the trained RNN
    for input_data in X:
        output = rnn.forward(input_data)
        print(f"Input: {input_data}, Predicted Output: {output}")
