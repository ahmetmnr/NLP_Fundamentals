import numpy as np

class FeedforwardNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        """
        Initializes the feedforward neural network with one hidden layer.
        Tek gizli katmanlı feedforward sinir ağını başlatır.

        :param input_size: Number of input neurons (Girdi nöronlarının sayısı)
        :param hidden_size: Number of hidden neurons (Gizli katmandaki nöronların sayısı)
        :param output_size: Number of output neurons (Çıktı nöronlarının sayısı)
        """
        # Ağırlıkları rastgele başlat
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.weights_input_hidden = np.random.randn(input_size, hidden_size)
        self.weights_hidden_output = np.random.randn(hidden_size, output_size)
        
        self.bias_hidden = np.random.randn(hidden_size)
        self.bias_output = np.random.randn(output_size)
    
    def sigmoid(self, z):
        """
        Sigmoid activation function.
        Sigmoid aktivasyon fonksiyonu.

        :param z: Weighted sum of inputs (Girdi değerlerinin ağırlıklı toplamı)
        :return: Sigmoid function output (Sigmoid fonksiyonunun çıktısı)
        """
        return 1 / (1 + np.exp(-z))
    
    def sigmoid_derivative(self, z):
        """
        Derivative of the sigmoid function (backpropagation için).
        Sigmoid fonksiyonunun türevi (backpropagation için).

        :param z: Sigmoid fonksiyonunun çıktısı
        :return: Türev
        """
        return z * (1 - z)
    
    def forward(self, X):
        """
        Forward pass through the network.
        Sinir ağı üzerinden ileri besleme işlemi gerçekleştirir.

        :param X: Input data (Girdi verisi)
        :return: Output after forward pass (İleri besleme sonrası çıktı)
        """
        # Girdi ile gizli katman arasındaki forward pass
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = self.sigmoid(self.hidden_input)
        
        # Gizli katman ile çıktı katmanı arasındaki forward pass
        self.output_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.output_output = self.sigmoid(self.output_input)
        
        return self.output_output
    
    def backward(self, X, y, learning_rate):
        """
        Backpropagation algorithm to update weights and biases.
        Backpropagation algoritması ile ağırlık ve bias değerlerini günceller.

        :param X: Input data (Girdi verisi)
        :param y: True labels (Gerçek etiketler)
        :param learning_rate: Öğrenme oranı
        """
        # Çıktı hatası
        output_error = y - self.output_output
        output_delta = output_error * self.sigmoid_derivative(self.output_output)

        # Gizli katman hatası
        hidden_error = np.dot(output_delta, self.weights_hidden_output.T)
        hidden_delta = hidden_error * self.sigmoid_derivative(self.hidden_output)

        # Ağırlıkları güncelle
        self.weights_hidden_output += np.dot(self.hidden_output.T, output_delta) * learning_rate
        self.weights_input_hidden += np.dot(X.T, hidden_delta) * learning_rate

        # Bias'ları güncelle
        self.bias_output += np.sum(output_delta, axis=0) * learning_rate
        self.bias_hidden += np.sum(hidden_delta, axis=0) * learning_rate
    
    def train(self, X, y, epochs, learning_rate):
        """
        Trains the neural network using backpropagation.
        Backpropagation kullanarak sinir ağını eğitir.

        :param X: Input data (Girdi verisi)
        :param y: True labels (Gerçek etiketler)
        :param epochs: Training iterations (Eğitim iterasyonları)
        :param learning_rate: Öğrenme oranı
        """
        for epoch in range(epochs):
            # İleri besleme (forward pass)
            output = self.forward(X)

            # Geri yayılım (backpropagation)
            self.backward(X, y, learning_rate)

            # Eğitim ilerlemesini yazdır
            if (epoch + 1) % 100 == 0:
                loss = np.mean(np.square(y - output))  # Mean squared error (MSE)
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")


# Example usage (Örnek kullanım)
if __name__ == "__main__":
    # XOR problemini çözmek için verileri hazırlayalım
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Girdiler
    y = np.array([[0], [1], [1], [0]])  # Çıktılar (XOR)

    # Feedforward Neural Network'i başlat
    input_size = 2  # XOR problemi için 2 girdi
    hidden_size = 2  # 2 gizli katman nöronu
    output_size = 1  # Tek çıktı (XOR)

    neural_net = FeedforwardNeuralNetwork(input_size, hidden_size, output_size)

    # Sinir ağını eğit
    neural_net.train(X, y, epochs=1000, learning_rate=0.1)

    # Eğitilen sinir ağını test et
    print("Testing the neural network on XOR inputs:")
    for input_data in X:
        output = neural_net.forward(input_data)
        print(f"Input: {input_data}, Output: {output}")
