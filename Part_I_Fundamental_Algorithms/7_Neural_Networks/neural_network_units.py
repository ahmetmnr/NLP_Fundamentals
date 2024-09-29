import numpy as np

class NeuralNetworkUnit:
    def __init__(self, input_size, output_size):
        """
        Initializes the neural network unit with random weights and bias.
        Rastgele ağırlıklar ve bias ile neural network unit'i başlatır.

        :param input_size: Number of inputs to the neuron (Nöronun girdi sayısı)
        :param output_size: Number of outputs from the neuron (Nöronun çıktı sayısı)
        """
        # Rastgele ağırlıklar ve bias başlat
        self.weights = np.random.randn(input_size, output_size)
        self.bias = np.random.randn(output_size)
    
    def sigmoid(self, z):
        """
        Sigmoid activation function.
        Sigmoid aktivasyon fonksiyonu.

        :param z: Weighted sum of inputs (Girdi değerlerinin ağırlıklı toplamı)
        :return: Output of the sigmoid function (Sigmoid fonksiyonunun çıktısı)
        """
        return 1 / (1 + np.exp(-z))
    
    def relu(self, z):
        """
        ReLU activation function.
        ReLU aktivasyon fonksiyonu.

        :param z: Weighted sum of inputs (Girdi değerlerinin ağırlıklı toplamı)
        :return: Output of the ReLU function (ReLU fonksiyonunun çıktısı)
        """
        return np.maximum(0, z)
    
    def forward(self, inputs, activation_function='sigmoid'):
        """
        Forward pass through the neuron with a given activation function.
        Verilen aktivasyon fonksiyonu ile nörondan forward geçişi gerçekleştirir.

        :param inputs: Input data (Girdi verileri)
        :param activation_function: Activation function to use ('sigmoid' or 'relu') (Kullanılacak aktivasyon fonksiyonu)
        :return: Output from the neuron (Nöronun çıktısı)
        """
        # Ağırlıklı toplam: z = W * x + b
        z = np.dot(inputs, self.weights) + self.bias
        
        # Aktivasyon fonksiyonunu seç
        if activation_function == 'sigmoid':
            return self.sigmoid(z)
        elif activation_function == 'relu':
            return self.relu(z)
        else:
            raise ValueError(f"Unsupported activation function: {activation_function}")
    
    def __repr__(self):
        return f"Weights: {self.weights}, Bias: {self.bias}"


# Example usage (Örnek kullanım)
if __name__ == "__main__":
    # Girdi ve çıktı sayısını belirle (Örneğin 3 giriş, 2 çıkış)
    input_size = 3
    output_size = 2
    
    # NeuralNetworkUnit sınıfını başlat
    nn_unit = NeuralNetworkUnit(input_size, output_size)
    
    # Giriş verisi oluştur
    inputs = np.array([0.5, 0.2, 0.1])
    
    # Sigmoid aktivasyon fonksiyonu ile çıktıyı hesapla
    output_sigmoid = nn_unit.forward(inputs, activation_function='sigmoid')
    print(f"Output (Sigmoid): {output_sigmoid}")
    
    # ReLU aktivasyon fonksiyonu ile çıktıyı hesapla
    output_relu = nn_unit.forward(inputs, activation_function='relu')
    print(f"Output (ReLU): {output_relu}")
    
    # Ağırlıkları ve bias'ı yazdır
    print(nn_unit)
