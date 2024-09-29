import numpy as np

class MiniBatchLogisticRegression:
    def __init__(self, learning_rate=0.01, epochs=1000, batch_size=32):
        """
        Initializes the logistic regression model with mini-batch training.
        Mini-batch eğitimi ile logistic regression modelini başlatır.
        
        :param learning_rate: The step size for weight updates (Ağırlık güncellemeleri için adım büyüklüğü)
        :param epochs: The number of iterations to train the model (Modeli eğitmek için epoch sayısı)
        :param batch_size: The size of each mini-batch (Her mini-batch'in boyutu)
        """
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.weights = None  # Model ağırlıkları
        self.bias = None  # Model bias'ı

    def sigmoid(self, z):
        """
        Sigmoid function to calculate probabilities.
        Olasılıkları hesaplamak için sigmoid fonksiyonu.
        
        :param z: Input value (Girdi değeri)
        :return: Sigmoid of z (z'nin sigmoid değeri)
        """
        return 1 / (1 + np.exp(-z))

    def cross_entropy_loss(self, y_true, y_pred):
        """
        Cross-entropy loss function for binary classification.
        İkili sınıflandırma için cross-entropy kayıp fonksiyonu.
        
        :param y_true: True labels (Gerçek etiketler)
        :param y_pred: Predicted probabilities (Tahmin edilen olasılıklar)
        :return: Cross-entropy loss (Cross-entropy kayıp değeri)
        """
        n_samples = len(y_true)
        y_pred = np.clip(y_pred, 1e-9, 1 - 1e-9)  # Avoid log(0) error (log(0) hatalarını önlemek için)
        loss = -1 / n_samples * np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss

    def compute_gradients(self, X, y, y_predicted):
        """
        Computes gradients of the loss function with respect to weights and bias.
        Kayıp fonksiyonunun ağırlıklar ve bias'a göre gradyanlarını hesaplar.
        
        :param X: Feature matrix (Özellik matrisi)
        :param y: True labels (Gerçek etiketler)
        :param y_predicted: Predicted probabilities (Tahmin edilen olasılıklar)
        :return: Gradients of weights and bias (Ağırlıklar ve bias'ın gradyanları)
        """
        n_samples = X.shape[0]
        dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))  # Weight gradient (Ağırlık gradyanı)
        db = (1 / n_samples) * np.sum(y_predicted - y)  # Bias gradient (Bias gradyanı)
        return dw, db

    def fit(self, X, y):
        """
        Trains the logistic regression model using mini-batch gradient descent.
        Mini-batch gradient descent kullanarak logistic regression modelini eğitir.
        
        :param X: Feature matrix (Özellik matrisi)
        :param y: Target labels (Hedef etiketler)
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)  # Ağırlıklar sıfırla başlatılır
        self.bias = 0  # Başlangıçta bias sıfırdır

        # Mini-batch gradient descent
        for epoch in range(self.epochs):
            # Shuffle data at the start of each epoch (Her epoch başında verileri karıştır)
            indices = np.random.permutation(n_samples)
            X = X[indices]
            y = y[indices]

            for batch_idx in range(0, n_samples, self.batch_size):
                X_batch = X[batch_idx:batch_idx + self.batch_size]
                y_batch = y[batch_idx:batch_idx + self.batch_size]

                # Compute predictions (Tahminleri hesapla)
                linear_model = np.dot(X_batch, self.weights) + self.bias
                y_predicted = self.sigmoid(linear_model)

                # Compute gradients (Gradyanları hesapla)
                dw, db = self.compute_gradients(X_batch, y_batch, y_predicted)

                # Update weights and bias (Ağırlıkları ve bias'ı güncelle)
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db

            # Optional: Print loss every 100 epochs (İsteğe bağlı: Her 100 epoch'ta kayıp değeri yazdır)
            if (epoch + 1) % 100 == 0:
                y_pred = self.sigmoid(np.dot(X, self.weights) + self.bias)
                loss = self.cross_entropy_loss(y, y_pred)
                print(f"Epoch {epoch + 1}: Loss = {loss:.4f}")

    def predict_proba(self, X):
        """
        Predicts probabilities for the input data using the trained model.
        Eğitilen model kullanılarak giriş verileri için olasılıkları tahmin eder.
        
        :param X: Feature matrix (Özellik matrisi)
        :return: Predicted probabilities (Tahmin edilen olasılıklar)
        """
        linear_model = np.dot(X, self.weights) + self.bias
        return self.sigmoid(linear_model)

    def predict(self, X):
        """
        Predicts binary class labels (0 or 1) for the input data.
        Giriş verileri için ikili sınıf etiketlerini (0 veya 1) tahmin eder.
        
        :param X: Feature matrix (Özellik matrisi)
        :return: Predicted class labels (Tahmin edilen sınıf etiketleri)
        """
        y_pred_proba = self.predict_proba(X)
        return [1 if i > 0.5 else 0 for i in y_pred_proba]


# Example usage (Örnek kullanım)
if __name__ == "__main__":
    # Sample dataset (Örnek veri seti)
    X = np.array([[0.1, 0.2], [0.2, 0.4], [0.3, 0.6], [0.4, 0.8], [0.5, 1.0]])
    y = np.array([0, 0, 1, 1, 1])

    # Initialize the model with mini-batch gradient descent (Mini-batch gradient descent ile modeli başlat)
    classifier = MiniBatchLogisticRegression(learning_rate=0.01, epochs=1000, batch_size=2)

    # Train the model (Modeli eğit)
    classifier.fit(X, y)

    # Predict class labels (Sınıf etiketlerini tahmin et)
    predictions = classifier.predict(X)
    print(f"Predicted class labels: {predictions}")
