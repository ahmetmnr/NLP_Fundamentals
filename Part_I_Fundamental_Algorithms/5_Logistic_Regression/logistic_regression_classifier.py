import numpy as np

class LogisticRegressionClassifier:
    def __init__(self, learning_rate=0.01, epochs=1000):
        """
        Initializes the Logistic Regression classifier with learning rate and epochs.
        Logistic Regression sınıflandırıcısını öğrenme oranı ve epoch sayısı ile başlatır.
        
        :param learning_rate: The step size for weight updates (Ağırlık güncellemeleri için adım büyüklüğü)
        :param epochs: The number of iterations to train the model (Modeli eğitmek için yapılacak yineleme sayısı)
        """
        self.learning_rate = learning_rate  # Öğrenme oranı
        self.epochs = epochs  # Epoch sayısı
        self.weights = None  # Model ağırlıkları
        self.bias = None  # Model bias'ı

    def sigmoid(self, z):
        """
        Sigmoid function to calculate probabilities.
        Olasılıkları hesaplamak için sigmoid fonksiyonu.
        
        :param z: The input value (Girdi değeri)
        :return: Sigmoid of z (z'nin sigmoid değeri)
        """
        return 1 / (1 + np.exp(-z))

    def cross_entropy_loss(self, y_true, y_pred):
        """
        Cross-entropy loss function for binary classification.
        İkili sınıflandırma için cross-entropy loss fonksiyonu.
        
        :param y_true: True labels (Gerçek etiketler)
        :param y_pred: Predicted probabilities (Tahmin edilen olasılıklar)
        :return: Cross-entropy loss (Cross-entropy loss değeri)
        """
        n_samples = len(y_true)
        # Avoid log(0) error (log(0) hatasını önlemek için)
        y_pred = np.clip(y_pred, 1e-9, 1 - 1e-9)
        loss = -1 / n_samples * np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss

    def fit(self, X, y):
        """
        Trains the logistic regression model using stochastic gradient descent.
        Stokastik gradient descent kullanarak logistic regression modelini eğitir.
        
        :param X: Feature matrix (Özellik matrisi)
        :param y: Target labels (Hedef etiketler)
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)  # Ağırlıklar başlangıçta sıfır
        self.bias = 0  # Başlangıçta bias sıfır

        # Stochastic Gradient Descent (Stokastik Gradient Descent)
        for epoch in range(self.epochs):
            for i in range(n_samples):
                # Linear combination (Doğrusal kombinasyon)
                linear_model = np.dot(X[i], self.weights) + self.bias
                # Predicted probability (Tahmin edilen olasılık)
                y_predicted = self.sigmoid(linear_model)

                # Compute gradients (Gradyanları hesapla)
                dw = (y_predicted - y[i]) * X[i]  # Weight gradient (Ağırlık gradyanı)
                db = y_predicted - y[i]  # Bias gradient (Bias gradyanı)

                # Update weights and bias (Ağırlıkları ve bias'ı güncelle)
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db

            # Optional: Print loss every 100 epochs (İsteğe bağlı: Her 100 epoch'ta loss değeri yazdır)
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
        y_pred_labels = [1 if i > 0.5 else 0 for i in y_pred_proba]
        return np.array(y_pred_labels)


# Example usage (Örnek kullanım)
if __name__ == "__main__":
    # Sample dataset (Örnek veri seti)
    X = np.array([[0.1, 0.2], [0.2, 0.4], [0.3, 0.8], [0.4, 0.5], [0.5, 0.7]])
    y = np.array([0, 0, 1, 1, 1])

    # Initialize the model (Modeli başlat)
    classifier = LogisticRegressionClassifier(learning_rate=0.01, epochs=1000)

    # Train the model (Modeli eğit)
    classifier.fit(X, y)

    # Predict probabilities (Olasılıkları tahmin et)
    probabilities = classifier.predict_proba(X)
    print(f"Predicted probabilities: {probabilities}")

    # Predict class labels (Sınıf etiketlerini tahmin et)
    predictions = classifier.predict(X)
    print(f"Predicted class labels: {predictions}")
