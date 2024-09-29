import numpy as np

class LogisticRegressionWithRegularization:
    def __init__(self, learning_rate=0.01, epochs=1000, regularization=None, lambda_=0.1):
        """
        Initializes the Logistic Regression model with L1 or L2 regularization.
        L1 veya L2 regularization ile Logistic Regression modelini başlatır.
        
        :param learning_rate: The step size for weight updates (Ağırlık güncellemeleri için adım büyüklüğü)
        :param epochs: The number of iterations to train the model (Modeli eğitmek için epoch sayısı)
        :param regularization: Type of regularization: 'l1', 'l2', or None (Regularization türü: 'l1', 'l2', veya None)
        :param lambda_: Regularization strength (Regularization gücü)
        """
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.regularization = regularization  # Regularization türü (L1 veya L2)
        self.lambda_ = lambda_  # Regularization katsayısı
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
        Cross-entropy loss function for binary classification, with optional regularization.
        İkili sınıflandırma için cross-entropy kayıp fonksiyonu, regularization ile.
        
        :param y_true: True labels (Gerçek etiketler)
        :param y_pred: Predicted probabilities (Tahmin edilen olasılıklar)
        :return: Cross-entropy loss with regularization (Regularization ile cross-entropy kayıp değeri)
        """
        n_samples = len(y_true)
        y_pred = np.clip(y_pred, 1e-9, 1 - 1e-9)  # log(0)'ı önlemek için
        base_loss = -1 / n_samples * np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

        # Add regularization terms (Regularization terimlerini ekle)
        if self.regularization == 'l2':
            regularization_term = self.lambda_ * np.sum(self.weights ** 2) / (2 * n_samples)
            base_loss += regularization_term
        elif self.regularization == 'l1':
            regularization_term = self.lambda_ * np.sum(np.abs(self.weights)) / n_samples
            base_loss += regularization_term

        return base_loss

    def fit(self, X, y):
        """
        Trains the logistic regression model with regularization using gradient descent.
        Gradient descent kullanarak regularization ile logistic regression modelini eğitir.
        
        :param X: Feature matrix (Özellik matrisi)
        :param y: Target labels (Hedef etiketler)
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)  # Ağırlıklar sıfırla başlatılır
        self.bias = 0  # Başlangıçta bias sıfırdır

        # Gradient descent (Gradient descent)
        for epoch in range(self.epochs):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self.sigmoid(linear_model)

            # Compute gradients (Gradyanları hesapla)
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # Apply regularization to gradients (Regularization'u gradyanlara uygula)
            if self.regularization == 'l2':
                dw += (self.lambda_ / n_samples) * self.weights
            elif self.regularization == 'l1':
                dw += (self.lambda_ / n_samples) * np.sign(self.weights)

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
    X = np.array([[0.1, 0.2], [0.2, 0.4], [0.3, 0.8], [0.4, 0.5], [0.5, 0.7]])
    y = np.array([0, 0, 1, 1, 1])

    # Initialize the model with L2 regularization (L2 regularization ile modeli başlat)
    classifier_l2 = LogisticRegressionWithRegularization(learning_rate=0.01, epochs=1000, regularization='l2', lambda_=0.1)

    # Train the model (Modeli eğit)
    classifier_l2.fit(X, y)

    # Predict class labels (Sınıf etiketlerini tahmin et)
    predictions_l2 = classifier_l2.predict(X)
    print(f"Predictions with L2 regularization: {predictions_l2}")

    # Initialize the model with L1 regularization (L1 regularization ile modeli başlat)
    classifier_l1 = LogisticRegressionWithRegularization(learning_rate=0.01, epochs=1000, regularization='l1', lambda_=0.1)

    # Train the model (Modeli eğit)
    classifier_l1.fit(X, y)

    # Predict class labels (Sınıf etiketlerini tahmin et)
    predictions_l1 = classifier_l1.predict(X)
    print(f"Predictions with L1 regularization: {predictions_l1}")
