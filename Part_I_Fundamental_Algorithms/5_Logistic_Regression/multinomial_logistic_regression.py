import numpy as np

class MultinomialLogisticRegression:
    def __init__(self, learning_rate=0.01, epochs=1000, batch_size=32):
        """
        Initializes the multinomial logistic regression classifier with learning rate, epochs, and batch size.
        Multinomial logistic regression sınıflandırıcısını öğrenme oranı, epoch sayısı ve mini-batch boyutu ile başlatır.
        
        :param learning_rate: The step size for weight updates (Ağırlık güncellemeleri için adım büyüklüğü)
        :param epochs: The number of iterations to train the model (Modeli eğitmek için epoch sayısı)
        :param batch_size: Size of mini-batches for gradient descent (Gradient descent için mini-batch boyutu)
        """
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.weights = None  # Model ağırlıkları
        self.bias = None  # Model bias'ı

    def softmax(self, z):
        """
        Softmax function to calculate probabilities for each class.
        Her sınıf için olasılıkları hesaplamak için softmax fonksiyonu.
        
        :param z: Input array (Girdi dizisi)
        :return: Softmax probabilities (Softmax olasılıkları)
        """
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # Numerical stability (Sayısal kararlılık için)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def cross_entropy_loss(self, y_true, y_pred):
        """
        Cross-entropy loss function for multinomial classification.
        Çok sınıflı sınıflandırma için cross-entropy kayıp fonksiyonu.
        
        :param y_true: True labels (Gerçek etiketler)
        :param y_pred: Predicted probabilities (Tahmin edilen olasılıklar)
        :return: Cross-entropy loss (Cross-entropy kayıp değeri)
        """
        n_samples = y_true.shape[0]
        # Avoid log(0) errors (log(0) hatasını önlemek için)
        y_pred = np.clip(y_pred, 1e-9, 1 - 1e-9)
        # Cross-entropy loss calculation (Cross-entropy kayıp hesabı)
        loss = -np.sum(y_true * np.log(y_pred)) / n_samples
        return loss

    def one_hot_encode(self, y, num_classes):
        """
        Converts class labels to one-hot encoding.
        Sınıf etiketlerini one-hot encoding formatına dönüştürür.
        
        :param y: Class labels (Sınıf etiketleri)
        :param num_classes: Number of unique classes (Sınıf sayısı)
        :return: One-hot encoded labels (One-hot encoded etiketler)
        """
        one_hot = np.zeros((y.shape[0], num_classes))
        one_hot[np.arange(y.shape[0]), y] = 1
        return one_hot

    def fit(self, X, y):
        """
        Trains the multinomial logistic regression model using mini-batch gradient descent.
        Mini-batch gradient descent kullanarak multinomial logistic regression modelini eğitir.
        
        :param X: Feature matrix (Özellik matrisi)
        :param y: Class labels (Sınıf etiketleri)
        """
        n_samples, n_features = X.shape
        num_classes = len(np.unique(y))  # Unique class count (Benzersiz sınıf sayısı)

        # Initialize weights and bias (Ağırlıklar ve bias'ı başlat)
        self.weights = np.zeros((n_features, num_classes))
        self.bias = np.zeros((1, num_classes))

        # Convert class labels to one-hot encoding (Sınıf etiketlerini one-hot encoding'e çevir)
        y_one_hot = self.one_hot_encode(y, num_classes)

        # Mini-batch gradient descent (Mini-batch gradient descent)
        for epoch in range(self.epochs):
            for batch_idx in range(0, n_samples, self.batch_size):
                X_batch = X[batch_idx:batch_idx + self.batch_size]
                y_batch = y_one_hot[batch_idx:batch_idx + self.batch_size]

                # Compute the linear predictions (Doğrusal tahminleri hesapla)
                linear_model = np.dot(X_batch, self.weights) + self.bias
                # Compute softmax probabilities (Softmax olasılıklarını hesapla)
                y_pred = self.softmax(linear_model)

                # Compute gradients (Gradyanları hesapla)
                dw = (1 / X_batch.shape[0]) * np.dot(X_batch.T, (y_pred - y_batch))  # Weight gradient (Ağırlık gradyanı)
                db = (1 / X_batch.shape[0]) * np.sum(y_pred - y_batch, axis=0)  # Bias gradient (Bias gradyanı)

                # Update weights and bias (Ağırlıkları ve bias'ı güncelle)
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db

            # Optional: Print loss every 100 epochs (İsteğe bağlı: Her 100 epoch'ta loss değeri yazdır)
            if (epoch + 1) % 100 == 0:
                linear_model = np.dot(X, self.weights) + self.bias
                y_pred = self.softmax(linear_model)
                loss = self.cross_entropy_loss(y_one_hot, y_pred)
                print(f"Epoch {epoch + 1}: Loss = {loss:.4f}")

    def predict_proba(self, X):
        """
        Predicts probabilities for each class using the trained model.
        Eğitilen model kullanılarak her sınıf için olasılıklar tahmin edilir.
        
        :param X: Feature matrix (Özellik matrisi)
        :return: Predicted probabilities for each class (Her sınıf için tahmin edilen olasılıklar)
        """
        linear_model = np.dot(X, self.weights) + self.bias
        return self.softmax(linear_model)

    def predict(self, X):
        """
        Predicts class labels for the input data.
        Giriş verileri için sınıf etiketlerini tahmin eder.
        
        :param X: Feature matrix (Özellik matrisi)
        :return: Predicted class labels (Tahmin edilen sınıf etiketleri)
        """
        y_pred_proba = self.predict_proba(X)
        return np.argmax(y_pred_proba, axis=1)


# Example usage (Örnek kullanım)
if __name__ == "__main__":
    # Sample dataset (Örnek veri seti)
    X = np.array([[0.1, 0.2], [0.2, 0.4], [0.3, 0.6], [0.4, 0.8], [0.5, 1.0]])
    y = np.array([0, 1, 2, 1, 0])  # Multi-class labels (Çok sınıflı etiketler)

    # Initialize the model (Modeli başlat)
    classifier = MultinomialLogisticRegression(learning_rate=0.01, epochs=1000, batch_size=2)

    # Train the model (Modeli eğit)
    classifier.fit(X, y)

    # Predict probabilities (Olasılıkları tahmin et)
    probabilities = classifier.predict_proba(X)
    print(f"Predicted probabilities: {probabilities}")

    # Predict class labels (Sınıf etiketlerini tahmin et)
    predictions = classifier.predict(X)
    print(f"Predicted class labels: {predictions}")
