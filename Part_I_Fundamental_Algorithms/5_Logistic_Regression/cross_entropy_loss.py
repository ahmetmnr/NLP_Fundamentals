import numpy as np

class CrossEntropyLoss:
    @staticmethod
    def binary_cross_entropy(y_true, y_pred):
        """
        Computes the binary cross-entropy loss function for binary classification.
        İkili sınıflandırma için binary cross-entropy kayıp fonksiyonunu hesaplar.
        
        :param y_true: True labels (Gerçek etiketler)
        :param y_pred: Predicted probabilities (Tahmin edilen olasılıklar)
        :return: Binary cross-entropy loss (Binary cross-entropy kaybı)
        """
        # Avoid log(0) errors by clipping values (log(0) hatalarını önlemek için değerleri sınırla)
        y_pred = np.clip(y_pred, 1e-9, 1 - 1e-9)
        n_samples = len(y_true)

        # Binary cross-entropy loss calculation (Binary cross-entropy kaybı hesaplama)
        loss = -1 / n_samples * np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss

    @staticmethod
    def multinomial_cross_entropy(y_true, y_pred):
        """
        Computes the multinomial cross-entropy loss function for multi-class classification.
        Çok sınıflı sınıflandırma için multinomial cross-entropy kayıp fonksiyonunu hesaplar.
        
        :param y_true: One-hot encoded true labels (One-hot encoded gerçek etiketler)
        :param y_pred: Predicted probabilities for each class (Her sınıf için tahmin edilen olasılıklar)
        :return: Multinomial cross-entropy loss (Multinomial cross-entropy kaybı)
        """
        # Avoid log(0) errors by clipping values (log(0) hatalarını önlemek için değerleri sınırla)
        y_pred = np.clip(y_pred, 1e-9, 1 - 1e-9)
        n_samples = y_true.shape[0]

        # Multinomial cross-entropy loss calculation (Multinomial cross-entropy kaybı hesaplama)
        loss = -1 / n_samples * np.sum(y_true * np.log(y_pred))
        return loss

# Example usage (Örnek kullanım)
if __name__ == "__main__":
    # Example binary classification (İkili sınıflandırma örneği)
    y_true_binary = np.array([1, 0, 1, 1, 0])
    y_pred_binary = np.array([0.9, 0.2, 0.8, 0.6, 0.1])

    binary_loss = CrossEntropyLoss.binary_cross_entropy(y_true_binary, y_pred_binary)
    print(f"Binary Cross-Entropy Loss: {binary_loss:.4f}")

    # Example multinomial classification (Çok sınıflı sınıflandırma örneği)
    y_true_multinomial = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0]])
    y_pred_multinomial = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.3, 0.5], [0.9, 0.05, 0.05], [0.2, 0.7, 0.1]])

    multinomial_loss = CrossEntropyLoss.multinomial_cross_entropy(y_true_multinomial, y_pred_multinomial)
    print(f"Multinomial Cross-Entropy Loss: {multinomial_loss:.4f}")
