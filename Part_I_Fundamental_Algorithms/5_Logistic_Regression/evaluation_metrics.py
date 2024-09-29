import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

class EvaluationMetrics:
    @staticmethod
    def accuracy(y_true, y_pred):
        """
        Computes the accuracy metric.
        Doğruluk (accuracy) metriğini hesaplar.
        
        :param y_true: True labels (Gerçek etiketler)
        :param y_pred: Predicted labels (Tahmin edilen etiketler)
        :return: Accuracy score (Doğruluk skoru)
        """
        return accuracy_score(y_true, y_pred)

    @staticmethod
    def precision(y_true, y_pred):
        """
        Computes the precision metric.
        Kesinlik (precision) metriğini hesaplar.
        
        :param y_true: True labels (Gerçek etiketler)
        :param y_pred: Predicted labels (Tahmin edilen etiketler)
        :return: Precision score (Kesinlik skoru)
        """
        return precision_score(y_true, y_pred)

    @staticmethod
    def recall(y_true, y_pred):
        """
        Computes the recall metric.
        Duyarlılık (recall) metriğini hesaplar.
        
        :param y_true: True labels (Gerçek etiketler)
        :param y_pred: Predicted labels (Tahmin edilen etiketler)
        :return: Recall score (Duyarlılık skoru)
        """
        return recall_score(y_true, y_pred)

    @staticmethod
    def f1_score(y_true, y_pred):
        """
        Computes the F1-score metric.
        F1-score metriğini hesaplar.
        
        :param y_true: True labels (Gerçek etiketler)
        :param y_pred: Predicted labels (Tahmin edilen etiketler)
        :return: F1-score (F1 skoru)
        """
        return f1_score(y_true, y_pred)

    @staticmethod
    def confusion_matrix(y_true, y_pred):
        """
        Computes the confusion matrix.
        Karışıklık matrisi (confusion matrix) hesaplar.
        
        :param y_true: True labels (Gerçek etiketler)
        :param y_pred: Predicted labels (Tahmin edilen etiketler)
        :return: Confusion matrix (Karışıklık matrisi)
        """
        return confusion_matrix(y_true, y_pred)

# Example usage (Örnek kullanım)
if __name__ == "__main__":
    # Sample true labels and predicted labels (Örnek gerçek etiketler ve tahmin edilen etiketler)
    y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 0])
    y_pred = np.array([1, 0, 1, 0, 0, 1, 0, 0, 1, 1])

    # Initialize the evaluation metrics class (Değerlendirme metrikleri sınıfını başlat)
    evaluator = EvaluationMetrics()

    # Calculate accuracy (Doğruluk hesapla)
    accuracy = evaluator.accuracy(y_true, y_pred)
    print(f"Accuracy: {accuracy:.2f}")

    # Calculate precision (Kesinlik hesapla)
    precision = evaluator.precision(y_true, y_pred)
    print(f"Precision: {precision:.2f}")

    # Calculate recall (Duyarlılık hesapla)
    recall = evaluator.recall(y_true, y_pred)
    print(f"Recall: {recall:.2f}")

    # Calculate F1-score (F1-score hesapla)
    f1 = evaluator.f1_score(y_true, y_pred)
    print(f"F1-score: {f1:.2f}")

    # Calculate confusion matrix (Karışıklık matrisi hesapla)
    conf_matrix = evaluator.confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(conf_matrix)
