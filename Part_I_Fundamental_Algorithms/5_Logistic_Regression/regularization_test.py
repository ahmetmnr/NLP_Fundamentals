import numpy as np
from sklearn.model_selection import train_test_split
from evaluation_metrics import EvaluationMetrics
from regularization import LogisticRegressionWithRegularization  # regularization.py dosyasını kullanıyoruz

# Function to test logistic regression with and without regularization
# Regularization ile ve olmadan logistic regression'ı test fonksiyonu
def test_regularization(X, y):
    """
    Tests logistic regression with L1 and L2 regularization, and without regularization.
    Logistic regression'ı L1, L2 regularization ile ve olmadan test eder.
    
    :param X: Feature matrix (Özellik matrisi)
    :param y: Target labels (Hedef etiketler)
    """
    # Split the data into training and testing sets (Verileri eğitim ve test setlerine ayır)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize evaluation metrics class (Değerlendirme metrikleri sınıfını başlat)
    evaluator = EvaluationMetrics()

    # 1. Logistic Regression without Regularization (Regularization olmadan logistic regression)
    classifier_no_reg = LogisticRegressionWithRegularization(learning_rate=0.01, epochs=1000, regularization=None)
    classifier_no_reg.fit(X_train, y_train)
    y_pred_no_reg = classifier_no_reg.predict(X_test)
    
    # Calculate metrics for no regularization (Regularization olmadan metrikleri hesapla)
    print("\nLogistic Regression without Regularization:")
    print(f"Accuracy: {evaluator.accuracy(y_test, y_pred_no_reg):.2f}")
    print(f"Precision: {evaluator.precision(y_test, y_pred_no_reg):.2f}")
    print(f"Recall: {evaluator.recall(y_test, y_pred_no_reg):.2f}")
    print(f"F1-score: {evaluator.f1_score(y_test, y_pred_no_reg):.2f}")

    # 2. Logistic Regression with L2 Regularization (L2 Regularization ile logistic regression)
    classifier_l2 = LogisticRegressionWithRegularization(learning_rate=0.01, epochs=1000, regularization='l2', lambda_=0.1)
    classifier_l2.fit(X_train, y_train)
    y_pred_l2 = classifier_l2.predict(X_test)
    
    # Calculate metrics for L2 regularization (L2 regularization ile metrikleri hesapla)
    print("\nLogistic Regression with L2 Regularization:")
    print(f"Accuracy: {evaluator.accuracy(y_test, y_pred_l2):.2f}")
    print(f"Precision: {evaluator.precision(y_test, y_pred_l2):.2f}")
    print(f"Recall: {evaluator.recall(y_test, y_pred_l2):.2f}")
    print(f"F1-score: {evaluator.f1_score(y_test, y_pred_l2):.2f}")

    # 3. Logistic Regression with L1 Regularization (L1 Regularization ile logistic regression)
    classifier_l1 = LogisticRegressionWithRegularization(learning_rate=0.01, epochs=1000, regularization='l1', lambda_=0.1)
    classifier_l1.fit(X_train, y_train)
    y_pred_l1 = classifier_l1.predict(X_test)
    
    # Calculate metrics for L1 regularization (L1 regularization ile metrikleri hesapla)
    print("\nLogistic Regression with L1 Regularization:")
    print(f"Accuracy: {evaluator.accuracy(y_test, y_pred_l1):.2f}")
    print(f"Precision: {evaluator.precision(y_test, y_pred_l1):.2f}")
    print(f"Recall: {evaluator.recall(y_test, y_pred_l1):.2f}")
    print(f"F1-score: {evaluator.f1_score(y_test, y_pred_l1):.2f}")

# Example usage (Örnek kullanım)
if __name__ == "__main__":
    # Sample dataset (Örnek veri seti)
    X = np.array([[0.1, 0.2], [0.2, 0.4], [0.3, 0.6], [0.4, 0.8], [0.5, 1.0], [0.1, 0.3], [0.2, 0.5], [0.3, 0.7], [0.4, 0.9], [0.5, 1.1]])
    y = np.array([0, 0, 1, 1, 1, 0, 0, 1, 1, 1])

    # Test logistic regression with and without regularization (Regularization ile ve olmadan logistic regression testi)
    test_regularization(X, y)
