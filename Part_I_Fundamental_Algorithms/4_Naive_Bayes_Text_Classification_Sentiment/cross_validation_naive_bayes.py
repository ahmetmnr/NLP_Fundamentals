from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sentiment_analysis_with_negation import NaiveBayesWithNegation  # Negation handling modelini import ediyoruz
from nltk.tokenize import word_tokenize
import numpy as np
import nltk

# Download necessary data for tokenization
# Tokenizasyon için gerekli verileri indir
nltk.download('punkt')

# Function to perform K-fold cross-validation (K-fold cross-validation'ı gerçekleştiren fonksiyon)
def cross_validate_naive_bayes(documents, labels, k=5):
    """
    Perform K-fold cross-validation on Naive Bayes Classifier and calculate average performance metrics.
    Naive Bayes Sınıflandırıcısı üzerinde K-fold cross-validation gerçekleştirir ve ortalama performans metriklerini hesaplar.
    
    :param documents: List of documents (strings) to be used for cross-validation (Cross-validation için kullanılacak belgelerin listesi)
    :param labels: List of labels corresponding to the documents (Belgelere karşılık gelen etiketlerin listesi)
    :param k: Number of folds for cross-validation (Cross-validation için K katlama sayısı)
    :return: Dictionary with average precision, recall, F1-score, and accuracy (Ortalama kesinlik, duyarlılık, F1-score ve doğruluk)
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=42)  # K-Fold Cross Validation split
    precision_scores = []
    recall_scores = []
    f1_scores = []
    accuracy_scores = []

    fold_index = 1
    for train_index, test_index in kf.split(documents):
        # Split the data into training and testing sets (Veriyi eğitim ve test kümelerine böl)
        train_documents = [documents[i] for i in train_index]
        train_labels = [labels[i] for i in train_index]
        test_documents = [documents[i] for i in test_index]
        test_labels = [labels[i] for i in test_index]

        # Train the Naive Bayes classifier (Naive Bayes sınıflandırıcısını eğit)
        classifier = NaiveBayesWithNegation()
        classifier.train(train_documents, train_labels)

        # Predict on the test set (Test verisi üzerinde tahmin yap)
        predicted_labels = [classifier.predict(doc) for doc in test_documents]

        # Calculate evaluation metrics (Değerlendirme metriklerini hesapla)
        precision = precision_score(test_labels, predicted_labels, pos_label="positive")
        recall = recall_score(test_labels, predicted_labels, pos_label="positive")
        f1 = f1_score(test_labels, predicted_labels, pos_label="positive")
        accuracy = accuracy_score(test_labels, predicted_labels)

        # Store the metrics for this fold (Bu katlama ait metrikleri sakla)
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)
        accuracy_scores.append(accuracy)

        print(f"Fold {fold_index} - Precision: {precision:.2f}, Recall: {recall:.2f}, F1-score: {f1:.2f}, Accuracy: {accuracy:.2f}")
        fold_index += 1

    # Calculate average metrics across all folds (Tüm katlamalar arasında ortalama metrikleri hesapla)
    avg_precision = np.mean(precision_scores)
    avg_recall = np.mean(recall_scores)
    avg_f1 = np.mean(f1_scores)
    avg_accuracy = np.mean(accuracy_scores)

    # Print average metrics (Ortalama metrikleri yazdır)
    print("\nAverage Metrics across all folds:")
    print(f"Average Precision: {avg_precision:.2f}")
    print(f"Average Recall: {avg_recall:.2f}")
    print(f"Average F1-score: {avg_f1:.2f}")
    print(f"Average Accuracy: {avg_accuracy:.2f}")

    # Return average metrics (Ortalama metrikleri döndür)
    return {
        "precision": avg_precision,
        "recall": avg_recall,
        "f1_score": avg_f1,
        "accuracy": avg_accuracy
    }

# Example usage (Örnek kullanım)
if __name__ == "__main__":
    # Sample documents and labels (Örnek belgeler ve etiketler)
    documents = [
        "I love this movie. It is fantastic!",
        "I don't like this film. It was boring.",
        "The acting was wonderful!",
        "The plot was terrible. I didn't enjoy it.",
        "This movie is great.",
        "I wouldn't recommend this movie.",
        "I like the direction but not the acting.",
        "The film was absolutely amazing and enjoyable.",
        "It was a disaster. I hated it.",
        "What a fantastic story!"
    ]

    labels = ["positive", "negative", "positive", "negative", "positive", "negative", "negative", "positive", "negative", "positive"]

    # Perform 5-fold cross-validation (5 katlı cross-validation gerçekleştirin)
    cross_validate_naive_bayes(documents, labels, k=5)
