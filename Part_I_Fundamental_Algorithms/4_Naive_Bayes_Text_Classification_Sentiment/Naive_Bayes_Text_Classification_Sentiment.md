
# Naive Bayes Text Classification and Sentiment Analysis

This repository contains Python scripts to perform sentiment analysis using the Naive Bayes classifier, with variations such as Binary Naive Bayes, negation handling, and model evaluation using cross-validation and bootstrap testing. Below are the key components of this repository with both English and Turkish explanations.

Bu depo, Naive Bayes sınıflandırıcısını kullanarak duygu analizi gerçekleştirmek için Python betiklerini içerir. Binary Naive Bayes, olumsuzlama işleme ve çapraz doğrulama ile bootstrap testi kullanarak model değerlendirmeleri gibi varyasyonlar bulunmaktadır. Aşağıda, bu deponun ana bileşenleri verilmiştir.

---

## `naive_bayes_classifier.py`

This script implements the basic Naive Bayes classifier using bag-of-words representation. It calculates class priors and likelihoods using training data and applies the classifier to predict the sentiment of test documents.

Bu betik, bag-of-words gösterimi kullanarak temel Naive Bayes sınıflandırıcısını uygular. Eğitim verilerini kullanarak sınıf önceliklerini ve olasılıkları hesaplar ve test belgelerinin duygu durumunu tahmin etmek için sınıflandırıcıyı uygular.

---

## `train_naive_bayes.py`

This script trains the Naive Bayes model on a dataset. It handles the calculation of likelihoods with add-one smoothing (Laplace Smoothing) and outputs the trained model for future predictions.

Bu betik, Naive Bayes modelini bir veri kümesi üzerinde eğitir. Olasılıkları Laplace Düzeltmesi ile hesaplar ve eğitilen modeli gelecekteki tahminler için dışa aktarır.

---

## `binary_naive_bayes.py`

This script implements a variant of Naive Bayes called Binary Multinomial Naive Bayes. It clips the word counts in each document at 1 (binary representation). This approach is useful for sentiment classification where the binary occurrence of words is more important than their frequency.

Bu betik, Binary Multinomial Naive Bayes adlı bir Naive Bayes varyantını uygular. Belgedeki kelime sayımlarını 1 ile sınırlar (ikili gösterim). Bu yaklaşım, kelimelerin frekansından ziyade varlığının önemli olduğu duygu sınıflandırması için kullanışlıdır.

---

## `sentiment_analysis_with_negation.py`

This script enhances the Naive Bayes classifier by handling negation in text (e.g., "didn't like"). It implements negation handling by adding a "NOT_" prefix to words following negation tokens (e.g., "not", "no").

Bu betik, metindeki olumsuzlamayı (örneğin, "didn't like") işleyerek Naive Bayes sınıflandırıcısını geliştirir. Olumsuzlama kelimelerinden sonra gelen kelimelere "NOT_" öneki ekleyerek olumsuzlamayı işler.

---

## `sentiment_evaluation.py`

This script evaluates the trained Naive Bayes classifier using metrics like Precision, Recall, and F1-score. It also implements a confusion matrix and calculates accuracy for performance evaluation.

Bu betik, Precision, Recall ve F1-score gibi metrikler kullanarak eğitilen Naive Bayes sınıflandırıcısını değerlendirir. Aynı zamanda bir karışıklık matrisi oluşturur ve performans değerlendirmesi için doğruluk hesaplar.

---

## `cross_validation_naive_bayes.py`

This script performs K-fold cross-validation on the Naive Bayes classifier and calculates average accuracy, precision, recall, and F1-score. Cross-validation ensures that the model's performance is evaluated on different subsets of the data.

Bu betik, Naive Bayes sınıflandırıcısı üzerinde K katlı çapraz doğrulama yapar ve ortalama doğruluk, kesinlik, duyarlılık ve F1-score hesaplar. Çapraz doğrulama, modelin performansının farklı veri alt kümelerinde değerlendirilmesini sağlar.

---

## `bootstrap_test.py`

This script performs bootstrap testing to compare the performance of two different Naive Bayes models (e.g., standard vs. binary). It calculates the statistical significance of differences in performance.

Bu betik, iki farklı Naive Bayes modelinin performansını (örneğin, standart ve binary) karşılaştırmak için bootstrap testi gerçekleştirir. Performans farklarının istatistiksel anlamlılığını hesaplar.

---

## How to Use (Nasıl Kullanılır):

1. **Training the Classifier (Sınıflandırıcıyı Eğitme)**:
   - Train the classifier using `train_naive_bayes.py` or `binary_naive_bayes.py` scripts.
   - Sınıflandırıcıyı `train_naive_bayes.py` veya `binary_naive_bayes.py` betiklerini kullanarak eğitin.

2. **Making Predictions (Tahmin Yapma)**:
   - Use the trained model to classify new documents using the `predict` function in any of the scripts.
   - Eğitilen modeli kullanarak yeni belgeleri sınıflandırmak için betiklerdeki `predict` fonksiyonunu kullanın.

3. **Evaluating the Model (Modeli Değerlendirme)**:
   - Evaluate the classifier using `sentiment_evaluation.py` and perform cross-validation using `cross_validation_naive_bayes.py`.
   - Sınıflandırıcıyı `sentiment_evaluation.py` kullanarak değerlendirin ve `cross_validation_naive_bayes.py` ile çapraz doğrulama gerçekleştirin.

4. **Statistical Testing (İstatistiksel Test)**:
   - Perform statistical comparison between two models using `bootstrap_test.py`.
   - İki model arasında istatistiksel karşılaştırma yapmak için `bootstrap_test.py` betiğini kullanın.

---

## Requirements (Gereksinimler):

- Python 3.x
- NLTK
- Scikit-learn
- NumPy

Install the required packages using the following command:

Gerekli paketleri aşağıdaki komut ile yükleyin:

```bash
pip install numpy nltk scikit-learn
```

---

By following these scripts, you will be able to perform sentiment analysis using Naive Bayes, handle negations in text, evaluate your model, and statistically compare two models.

Bu betikleri takip ederek Naive Bayes kullanarak duygu analizi yapabilir, metindeki olumsuzlamaları işleyebilir, modelinizi değerlendirebilir ve iki modeli istatistiksel olarak karşılaştırabilirsiniz.

