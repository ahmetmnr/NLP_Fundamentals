# Logistic Regression with Regularization: Test and Implementation

This repository contains Python scripts to test logistic regression with L1 (lasso) and L2 (ridge) regularization. Regularization is useful for improving model generalization and preventing overfitting, especially when working with high-dimensional datasets.

Bu depo, L1 (lasso) ve L2 (ridge) regularization ile logistic regression'ı test eden Python betiklerini içerir. Regularization, model genellemesini iyileştirmek ve özellikle yüksek boyutlu veri setlerinde aşırı öğrenmeyi (overfitting) önlemek için kullanışlıdır.

---

## `regularization.py`

This file implements logistic regression with optional L1 and L2 regularization.
This allows you to add penalty terms to the model's weights, helping prevent overfitting.

Bu dosya, isteğe bağlı olarak L1 ve L2 regularization içeren logistic regression'ı uygular.
Bu sayede, modele ağırlıklar için ceza terimleri eklenerek aşırı öğrenmenin önüne geçilir.

### Key Features (Ana Özellikler):
- **L1 (Lasso) Regularization**: Adds the absolute value of the weights as a penalty term, useful for feature selection.
- **L1 (Lasso) Regularization**: Ağırlıkların mutlak değerini ceza terimi olarak ekler, özellik seçimi için kullanışlıdır.
  
- **L2 (Ridge) Regularization**: Adds the square of the weights as a penalty term, helps reduce all weights, preventing overfitting.
- **L2 (Ridge) Regularization**: Ağırlıkların karesini ceza terimi olarak ekler, tüm ağırlıkları küçülterek aşırı öğrenmeyi önler.

---

## `regularization_test.py`

This file tests the logistic regression model with and without regularization (L1, L2).
It compares the model's performance across three scenarios: no regularization, L1 regularization, and L2 regularization.

Bu dosya, regularization (L1, L2) ile ve olmadan logistic regression modelini test eder.
Modelin performansını üç senaryoda karşılaştırır: regularization olmadan, L1 regularization ile ve L2 regularization ile.

### Key Features (Ana Özellikler):
- **Comparison of Models**: Evaluates the model's accuracy, precision, recall, and F1-score in each scenario.
- **Modellerin Karşılaştırılması**: Her senaryoda modelin doğruluk, kesinlik, duyarlılık ve F1-score değerlerini değerlendirir.

---

## Regularization Benefits (Regularization'ın Faydaları):

- **Prevents Overfitting**: By adding penalty terms, regularization helps the model generalize better to unseen data, reducing overfitting.
- **Aşırı Öğrenmeyi Önler**: Ceza terimleri ekleyerek regularization, modelin görülmeyen verilere daha iyi genelleme yapmasına yardımcı olur.

- **Improves Generalization**: Particularly useful in high-dimensional datasets, regularization improves model generalization by preventing the model from fitting noise in the data.
- **Genelleme Yeteneğini İyileştirir**: Özellikle yüksek boyutlu veri setlerinde, regularization, modelin verideki gürültüye uyum sağlamasını engelleyerek genelleme yeteneğini artırır.

---

## Example Usage (Örnek Kullanım):

```python
# Import the necessary classes from regularization.py and regularization_test.py
from regularization import LogisticRegressionWithRegularization
from regularization_test import test_regularization

# Example dataset
X = np.array([[0.1, 0.2], [0.2, 0.4], [0.3, 0.6], [0.4, 0.8], [0.5, 1.0]])
y = np.array([0, 0, 1, 1, 1])

# Test the model with and without regularization
test_regularization(X, y)