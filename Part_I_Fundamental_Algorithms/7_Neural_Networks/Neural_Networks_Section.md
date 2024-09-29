# Part I: Fundamental Algorithms - Neural Networks (Sinir Ağları)

This section focuses on feedforward neural networks and their applications in tasks such as sentiment analysis and language modeling. It introduces neural network components, activation functions, and training techniques such as backpropagation.

Bu bölümde, ileri beslemeli sinir ağları ve duygu analizi ile dil modelleme gibi görevlerde nasıl kullanılacağına odaklanılıyor. Sinir ağı bileşenleri, aktivasyon fonksiyonları ve geri yayılım gibi eğitim teknikleri ele alınıyor.

## 1. Neural Network Units (Sinir Ağı Birimleri)
- **Description**: Implements basic neural network units, including weights, bias, and activation functions (sigmoid, ReLU).
- **Açıklama**: Ağırlıklar, bias ve aktivasyon fonksiyonları (sigmoid, ReLU) dahil olmak üzere temel sinir ağı birimlerini uygular.

### Files (Dosyalar):
- `neural_network_units.py`: Basic implementation of neurons with activation functions and forward pass. (Aktivasyon fonksiyonları ve ileri besleme işlemi ile temel nöron yapısı.)

---

## 2. Feedforward Neural Networks (İleri Beslemeli Sinir Ağları)
- **Description**: Creates a simple feedforward neural network with one hidden layer, supporting forward and backward propagation.
- **Açıklama**: Tek gizli katmanlı basit bir ileri beslemeli sinir ağı oluşturur, ileri ve geri yayılımı destekler.

### Files (Dosyalar):
- `feedforward_network.py`: Implements the feedforward network with forward and backward propagation. (İleri ve geri yayılım işlemi ile ileri beslemeli sinir ağı uygular.)

---

## 3. Feedforward Networks for NLP (NLP İçin İleri Beslemeli Ağlar)
- **Description**: Uses a feedforward neural network for binary sentiment classification using a simple bag-of-words model.
- **Açıklama**: Basit bir kelime torbası modeli ile ikili duygu analizi için ileri beslemeli bir sinir ağı kullanır.

### Files (Dosyalar):
- `feedforward_nlp_classification.py`: Implements sentiment classification using a feedforward neural network. (İleri beslemeli sinir ağı kullanarak duygu sınıflandırmasını uygular.)

---

## 4. Feedforward Neural Language Model (İleri Beslemeli Sinir Dil Modeli)
- **Description**: Builds a feedforward neural network to predict the next word in a sentence.
- **Açıklama**: Bir cümledeki bir sonraki kelimeyi tahmin etmek için ileri beslemeli bir sinir ağı oluşturur.

### Files (Dosyalar):
- `feedforward_language_model.py`: Builds a feedforward neural network for language modeling tasks. (Dil modelleme görevleri için ileri beslemeli bir sinir ağı oluşturur.)

---

## 5. Neural Language Model Training (Sinir Dil Modeli Eğitimi)
- **Description**: Trains the feedforward language model using backpropagation and evaluates it on the test set.
- **Açıklama**: Geri yayılım kullanarak ileri beslemeli dil modelini eğitir ve test seti üzerinde değerlendirir.

### Files (Dosyalar):
- `language_model_training.py`: Trains and evaluates the feedforward neural language model. (İleri beslemeli sinir dil modelini eğitir ve değerlendirir.)

---

## Tasks Covered (Kapsanan Görevler)
1. **Feedforward Network Implementation**: Creates and trains a feedforward network for basic tasks like classification and language modeling.
2. **Sentiment Analysis**: Applies the neural network for binary sentiment classification.
3. **Language Modeling**: Predicts the next word in a sentence using a neural language model.

1. **İleri Beslemeli Ağ Uygulaması**: Sınıflandırma ve dil modelleme gibi temel görevler için ileri beslemeli ağ oluşturur ve eğitir.
2. **Duygu Analizi**: İkili duygu sınıflandırması için sinir ağı uygular.
3. **Dil Modelleme**: Bir sinir dil modeli kullanarak bir cümledeki bir sonraki kelimeyi tahmin eder.
