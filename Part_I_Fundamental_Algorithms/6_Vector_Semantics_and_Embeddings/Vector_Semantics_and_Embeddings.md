# Part I: Fundamental Algorithms - 6. Vector Semantics and Embeddings

This section focuses on various approaches to learn and represent word vectors (embeddings). We explore several models that generate embeddings for words based on their context and usage in text data, helping machines understand word semantics.

## 1. Word2Vec (Skip-Gram Model)
- **Description**: Learns word embeddings using the skip-gram model. Predicts the context words given a center word.
- **Features**:
  - Uses stochastic gradient descent to optimize the embeddings.
  - Learns embeddings based on context words surrounding the target word.
  
### Files:
- `word2vec_skipgram.py`: Implements the skip-gram model for word2vec to learn word embeddings. Contains cosine similarity and other distance metrics.

---

## 2. GloVe (Global Vectors for Word Representation)
- **Description**: Global Vectors for Word Representation (GloVe) model generates word vectors by factoring in co-occurrence statistics from a corpus.
- **Features**:
  - Uses a global co-occurrence matrix to generate word vectors.
  - Suitable for capturing semantic relationships between words.
  
### Files:
- `glove_embeddings.py`: Downloads, extracts, and loads pretrained GloVe embeddings, providing functions to retrieve word embeddings for word similarity tasks.

---

## 3. FastText
- **Description**: FastText is an extension of Word2Vec that incorporates subword information into word embeddings, allowing it to handle rare and misspelled words.
- **Features**:
  - Uses subword (character n-gram) information to handle out-of-vocabulary words.
  - Learns embeddings for both words and subwords, enhancing word similarity and analogy tasks.
  
### Files:
- `fasttext_embeddings.py`: Implements FastText embeddings by downloading pretrained FastText models. Provides methods for generating word vectors, subword embeddings, and finding similar words.

---

## 4. BERT (Bidirectional Encoder Representations from Transformers)
- **Description**: BERT generates contextualized embeddings for each word by understanding the surrounding context in both directions (left-to-right and right-to-left).
- **Features**:
  - Generates different embeddings for the same word based on the sentence context.
  - Hugging Face’s transformers library provides pretrained models for generating embeddings.
  
### Files:
- `bert_embeddings.py`: Uses BERT to extract contextualized word embeddings for words and sentences. Supports different BERT models (e.g., `bert-base-uncased` or `bert-large-uncased`).

---

## 5. Bias Detection in Embeddings
- **Description**: Analyzes and detects biases (e.g., gender or racial) in word embeddings using methods like WEAT (Word Embedding Association Test).
  
### Files:
- `bias_detection_embeddings.py`: Detects and analyzes bias in word embeddings by projecting word vectors onto a bias subspace. Uses WEAT to evaluate bias.

---

## 6. Embedding Visualization
- **Description**: Visualizes word embeddings using dimensionality reduction techniques such as t-SNE and PCA.
- **Features**:
  - Helps in understanding word similarity and clustering by visualizing embeddings in 2D.
  
### Files:
- `embedding_visualization.py`: Generates visualizations of word embeddings using t-SNE and PCA, saving plots in specified directories.

---

## 7. Static Embeddings Evaluation
- **Description**: Evaluates the quality of static word embeddings on tasks like word similarity and analogy detection using datasets such as WordSim-353 or TOEFL.
  
### Files:
- `static_embeddings_evaluation.py`: Computes similarity scores for word pairs and performs analogy detection tasks to evaluate the effectiveness of static embeddings.

---

## 8. Future Extensions
- GPT models or other transformer-based architectures could be integrated to provide embeddings for contextualized language understanding, especially for complex tasks requiring sentence-level embeddings.

---

## Tasks Covered
1. **Word Similarity**: Calculate the similarity between words using cosine similarity on word vectors.
2. **Analogy Detection**: Detect analogies like "king:man::queen:woman" using vector arithmetic.
3. **Bias Detection**: Analyze and detect biases in word vectors, e.g., gender or racial bias.
4. **Embedding Visualization**: Visualize embeddings in 2D using dimensionality reduction methods.
