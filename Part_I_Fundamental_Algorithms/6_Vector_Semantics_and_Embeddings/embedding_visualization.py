import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from word2vec_skipgram import Word2VecSkipGram  # Daha önce oluşturduğumuz Word2VecSkipGram sınıfı

class EmbeddingVisualization:
    def __init__(self, embeddings, word_to_index, index_to_word):
        """
        Initializes the embedding visualization class with word embeddings and mappings.
        Kelime gömmeleri ve indeks-kelime eşlemeleri ile gömme görselleştirme sınıfını başlatır.
        
        :param embeddings: Word embeddings matrix (Kelime gömme matrisi)
        :param word_to_index: Dictionary mapping words to their indices (Kelime-indeks eşlemesi yapan sözlük)
        :param index_to_word: Dictionary mapping indices to their corresponding words (İndeks-kelime eşlemesi yapan sözlük)
        """
        self.embeddings = embeddings
        self.word_to_index = word_to_index
        self.index_to_word = index_to_word
    
    def visualize_embeddings(self, words, method='tsne', save_path=None):
        """
        Visualizes the word embeddings using dimensionality reduction (t-SNE or PCA).
        Boyut indirgeme yöntemleriyle (t-SNE veya PCA) kelime gömmelerini görselleştirir.
        
        :param words: List of words to visualize (Görselleştirilecek kelimeler listesi)
        :param method: Dimensionality reduction method ('tsne' or 'pca') (Boyut indirgeme yöntemi: 'tsne' veya 'pca')
        :param save_path: Path to save the plot (Grafiği kaydetmek için yol)
        """
        # Kelime gömmelerini al
        word_indices = [self.word_to_index[word] for word in words]
        word_embeddings = np.array([self.embeddings[idx] for idx in word_indices])

        # Boyut indirgeme yöntemi seç
        if method == 'tsne':
            perplexity_value = min(5, len(words) - 1)  # Perplexity'yi kelime sayısının altında bir değere ayarla
            reducer = TSNE(n_components=2, random_state=42, perplexity=perplexity_value)
        elif method == 'pca':
            reducer = PCA(n_components=2)
        else:
            raise ValueError("Invalid method. Choose 'tsne' or 'pca'.")

        # Embedding'leri 2D'ye indir
        reduced_embeddings = reducer.fit_transform(word_embeddings)

        # Plot oluştur
        self._plot_embeddings(reduced_embeddings, words, save_path)
    
    def _plot_embeddings(self, reduced_embeddings, words, save_path):
        """
        Creates a scatter plot of the reduced embeddings with word labels.
        Kelime etiketleriyle birlikte indirgenmiş gömmelerin scatter plot'unu oluşturur.
        
        :param reduced_embeddings: 2D reduced embeddings (2 boyutlu indirgenmiş gömmeler)
        :param words: List of words (Kelime listesi)
        :param save_path: Path to save the plot (Grafiği kaydetmek için yol)
        """
        plt.figure(figsize=(10, 10))
        for i, word in enumerate(words):
            x, y = reduced_embeddings[i]
            plt.scatter(x, y)
            plt.text(x + 0.03, y + 0.03, word, fontsize=12)
        
        plt.title('Word Embeddings Visualization')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)  # Grafiği belirtilen yolda kaydet
            print(f"Visualization saved to {save_path}")
        
        plt.show()

# Example usage (Örnek kullanım)
if __name__ == "__main__":
    # Örnek corpus
    corpus = [
        ["i", "love", "machine", "learning"],
        ["machine", "learning", "is", "great"],
        ["natural", "language", "processing", "is", "part", "of", "machine", "learning"],
        ["i", "enjoy", "learning", "about", "natural", "language", "processing"]
    ]
    
    # Kelime dağarcığını oluştur
    vocab = set(word for sentence in corpus for word in sentence)
    word_to_index = {word: i for i, word in enumerate(vocab)}
    index_to_word = {i: word for word, i in word_to_index.items()}
    
    # Word2VecSkipGram modelini başlat
    vocab_size = len(vocab)
    embedding_dim = 100  # Kelime gömme boyutu
    word2vec = Word2VecSkipGram(vocab_size, embedding_dim, learning_rate=0.01, window_size=2)
    
    # Eğitim verisini oluştur
    training_data = word2vec.generate_training_data(corpus, word_to_index)
    
    # Modeli eğit
    word2vec.train(training_data, epochs=1000)
    
    # Eğitilen kelime gömmelerini al
    embeddings = word2vec.center_embeddings
    
    # Görselleştirme sınıfını başlat
    vis = EmbeddingVisualization(embeddings, word_to_index, index_to_word)

    # Kayıt dizinini belirt
    save_dir = "Part_I_Fundamental_Algorithms/6_Vector_Semantics_and_Embeddings"
    os.makedirs(save_dir, exist_ok=True)  # Dizin yoksa oluştur

    # t-SNE görselleştirmesi ve kaydetme
    tsne_save_path = os.path.join(save_dir, "tsne_visualization.png")
    words_to_visualize = ['machine', 'learning', 'natural', 'language', 'processing']
    vis.visualize_embeddings(words_to_visualize, method='tsne', save_path=tsne_save_path)
    
    # PCA görselleştirmesi ve kaydetme
    pca_save_path = os.path.join(save_dir, "pca_visualization.png")
    vis.visualize_embeddings(words_to_visualize, method='pca', save_path=pca_save_path)
