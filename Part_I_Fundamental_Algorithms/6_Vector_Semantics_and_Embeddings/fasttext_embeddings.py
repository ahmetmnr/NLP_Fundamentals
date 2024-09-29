import fasttext
import fasttext.util

class FastTextEmbeddings:
    def __init__(self, language_code='en', dim=300):
        """
        Initializes the FastText model using pretrained vectors.
        Hazır eğitilmiş FastText modeli ile başlatır.
        
        :param language_code: Pretrained FastText model language code (Hazır eğitilmiş FastText modelinin dil kodu, örn: 'en')
        :param dim: Dimension of the word vectors (Kelime vektörlerinin boyutu, örn: 300)
        """
        self.language_code = language_code
        self.dim = dim

        # FastText dil modelini indir ve yükle
        print(f"Downloading and loading FastText model for language: {self.language_code}...")
        fasttext.util.download_model(self.language_code, if_exists='ignore')  # Dil modelini indirir
        self.model = fasttext.load_model(f'cc.{self.language_code}.{self.dim}.bin')  # İndirilen modeli yükler
        print("FastText model loaded successfully.")
    
    def get_embedding(self, word):
        """
        Retrieves the FastText embedding for a specific word.
        Belirli bir kelime için FastText gömme vektörünü getirir.
        
        :param word: Word to retrieve the embedding for (Gömme vektörünü almak istediğiniz kelime)
        :return: FastText embedding vector (FastText gömme vektörü)
        """
        return self.model.get_word_vector(word)
    
    def get_subword_embedding(self, word):
        """
        Retrieves the subword-based embedding for a specific word (useful for out-of-vocabulary words).
        Belirli bir kelime için alt kelime (subword) temelli gömme vektörünü getirir (Kelime dağarcığı dışındaki kelimeler için yararlı).
        
        :param word: Word to retrieve the subword-based embedding for (Subword temelli gömme vektörünü almak istediğiniz kelime)
        :return: FastText subword-based embedding vector (FastText subword temelli gömme vektörü)
        """
        return self.model.get_word_vector(word)
    
    def find_similar_words(self, word, k=5):
        """
        Finds the top k most similar words to the given word using FastText embeddings.
        FastText gömmelerini kullanarak belirli bir kelimeye en benzer k kelimeyi bulur.
        
        :param word: Word to find similar words for (Benzer kelimeleri bulmak istediğiniz kelime)
        :param k: Number of similar words to retrieve (Kaç benzer kelime getirileceği)
        :return: List of top k similar words (En benzer k kelime listesi)
        """
        return self.model.get_nearest_neighbors(word, k=k)

# Example usage (Örnek kullanım)
if __name__ == "__main__":
    # FastText Embeddings sınıfını başlat
    fasttext_embeddings = FastTextEmbeddings(language_code='en', dim=300)
    
    # 'king' kelimesinin gömme vektörünü al
    king_embedding = fasttext_embeddings.get_embedding('king')
    print(f"Embedding for 'king': {king_embedding}")
    
    # OOV (kelime dağarcığı dışı) bir kelimenin subword temelli gömme vektörünü al
    oov_word = 'technolgy'  # Yanlış yazılmış bir kelime (Doğrusu 'technology')
    oov_embedding = fasttext_embeddings.get_subword_embedding(oov_word)
    print(f"Subword-based embedding for OOV word '{oov_word}': {oov_embedding}")
    
    # 'king' kelimesine en benzer 5 kelimeyi bul
    similar_words = fasttext_embeddings.find_similar_words('king', k=5)
    print(f"Top 5 words similar to 'king': {similar_words}")
