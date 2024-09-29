import os
import requests
import zipfile
import numpy as np

class GloVeEmbeddings:
    def __init__(self, glove_url, glove_dir, glove_file):
        """
        Initializes the GloVe embeddings by downloading and extracting the GloVe model.
        GloVe modelini indirip çıkartarak gömme vektörlerini başlatır.
        
        :param glove_url: URL to the GloVe model zip file (GloVe model zip dosyasının URL'si)
        :param glove_dir: Directory to save the extracted GloVe files (GloVe dosyalarının kaydedileceği dizin)
        :param glove_file: GloVe text file to load embeddings from (Gömme vektörlerini yüklemek için GloVe dosyası)
        """
        self.glove_url = glove_url
        self.glove_dir = glove_dir
        self.glove_file = glove_file
        self.word_to_index = {}

        # Dizin yoksa oluştur
        if not os.path.exists(self.glove_dir):
            os.makedirs(self.glove_dir)
        
        # GloVe dosyası yoksa indir ve çıkart
        if not os.path.exists(os.path.join(self.glove_dir, self.glove_file)):
            self._download_and_extract_glove()
        
        # GloVe dosyasını yükle
        self.embeddings = self._load_glove_embeddings(os.path.join(self.glove_dir, self.glove_file))
    
    def _download_and_extract_glove(self):
        """
        Downloads and extracts the GloVe model if not already downloaded.
        GloVe modelini indirir ve çıkartır (eğer henüz indirilmemişse).
        """
        glove_zip_path = os.path.join(self.glove_dir, 'glove.6B.zip')

        # GloVe zip dosyasını indir
        if not os.path.exists(glove_zip_path):
            print(f"Downloading GloVe from {self.glove_url}...")
            response = requests.get(self.glove_url, stream=True)
            with open(glove_zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=128):
                    f.write(chunk)
            print(f"Downloaded GloVe model to {glove_zip_path}")
        
        # Zip dosyasını çıkart
        print(f"Extracting GloVe model to {self.glove_dir}...")
        with zipfile.ZipFile(glove_zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.glove_dir)
        print("GloVe model extracted.")

    def _load_glove_embeddings(self, glove_file):
        """
        Loads GloVe embeddings from a text file.
        GloVe gömmelerini bir metin dosyasından yükler.
        
        :param glove_file: Path to the GloVe embeddings file (GloVe gömme dosyasının yolu)
        :return: Embeddings matrix (Gömme matrisi)
        """
        embeddings = {}
        with open(glove_file, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                values = line.split()
                word = values[0]
                vector = np.array(values[1:], dtype='float32')
                embeddings[word] = vector
                self.word_to_index[word] = idx
        return embeddings
    
    def get_embedding(self, word):
        """
        Retrieves the GloVe embedding for a specific word.
        Belirli bir kelime için GloVe gömme vektörünü getirir.
        
        :param word: Word to retrieve the embedding for (Gömme vektörünü almak istediğiniz kelime)
        :return: GloVe embedding vector (GloVe gömme vektörü)
        """
        return self.embeddings.get(word)

# Example usage (Örnek kullanım)
if __name__ == "__main__":
    glove_url = "https://nlp.stanford.edu/data/glove.6B.zip"
    glove_dir = "Part_I_Fundamental_Algorithms/6_Vector_Semantics_and_Embeddings/glove"
    glove_file = "glove.6B.100d.txt"  # 100 boyutlu GloVe vektör dosyasını seçiyoruz

    # GloVe Embeddings sınıfını başlat
    glove = GloVeEmbeddings(glove_url, glove_dir, glove_file)
    
    # 'king' kelimesinin gömme vektörünü al
    king_embedding = glove.get_embedding('king')
    print(f"Embedding for 'king': {king_embedding}")
