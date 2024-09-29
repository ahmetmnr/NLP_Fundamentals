from transformers import BertModel, BertTokenizer
import torch

class BERTEmbeddings:
    def __init__(self, model_name='bert-base-uncased'):
        """
        Initializes the BERT model and tokenizer from Hugging Face's transformers library.
        Hugging Face'in transformers kütüphanesinden BERT modelini ve tokenizer'ı başlatır.
        
        :param model_name: Pretrained BERT model name (Hazır eğitilmiş BERT modelinin adı)
        """
        self.model_name = model_name
        print(f"Loading BERT model: {self.model_name}...")
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.model = BertModel.from_pretrained(self.model_name)
        print(f"BERT model {self.model_name} loaded successfully.")
    
    def get_embeddings(self, sentence, word):
        """
        Retrieves the BERT embedding for a specific word in a sentence.
        Bir cümlede belirli bir kelime için BERT gömme vektörünü getirir.
        
        :param sentence: The sentence containing the word (Kelimeyi içeren cümle)
        :param word: The word to retrieve the embedding for (Gömme vektörünü almak istediğiniz kelime)
        :return: BERT embedding vector for the word (Kelime için BERT gömme vektörü)
        """
        # Tokenize the sentence
        inputs = self.tokenizer(sentence, return_tensors='pt', add_special_tokens=True)
        tokenized_input = self.tokenizer.tokenize(sentence)
        
        # Get model output
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Get embeddings from the last hidden state
        embeddings = outputs.last_hidden_state
        
        # Find the index of the word in the tokenized sentence
        try:
            token_idx = tokenized_input.index(word)
        except ValueError:
            # Eğer kelime cümlede bulunamazsa OOV olabilir, subword token olabilir
            print(f"Word '{word}' not found in tokenized sentence.")
            return None
        
        # Kelimenin tüm tokenlarını alın (örn. "playing" -> "play", "##ing")
        token_ids = [i for i, token in enumerate(tokenized_input) if token.startswith(word) or token == word]
        
        # Token embeddinglerini birleştirin (ortalama alınabilir)
        word_embedding = torch.mean(embeddings[0, token_ids], dim=0)
        return word_embedding.detach().numpy()

    def get_sentence_embedding(self, sentence):
        """
        Retrieves the sentence embedding by averaging all word embeddings in the sentence.
        Cümledeki tüm kelime gömmelerinin ortalamasını alarak cümle için gömme vektörü oluşturur.
        
        :param sentence: The sentence to retrieve the embedding for (Gömme vektörünü almak istediğiniz cümle)
        :return: BERT embedding vector for the entire sentence (Cümle için BERT gömme vektörü)
        """
        # Tokenize the sentence
        inputs = self.tokenizer(sentence, return_tensors='pt', add_special_tokens=True)
        
        # Get model output
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Cümlenin tüm kelimeleri için gömmelerden ortalama alın
        sentence_embedding = torch.mean(outputs.last_hidden_state, dim=1)
        return sentence_embedding[0].detach().numpy()

# Example usage (Örnek kullanım)
if __name__ == "__main__":
    # Parametrik olarak farklı BERT modellerini seçebiliriz
    bert_model_name = 'bert-base-uncased'  # Başka bir model için 'bert-large-uncased' veya 'bert-base-multilingual-cased'
    
    # BERT Embeddings sınıfını başlat
    bert_embeddings = BERTEmbeddings(model_name=bert_model_name)
    
    # Bir cümledeki bir kelimenin bağlama dayalı BERT gömme vektörünü al
    sentence = "The king is a wise ruler."
    word = "king"
    word_embedding = bert_embeddings.get_embeddings(sentence, word)
    print(f"Embedding for word '{word}' in sentence: {word_embedding}")
    
    # Bir cümle için BERT gömme vektörü al
    sentence_embedding = bert_embeddings.get_sentence_embedding(sentence)
    print(f"Sentence embedding for '{sentence}': {sentence_embedding}")
