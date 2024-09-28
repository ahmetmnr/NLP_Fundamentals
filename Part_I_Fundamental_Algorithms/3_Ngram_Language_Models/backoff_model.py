import random
from collections import defaultdict
import nltk
from nltk.util import ngrams
from nltk.tokenize import word_tokenize

# Sample text corpus for training (Eğitim için örnek metin corpus'u)
corpus = "I love natural language processing. Natural language processing is fun. I love machine learning."

# Tokenize the text into words (Metni kelimelere böl)
nltk.download('punkt')
tokens = word_tokenize(corpus.lower())

# Function to calculate n-grams (N-gram'ları hesaplayan fonksiyon)
def calculate_ngrams(tokens, n):
    """Calculate n-grams from the tokenized text. (Tokenize edilmiş metinden n-gram'ları hesapla.)"""
    return list(ngrams(tokens, n))

# Unigrams, Bigrams, Trigrams (Unigram'lar, Bigram'lar, Trigram'lar)
unigrams = calculate_ngrams(tokens, 1)
bigrams = calculate_ngrams(tokens, 2)
trigrams = calculate_ngrams(tokens, 3)

# Function to calculate N-gram frequencies (N-gram frekanslarını hesaplayan fonksiyon)
def calculate_ngram_frequencies(n_grams):
    """Calculate the frequency of each N-gram. (Her N-gram'ın frekansını hesapla.)"""
    ngram_freqs = defaultdict(int)
    for ngram in n_grams:
        ngram_freqs[ngram] += 1
    return ngram_freqs

# Calculate frequencies for unigrams, bigrams, and trigrams (Unigram, bigram ve trigram frekanslarını hesaplayın)
unigram_freqs = calculate_ngram_frequencies(unigrams)
bigram_freqs = calculate_ngram_frequencies(bigrams)
trigram_freqs = calculate_ngram_frequencies(trigrams)

# Vocabulary size (Kelime dağarcığının büyüklüğü)
vocab_size = len(set(tokens))

# Function for Stupid Backoff (Stupid Backoff fonksiyonu)
# If the trigram is not found, it falls back to bigram, and then to unigram.
# Trigram bulunamazsa bigram'a, o da bulunamazsa unigram'a geri döner.
def stupid_backoff(ngram, unigram_freqs, bigram_freqs, trigram_freqs, alpha=0.4):
    """ 
    Stupid Backoff algorithm for N-gram probabilities.
    N-gram olasılıkları için Stupid Backoff algoritması.
    """
    # Check trigram first (Önce trigram'ı kontrol et)
    if ngram in trigram_freqs:
        return trigram_freqs[ngram] / bigram_freqs[ngram[:-1]]
    
    # Check bigram if trigram is not found (Trigram bulunamazsa bigram'a bak)
    elif ngram[1:] in bigram_freqs:
        return alpha * (bigram_freqs[ngram[1:]] / unigram_freqs[ngram[1:-1]])
    
    # Fallback to unigram (Unigram'a geri dön)
    else:
        return alpha * (unigram_freqs[ngram[-1],] / vocab_size)

# Example backoff for a sequence (Bir dizinin backoff ile hesaplanması)
def calculate_sequence_backoff(sequence, unigram_freqs, bigram_freqs, trigram_freqs):
    """Calculate the probability of a sequence using Stupid Backoff. 
    (Stupid Backoff kullanarak bir dizinin olasılığını hesaplayın.)"""
    
    tokens = word_tokenize(sequence.lower())
    trigrams = list(ngrams(tokens, 3))
    prob = 1.0
    
    for trigram in trigrams:
        prob *= stupid_backoff(trigram, unigram_freqs, bigram_freqs, trigram_freqs)
    
    return prob

# Calculate smoothed bigram and trigram frequencies (Düzeltmeli bigram ve trigram frekanslarını hesaplayın)
unigram_probs = {unigram: count / sum(unigram_freqs.values()) for unigram, count in unigram_freqs.items()}
bigram_probs = {bigram: count / sum(bigram_freqs.values()) for bigram, count in bigram_freqs.items()}
trigram_probs = {trigram: count / sum(trigram_freqs.values()) for trigram, count in trigram_freqs.items()}

# Test sequence (Test cümlesi)
test_sentence = "I love natural language processing"

# Calculate probability of the sequence using Stupid Backoff (Stupid Backoff kullanarak dizinin olasılığını hesapla)
backoff_prob = calculate_sequence_backoff(test_sentence, unigram_freqs, bigram_freqs, trigram_freqs)
print(f"Stupid Backoff Probability for the sentence '{test_sentence}': {backoff_prob}")

