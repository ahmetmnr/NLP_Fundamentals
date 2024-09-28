from collections import defaultdict
import nltk
from nltk.util import ngrams
from nltk.tokenize import word_tokenize

# Sample text corpus (Örnek metin corpus'u)
corpus = "I love natural language processing. Natural language processing is fun. I love machine learning."

# Tokenize the text into words (Metni kelimelere böl)
nltk.download('punkt')
tokens = word_tokenize(corpus.lower())

# Function to calculate n-grams (N-gram'ları hesaplayan fonksiyon)
def calculate_ngrams(tokens, n):
    """ Calculate n-grams from the tokenized text. (Tokenize edilmiş metinden n-gram'ları hesapla.) """
    n_grams = list(ngrams(tokens, n))
    return n_grams

# Unigrams, Bigrams, Trigrams (Unigram'lar, Bigram'lar, Trigram'lar)
unigrams = calculate_ngrams(tokens, 1)
bigrams = calculate_ngrams(tokens, 2)
trigrams = calculate_ngrams(tokens, 3)

# Function to calculate N-gram frequencies (N-gram frekanslarını hesaplayan fonksiyon)
def calculate_ngram_frequencies(n_grams):
    """ Calculate the frequency of each N-gram. (Her N-gram'ın frekansını hesapla.) """
    ngram_freqs = defaultdict(int)
    for ngram in n_grams:
        ngram_freqs[ngram] += 1
    return ngram_freqs

# Calculate frequencies for unigrams, bigrams, and trigrams (Unigram, bigram ve trigram frekanslarını hesaplayın)
unigram_freqs = calculate_ngram_frequencies(unigrams)
bigram_freqs = calculate_ngram_frequencies(bigrams)
trigram_freqs = calculate_ngram_frequencies(trigrams)

# Function to calculate MLE for bigrams and trigrams
# Bigramlar ve trigramlar için MLE hesaplayan fonksiyon
def calculate_mle_ngram_probabilities(n_gram_freqs, lower_order_gram_freqs):
    """ 
    Calculate Maximum Likelihood Estimation (MLE) for N-grams. 
    N-gram'lar için Maximum Likelihood Estimation (MLE) hesapla.
    """
    ngram_probs = {}
    for ngram, count in n_gram_freqs.items():
        # Calculate MLE: P(word_n | word_n-1) = Count(word_n-1, word_n) / Count(word_n-1)
        ngram_probs[ngram] = count / lower_order_gram_freqs[ngram[:-1]]
    return ngram_probs

# MLE for bigrams (Bigram'lar için MLE)
bigram_mle_probs = calculate_mle_ngram_probabilities(bigram_freqs, unigram_freqs)

# MLE for trigrams (Trigram'lar için MLE)
trigram_mle_probs = calculate_mle_ngram_probabilities(trigram_freqs, bigram_freqs)

# Print MLE probabilities (MLE olasılıklarını yazdırma)
print("Bigram MLE Probabilities:")
for bigram, prob in bigram_mle_probs.items():
    print(f"{bigram}: {prob}")

print("\nTrigram MLE Probabilities:")
for trigram, prob in trigram_mle_probs.items():
    print(f"{trigram}: {prob}")

# Function to calculate sequence probability using MLE
# MLE kullanarak bir dizinin olasılığını hesaplayan fonksiyon
def calculate_mle_sequence_probability(sequence, ngram_probs, n):
    """ 
    Calculate the probability of a given sequence using the MLE N-gram model. 
    MLE N-gram modelini kullanarak verilen bir dizinin olasılığını hesapla.
    """
    tokens = word_tokenize(sequence.lower())
    sequence_ngrams = list(ngrams(tokens, n))
    prob = 1.0
    for ngram in sequence_ngrams:
        prob *= ngram_probs.get(ngram, 1e-5)  # Use a small probability for unseen N-grams (Görülmeyen N-gram'lar için küçük bir olasılık kullanın)
    return prob

# Example sequence probability using Bigram MLE (Bigram MLE ile örnek dizi olasılığı)
sequence = "I love natural language"
sequence_prob = calculate_mle_sequence_probability(sequence, bigram_mle_probs, 2)
print(f"\nBigram MLE Probability of the sequence '{sequence}': {sequence_prob}")
