# Import necessary libraries (Gerekli kütüphaneleri içe aktar)
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize, regexp_tokenize, TweetTokenizer

# NLTK araçlarını indirme (Bu adım bir defalık gereklidir)
nltk.download('punkt')
nltk.download('punkt_tab')


# Example 1: Word Tokenization (Kelime Tokenizasyonu)
# Tokenizing a sentence into individual words.
# Bir cümleyi kelimelere ayırma.
text = "Natural Language Processing (NLP) is fascinating!"

# Basic word tokenization using NLTK
# NLTK kullanarak temel kelime tokenizasyonu
words = word_tokenize(text)
print(f"Word Tokenization: {words}")  # ['Natural', 'Language', 'Processing', '(', 'NLP', ')', 'is', 'fascinating', '!']

# Example 2: Sentence Tokenization (Cümle Tokenizasyonu)
# Tokenizing text into individual sentences.
# Bir metni cümlelere ayırma.
text = "Hello world. Welcome to the field of NLP. Let's explore tokenization."

# Sentence tokenization using NLTK
# NLTK kullanarak cümle tokenizasyonu
sentences = sent_tokenize(text)
print(f"Sentence Tokenization: {sentences}")
# ['Hello world.', 'Welcome to the field of NLP.', "Let's explore tokenization."]

# Example 3: Tokenization using Regular Expressions (Düzenli İfadelerle Tokenizasyon)
# Tokenization based on custom patterns.
# Özel kalıplara göre tokenizasyon.

text = "The price of the book is $19.99. Contact us at support@example.com."

# Regex-based tokenization: extract words, prices, and email addresses
# Düzenli ifadeye dayalı tokenizasyon: kelimeleri, fiyatları ve e-posta adreslerini ayırma
pattern = r'\w+|\$[\d\.]+|\S+@\S+'
tokens = regexp_tokenize(text, pattern)
print(f"Regex Tokenization: {tokens}")
# ['The', 'price', 'of', 'the', 'book', 'is', '$19.99', '.', 'Contact', 'us', 'at', 'support@example.com', '.']

# Example 4: Tokenizing Tweets and Informal Text (Tweet ve Resmi Olmayan Metinlerin Tokenizasyonu)
# Tokenizing tweets or short messages where informal language is common.
# Gayri resmi dilin yaygın olduğu tweetlerin veya kısa mesajların tokenizasyonu.

tweet = "I love #NLP! 😍 Let's tokenize this tweet @example #AI"

# Using TweetTokenizer from NLTK
# NLTK'deki TweetTokenizer'ı kullanma
tweet_tokenizer = TweetTokenizer()
tweet_tokens = tweet_tokenizer.tokenize(tweet)
print(f"Tweet Tokenization: {tweet_tokens}")
# ['I', 'love', '#NLP', '😍', "Let's", 'tokenize', 'this', 'tweet', '@example', '#AI']

# Example 5: Custom Tokenization Rules (Özelleştirilmiş Tokenizasyon Kuralları)
# You can create custom tokenizers for specific use cases.
# Özel kullanım durumları için özelleştirilmiş tokenizasyon kuralları oluşturabilirsiniz.

# A custom tokenizer for splitting on spaces, keeping punctuation as separate tokens.
# Boşluklara göre ayıran, noktalama işaretlerini ayrı tokenlar olarak tutan bir özelleştirilmiş tokenizer.
text = "Welcome to the world of NLP, where things move fast!"

custom_tokens = text.split()
print(f"Custom Tokenization: {custom_tokens}")
# ['Welcome', 'to', 'the', 'world', 'of', 'NLP,', 'where', 'things', 'move', 'fast!']

# Example 6: Handling Multi-Word Expressions (Çok Kelimeli İfadelerin İşlenmesi)
# Sometimes you may want to treat multi-word expressions as a single token.
# Bazen çok kelimeli ifadeleri tek bir token olarak ele almak isteyebilirsiniz.

# Example of multi-word expressions like "New York" or "San Francisco".
# "New York" veya "San Francisco" gibi çok kelimeli ifadelerle örnek.
multiword_text = "I live in New York and work in San Francisco."

# Defining a pattern to capture multi-word expressions
# Çok kelimeli ifadeleri yakalamak için bir kalıp tanımlama
multiword_pattern = r'\bNew York\b|\bSan Francisco\b|\w+'
multiword_tokens = regexp_tokenize(multiword_text, multiword_pattern)
print(f"Multi-Word Tokenization: {multiword_tokens}")
# ['I', 'live', 'in', 'New York', 'and', 'work', 'in', 'San Francisco', '.']

# Example 7: Subword Tokenization (Alt Kelime Tokenizasyonu)
# Tokenizing into parts of words or subwords, useful in some machine learning models.
# Kelimeleri alt parçalara veya alt kelimelere ayırma, bazı makine öğrenmesi modellerinde kullanışlıdır.

from nltk.tokenize import SyllableTokenizer
text = "naturalization"

# Using NLTK's SyllableTokenizer
# NLTK'nin SyllableTokenizer'ını kullanma
syllable_tokenizer = SyllableTokenizer()
syllables = syllable_tokenizer.tokenize(text)
print(f"Syllable Tokenization: {syllables}")
# ['na', 'tu', 'ra', 'li', 'za', 'tion']

# Example 8: Byte-Pair Encoding (BPE) - Alt Kelime Tokenizasyonu
# BPE is a subword tokenization algorithm often used in large language models.
# BPE, büyük dil modellerinde sıklıkla kullanılan bir alt kelime tokenizasyon algoritmasıdır.

# This would typically be done using libraries such as Hugging Face's `transformers`.
# Genellikle Hugging Face'in `transformers` gibi kütüphanelerle yapılır.
# Here's a pseudo-code example to illustrate the idea (Bu, fikri göstermek için bir psödo-koddur):


from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokens = tokenizer.tokenize("natural language processing")
print(tokens)


# Output: ['natural', 'language', 'processing'] -> ['nat', '##ural', 'lang', '##uage', 'process', '##ing']

# This example shows how BPE splits words into smaller subwords or character sequences.
# Bu örnek, BPE'nin kelimeleri daha küçük alt kelimelere veya karakter dizilerine nasıl böldüğünü gösterir.
