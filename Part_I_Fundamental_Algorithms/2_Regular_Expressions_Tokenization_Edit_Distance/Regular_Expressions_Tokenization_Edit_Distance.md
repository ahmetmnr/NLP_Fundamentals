# Regular Expressions, Tokenization, and Edit Distance

## Overview
This section contains Python scripts for three fundamental techniques in Natural Language Processing (NLP): **Regular Expressions**, **Tokenization**, and **Edit Distance**. Each of these techniques is crucial for text processing tasks. The provided scripts demonstrate how to use them effectively in Python.

Bu bölüm, Doğal Dil İşleme (NLP) alanında kullanılan üç temel tekniğin Python betiklerini içerir: **Düzenli İfadeler (Regex)**, **Tokenizasyon** ve **Edit Mesafesi**. Her biri, metin işleme görevleri için kritik öneme sahiptir. Sağlanan betikler, bu tekniklerin Python'da nasıl kullanıldığını göstermektedir.

## File Structure
1. **`regex_example.py`**: Demonstrates various regular expression techniques.
2. **`tokenization_example.py`**: Shows how to tokenize text into words and sentences.
3. **`edit_distance_example.py`**: Calculates the edit distance between two words or sentences.

---

### `regex_example.py`

This script covers the basics and advanced usage of **regular expressions** in Python using the `re` library. Regular expressions are useful for pattern matching, searching, and string manipulation tasks.

Bu betik, Python'da `re` kütüphanesini kullanarak **düzenli ifadelerin** (regex) temel ve ileri kullanımını ele alır. Düzenli ifadeler, desen eşleştirme, arama ve dizeleri işleme görevlerinde kullanışlıdır.

#### Key Examples:
1. **Basic Pattern Matching:** Searching for simple patterns like specific words.
2. **Character Sets:** Finding characters from a defined set, like vowels.
3. **Quantifiers:** Using `+`, `*`, and `?` to handle repetition and optional characters.
4. **Anchors:** Using `^` and `$` to match patterns at the start or end of a line.
5. **Lookahead and Lookbehind:** Performing advanced assertions to check patterns before or after another string.

#### Example Usage:

```python
import re
pattern = r'\d+'
text = "I have 100 apples and 200 oranges."
numbers = re.findall(pattern, text)
print(numbers)  # Output: ['100', '200']

tokenization_example.py
This script focuses on tokenization, the process of splitting text into smaller units such as words or sentences. Tokenization is the first step in most NLP tasks, including text analysis and machine learning.

Bu betik, metinleri kelimelere veya cümlelere ayırma işlemi olan tokenizasyon üzerine odaklanır. Tokenizasyon, çoğu NLP görevinde ilk adımdır ve metin analizi ile makine öğrenmesi için gereklidir.

Key Examples:
Word Tokenization: Splitting a sentence into individual words.
Sentence Tokenization: Dividing text into sentences.
Regex-Based Tokenization: Using regular expressions to tokenize based on custom patterns.
Tweet Tokenization: Tokenizing informal text, such as tweets.
Subword Tokenization: Breaking words into subword units, useful for language models.
Example Usage:
from nltk.tokenize import word_tokenize
text = "Natural Language Processing is fascinating!"
tokens = word_tokenize(text)
print(tokens)  # Output: ['Natural', 'Language', 'Processing', 'is', 'fascinating', '!']
