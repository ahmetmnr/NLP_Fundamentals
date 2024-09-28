# Importing necessary library (Gerekli kütüphaneyi içe aktarma)
from nltk.metrics import edit_distance

# Example 1: Basic Edit Distance (Temel Edit Mesafesi)
# Edit distance measures the minimum number of operations required to convert one string to another.
# Edit mesafesi, bir dizeyi başka bir dizeye dönüştürmek için gereken minimum işlem sayısını ölçer.
word1 = "kitten"
word2 = "sitting"

# Calculate the edit distance between 'kitten' and 'sitting'
# 'kitten' ve 'sitting' arasındaki edit mesafesini hesapla
distance = edit_distance(word1, word2)
print(f"Edit distance between '{word1}' and '{word2}': {distance}")


# Example 2: Edit Distance with Insertions, Deletions, and Substitutions (Ekleme, Silme ve Değiştirme ile Edit Mesafesi)
word3 = "intention"
word4 = "execution"

# Calculate the edit distance between 'intention' and 'execution'
# 'intention' ve 'execution' arasındaki edit mesafesini hesapla
distance2 = edit_distance(word3, word4)
print(f"Edit distance between '{word3}' and '{word4}': {distance2}")


# Example 3: Edit Distance with Custom Costs (Özel Maliyetlerle Edit Mesafesi)
# You can assign different costs to insertions, deletions, and substitutions.
# Ekleme, silme ve değiştirme işlemlerine farklı maliyetler atayabilirsiniz.
word5 = "flaw"
word6 = "lawn"

# Custom edit distance with substitution cost of 2
# Değiştirme maliyetinin 2 olduğu özel edit mesafesi
distance3 = edit_distance(word5, word6, substitution_cost=2)
print(f"Custom edit distance between '{word5}' and '{word6}' (substitution cost = 2): {distance3}")


# Example 4: Case-Sensitivity in Edit Distance (Büyük/Küçük Harf Duyarlılığı)
# Edit distance is case-sensitive by default.
# Edit mesafesi varsayılan olarak büyük/küçük harf duyarlıdır.
word7 = "Apple"
word8 = "apple"

# Calculate the edit distance considering case-sensitivity
# Büyük/küçük harf duyarlılığını göz önünde bulundurarak edit mesafesini hesaplayın
distance4 = edit_distance(word7, word8)
print(f"Case-sensitive edit distance between '{word7}' and '{word8}': {distance4}")


# Example 5: Edit Distance in Real-World Scenarios (Gerçek Dünya Senaryolarında Edit Mesafesi)
# Edit distance is useful in spell checking, DNA sequence analysis, and other applications.
# Edit mesafesi, yazım denetimi, DNA dizilim analizi ve diğer uygulamalarda kullanışlıdır.
sentence1 = "I love natural language processing"
sentence2 = "I love natural language programming"

# Calculate the edit distance between two sentences
# İki cümle arasındaki edit mesafesini hesaplayın
distance5 = edit_distance(sentence1, sentence2)
print(f"Edit distance between sentences: {distance5}")


# Example 6: Optimal String Alignment (Optimal Dizi Hizalaması)
# Edit distance with transpositions (değiş tokuşlarla edit mesafesi)
# Optimal string alignment allows transpositions, meaning it accounts for swapped adjacent characters.
# Optimal dizi hizalaması, bitişik karakterlerin yer değiştirmesini hesaba katar.
word9 = "abcdef"
word10 = "abcfde"

# Edit distance with transpositions
# Değiş tokuşları hesaba katan edit mesafesi
distance6 = edit_distance(word9, word10, transpositions=True)
print(f"Edit distance with transpositions between '{word9}' and '{word10}': {distance6}")


