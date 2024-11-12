#BIGRAM
from nltk import ngrams
sentence='The greatest glory in living lies not in never falling, but in rising every time we fall'
n=2
bigram=ngrams(sentence.split(),n)
for gram in bigram:
    print(gram)
#TRIGRAM
sentence='The greatest glory in living lies not in never falling, but in rising every time we fall'
n=3
trigram=ngrams(sentence.split(),n)
for grams in trigram:
    print(grams)