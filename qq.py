#UNIGRAM
from nltk import ngrams
sentence='artificial intelligence is a branch of computer science'
n=1
unigrams=ngrams(sentence.split(),n)
for items in unigrams:
    print(items)
#BIGRAM
from nltk import ngrams
sentence='artificial intelligence is a branch of computer science'
n=2
unigrams=ngrams(sentence.split(),n)
for items in unigrams:
    print(items)
#TRIGRAM
from nltk import ngrams
sentence='artificial intelligence is a branch of computer science'
n=3
unigrams=ngrams(sentence.split(),n)
for items in unigrams:
    print(items)