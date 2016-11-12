from nltk.corpus import wordnet as wn 

print(wn.synsets('dog')[0].hypernyms())