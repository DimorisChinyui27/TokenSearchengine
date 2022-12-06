from rank_bm25 import BM25L
#from rank_bm25 import BM25LPlus  #Large and Plus are good for documents with uneven lenght of paragraphs. But Plus is the most recent 
#and most likely the best but I am not sure
#from rank_bm25 import BM25Okapi #Okapi is limited and works only for smaller docs but is less resource intensive
#BM algorithms are tokenized search engines not semantic but can be combine with semantic search engines to speed them up since 
# semantic search engines are more resource intensive and slow down with larger text bodies
import string

corpus = [
    "Hello there good man!",
    "It is quite windy in London",
    "How is the weather today?"
]

def remove_puncts(input_string, string):
    return input_string.translate(str.maketrans('', '', string.punctuation)).lower()

cleaned_corpus = []
for doc in corpus:
    #print(doc)
    clean_doc = remove_puncts(doc, string)
    #print(clean_doc)
    cleaned_corpus.append(clean_doc)

tokenized_clean_corpus = []
for doc in cleaned_corpus:
    #print(doc)
    doc = doc.split()
    tokenized_clean_corpus.append(doc)

bm25 = BM25L(tokenized_clean_corpus)

query = "women live in the weather"
tokenized_query = remove_puncts(query, string).split(" ")
print(bm25.get_top_n(tokenized_query, corpus, n=1))

