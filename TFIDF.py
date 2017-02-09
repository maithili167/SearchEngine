'''
Name: Maithili Deshpande
Student ID: 1001230528
Programming assignment 1
'''


import os
from collections import Counter
import math
from nltk import FreqDist
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import datetime


tokens={}
documents={}
idf_dict={}
postingList={}
scored={}
length={}
query_vector={}
docVector={}
document_vector={}
topK ={}
upperBound ={}

corpusroot = './presidential_debates'
N=30
tokenizer =  RegexpTokenizer(r'[a-zA-Z]+')
stemmer = PorterStemmer()
#All stopwords are cached into cachedStopWords variable
cachedStopWords = stopwords.words("english")

''' Tokenization,stemming is the part of preprocessing.
This function also takes care of calculating frequency of every token in the input document/query '''
def preprocessData(doc):
    doc = doc.lower()
    tokens = tokenizer.tokenize(doc)
    filtered_words = [stemmer.stem(word) for word in tokens if word not in cachedStopWords]
    return FreqDist(filtered_words)


#This function calculates IDF vector for a collection of documents
def calculateIDF(sample):
    if sample not in idf_dict:
        idf_dict[sample] = 1
    else:
        idf_dict[sample] += 1

#Below function reads file from the specified location calculates tf and idf for each document
def readFiles():
    for filename in os.listdir(corpusroot):
        file = open(os.path.join(corpusroot, filename), "r", encoding='UTF-8')
        doc = file.read()
        file.close()
        wordfrequency=preprocessData(doc)
        tf_dict = {}
        for sample in wordfrequency:
            calculateIDF(sample)
            #calculates tf for a specific token
            tf_dict[sample]=1 + math.log10(wordfrequency[sample])
        # creates tf vector for a document
        documents[filename]=tf_dict

#This function calculates idf of a specific token
def getidf(sample):
    if sample not in idf_dict:
        return -1
    return (math.log10((N/idf_dict[sample])))


''' This function creates document vectors in which each document is a key and values are
 token and respective token tf-idf weights'''
def createDocumentVector():
    for key,value in documents.items():
        tf_idf = {}
        for keytoken,tWeight in value.items():
            tf_idf[keytoken]=documents[key][keytoken] * getidf(keytoken)
        document_vector[key] = tf_idf


#This function calculates magnitude of a vector
def mag(x):
    return math.sqrt(sum(j**2 for i,j in x.items()))

#This function normalizes the vector to convert it into a unit vector
def normalizeVector():
    #calculate length of a vector
    for key,value in document_vector.items():
        length[key] = mag(document_vector.get(key))
    # Divide every token weight by length of a vector
    for key,value in document_vector.items():
        normalize={}
        for keytoken,tWeight in value.items():
            normalize[keytoken] = document_vector.get(key)[keytoken] / length[key]
        document_vector[key]=normalize

''' Create a list of token with key as a token and values as number of documents containing
specified token and its weight'''
def createPostingList():
    for key,value in document_vector.items():
        for keytoken,tWeight in value.items():
            if keytoken not in postingList:
                postingList[keytoken]={}
                postingList[keytoken][key]=tWeight
            else:
                postingList[keytoken][key] = tWeight

#This function finds 10 documents with highest weight from the posting list
def findTopK(key_query):
    if key_query in postingList:
        topK[key_query] = dict(Counter(postingList.get(key_query)).most_common(10))
        return min(topK[key_query].values())


'''Add score to document if the query token is present in the document else
add upper bound score for the specified key word'''
def calculateScore(key_query, doc):
    if key_query not in docVector.get(doc):
        return scored[doc] + query_vector[key_query] * upperBound[key_query]
    else:
        return scored[doc] + docVector.get(doc)[key_query] * query_vector[key_query]

#This function takes query key words as an input and returns the document with highest score
def query(query):
    #preprocess the query
    wordfrequency=preprocessData(query)

    #calculate term frequency for query
    for sample in wordfrequency:
        query_vector[sample]=1+math.log10(wordfrequency[sample])

    #normalize query vector
    lenghQuery=mag(query_vector)
    for keyq in query_vector:
        query_vector[keyq]=query_vector[keyq]/lenghQuery

    #find upper bound of the documents with respect to each query token
    for key_query in query_vector:
        if key_query in postingList:
            upperBound[key_query] = findTopK(key_query)

    #Create a document vector for top 10 documents
    for keytoken,value in topK.items():
        for keydoc,tWeight in value.items():
            if keydoc not in docVector:
                scored[keydoc]=0
                docVector[keydoc]={}
                docVector[keydoc][keytoken]=tWeight
            else:
                docVector[keydoc][keytoken] = tWeight

    #Calculate cosine score for every document
    for key_query in query_vector:
        for doc,value in docVector.items():
            if key_query in postingList:
                scored[doc]=calculateScore(key_query,doc)


    actualScore=True
    #Find document with the highest score
    finalDoc=dict(Counter(scored).most_common(1))
    '''check if all the query strings are present in the document in order to determine
    if the calculated score is an actual score'''
    for doc,value in finalDoc.items():
        for key_query in query_vector:
            if key_query not in docVector.get(doc) and key_query in postingList:
                actualScore=False
                break

    #If no document exist with the entered keywords
    if not scored:
        return ("None",0)
    else:
        if actualScore:
            for key,value in dict(Counter(scored).most_common(1)).items():
                return (key,value)
        else:
            # If the score is not an actual score, fetch more than 10 documents
            return ("fetch more", 0)


'''Function takes document and query token as an input and returns the respective
 token weight in that document'''
def getweight(key, keytoken):
    if keytoken not in document_vector.get(key):
        return 0
    else:
        return document_vector.get(key)[keytoken]

if __name__ == '__main__':
    # print(datetime.datetime.now())
    readFiles()
    createDocumentVector()
    normalizeVector()
    createPostingList()

# print("%.12f" % getidf("agenda"))
print("(%s, %.12f)" % query("Terror Attack"))
print("(%s, %.12f)" % query("Terror Attack Aniket"))
print("(%s, %.12f)" % query("health insurance wall street"))
# print("(%s, %.12f)" % query("Terror Attack health"))
# print("%.12f" % getidf("reason"))
# print("%.12f" % getidf("kennedi"))
# print("%.12f" % getidf("vector"))
# print("%.12f" % getweight("1960-10-13.txt","kennedi"))
# print("%.12f" % getweight("2012-10-16.txt","hispan"))
# print(datetime.datetime.now())


#REFERENCE

'''
[1] [fdist1 = FreqDist(text1)]
    This was adopted from a post on: http: // www.nltk.org / book / ch01.html

[2] [text = ' '.join([word for word in text.split() if word not in cachedStopWords])]
    This was adopted from a post on: http://stackoverflow.com/questions/19560498/faster-way-to-remove-stop-words-in-python

[3] All the loops for the dictionary are referred from the post on:
    https://www.quora.com/In-Python-can-we-add-a-dictionary-inside-a-dictionary-If-yes-how-can-we-access-the-inner-dictionary-using-the-key-in-the-primary-dictionary

[4] [Counter('abracadabra').most_common(3)]
    This is referred from the python documentation from https://docs.python.org/2/library/collections.html#collections.Counter

[5] [math.sqrt(sum(j**2 for i,j in x.items())]
    This code snippet is referred from http://stackoverflow.com/questions/9171158/how-do-you-get-the-magnitude-of-a-vector-in-numpy

[6] [lowest = min(city.values())]
    This code snippet is referred from http://stackoverflow.com/questions/27009250/finding-the-lowest-value-in-dictionary

[7] https://nbviewer.jupyter.org/url/crystal.uta.edu/~cli/cse5334/ipythonnotebook/P1.ipynb

[8] [TSK] Pang-Ning Tan, Michael Steinbach, and Vipin Kumar. Introduction to Data Mining, Addison-Wesley, 2006. ISBN 0-321-32136-7.


'''

