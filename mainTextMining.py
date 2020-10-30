from sklearn.feature_extraction.text import CountVectorizer  # library for tokenization & vectorization
from sklearn.naive_bayes import MultinomialNB
import pandas as pd

# you can create the dataset using StringIO
# N.B: it is important that there are no spaces between words

'''
from io import StringIO

dataset = StringIO("""text,class
oggi c'è un bel sole è caldo si sta bene al mare,ESTATE
Ho voglia di acqua e di mare,ESTATE
Che freddo con questa neve! si sta bene sotto le coperte,INVERNO
Le previsioni danno ghiaccio e neve,INVERNO
Mi piace la vendemmia… Il marrone delle foglie…,AUTUNNO
Le foglie stanno cadendo tutte… è proprio autunnno,AUTUNNO
Domenica se fa caldo faremo un bel picnic in mezzo ai fiori,PRIMAVERA
In questa stagione è bello camminare anche sotto le stelle,PRIMAVERA
""")
'''

'''data = pd.read_csv(dataset, sep=",")'''  # transform the dataset in a structured way

# you can also write the dataset in a excel file saved as ".csv"
# you can use the read_csv method to transform the dataset in a structured way

data = pd.read_csv("phrases_dataset.csv", sep=",") # venv\Lib\site-packages\sklearn\datasets\data\dataset_frasi.csv

print("\nDATASET: ")
print(data)

# separation of features (X) form classes (y)

X = data["text"]  # One-dimensional ndarray with axis labels
                    # X is an array that contains phrases

y = data["class"]  # One-dimensional ndarray with axis labels
                    # y is an array that contains data label

ordered_class_list = list(set(y))  # creating a list that contains all y's elements without repetition
ordered_class_list.sort()  # order the previous list

print("\nOredered Classess: ")
print(ordered_class_list)

vectorizer_train = CountVectorizer(min_df=0, binary=True)  # convert a collection of text documents to a matrix of token counts

vectorizer_train.fit(X)  # method used to learn form data

x_train_array = vectorizer_train.transform(X).toarray()  # transformation of X in  array representation

tokens = vectorizer_train.get_feature_names()  # array that contains all the words of all phrases

print("\nTokens: ")
print(tokens)

print("\nArray: ")
print(x_train_array)  # it contains all phrases expressed as a matrix of vectors

clf = MultinomialNB().fit(x_train_array, y)  # classifier --> the most used is Naive Bayes classifier

msg = "adoro giocare con l'acqua e la sabbia sotto il sole caldo"  # testing phrase 1

msg_array = vectorizer_train.transform([msg]).toarray()  # you need to convert the msg in vectorizer form

print("\nMSG_ARRAY: ")
print(msg_array, "\n")

class_probabilities = clf.predict_proba(msg_array)[0]  # calculating

print(list(zip(ordered_class_list, class_probabilities)))  # creating a list that is the merge of two lists

print("\nResult: ", clf.predict(msg_array))

msg1 = "autunno estate inverno primavera"  # testing phrase 2

msg_array1 = vectorizer_train.transform([msg1]).toarray()

print("\nMSG_ARRAY1: ")
print(msg_array1, "\n")

class_probabilities1 = clf.predict_proba(msg_array1)[0]  # calculation of the probability of belonging to classes
                                                            # '[0]' is inserted to indicate the first line (i.e vector) of the matrix
print(list(zip(ordered_class_list, class_probabilities1)))

print("\nResult: ", clf.predict(msg_array1))
