
# coding: utf-8

# In[200]:

import csv
import re
import numpy as np
import urllib
#import scipy.optimize
import random
import collections
import matplotlib.pyplot as plt
import string
import math
from sklearn import svm
from sklearn.metrics import mean_squared_error
import pandas
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
from nltk.stem.lancaster import LancasterStemmer


# In[201]:

with open('C:/Users/aravi/Desktop/158Assignment2/Amazon_Unlocked_Mobile.csv','r') as allData:
    allDataReader = csv.reader(allData, delimiter=",")
    myData = list(allDataReader)


# In[202]:

del myData[0]
random.shuffle(myData)


# In[203]:

for x in myData:
    x[4] = x[4].decode('utf8')


# In[204]:

train_reviews = myData[1:100001]
test_reviews = myData[100001:200001]
all_reviews = myData[1:400000]


# In[205]:

train_reviews[96228]


# In[206]:

price = 0
na = 0
for x in train_reviews:
    if(x[2] != 'NA'):
        price += float(x[2])
    else: 
        na += 1
print price, price / (len(train_reviews) - na)


# In[207]:

five = 0
count = 0
for x in train_reviews:
    count +=1
    if (int(x[3])==5):
        five += 1
print five * 1.0 / len(train_reviews)

print count


# In[208]:

def changeTypes(reviews):
    x = []
    y = []
    for r in reviews:
        if(r[2] != 'NA'):
            r[2] = float(r[2])
        if(r[5] == ''):
            r[5] =0
        else:
            r[5] = int(r[5])
        x.append([r[0],r[1],r[2], r[4], r[5]])
        y.append(r[3])
    return x , y


# In[209]:

X_train, y_train = changeTypes(train_reviews)
X_test, y_test = changeTypes(test_reviews)


# In[ ]:




# In[210]:

#Find Average Rating
average = 0
na = 0
for y in all_reviews: 
     average += int(y[3])

 
average = average * 1.0 / (len(all_reviews) - na)
print average


# In[120]:

one = 0; two = 0; three = 0;  four = 0; five = 0;
for z in y_train:
    y = int(z)
    if(y==1):
        one +=1
    elif(y==2):
        two += 1
    elif(y==3):
        three += 1
    elif(y==4):
        four += 1
    elif(y==5):
        five += 1


# In[51]:

objects = ('1', '2', '3', '4', '5')
y_pos = np.arange(len(objects))
performance = [one,two,three,four, five]
 
plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Number of Reviews')
plt.xlabel('Star Rating')
plt.title('Star Rating vs Number of Reviews for Training Data')
 
#plt.show()


# In[211]:

# Remove stop words, punctation, empty strings
def remove_stopwords( review, n):
    st = LancasterStemmer()
    stops = set(stopwords.words("english"))
    #review_text = re.sub("[^a-zA-Z]"," ", review[n])
    words = review[n].lower().split()
    words = [''.join(c for c in s if c not in string.punctuation) for s in words]
    words = [s for s in words if s]
    words = [st.stem(w) for w in words if not w in stops]
    review[n] = words

for x in X_train:
    remove_stopwords(x , 3)



# In[ ]:




# In[212]:

def string_to_int(rating):
    rating = int(rating)

for y in y_train:
    string_to_int(y)

reviews = []
for x in X_train:
    review = [w.decode('utf-8') for w in x[3]]
    reviews.append(x[3])


# In[213]:

#Test data Reviews need to be changed too
for x in X_test:
    remove_stopwords(x , 3)
    
reviews_test = []
for x in X_test:
    review = [w.decode('utf-8') for w in x[3]]
    reviews_test.append(x[3])
    
for y in y_test:
    string_to_int(y)


# In[232]:

CV = CountVectorizer(tokenizer=lambda i:i, max_df=0.95, min_df=2, max_features=200000, stop_words='english',  lowercase=False)
CV.fit(reviews)
CV_train_features = CV.transform(reviews)
CV_test_features = CV.transform(reviews_test)

CV_NB = MultinomialNB().fit(CV_train_features, y_train)
CV_predictions = CV_NB.predict(CV_test_features)



# In[262]:

print("simple bag of words with Stemming Multinomial classifier")
find_Accuracy(CV_predictions, y_test)
find_minority_accuracy(CV_predictions, y_test)


# In[266]:

#Tdidf modeling to achive feature matrix
#vectorizer = TfidfVectorizer( min_df=2, max_df=0.95, max_features = 200000, ngram_range = ( 1, 4 ),
                              #sublinear_tf = True )
tfidf = TfidfVectorizer(tokenizer=lambda i:i,max_features=200000, ngram_range = (1 , 4), lowercase=False)
tfidf.fit(reviews)
#feature matrix
train_Features = tfidf.transform(reviews)
test_Features = tfidf.transform(reviews_test)


# In[267]:

clfNB = MultinomialNB(alpha = 0.001)
clfNB.fit(train_Features, y_train)
predictions = clfNB.predict(test_Features)


# In[269]:

print("TFID ngrams 1-4")
find_Accuracy(predictions, y_test)
find_minority_accuracy(predictions, y_test)


# In[270]:


def find_Accuracy(predictions, y_test):
    AE = 0
    for y in range(0,len(predictions)):
        if(predictions[y]!=y_test[y]):
            AE +=1
    print("Total Accuracy : ", (len(y_test) - AE)*1.0 / len(y_test) )


# In[261]:

def find_minority_accuracy(predictions, y_test):
    acc = 0
    majority = 0
    for y in range(0, len(predictions)):
        if(y_test[y] != '5'):
            if(predictions[y]==y_test[y]):
                acc +=1
        else:
            majority += 1
    print("Minority Accuracy : ", (acc)*1.0 / (len(y_test)-majority) )    


# In[172]:

print "fitting NB"
model1 = MultinomialNB(alpha=0.001)
model1.fit( train_Features, y_train )
print "NB done"
print "Fitting SGD Classifier"
model2 = SGDClassifier(loss='modified_huber', n_iter=5, random_state=0, shuffle=True)
model2.fit( train_Features, y_train )
print "SGD Done"




# In[173]:

print "predicting"
pred_1 = model1.predict( test_Features)
pred_2 = model2.predict( test_Features)
pred_3 = model3.predict( test_Features)

print "done"


# In[174]:

print "MultinomialNB"
find_Accuracy(pred_1, y_test)
print "SGD Classifier"
find_Accuracy(pred_2, y_test)



# In[223]:

AE = 0
for y in range(0, len(y_test)):
    if(int(y_test[y]) != 5):
        AE +=1

print("Accuracy : ", (len(y_test) - AE)*1.0 / len(y_test) )


# In[ ]:

print "fitting RFC"
RFC = RandomForestClassifier(verbose=3, class_weight={'1':0.22, '2':0.22,'3':0.22, '4':0.22, '5':0.12})
RFC.fit( train_Features, y_train, sample_weight=np.array([1 if y == 5 else 5 for y in y_train]))
print "RFC done"


# In[ ]:

RFC_predictions = RFC.predict( test_Features)
print "CM-4"
find_Accuracy(RFC_predictions, y_test)
find_minority_accuracy(RFC_predictions, y_test)


# In[272]:

RFC1 = RandomForestClassifier(verbose=3)
RFC1.fit( train_Features, y_train)
RFC_predictions1 = RFC1.predict( test_Features)
print "CM-1"
find_Accuracy(RFC_predictions1, y_test)
find_minority_accuracy(RFC_predictions1, y_test)


# In[273]:

RFC2 = RandomForestClassifier(verbose=3, class_weight={'1':0.22, '2':0.22,'3':0.22, '4':0.22, '5':0.12})
RFC2.fit( train_Features, y_train)
RFC_predictions2 = RFC2.predict( test_Features)
print "CM-2"
find_Accuracy(RFC_predictions2, y_test)
find_minority_accuracy(RFC_predictions2, y_test)


# In[276]:

RFC3 = RandomForestClassifier(verbose=3)
RFC3.fit( train_Features, y_train, sample_weight=np.array([1 if y == 5 else 10 for y in y_train]))
RFC_predictions3 = RFC3.predict( test_Features)


# In[277]:

print "CM-3"
find_Accuracy(RFC_predictions3, y_test)
find_minority_accuracy(RFC_predictions3, y_test)


# In[ ]:



