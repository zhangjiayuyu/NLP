# 1
# read file and transform to a list
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from tensorflow import keras

reviews = pd.read_excel('Assignment 3.xlsx', header=0)
# Build a training dataset with the first 400 restaurant reviews and the first 400 movie reviews
train_review = reviews.iloc[:400, 1].tolist() + reviews.iloc[500:900, 1].tolist()
train_c = reviews.iloc[:400, 2].tolist()  + reviews.iloc[500:900, 2].tolist()
test_review = reviews.iloc[400:500, 1].tolist() + reviews.iloc[900:, 1].tolist()
test_c = reviews.iloc[400:500, 2].tolist() + reviews.iloc[900:, 2].tolist()

# 2
# Tokenize each review in the collection
import nltk
token_train_review = [nltk.word_tokenize(each) for each in train_review]
token_test_review = [nltk.word_tokenize(each) for each in test_review]

# lemmatize words
lemmatizer = nltk.stem.WordNetLemmatizer()
lemmatized_token_train_review = []
lemmatized_token_test_review = []
for each in token_train_review:
    lemmatized_token_train_review.append([lemmatizer.lemmatize(word.lower()) for word in each])
for each in token_test_review:
    lemmatized_token_test_review.append([lemmatizer.lemmatize(word.lower()) for word in each])

# remove stop-words and punctuations
from nltk.corpus import stopwords
stop_words_removed_train = []
stop_words_removed_test = []
for each in lemmatized_token_train_review:
    stop_words_removed_train.append([token for token in each if not token in stopwords.words("english") if token.isalpha()])
for each in lemmatized_token_test_review:
    stop_words_removed_test.append([token for token in each if not token in stopwords.words("english") if token.isalpha()])

# transform list to string
normalized_train_review = []
normalized_test_review = []
for each in stop_words_removed_train:
    normalized_train_review.append(" ".join(each))
for each in stop_words_removed_test:
    normalized_test_review.append(" ".join(each))

# set the minimal document frequency for each term and include unigram and bi-gram
# convert each of reviews in a TD-IDF vector
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=5)
vectorizer.fit(normalized_train_review)
train_x = vectorizer.transform(normalized_train_review)
test_x = vectorizer.transform(normalized_test_review)


# 3
## Naive Bayes
from sklearn.naive_bayes import MultinomialNB
NBmodel = MultinomialNB()
# training
NBmodel.fit(train_x, train_c)
y_pred_NB = NBmodel.predict(test_x)
# evaluation
acc_NB = accuracy_score(test_c, y_pred_NB)
print("Naive Bayes model Accuracy::{:.2f}%".format(acc_NB*100))


## Logit
from sklearn.linear_model import LogisticRegression
Logitmodel = LogisticRegression()
# training
Logitmodel.fit(train_x, train_c)
y_pred_logit = Logitmodel.predict(test_x)
# evaluation
from sklearn.metrics import accuracy_score
acc_logit = accuracy_score(test_c, y_pred_logit)
print("Logit model Accuracy:: {:.2f}%".format(acc_logit*100))


## Random forest
from sklearn.ensemble import RandomForestClassifier
RFmodel = RandomForestClassifier(n_estimators=50, max_depth=3, bootstrap=True, random_state=0)
# training
RFmodel.fit(train_x, train_c)
y_pred_RF = RFmodel.predict(test_x)
# evaluation
acc_RF = accuracy_score(test_c, y_pred_RF)
print("Random Forest Model Accuracy: {:.2f}%".format(acc_RF*100))

## SVM
from sklearn.svm import LinearSVC
SVMmodel = LinearSVC()
# training
SVMmodel.fit(train_x, train_c)
y_pred_SVM = SVMmodel.predict(test_x)
# evaluation
acc_SVM = accuracy_score(test_c, y_pred_SVM)
print("SVM model Accuracy:{:.2f}%".format(acc_SVM*100))


## Neural Network
from sklearn.neural_network import MLPClassifier
DLmodel = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(4), random_state=1)
# training
DLmodel.fit(train_x, train_c)
y_pred_DL= DLmodel.predict(test_x)
# evaluation
acc_DL = accuracy_score(test_c, y_pred_DL)
print("DL model Accuracy: {:.2f}%".format(acc_DL*100))



# 4
tokenized_train = [nltk.word_tokenize(doc.lower()) for doc in train_review]
tokenized_test = [nltk.word_tokenize(doc.lower()) for doc in test_review]
# A set for all possible words
words = [j for i in tokenized_train for j in i] + [j for i in tokenized_test for j in i]
# use package
from sklearn.preprocessing import LabelEncoder
index_encoder = LabelEncoder()
index_encoder = index_encoder.fit(words) # define vocabulary
# encoding
index_encoded_train = [index_encoder.transform(doc).tolist() for doc in tokenized_train]
index_encoded_test = [index_encoder.transform(doc).tolist() for doc in tokenized_test]
# padding
from keras.preprocessing import sequence
x_train = np.array(sequence.pad_sequences(index_encoded_train, maxlen=100))
x_test = np.array(sequence.pad_sequences(index_encoded_test, maxlen=100))


# 5
# build model
from keras.models import Sequential
from keras.layers import Dense, Embedding, Dropout
from keras.layers import LSTM
max_features = len(words)
maxlen = 100
batch_size = 100
# dummy label
y_train = np.array([1 if label =='restaurant' else 0 for label in train_c])
y_test = np.array([1 if label =='restaurant' else 0 for label in test_c])
# model architecture
model = Sequential()
model.add(Embedding(max_features, 20, input_length=maxlen))
model.add(LSTM(40, dropout=0.20, recurrent_dropout=0.20))
model.add(Dropout(rate=0.10, noise_shape=None, seed=None))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=batch_size, epochs=10, validation_data=(x_test, y_test))


# 1
import keras
from keras.datasets import cifar10
# Split the data between train and test
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print(x_train.shape)
# Visualize the first 20 images of test set
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
figure(figsize=(16, 8), dpi=80)
for i in range(20):
    plt.subplot(4, 5, i+1)
    plt.xticks([])
    plt.imshow(x_test[i])
    plt.xlabel(y_test[i])
plt.show()
# bc the expected y is binary class matrices
y_train = keras.utils.np_utils.to_categorical(y_train, 10)
y_test = keras.utils.np_utils.to_categorical(y_test, 10)


# 2
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
# define model structure
CNNmodel = Sequential()
CNNmodel.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)))
CNNmodel.add(Dropout(0.20))
CNNmodel.add(Conv2D(32, (3, 3), activation='relu'))
CNNmodel.add(MaxPooling2D(pool_size=(2, 2)))
CNNmodel.add(Conv2D(64, (3, 3), activation='relu'))
CNNmodel.add(Dropout(0.20))
CNNmodel.add(Conv2D(64, (3, 3), activation='relu'))
CNNmodel.add(MaxPooling2D(pool_size=(2, 2)))
CNNmodel.add(Flatten())
CNNmodel.add(Dense(256, activation='relu'))
CNNmodel.add(Dropout(0.20))
CNNmodel.add(Dense(10, activation='softmax'))
CNNmodel.compile(loss=keras.losses.categorical_crossentropy, optimizer='Adam', metrics=['accuracy'])
## model fit
CNNmodel.fit(x_train, y_train, batch_size=256, epochs=5, validation_data=(x_test, y_test))
## model performance
performance = CNNmodel.evaluate(x_test, y_test)
print('Test accuracy:', performance[1])



# 3
# define model2 structure
CNNmodel2 = Sequential()
CNNmodel2.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)))
CNNmodel2.add(Dropout(0.20))
CNNmodel2.add(Conv2D(32, (3, 3), activation='relu'))
CNNmodel2.add(MaxPooling2D(pool_size=(2, 2)))
CNNmodel2.add(Conv2D(64, (3, 3), activation='relu'))
CNNmodel2.add(Dropout(0.20))
CNNmodel2.add(Conv2D(64, (3, 3), activation='relu'))
CNNmodel2.add(MaxPooling2D(pool_size=(2, 2)))
CNNmodel2.add(Conv2D(128, (3, 3), activation='relu'))
CNNmodel2.add(Dropout(0.20))
CNNmodel2.add(Conv2D(128, (3, 3), activation='relu'))
CNNmodel2.add(Flatten())
CNNmodel2.add(Dense(256, activation='relu'))
CNNmodel2.add(Dropout(0.20))
CNNmodel2.add(Dense(10, activation='softmax'))
CNNmodel2.compile(loss=keras.losses.categorical_crossentropy, optimizer='Adam', metrics=['accuracy'])
## model fit
CNNmodel2.fit(x_train, y_train, batch_size=256, epochs=5, validation_data=(x_test, y_test))
## model performance
performance = CNNmodel2.evaluate(x_test, y_test)
print('Test accuracy:', performance[1])

# 4
# define model2 structure
CNNmodel2.compile(loss=keras.losses.categorical_crossentropy, optimizer='Adam', metrics=['accuracy'])
## model fit 20 epochs
CNNmodel2.fit(x_train, y_train, batch_size=256, epochs=20, validation_data=(x_test, y_test))
## model performance
performance = CNNmodel2.evaluate(x_test, y_test)
print('Test accuracy:', performance[1])

# 5
# Reload Data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# Flatten
x_train_flatten = np.array([x_train[i].flatten() for i in range(x_train.shape[0])])
x_test_flatten = np.array([x_test[i].flatten() for i in range(x_test.shape[0])])
y_train = y_train.reshape(50000)
y_test = y_test.reshape(10000)

## Naive Bayes
from sklearn.naive_bayes import MultinomialNB
NBmodel = MultinomialNB()
# training
NBmodel.fit(x_train_flatten, y_train)
y_pred_NB = NBmodel.predict(x_test_flatten)
# evaluation
acc_NB = accuracy_score(y_test, y_pred_NB)
print("Naive Bayes model Accuracy::{:.2f}%".format(acc_NB*100))

## Random forest
from sklearn.ensemble import RandomForestClassifier
RFmodel = RandomForestClassifier(n_estimators=100, max_depth=10, bootstrap=True, random_state=0)
# training
RFmodel.fit(x_train_flatten, y_train)
y_pred_RF = RFmodel.predict(x_test_flatten)
# evaluation
acc_RF = accuracy_score(y_test, y_pred_RF)
print("Random Forest Model Accuracy: {:.2f}%".format(acc_RF*100))