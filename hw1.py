# Part1 Text representation

# read file and transform to a list
import pandas as pd
import numpy as np
reviews = pd.read_csv('Assignment 1.csv', header=None)
reviews = np.array(reviews)[:, 1].tolist()

# 1. Tokenize each review in the collection
import nltk
token_reviews = [nltk.word_tokenize(each) for each in reviews]

# 2. lemmatize words
lemmatizer = nltk.stem.WordNetLemmatizer()
lemmatized_token_reviews = []
for each in token_reviews:
    lemmatized_token_reviews.append([lemmatizer.lemmatize(word.lower()) for word in each])


# 3. remove stop-words and punctuations
from nltk.corpus import stopwords
stop_words_removed = []
for each in lemmatized_token_reviews:
    stop_words_removed.append([token for token in each if not token in stopwords.words("english") if token.isalpha()])


# 4. convert each of reviews in a TD-IDF vector
from sklearn.feature_extraction.text import TfidfVectorizer

# transform list to string
normalized_reviews = []
for each in stop_words_removed:
    normalized_reviews.append(" ".join(each))

vectorizer1 = TfidfVectorizer(ngram_range=(1,2), min_df=3)
vectorizer1.fit(normalized_reviews)
v1 = vectorizer1.transform(normalized_reviews)

# transfer v1 to dataframe and save to csv
df_v1 = pd.DataFrame(v1.toarray())
df_v1.to_csv('p1_step4.csv')


# 5. POS-tag and TD-IDF vectorization
POS_reviews = []
for each in token_reviews:
    POS_token_doc = nltk.pos_tag(each)
    POS_token_temp = []
    for i in POS_token_doc:
        POS_token_temp.append(i[0] + i[1])
    POS_reviews.append(" ".join(POS_token_temp))

vectorizer2 = TfidfVectorizer(min_df=4)
vectorizer2.fit(POS_reviews)
POS_v = vectorizer2.transform(POS_reviews)

df_POS_v = pd.DataFrame(POS_v.toarray())
df_POS_v.to_csv('p1_step5.csv')


# Part2. Word Embedding

# 1. index-based encoding
# Choose the first 10 tokenized documents
tokenized_top10 = token_reviews[:10]
# a set for all possible words
words = [j for i in tokenized_top10 for j in i]

from sklearn.preprocessing import LabelEncoder
index_encoder = LabelEncoder()
index_encoder = index_encoder.fit(words)
# encoding
index_encoded = [index_encoder.transform(doc).tolist() for doc in tokenized_top10]

# save to csv
df_index_encoded = pd.DataFrame(index_encoded)
df_index_encoded.to_csv('p2_step1.csv')


# 2. one-hot encoding
from sklearn.preprocessing import OneHotEncoder
# vocabulary
indices_list = [[j] for i in index_encoded for j in i]

onehot_encoder = OneHotEncoder(sparse=False)
onehot_encoder = onehot_encoder.fit(indices_list)
# encoding
onehot_encoded = [onehot_encoder.transform([[i] for i in doc_i]).tolist() for doc_i in index_encoded]

# save to txt
with open('p2_step2.txt', 'w') as f:
    for i in onehot_encoded:
        f.write('%s\n' % i)


# 3. pre-trained glove
from gensim.scripts.glove2word2vec import glove2word2vec
glove_input_file = 'glove.6B.50d.txt'
word2vec_output_file = 'glove.6B.50d.txt.word2vec'
glove2word2vec(glove_input_file, word2vec_output_file)

from gensim.models import KeyedVectors
filename = 'glove.6B.50d.txt.word2vec'
model = KeyedVectors.load_word2vec_format(filename, binary= False)

# embed each word as a 50d vector
glove_top10 = []
for doc in tokenized_top10:
    glove_doc = []
    for word in doc:
        if word in model:
            glove_doc.append(model[word].tolist())
    glove_top10.append(glove_doc)

# save to txt
with open('p2_step3.txt', 'w') as f:
    for i in glove_top10:
        f.write('%s\n' % i)

