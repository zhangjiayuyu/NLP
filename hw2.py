from PIL import Image
from pylab import *
import os
import pandas as pd

os.chdir('C:\\Users\\Lenovo\\OneDrive - purdue.edu\\AUD_59000\\04_HW\\hw2\\Assignment 2 images\\image')

imgs = []
for i in range(1, 11):
    # Read in all of the 10 images
    im = Image.open(str(i)+'.png')
    # resize a 100 by 100 pixels format
    im = im.resize((100, 100))
    imgs.append(im)

# convert images to greyscale arrays
imgs_grey = []
for i in imgs:
    im_grey = array(i.convert("L"))
    imgs_grey.append(im_grey)

# flatten the 2-D array to a 1-D array
imgs_vector = []
for i in imgs_grey:
    im_v = i.flatten()
    imgs_vector.append(im_v)

# transfer imgs_vector to dataframe and save to csv
df_imgs_vector = pd.DataFrame(imgs_vector)
df_imgs_vector.to_csv('p1_step3.csv')

# draw histogram for each image
for i in imgs_vector:
    figure()
    hist(i, 256)

# normalize each image and draw a histogram
for i in imgs_vector:
    imhist, bins = histogram(i, 256, normed= True)
    cdf = imhist.cumsum() # cumulative distribution function
    cdf = 255 * cdf / cdf[-1] # normalize
    i2 = interp(i, bins[:-1], cdf)
    figure()
    hist(i2, 256)


# read file and transform to a list
import pandas as pd
import numpy as np
reviews = pd.read_excel('Assignment 2 text.xlsx', header=0)
reviews = np.array(reviews)[:, 1].tolist()

# Tokenize each review in the collection
import nltk
token_reviews = [nltk.word_tokenize(each) for each in reviews]

# lemmatize words
lemmatizer = nltk.stem.WordNetLemmatizer()
lemmatized_token_reviews = []
for each in token_reviews:
    lemmatized_token_reviews.append([lemmatizer.lemmatize(word.lower()) for word in each])

# remove stop-words and punctuations
from nltk.corpus import stopwords
stop_words_removed = []
for each in lemmatized_token_reviews:
    stop_words_removed.append([token for token in each if not token in stopwords.words("english") if token.isalpha()])

# transform list to string
normalized_reviews = []
for each in stop_words_removed:
    normalized_reviews.append(" ".join(each))

# set the minimal document frequency for each term and include unigram and bi-gram
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(ngram_range=(1,2), min_df=5)
v = vectorizer.fit_transform(normalized_reviews)

# use the LDA model to extract the topics of each document
from sklearn.decomposition import LatentDirichletAllocation
lda = LatentDirichletAllocation(n_components=6).fit(v)

# topic distribution of the first 10 restaurant reviews(id=[1:10])
topic_dist_top10_rest = lda.transform(v[:10])
print(topic_dist_top10_rest)
for i in range(10):
    print(np.argsort(topic_dist_top10_rest[i])[-1:-3:-1])

# topic distribution of the first 10 movie reviews (id = [501:510])
topic_dist_top10_movie = lda.transform(v[500:510])
print(topic_dist_top10_movie)
for i in range(10):
    print(np.argsort(topic_dist_top10_movie[i])[-1:-3:-1])

# get a list of every feature name (terms)
terms = vectorizer.get_feature_names()
# top-5 terms for each of the 6 topics
for topic_idx, topic in enumerate(lda.components_):
    print("Topic %d:" % topic_idx)
    print(" ".join([terms[i] for i in topic.argsort()[:-5-1:-1]]))



