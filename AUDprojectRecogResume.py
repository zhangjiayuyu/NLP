import pandas as pd
import nltk
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer

# Extract review of all from resume, activity and community file
df1 = pd.read_csv("Data_Description2.csv")
df2 = pd.read_csv('Resume.csv')
df2['class'] = [1 for x in range(len(df2))]
df3 = pd.read_csv('Activity.csv')
df3['class'] = [0 for x in range(len(df3))]
df4 = pd.read_csv('Community.csv')
df4['class'] = [0 for x in range(len(df4))]

# Transform reviews into list of list
df = pd.concat([df1, df2, df3, df4], ignore_index=True)
df['review'] = df['Title'].str.cat(df['Description'], sep='. ')
df.drop(['Unnamed: 0', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
df.dropna(inplace=True)
reviews = df['review'].tolist()


# Clean the review by lemmatization and stemming, wipe off stop words and make cleaned words become reviews again
# Define a fomula to clean the review
def clean_review_list(reviews):
    lemmatizer = nltk.stem.WordNetLemmatizer()
    stemmer = nltk.stem.PorterStemmer()
    t_d_matrix_words = []
    for review in reviews:
        review_token = nltk.word_tokenize(re.sub(r"(&gt|&lt|&#39|&quot)", "", review))
        r_token = []
        for token in review_token:
            token = stemmer.stem(token.lower())
            lemmatized_token = lemmatizer.lemmatize(token.lower())
            if not lemmatized_token in stopwords.words('English') and lemmatized_token.isalpha():
                r_token.append(lemmatized_token)
        t_d_matrix_words.append(r_token)
    clean_review = []
    for i in range(len(t_d_matrix_words)):
        clean_review.append(" ".join(t_d_matrix_words[i]))
    return clean_review


clean_review = clean_review_list(reviews)

# Words from non-resume documents have high chances not showing up in resume documents, so tf_idf is better than
# normal vectorization


tfidfvectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
tfidfvectorizer.fit(clean_review)
# print(len(tfidfvectorizer.vocabulary_))

# tfidf transform each review to vector
tfidfvectorized_review = tfidfvectorizer.transform(clean_review)
tfidfvec = tfidfvectorized_review.toarray()

# try clustering
from sklearn.cluster import KMeans
binaryclass = KMeans(n_clusters=2, random_state=0).fit(tfidfvec)
acc_cluster = accuracy_score(df['class'], binaryclass.labels_)
print("Kmeans Accuracy: {:.2f}%".format(acc_cluster*100))

# fit different to see the performance of supervised learning models

# split data
from sklearn.model_selection import train_test_split

train_x, test_x, train_y, test_y = train_test_split(tfidfvec, df['class'], test_size=0.2)

# use SVM
from sklearn.svm import LinearSVC
SVMmodel = LinearSVC()
# training
SVMmodel.fit(train_x, train_y)
topic_pred_SVM = SVMmodel.predict(test_x)
# evaluation
acc_SVM = accuracy_score(test_y, topic_pred_SVM)
print("SVM model Accuracy:{:.2f}%".format(acc_SVM * 100))

# Supervised Decision Tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
DTmodel = DecisionTreeClassifier()
RFmodel = RandomForestClassifier(n_estimators=50, max_depth=3,bootstrap=True, random_state=0)
# training
DTmodel.fit(train_x, train_y)
y_pred_DT = DTmodel.predict(test_x)
RFmodel.fit(train_x, train_y)
y_pred_RF = RFmodel.predict(test_x)
# evaluation
acc_DT = accuracy_score(test_y, y_pred_DT)
print("Decision Tree Model Accuracy: {:.2f}%".format(acc_DT * 100))
acc_RF = accuracy_score(test_y, y_pred_RF)
print("Random Forest Model Accuracy: {:.2f}%".format(acc_RF * 100))

# Logistic
from sklearn.linear_model import LogisticRegression
Logitmodel = LogisticRegression()
# training
Logitmodel.fit(train_x, train_y)
y_pred_logit = Logitmodel.predict(test_x)
# evaluation
from sklearn.metrics import accuracy_score
acc_logit = accuracy_score(test_y, y_pred_logit)
print("Logit model Accuracy:: {:.2f}%".format(acc_logit * 100))

# KNN
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(train_x, train_y)
y_pred_knn = neigh.predict(test_x)
# evaluation
acc_knn = accuracy_score(test_y, y_pred_knn)
print("KNN model Accuracy:: {:.2f}%".format(acc_knn * 100))

# Use SVM, the best performed model to predict the a small test data.
dft = pd.read_excel("Test.xlsx")
dft['review'] = dft['Title'].str.cat(dft['Description'], sep='. ')
testreviews = dft['review'].tolist()
tclean_review = clean_review_list(testreviews)

tfidfvectorized_reviewt = tfidfvectorizer.transform(tclean_review)
tfidfvect = tfidfvectorized_reviewt.toarray()

testtopic_pred_SVM = SVMmodel.predict(tfidfvect)
dft['predict'] = testtopic_pred_SVM
acc_SVM = accuracy_score(dft['class'], testtopic_pred_SVM)
print("Test Accuracy:: {:.2f}%".format(acc_SVM * 100))

# Part 2 labelling

# [:2196] are all from resume
# create df only have resume
df_resume = df.iloc[:2196]
topic_pred_SVM = SVMmodel.predict(tfidfvec[0:2196])
df_resume.loc[:, 'recog'] = topic_pred_SVM.tolist()

df_resume = df_resume[df_resume['recog'] == 1]
resume_review = df_resume['review'].tolist()

# clean the resume review
clean_review_r = clean_review_list(resume_review)


# use new tfidf model to transform the review because those review already classified as resume, so new model will be good
tfidfvectorizer2 = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
tfidfvectorizer2.fit(clean_review_r)
tfidfvectorized_resume_review = tfidfvectorizer2.transform(clean_review_r)
tfidf_rvec = tfidfvectorized_resume_review.toarray()

#the bag of words
terms = tfidfvectorizer2.get_feature_names()
# Try Unsupervised LDA to automatically clustering, to see if the topic can use as tools to label.
from sklearn.decomposition import LatentDirichletAllocation
from time import time
import matplotlib.pyplot as plt

lda = LatentDirichletAllocation(n_components=12).fit(tfidfvectorized_resume_review)
for topic_idx, topic in enumerate(lda.components_):
    print("Topic %d:" % (topic_idx+1))
    print(" ".join([terms[i] for i in topic.argsort()[:-4-1:-1]]))

#plot the LDA topic to see if it meets our needs
def plot_top_words(model, feature_names, n_top_words, title):
    fig, axes = plt.subplots(3, 4, figsize=(30, 15), sharex=True)
    axes = axes.flatten()
    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[: -n_top_words - 1 : -1]
        top_features = [feature_names[i] for i in top_features_ind]
        weights = topic[top_features_ind]

        ax = axes[topic_idx]
        ax.barh(top_features, weights, height=0.7)
        ax.set_title(f"Topic {topic_idx +1}", fontdict={"fontsize": 30})
        ax.invert_yaxis()
        ax.tick_params(axis="both", which="major", labelsize=20)
        for i in "top right left".split():
            ax.spines[i].set_visible(False)
        fig.suptitle(title, fontsize=40)

    plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
    plt.show()


plot_top_words(lda,terms,5,"Topics in LDA model")


# load manually labelled review, turn it into x & y
labeleddf = pd.read_csv('labelleddf.csv')
dfjobclass = labeleddf.iloc[:,8:]
# y
class_vec = [dfjobclass.iloc[i].tolist() for i in range(len(dfjobclass))]

labeleddf['review'] = labeleddf['Title'].str.cat(labeleddf['Description'],sep = '. ')
labeled_reviews = labeleddf['review'].tolist()
clean_lr = clean_review_list(labeled_reviews)
# x
clr_vec = tfidfvectorizer2.transform(clean_lr)

# split data
train_rx,test_rx,train_ry,test_ry = train_test_split(clr_vec,dfjobclass,test_size=0.2)

# label list
label_list = dfjobclass.columns.tolist()

# Train the model for each label, save each model in a dict
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
predict_model_dict = {}
for job in label_list:
    SVMmodel = LinearSVC()
    SVMmodel.fit(train_rx, train_ry[job])
    predict_model_dict[job] = SVMmodel
    pred_SVM = SVMmodel.predict(test_rx)
    acc_SVM = accuracy_score(test_ry[job],pred_SVM)
    print(job," SVM model Accuracy:{:.2f}%".format(acc_SVM*100))

# use model to label each review
dfboston = pd.read_excel("boston.xlsx")
dfboston['review'] = dfboston['Title'].str.cat(dfboston['Description'],sep = '. ')
breviews = dfboston['review'].tolist()

# clean of boston reviews
clean_review_b = clean_review_list(breviews)

# use the previous resume's tfidf to transform the clean review of boston
tfidfvectorized_b_review = tfidfvectorizer2.transform(clean_review_b)
tfidf_bvec = tfidfvectorized_b_review.toarray()

# create df for label from jobdict
bjobclass_matrix = {}
for job in label_list:
    bjobclass_matrix[job] = [0 for j in range(len(dfboston))]
bdfjobclass = pd.DataFrame(bjobclass_matrix)

# save labeled review
for job in label_list:
    model = predict_model_dict[job]
    label_predicted = model.predict(tfidf_bvec)
    bdfjobclass[job] = label_predicted

dfb_labelled = pd.concat([dfboston,bdfjobclass],axis = 1)
dfb_labelled.to_csv('dfb_labelled.csv')
