#!/usr/bin/env python
# coding: utf-8

# # Chapter 3 - Document encoding (TF-IDF), topic modeling, sentiment analysis, building text classifiers

# 2022 February 2

# In[1]:


import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.decomposition import TruncatedSVD
import nltk
from nltk.corpus import movie_reviews
import numpy as np
from sklearn.utils import shuffle
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import roc_curve, roc_auc_score, classification_report, accuracy_score, confusion_matrix 

import warnings
warnings.filterwarnings("ignore", category = DeprecationWarning)


# ## Document encoding for machine learning
# 
# In the last chapter you saw that we do not change text to numbers, but instead changed the _representation_ of the text to the numbers in sparse matrix format. 
# 
# In this format, each row represents a document and each column represents a token from the shared text vocabulary called a **feature**. 

# ## Key terms

# * **Document term matrix:** contains the frequencies (or TF-IDF scores) of vocabulary terms in a collection of documents in sparse format. 
#     * Each row is a document in the corpus.
#     * Each column represents a term (uni-gram, bi-gram, etc.) called a feature.
# 
# * **Bag of words:** The simplest text analysis model that standardizes text in a document by removing punctuation, converting the words to lowercase, and counting the token frequencies.
#     * Numeric values indicate that a particular feature is found in a document that number of times.
#     * A 0 indicates that the feature is _not_ found in that document. 
# 
# ![dtm](img/dtm.png)
# 
# [modified from "The Effects of Feature Scaling: From Bag-of-Words to Tf-Idf"](https://www.oreilly.com/library/view/feature-engineering-for/9781491953235/ch04.html)

# * **TF-IDF:** Term frequency–inverse document frequency; a weighted numerical statistic that indicates the uniqueness of a word is in a given document or corpus.
# 
# For TF-IDF sparse matrices:
# * A value closer to 1 indicate that a feature is more relevant to a particular document.
# * A value closer to 0 indicates that that feature is less/not relevant to that document.
# 
# ![tf1](img/tf1.png)
# 
# [Wikipedia](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)
# 
# ![tf2](img/tf2.png)
# 
# [towardsdatascience](https://towardsdatascience.com/tf-term-frequency-idf-inverse-document-frequency-from-scratch-in-python-6c2b61b78558)

# ## Topic modeling
# 
# ![topic](img/topic.png)
# 
# [Wikipedia](https://en.wikipedia.org/wiki/Topic_model)

# ![unhrc](img/unhrc.jpg)
# 
# ### Corpus definition: United Nations Human Rights Council Documentation
# 
# We will select eleven .txt files from the UN HRC as our corpus, stored within the subfolder "human_rights" folder inside the main "data" directory. 
# 
# These documents contain information about human rights recommendations made by member nations towards countries deemed to be in violation of the HRC. 
# 
# [Learn more about the UN HRC by clicking here.](https://www.ohchr.org/en/hrbodies/hrc/pages/home.aspx)

# ### Define the corpus directory
# 
# Set the directory's file path and print the files it contains.

# In[ ]:


import os
corpus = os.listdir('data/human_rights/')

# View the contents of this directory
corpus


# ### Store these documents in a data frame

# In[ ]:


import pandas as pd

# Store in an empty dictionary for conversion to data frame
empty_dictionary = {}

# Loop through the folder of documents to open and read each one
for document in corpus:
    with open('data/human_rights/' + document, 'r', encoding = 'utf-8') as to_open:
         empty_dictionary[document] = to_open.read()

# Populate the data frame with two columns: file name and document text
human_rights = (pd.DataFrame.from_dict(empty_dictionary, 
                                       orient = 'index')
                .reset_index().rename(index = str, 
                                      columns = {'index': 'file_name', 0: 'document_text'}))


# ### View the data frame

# In[ ]:


human_rights


# ### View the text of the first document

# In[ ]:


# first thousand characters
print(human_rights['document_text'][0][:1000])


# ## English text preprocessing
# 
# Create a new column named "clean_text" to store the text as it is preprocessed. 
# 
# * Remove non-alphanumeric characters/punctuation
# * Remove digits
# * Remove [unicode characters](https://en.wikipedia.org/wiki/List_of_Unicode_characters)
# * Remove extra spaces
# * Convert to lowercase
# * Lemmatize (optional for now)
# 
# Take a look at the first document after each step to see if you can notice what changed. 
# 
# [How else could you improve this process?](/SSDS-TAML/winter2022/Appendix.ipynb#appendix-b-more-on-text-preprocessing) 
# 
# > NOTE: Remember, this is just a bare bones, basic process. Furthermore, it will not likely work for many other languages. 

# ### Remove non-alphanumeric characters/punctuation

# In[ ]:


human_rights['clean_text'] = human_rights['document_text'].str.replace(r'[^\w\s]', ' ', regex = True)


# In[ ]:


print(human_rights['clean_text'][0][:1000])


# In[ ]:


# view third column
human_rights


# ### Remove digits

# In[ ]:


human_rights['clean_text'] = human_rights['clean_text'].str.replace(r'\d', ' ', regex = True)


# In[ ]:


print(human_rights['clean_text'][0][:1000])


# ### Remove unicode characters such as Ð and ð

# In[ ]:


# for more on text encodings: https://www.w3.org/International/questions/qa-what-is-encoding
human_rights['clean_text'] = human_rights['clean_text'].str.encode('ascii', 'ignore').str.decode('ascii')


# In[ ]:


print(human_rights['clean_text'][0][:1000])


# ### Remove extra spaces

# In[ ]:


import regex as re
human_rights['clean_text'] = human_rights['clean_text'].str.replace(r'\s+', ' ', regex = True)


# In[ ]:


print(human_rights['clean_text'][0][:1000])


# ### Convert to lowercase

# In[ ]:


human_rights['clean_text'] = human_rights['clean_text'].str.lower()


# In[ ]:


print(human_rights['clean_text'][0][:1000])


# ### Lemmatize

# In[ ]:


# import spacy


# In[ ]:


# !python -m spacy download en_core_web_sm
# !python -m spacy download en_core_web_lg


# In[ ]:


# nlp = spacy.load('en_core_web_lg')
# human_rights['clean_text'] = human_rights['clean_text'].apply(lambda row: ' '.join([w.lemma_ for w in nlp(row)]))


# In[ ]:


# print(human_rights['clean_text'][0])


# ### View the updated data frame

# In[ ]:


human_rights


# ## Unsupervised learning with `TfidfVectorizer()`
# 
# Remember `CountVectorizer()` for creating Bag of Word models? Bag of Words models are inputs for [Latent Dirichlet allocation](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation). 
# 
# However, let's extend this idea to `TfidfVectorizer()`. Each row will still be a colunm in our matrix and each column will still be a linguistic feature, but the cells will now be populated by the word uniqueness weights instead of frequencies. 
# 
# This will be the input for [Truncated Singular Value Decomposition](https://en.wikipedia.org/wiki/Singular_value_decomposition) instead of LDA. 

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer

tf_vectorizer = TfidfVectorizer(ngram_range = (1, 3), 
                                stop_words = 'english', 
                                max_df = 0.50
                                )
tf_sparse = tf_vectorizer.fit_transform(human_rights['clean_text'])


# In[ ]:


tf_sparse.shape


# In[ ]:


print(tf_sparse)


# ### Convert the tfidf sparse matrix to data frame

# In[ ]:


tfidf_df = pd.DataFrame(tf_sparse.todense(), columns = tf_vectorizer.get_feature_names())
tfidf_df


# ### View 20 highest weighted words

# In[ ]:


tfidf_df.max().sort_values(ascending = False).head(n = 20)


# ### Add country name to `tfidf_df`

# In[ ]:


# wrangle the country names from the human_rights data frame
countries = human_rights['file_name'].str.slice(stop = -8)
countries = list(countries)
countries


# In[ ]:


tfidf_df['COUNTRY'] = countries


# In[ ]:


tfidf_df


# ### Examine unique words by each document/country
# 
# Change the country names to view their highest rated terms.

# In[ ]:


country = tfidf_df[tfidf_df['COUNTRY'] == 'jordan']
country.max(numeric_only = True).sort_values(ascending = False).head(20)


# ### Singular value decomposition
# 
# ![tsvd](img/tsvd.png)
# 
# [Analytics Vidhya](https://www.analyticsvidhya.com/blog/2021/06/part-16-step-by-step-guide-to-master-nlp-topic-modelling-using-lsa/)
# 
# * Look ahead to Chapter 5 for new techniques in topic modeling - [BERTopic!](Chapter5.ipynb)

# In[ ]:


from sklearn.decomposition import TruncatedSVD

tsvd = TruncatedSVD(n_components = 5, 
                   random_state = 1, 
                   algorithm = 'arpack')
tsvd.fit(tf_sparse)


# In[ ]:


print(tsvd.explained_variance_ratio_)


# In[ ]:


print(tsvd.singular_values_)


# In[ ]:


def topics(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("\nTopic #{}:".format(topic_idx))
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))


# In[ ]:


tf_features = tf_vectorizer.get_feature_names()
topics(tsvd, tf_features, 20)


# ## UN HRC text analysis - what next? 
# 
# Keep in mind that we have not even begun to consider named entities and parts of speech. How might country names be swamping the five topics produced? 
# 
# [Read this stack overflow post to learn about the possibility of having too few documents in your corpus](https://stats.stackexchange.com/questions/302965/some-topics-with-all-equal-weights-when-using-latentdirichletallocation-from-sci)
# 
# [Also, read this post about how to grid search for the best topic models](https://www.machinelearningplus.com/nlp/topic-modeling-python-sklearn-examples/)
# 
# Use BERTopic (see Chapter 5 in this book)

# ## Sentiment analysis
# 
# Sentiment analysis is the contextual mining of text data that elicits abstract information in source materials to determine if data are positive, negative, or neutral. 

# ![sa](img/sa.jpg)
# 
# [Repustate](https://www.repustate.com/blog/sentiment-analysis-challenges-with-solutions/)

# ### Download the nltk built-in movie reviews dataset

# In[ ]:


import nltk
from nltk.corpus import movie_reviews
nltk.download("movie_reviews")


# ### Define x (reviews) and y (judgements) variables

# In[ ]:


# Extract our x (reviews) and y (judgements) variables
reviews = [movie_reviews.raw(fileid) for fileid in movie_reviews.fileids()]
judgements = [movie_reviews.categories(fileid)[0] for fileid in movie_reviews.fileids()]


# In[ ]:


# Save in a dataframe
movies = pd.DataFrame({"Reviews" : reviews, 
                      "Judgements" : judgements})
movies.head()


# In[ ]:


movies.shape


# ### Shuffle the reviews

# In[ ]:


import numpy as np
from sklearn.utils import shuffle
x, y = shuffle(np.array(movies.Reviews), np.array(movies.Judgements), random_state = 1)


# In[ ]:


# change x[0] and y[0] to see different reviews
x[0], print("Human review was:", y[0])


# ### Pipelines
# 
# scikit-learn offers hand ways to build machine learning pipelines: https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html

# ### One standard way

# In[ ]:


# standard training/test split (no cross validation)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.30, random_state = 0)

# get tfidf values
tfidf = TfidfVectorizer()
tfidf.fit(x)
x_train = tfidf.transform(x_train)
x_test = tfidf.transform(x_test)

# instantiate, train, and test an logistic regression model
logit_class = LogisticRegression(solver = 'liblinear',
                                 penalty = 'l2', 
                                 C = 1000, 
                                 random_state = 1)
model = logit_class.fit(x_train, y_train)


# In[ ]:


# test set accuracy
model.score(x_test, y_test)


# ### $k$-fold cross-validated model

# In[ ]:


# Cross-validated model!
text_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1, 3))),
                    ('tfidf', TfidfTransformer()),
                    ('clf', LogisticRegression(solver = 'liblinear',
                                               penalty = 'l2', 
                                               C = 1000, 
                                               random_state = 1))
                     ])

# for your own research, thesis, or publication
# you would select cv equal to 10 or 20
scores = cross_val_score(text_clf, x, y, cv = 3)

print(scores, np.mean(scores))


# ### Top 25 features for positive and negative reviews

# In[ ]:


feature_names = tfidf.get_feature_names()
top25pos = np.argsort(model.coef_[0])[-25:]
print("Top features for positive reviews:")
print(list(feature_names[j] for j in top25pos))
print()
print("Top features for negative reviews:")
top25neg = np.argsort(model.coef_[0])[:25]
print(list(feature_names[j] for j in top25neg))


# In[ ]:


new_bad_review = "This was the most awful worst super bad movie ever!"

features = tfidf.transform([new_bad_review])

model.predict(features)


# In[ ]:


new_good_review = 'WHAT A WONDERFUL, FANTASTIC MOVIE!!!'

features = tfidf.transform([new_good_review])

model.predict(features)


# In[ ]:


# type another review here
my_review = 'I hated this movie, even though my friend loved it'
my_features = tfidf.transform([my_review])
model.predict(my_features)


# ## Quiz - 20 newsgroups dataset
# 
# Go through the 20 newsgroups text dataset to get familiar with newspaper data: https://scikit-learn.org/0.19/datasets/twenty_newsgroups.html
# 
# "The 20 newsgroups dataset comprises around 18000 newsgroups posts on 20 topics split in two subsets: one for training (or development) and the other one for testing (or for performance evaluation). The split between the train and test set is based upon a messages posted before and after a specific date."
