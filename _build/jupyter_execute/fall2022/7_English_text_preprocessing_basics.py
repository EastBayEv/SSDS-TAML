#!/usr/bin/env python
# coding: utf-8

# # Chapter 7 - English text preprocessing basics
# 2022 August 26

# <a target="_blank" href="https://colab.research.google.com/github/EastBayEv/SSDS-TAML/blob/main/fall2022/7_English_text_preprocessing_basics.ipynb">
#   <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
# </a>

# ![text](img/text.png)

# Unstructured text - text you find in the wild in books and websites - is generally not amenable to analysis. Before it can be analyzed, the text needs to be standardized to a format so that Python can recognize each unit of meaning **(called a "token")** as unique, no matter how many times it occurs and how it is stylized. 
# 
# Although not an exhaustive list, some key steps in preprocessing text include:  
# * Standardize text casing and spacing 
# * Remove punctuation and special characters/symbols
# * Remove stop words
# * Stem or lemmatize: convert all non-base words to their base form 
# 
# Stemming/lemmatization and stop words (and some punctuation) are language-specific. The Natural Language ToolKit (NLTK) works for English out-of-the-box, but you'll need different code to work with other languages. Some languages (e.g. Chinese) also require *segmentation*: artificially inserting spaces between words. If you want to do text pre-processing for other languages, please let us know and we can help!

# In[1]:


# Ensure you have the proper nltk modules
import nltk
nltk.download('words')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('omw-1.4')


# In[2]:


from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from string import punctuation
import pandas as pd
import seaborn as sns
from collections import Counter
import regex as re

import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
import spacy
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


# ## Import dracula.txt

# In[3]:


get_ipython().system('wget -P data/novels/ https://raw.githubusercontent.com/EastBayEv/SSDS-TAML/main/fall2022/data/dracula.txt')
text = open("data/dracula.txt").read()

# print just the first 100 characters
print(text[:100])


# ## Standardize Text 
# 
# ### Casing and spacing
# 
# Oftentimes in text analysis, identifying occurences of key word(s) is a necessary step. To do so, we may want "apple," "ApPLe," and "apple      " to be treated the same; i.e., as an occurence of the token, 'apple.' To achieve this, we can standardize text casing and spacing: 

# In[4]:


# Converting all charazcters in a string to lowercase only requires one method: 
message = "Hello! Welcome      to        TAML!"
print(message.lower())

# To replace instances of multiple spaces with one, we can use the regex module's 'sub' function:
# Documentation on regex can be found at: https://docs.python.org/3/library/re.html
single_spaces_msg = re.sub('\s+', ' ', message)
print(single_spaces_msg)


# ### Remove punctuation
# 
# Remember that Python methods can be chained together. 
# 
# Below, a standard for loop loops through the `punctuation` module to replace any of these characters with nothing.

# In[5]:


print(punctuation)


# In[6]:


for char in punctuation:
    text = text.lower().replace(char, "")


# In[7]:


print(text[:100])


# ### Tokenize the text
# 
# Split each word on spaces.

# In[8]:


# .split() returns a list of the tokens in a string, separated by the specified delimiter (default: " ")
tokens = text.split()


# In[9]:


# View the first 20
print(tokens[:20])


# ### Remove stop words
# 
# 
# 
# Below is a list comprehension (a sort of shortcut for loop, or chunk of repeating code) that can accomplish this task for us.

# In[10]:


filtered_text = [word for word in tokens if word not in stopwords.words('english')]


# In[11]:


# show only the first 100 words
# do you see any stopwords?
print(filtered_text[:100])


# ### Lemmatizing/Stemming tokens
# 
# Lemmatizating and stemming are related, but are different practices. Both aim to reduce the inflectional forms of a token to a common base/root. However, how they go about doing so is the key differentiating factor.  
# 
# Stemming operates by removes the prefixs and/or suffixes of a word. Examples include: 
# * flooding to flood 
# * studies to studi
# * risky to risk 
# 
# Lemmatization attempts to contextualize a word, arriving at it's base meaning. Lemmatization reductions can occur across various dimensions of speech. Examples include: 
# * Plural to singular (corpora to corpus)
# * Condition (better to good)
# * Gerund (running to run)
# 
# One technique is not strictly better than the other - it's a matter of project needs and proper application. 

# In[12]:


stmer = nltk.PorterStemmer()

lmtzr = nltk.WordNetLemmatizer()


# In[14]:


# do you see any differences?
token_stem  = [ stmer.stem(token) for token in filtered_text]

token_lemma = [ lmtzr.lemmatize(token) for token in filtered_text ]

print(token_stem[:20])

print(token_lemma[:20])


# ## Part of speech tags
# 
# Part of speech tags are labels given to each word in a text such as verbs, adverbs, nouns, pronouns, adjectives, conjunctions, and their various derivations and subcategories. 

# In[15]:


tagged = nltk.pos_tag(token_lemma)

# Let's see a quick example: 
ex_string = 'They refuse to permit us to obtain the refuse permit.'
print(nltk.pos_tag(ex_string.split())) 


# The output of .pos_tag is a list of tuples (immutable pairs), where the first element is a text token and the second is a part of speech. Note that, in our example string, the token 'refuse' shows up twice - once as a verb, and once as a noun. In the output to .pos_tag, the first tuple with 'refuse' has the 'VBP' tag (present tense verb) and the second tuple has the 'NN' tag (noun). Nifty!

# In[20]:


chunked = nltk.chunk.ne_chunk(tagged)


# ## Convert to dataframe

# In[21]:


df = pd.DataFrame(chunked, columns=['word', 'pos'])


# In[22]:


df.head(n = 10)


# In[23]:


df.shape


# ## Visualize the 20 most frequent words

# In[24]:


top = df.copy()

count_words = Counter(top['word'])
count_words.most_common()[:20]


# In[25]:


words_df = pd.DataFrame(count_words.items(), columns=['word', 'count']).sort_values(by = 'count', ascending=False)


# In[26]:


words_df[:20]


# In[27]:


# What would you need to do to improve an approach to word visualization such as this one?
top_plot = sns.barplot(x = 'word', y = 'count', data = words_df[:20])
top_plot.set_xticklabels(top_plot.get_xticklabels(),rotation = 40);


# ## Corpus definition: United Nations Human Rights Council Documentation
# 
# ![unhrc](img/unhrc.jpg)
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


# Make the directory "human_rights" inside of data
get_ipython().system('mkdir data')
get_ipython().system('mkdir data/human_rights')


# In[ ]:


# Download elevent UN HRC files
get_ipython().system('wget -P data/human_rights/ https://github.com/EastBayEv/SSDS-TAML/blob/main/fall2022/data/human_rights/afghanistan2014.txt')
get_ipython().system('wget -P data/human_rights/ https://github.com/EastBayEv/SSDS-TAML/blob/main/fall2022/data/human_rights/bangladesh2013.txt')
get_ipython().system('wget -P data/human_rights/ https://github.com/EastBayEv/SSDS-TAML/blob/main/fall2022/data/human_rights/cotedivoire2014.txt')
get_ipython().system('wget -P data/human_rights/ https://github.com/EastBayEv/SSDS-TAML/blob/main/fall2022/data/human_rights/djibouti2013.txt')
get_ipython().system('wget -P data/human_rights/ https://github.com/EastBayEv/SSDS-TAML/blob/main/fall2022/data/human_rights/fiji2014.txt')
get_ipython().system('wget -P data/human_rights/ https://github.com/EastBayEv/SSDS-TAML/blob/main/fall2022/data/human_rights/jordan2013.txt')
get_ipython().system('wget -P data/human_rights/ https://github.com/EastBayEv/SSDS-TAML/blob/main/fall2022/data/human_rights/kazakhstan2014.txt')
get_ipython().system('wget -P data/human_rights/ https://github.com/EastBayEv/SSDS-TAML/blob/main/fall2022/data/human_rights/monaco2013.txt')
get_ipython().system('wget -P data/human_rights/ https://github.com/EastBayEv/SSDS-TAML/blob/main/fall2022/data/human_rights/sanmarino2014.txt')
get_ipython().system('wget -P data/human_rights/ https://github.com/EastBayEv/SSDS-TAML/blob/main/fall2022/data/human_rights/turkmenistan2013.txt')
get_ipython().system('wget -P data/human_rights/ https://github.com/EastBayEv/SSDS-TAML/blob/main/fall2022/data/human_rights/tuvalu2013.txt')


# In[2]:


# Check that we have eleven files, one for each country
get_ipython().system('ls data/human_rights/')


# In[28]:


import os
corpus = os.listdir('data/human_rights/')

# View the contents of this directory
corpus


# ### Store these documents in a data frame

# In[29]:


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

# In[30]:


human_rights


# ### View the text of the first document

# In[31]:


# first thousand characters
print(human_rights['document_text'][0][:1000])


# ## English text preprocessing
# 
# Create a new column named "clean_text" to store the text as it is preprocessed. 
# 
# ### What are some of the things we can do? 
# 
# These are just a few examples. How else could you improve this process? 
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
# > Remember: the process will likely be different for many other natural languages, which frequently require special considerations. 

# ### Remove non-alphanumeric characters/punctuation

# In[37]:


# Create a new column 'clean_text' to store the text we are standardizing
human_rights['clean_text'] = human_rights['document_text'].str.replace(r'[^\w\s]', ' ', regex = True)


# In[38]:


print(human_rights['clean_text'][0][:1000])


# In[39]:


# view third column
human_rights


# ### Remove digits

# In[40]:


human_rights['clean_text'] = human_rights['clean_text'].str.replace(r'\d', ' ', regex = True)


# In[41]:


print(human_rights['clean_text'][0][:1000])


# ### Remove unicode characters such as Ð and ð

# In[42]:


# for more on text encodings: https://www.w3.org/International/questions/qa-what-is-encoding
human_rights['clean_text'] = human_rights['clean_text'].str.encode('ascii', 'ignore').str.decode('ascii')


# In[43]:


print(human_rights['clean_text'][0][:1000])


# ### Remove extra spaces

# In[44]:


import regex as re
human_rights['clean_text'] = human_rights['clean_text'].str.replace(r'\s+', ' ', regex = True)


# In[45]:


print(human_rights['clean_text'][0][:1000])


# ### Convert to lowercase

# In[46]:


human_rights['clean_text'] = human_rights['clean_text'].str.lower()


# In[47]:


print(human_rights['clean_text'][0][:1000])


# ### Lemmatize

# In[48]:


# !python -m spacy download en_core_web_sm
# !python -m spacy download en_core_web_lg


# In[49]:


nlp = spacy.load('en_core_web_sm')
human_rights['clean_text'] = human_rights['clean_text'].apply(lambda row: ' '.join([w.lemma_ for w in nlp(row)]))


# In[50]:


print(human_rights['clean_text'][0])


# ### View the updated data frame

# In[51]:


human_rights


# ## Unsupervised learning with `TfidfVectorizer()`
# 
# Remember `CountVectorizer()` for creating Bag of Word models? We can extend this idea of counting words, to _counting unique words_ within a document relative to the rest of the corpus with `TfidfVectorizer()`. Each row will still be a document in the document term matrix and each column will still be a linguistic feature, but the cells will now be populated by the word uniqueness weights instead of frequencies. Remember that: 
# 
# * For TF-IDF sparse matrices:
#     * A value closer to 1 indicate that a feature is more relevant to a particular document.
#     * A value closer to 0 indicates that that feature is less/not relevant to that document.
# 
# ![tf1](img/tf1.png)
# 
# [Wikipedia](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)
# 
# ![tf2](img/tf2.png)
# 
# [towardsdatascience](https://towardsdatascience.com/tf-term-frequency-idf-inverse-document-frequency-from-scratch-in-python-6c2b61b78558)

# In[52]:


from sklearn.feature_extraction.text import TfidfVectorizer

tf_vectorizer = TfidfVectorizer(ngram_range = (1, 3), 
                                stop_words = 'english', 
                                max_df = 0.50
                                )
tf_sparse = tf_vectorizer.fit_transform(human_rights['clean_text'])


# In[53]:


tf_sparse.shape


# In[54]:


print(tf_sparse)


# ### Convert the tfidf sparse matrix to data frame

# In[55]:


tfidf_df = pd.DataFrame(tf_sparse.todense(), columns = tf_vectorizer.get_feature_names())
tfidf_df


# ### View 20 highest weighted words

# In[56]:


tfidf_df.max().sort_values(ascending = False).head(n = 20)


# ### Add country name to `tfidf_df`
# 
# This way, we will know which document is relative to which country.

# In[60]:


# wrangle the country names from the human_rights data frame
countries = human_rights['file_name'].str.slice(stop = -8)
countries = list(countries)
countries


# In[61]:


tfidf_df['COUNTRY'] = countries


# In[62]:


tfidf_df


# ### Examine unique words by each document/country
# 
# Change the country names to view their highest rated terms.

# In[63]:


country = tfidf_df[tfidf_df['COUNTRY'] == 'jordan']
country.max(numeric_only = True).sort_values(ascending = False).head(20)


# ## Sentiment analysis
# 
# Sentiment analysis is the contextual mining of text data that elicits abstract information in source materials to determine if data are positive, negative, or neutral. 

# ![sa](img/sa.jpg)
# 
# [Repustate](https://www.repustate.com/blog/sentiment-analysis-challenges-with-solutions/)

# ### Download the nltk built movie reviews dataset

# In[64]:


import nltk
from nltk.corpus import movie_reviews
nltk.download("movie_reviews")


# ### Define x (reviews) and y (judgements) variables

# In[65]:


# Extract our x (reviews) and y (judgements) variables
reviews = [movie_reviews.raw(fileid) for fileid in movie_reviews.fileids()]
judgements = [movie_reviews.categories(fileid)[0] for fileid in movie_reviews.fileids()]


# In[66]:


# Save in a dataframe
movies = pd.DataFrame({"Reviews" : reviews, 
                      "Judgements" : judgements})
movies.head()


# In[67]:


movies.shape


# ### Shuffle the reviews

# In[68]:


import numpy as np
from sklearn.utils import shuffle
x, y = shuffle(np.array(movies.Reviews), np.array(movies.Judgements), random_state = 1)


# In[69]:


# change x[0] and y[0] to see different reviews
x[0], print("Human review was:", y[0])


# ### Pipelines - one example
# 
# scikit-learn offers hand ways to build machine learning pipelines: https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html

# In[70]:


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


# In[71]:


# test set accuracy
model.score(x_test, y_test)


# ### $k$-fold cross-validated model

# In[72]:


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

# In[73]:


feature_names = tfidf.get_feature_names()
top25pos = np.argsort(model.coef_[0])[-25:]
print("Top features for positive reviews:")
print(list(feature_names[j] for j in top25pos))
print()
print("Top features for negative reviews:")
top25neg = np.argsort(model.coef_[0])[:25]
print(list(feature_names[j] for j in top25neg))


# In[74]:


new_bad_review = "This was the most awful worst super bad movie ever!"

features = tfidf.transform([new_bad_review])

model.predict(features)


# In[75]:


new_good_review = 'WHAT A WONDERFUL, FANTASTIC MOVIE!!!'

features = tfidf.transform([new_good_review])

model.predict(features)


# In[76]:


# try a more complex statement
my_review = 'I hated this movie, even though my friend loved it'
my_features = tfidf.transform([my_review])
model.predict(my_features)


# ## UN HRC text analysis - what next?
# 
# What next? Keep in mind that we have not even begun to consider named entities and parts of speech. What problems immediately jump out from the above examples, such as with the number and uniqueness of country names?
# 
# The next two chapters 8 and 9 introduce powerful text preprocessing and analysis techniques. Read ahead to see how we can handle roadblocks such as these. 

# ## Exercises - redwoods webscraping
# 
# This also works with data scraped from the web. Below is very brief BeautifulSoup example to save the contents of the Sequoioideae (redwood trees) Wikipedia page in a variable named `text`. 
# 
# 1. Read through the code below
# 2. Practice by repeating for a webpage of your choice
# 
# ![redwood](img/redwood.png)

# In[23]:


# import necessary libraries
from bs4 import BeautifulSoup
import requests
import regex as re
import nltk


# ## Three variables will get you started
# 
# 1. `url` - define the URL to be scraped 
# 2. `response` - perform the get request on the URL 
# 3. `soup` - create the soup object so we can parse the html 

# In[24]:


url = "https://en.wikipedia.org/wiki/Sequoioideae"
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html')


# ## Get the text
# 
# HTML (hypertext markup language) is used to structure a webpage and the content it contains, including text.
# 
# Below is a handy for loop that finds all everything within paragraph `<p>`, or paragraph tags. 

# In[25]:


# save in an empty string
text = ""

for paragraph in soup.find_all('p'):
    text += paragraph.text


# In[26]:


print(text)


# ## Regular expressions
# 
# Remember how we did some regular expression preprocessing above? You could even use a bunch of regular expressions in sequence or simultaneously. [Check out the tutorial](https://docs.python.org/3/library/re.html) and [cheatsheet](https://www.dataquest.io/blog/regex-cheatsheet/) to find out what the below symbols mean and write your own code.

# In[27]:


text = re.sub(r'\[[0-9]*\]',' ',text)
text = re.sub(r'\s+',' ',text)
text = re.sub(r'\d',' ',text)
text = re.sub(r'[^\w\s]','',text)
text = text.lower()
text = re.sub(r'\s+',' ',text)


# In[28]:


print(text)


# ## Exercises - text classification
# 
# The 20 newsgroups dataset comprises around 18000 newsgroups posts on 20 topics split in two subsets: one for training (or development) and the other one for testing (or for performance evaluation). The split between the train and test set is based upon a messages posted before and after a specific date."
# 
# 1. Investigate classic horror novel vocabulary. Create a single TF-IDF sparse matrix that contains the vocabulary for _Frankenstein_ and _Dracula_. You should only have two rows (one for each of these novels), but potentially thousands of columns to represent the vocabulary across the two texts. What are the 20 most unique words in each? Make a dataframe or visualization to illustrate the differences.
# 2. [Read through this 20 newsgroups dataset example](https://scikit-learn.org/0.19/datasets/twenty_newsgroups.html) to get familiar with newspaper data. 
# 3. Do you best to understand and explain what is happening at each step of the workflow. 

# ## Improving preprocessing accuracy and efficiency
# 
# Remember these are just the basics. There are more efficient ways to preprocess your text that you will want to consider. Read Chapter 8 "spaCy and textaCy" to learn more!

# In[ ]:




