#!/usr/bin/env python
# coding: utf-8

# # Chapter 1 - English text preprocessing basics

# 2022 January 19

# ![text](img/text.png)

# Unstructured text - text you find in the wild in books and websites - is generally not amenable to analysis. Before it can be analyzed, the text needs to be standardized to a format so that Python can recognize each unit of meaning (called a "token") as unique, no matter how many times it occurs and how it is stylized. 
# 
# * Remove punctuation and special characters/symbols
# * Remove stop words
# * Stem or lemmatize: convert all non-base words to their base form 
# 
# Stemming/lemmatization and stop words (and some punctuation) are language-specific. NLTK works for English out-of-the-box, but you'll need different code to work with other languages. Some languages (e.g. Chinese) also require *segmentation*: artificially inserting spaces between words. If you want to do text pre-processing for other languages, please let us know and we can put together a notebook for you.

# In[1]:


from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from string import punctuation
import pandas as pd
import seaborn as sns
from collections import Counter
import regex as re


# In[2]:


# ensure you have the proper nltk modules
import nltk
nltk.download('words')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')


# In[3]:


text = open("data/dracula.txt").read()

# print just the first 100 characters
print(text[:100])


# ## Remove punctuation
# 
# Remember that Python methods can be chained together. 
# 
# Below, a standard for loop loops through the `punctuation` module to replace any of these characters with nothing.

# In[4]:


print(punctuation)


# In[5]:


for char in punctuation:
    text = text.lower().replace(char, "")


# In[6]:


print(text[:100])


# ### Tokenize the text
# 
# Split each word on spaces.

# In[7]:


tokens = text.split()


# In[8]:


print(tokens[:20])


# ### Remove stop words
# 
# 
# 
# Below is a list comprehension (a sort of shortcut for loop) that can accomplish this task for us.

# In[9]:


filtered_text = [word for word in tokens if word not in stopwords.words('english')]


# In[10]:


# show only the first 100 words
print(filtered_text[:100])


# ### Lemmatize
# 
# Examples include: 
# * Plural to singular (corpora to corpus)
# * Condition (better to good)
# * Gerund (running to run)

# In[11]:


lmtzr = nltk.WordNetLemmatizer()


# In[12]:


token_lemma = [ lmtzr.lemmatize(token) for token in filtered_text ]


# ### Part of speech tags
# 
# Part of speech tags are labels given to each word in a text such as verbs, adverbs, nouns, pronouns, adjectives, conjunctions, and their various derivations and subcategories. 

# In[13]:


tagged = nltk.pos_tag(token_lemma)


# In[14]:


chunked = nltk.chunk.ne_chunk(tagged)


# ### Convert to dataframe

# In[15]:


df = pd.DataFrame(chunked, columns=['word', 'pos'])


# In[16]:


df.head()


# In[17]:


df.shape


# ### Visualize the 20 most frequent words

# In[18]:


top = df.copy()

count_words = Counter(top['word'])
count_words.most_common()[:20]


# In[19]:


words_df = pd.DataFrame(count_words.items(), columns=['word', 'count']).sort_values(by = 'count', ascending=False)


# In[20]:


words_df[:20]


# In[21]:


top_plot = sns.barplot(x = 'word', y = 'count', data = words_df[:20])
top_plot.set_xticklabels(top_plot.get_xticklabels(),rotation = 40);


# ![redwood](img/redwood.png)

# ## Quiz: Redwood webscraping
# 
# This also works with data scraped from the web. Below is very brief BeautifulSoup example to save the contents of the Sequoioideae (redwood trees) Wikipedia page to a variable named `text`. 
# 
# 1. Read through the code below
# 2. Practice by repeating for a webpage of your choice
# 3. Combine methods on this page to produce a ready-to-be analyzed copy of "Frankenstein.txt". This file is located in the `/data` folder

# In[22]:


# import necessary libraries
from bs4 import BeautifulSoup
import requests
import regex
import nltk


# ### Three variables will get you started
# 
# 1. `url` - define the URL to be scraped 
# 2. `response` - perform the get request on the URL 
# 3. `soup` - create the soup object so we can parse the html 

# In[23]:


url = "https://en.wikipedia.org/wiki/Sequoioideae"
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html')


# ### Get the text
# 
# HTML (hypertext markup language) is used to structure a webpage and the content it contains, including text.
# 
# Below is a handy for loop that finds all everything within paragraph `<p>` tags. 

# In[24]:


# save in an empty string
text = ""

for paragraph in soup.find_all('p'):
    text += paragraph.text


# In[25]:


print(text)


# ### Regular expressions
# 
# Remember how we did preprocessing the long way above? You might find that using egular expressions are easier. [Check out the tutorial](https://docs.python.org/3/library/re.html) and [cheatsheet](https://www.dataquest.io/blog/regex-cheatsheet/) to find out what the below symbols mean and write your own code.

# In[26]:


text = re.sub(r'\[[0-9]*\]',' ',text)
text = re.sub(r'\s+',' ',text)
text = re.sub(r'\d',' ',text)
text = re.sub(r'[^\w\s]','',text)
text = text.lower()


# In[27]:


print(text)


# ## Going further
# 
# We have used indivual words in this example, but what about [n-grams?](https://en.wikipedia.org/wiki/N-gram) Also read through this [n-gram language model with nltk](https://www.kaggle.com/alvations/n-gram-language-model-with-nltk). 
# 
# There are also more optimal ways to preprocess your text. Cehck out the [spaCy 101](https://spacy.io/usage/spacy-101) guide to try it out yourself and attend the CIDR Python Introduction to Text Analysis workshop on February 8, 2022. [Register here](https://appointments.library.stanford.edu/calendar/ssds/cidr-python-text). 
