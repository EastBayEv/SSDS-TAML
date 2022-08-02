#!/usr/bin/env python
# coding: utf-8

# # Chapter 1 - English text preprocessing basics

# 2022 January 19

# ![text](img/text.png)

# Unstructured text - text you find in the wild in books and websites - is generally not amenable to analysis. Before it can be analyzed, the text needs to be standardized to a format so that Python can recognize each unit of meaning (called a "token") as unique, no matter how many times it occurs and how it is stylized. 
# 
# Although not an exhaustive list, some key steps in preprocessing text include:  
# * Standardizing text casing and text spacing 
# * Remove punctuation and special characters/symbols
# * Remove stop words
# * Stem or lemmatize: convert all non-base words to their base form 
# 
# Stemming/lemmatization and stop words (and some punctuation) are language-specific. NLTK works for English out-of-the-box, but you'll need different code to work with other languages. Some languages (e.g. Chinese) also require *segmentation*: artificially inserting spaces between words. If you want to do text pre-processing for other languages, please let us know and we can put together a notebook for you.

# In[1]:


from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
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
nltk.download('omw-1.4')


# In[3]:


text = open("data/dracula.txt").read()

# print just the first 100 characters
print(text[:100])


# ### Standardize Text casing & spacing
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


print(tokens[:20])


# ### Remove stop words
# 
# 
# 
# Below is a list comprehension (a sort of shortcut for loop) that can accomplish this task for us.

# In[10]:


filtered_text = [word for word in tokens if word not in stopwords.words('english')]


# In[11]:


# show only the first 100 words
print(filtered_text[:100])


# ### Lemmatizing/Stemming tokens
# 
# Lemmatizating and stemming are related, but different practices. Both processes aim to reduce the inflectional forms of a token to a common base/root. However, how they go about doing so is the key differentiating factor.  
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


# In[13]:


token_stem  = [ stmer.stem(token) for token in filtered_text]

token_lemma = [ lmtzr.lemmatize(token) for token in filtered_text ]

print(token_stem[:10])

print(token_lemma[:10])


# ### Part of speech tags
# 
# Part of speech tags are labels given to each word in a text such as verbs, adverbs, nouns, pronouns, adjectives, conjunctions, and their various derivations and subcategories. 

# In[14]:


tagged = nltk.pos_tag(token_lemma)

# Let's see a quick example: 
ex_string = 'They refuse to permit us to obtain the refuse permit.'
print(nltk.pos_tag(ex_string.split())) 


# The output of .pos_tag is a list of tuples (pairs), where the first element is a text token and the second is a part of speech. Note that, in our example string, the token 'refuse' shows up twice - once as a verb, and once as a noun. In the output to .pos_tag, the first tuple with 'refuse' has the 'VBP' tag (present tense verb) and the second tuple has the 'NN' tag (noun). Nifty!

# In[15]:


chunked = nltk.chunk.ne_chunk(tagged)


# ### Convert to dataframe

# In[16]:


df = pd.DataFrame(chunked, columns=['word', 'pos'])


# In[17]:


df.head()


# In[18]:


df.shape


# ### Visualize the 20 most frequent words

# In[19]:


top = df.copy()

count_words = Counter(top['word'])
count_words.most_common()[:20]


# In[20]:


words_df = pd.DataFrame(count_words.items(), columns=['word', 'count']).sort_values(by = 'count', ascending=False)


# In[21]:


words_df[:20]


# In[22]:


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

# In[23]:


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

# In[24]:


url = "https://en.wikipedia.org/wiki/Sequoioideae"
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html')


# ### Get the text
# 
# HTML (hypertext markup language) is used to structure a webpage and the content it contains, including text.
# 
# Below is a handy for loop that finds all everything within paragraph `<p>` tags. 

# In[25]:


# save in an empty string
text = ""

for paragraph in soup.find_all('p'):
    text += paragraph.text


# In[26]:


print(text)


# ### Regular expressions
# 
# Remember how we did preprocessing the long way above? You might find that using egular expressions are easier. [Check out the tutorial](https://docs.python.org/3/library/re.html) and [cheatsheet](https://www.dataquest.io/blog/regex-cheatsheet/) to find out what the below symbols mean and write your own code.

# In[27]:


text = re.sub(r'\[[0-9]*\]',' ',text)
text = re.sub(r'\s+',' ',text)
text = re.sub(r'\d',' ',text)
text = re.sub(r'[^\w\s]','',text)
text = text.lower()


# In[28]:


print(text)


# ## Going further: n-grams
# 
# We have used indivual words in this example, but what about [n-grams?](https://en.wikipedia.org/wiki/N-gram) Also read through this [n-gram language model with nltk](https://www.kaggle.com/alvations/n-gram-language-model-with-nltk). 
# 
# There are also more optimal ways to preprocess your text. Check out the [spaCy 101](https://spacy.io/usage/spacy-101) guide to try it out yourself and attend the CIDR Python Introduction to Text Analysis workshop on February 8, 2022. [Register here](https://appointments.library.stanford.edu/calendar/ssds/cidr-python-text). 
