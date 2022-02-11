#!/usr/bin/env python
# coding: utf-8

# In[1]:


pwd


# In[2]:


import spacy
nlp = spacy.load('en_core_web_sm') #, disable=['parser', 'tagger', 'ner'])
# stopwords = en_core_web_sm.Defaults.stop_words
# stops = stopwords.words("english")
stops = nlp.Defaults.stop_words

def normalize(comment, lowercase, remove_stopwords):
    if lowercase:
        comment = comment.lower()
    comment = nlp(comment)
    lemmatized = list()
    for word in comment:
        lemma = word.lemma_.strip()
        if lemma:
            if not remove_stopwords or (remove_stopwords and lemma not in stops):
                lemmatized.append(lemma)
    return " ".join(lemmatized)

music['two'] = music['body'].apply(normalize, lowercase=True, remove_stopwords=True)
# Data['Text_After_Clean'] = Data['Text'].apply(normalize, lowercase=True, remove_stopwords=True)


# In[2]:


# [^A-Za-z0-9 ]+


# In[ ]:


def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(w) for w in text] 


# In[ ]:





# In[ ]:


# !python -m spacy download en_core_web_lg

# !python -m spacy download en_core_web_sm

# nlp = spacy.load('en_core_web_sm')
nlp = spacy.load('en_core_web_sm')

music["clean_text"].apply(lambda row: " ".join([w.lemma_ for w in nlp(row)]))


# In[ ]:





# In[ ]:


Build the vocabulary with CountVectorizer()

Check out this post to see how min_df = and max_df = can be used to change the size of the vocabulary.

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(stop_words = 'english',

                     analyzer = 'word',

                     min_df = 0.001,

                     max_df = 0.50)

cv_vec = cv.fit_transform(music['clean_text'])

cv_vec.shape

print(cv_vec)


# In[ ]:


### Tokenize the "clean_text" variable

# from nltk.tokenize import word_tokenize
# human_rights['clean_text'] = human_rights['clean_text'].apply(lambda x: word_tokenize(x))

# print(human_rights['clean_text'][0])


# In[ ]:





# In[ ]:





# In[ ]:


### Remove unicode characters such as Ð and ð

# First, save the "clean_text" variable as a single sring

long_string = ','.join(list(human_rights["clean_text"].values))

long_string

# encode as ascii

strencode = long_string.encode("ascii", "ignore")

# decode

strdecode = strencode.decode()

print(long_string)

output = ''.join([i if ord(i) < 128 else ' ' for i in long_string])

print(output)

import regex as re

o2 = re.sub(r'\s+',' ', output)

o2

# human_rights['clean_text'] = human_rights['clean_text'].str.replace(r'[\W\_]', ' ', regex = True)

print(human_rights['clean_text'][0])

# re.sub(ur'[\W_]+', u'', s, flags=re.UNICODE)

​

human_rights['Text_processed'][0]

# print(human_rights['Text_processed'][0])

# Save the "Text_processed" column as one long string

long_string = ','.join(list(human_rights["Text_processed"].values))

long_string

from nltk.corpus import stopwords

​

# Tokenize long_string

hr_tokens = long_string.split()

​

# Remove stopwords

stop = stopwords.words("english")

no_stops = [word for word in hr_tokens if word not in stopwords.words('english')]

freq_hr = Counter(no_stops)

​

# Print the 20 most common words

hr_df = pd.DataFrame(freq_hr.most_common(20), columns = ["Word", "Frequency"])

hr_df

​

​

# Encode the documents

vector = vectorizer.transform(human_rights["Text_processed"])

print(vector) #

#

#

#

#

print(vector.shape)

print(type(vector))

