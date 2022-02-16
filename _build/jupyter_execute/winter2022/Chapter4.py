#!/usr/bin/env python
# coding: utf-8

# # Chapter 4 - The BERT algorithm

# 2022 February 16

# ![bande](img/bande.png)

# [... but don't forget about Ernie!](https://www.sesamestreet.org/)

# ## Tensorflow
# 
# We will walkthrough the tensorflow "Classisfy Text with BERT" tutorial for this session: https://www.tensorflow.org/text/tutorials/classify_text_with_bert
# 
# Be sure to go through the below tutorials at some point, since they will help you better contextualize what is happening in the BERT tutorial:
# 1. basic text classification: https://www.tensorflow.org/tutorials/keras/text_classification
# 2. word embeddings: https://www.tensorflow.org/text/guide/word_embeddings
# 3. word2vec: https://www.tensorflow.org/tutorials/text/word2vec
# 
# <h1 style="font-size: 4rem; color: #8C1515">Access the notebooks</h1>
# 
# **It is strongly recommeneded that you download the notebooks (or setup your Colab environment) in advance of our meeting session.** 
# 
# At the top of each tutorial page, click the appropriate button to access the notebooks.
# 
# ![tfdl](img/tfdl.png)
# 
# <h1 style="font-size: 4rem; color: #00505C">Run all code</h1>
# 
# **Also be sure to run all code in advance.**
# 
# The models will likely take 1-2 hours to fit and we will not have time to do so during the walkthrough. 
# 
# <h1 style="font-size: 4rem; color: #2ecc71">Need help?</h1>
# 
# Contact muzzall {at} stanford {dot} edu

# ## Setup and software library installation instructions

# ### Install tensorflow
# 
# If the instructions in the tensorflow "Classify Text with Bert" notebook do not work, try the cell below: 

# In[1]:


# !pip install tensorflow


# Or, setup a virtual environment (you might find this more complicated, but it is worth it in the long run). 
# 
# View the instructions: https://www.tensorflow.org/install/pip

# ### A dependency of the preprocessing for BERT inputs

# In[2]:


# !pip install -q -U tensorflow-text==2.7.3


# ### AdamW optimizer
# 
# Use the AdamW optimizer from tensorflow/models: https://github.com/tensorflow/models

# In[3]:


# !pip install -q -U tensorflow-text==2.7.3


# ### Install pydot and graphviz

# In[4]:


# install pydot
# !pip install pydot


# graphviz installation instructions: https://graphviz.org/download/

# ## Visualizations

# ![accloss](img/accloss.png)

# ![weboard](img/weboard.png)

# ## Semantic similarity
# 
# See "Application Examples" section here: https://github.com/UKPLab/sentence-transformers

# In[5]:


from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('all-MiniLM-L6-v2')

# Two lists of sentences
sentences1 = ['The cat sits outside',
             'A man is playing guitar',
             'The new movie is awesome']

sentences2 = ['The dog plays in the garden',
              'A woman watches TV',
              'The new movie is so great']

#Compute embedding for both lists
embeddings1 = model.encode(sentences1, convert_to_tensor=True)
embeddings2 = model.encode(sentences2, convert_to_tensor=True)

#Compute cosine-similarits
cosine_scores = util.cos_sim(embeddings1, embeddings2)

#Output the pairs with their score
for i in range(len(sentences1)):
    print("{} \t\t {} \t\t Score: {:.4f}".format(sentences1[i], sentences2[i], cosine_scores[i][i]))


# ### Find pairs of sentences with highest cosine similarity scores

# In[6]:


from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')

# Single list of sentences
sentences = ['The cat sits outside',
             'A man is playing guitar',
             'I love pasta',
             'The new movie is awesome',
             'The cat plays in the garden',
             'A woman watches TV',
             'The new movie is so great',
             'Do you like pizza?']

#Compute embeddings
embeddings = model.encode(sentences, convert_to_tensor=True)

#Compute cosine-similarities for each sentence with each other sentence
cosine_scores = util.cos_sim(embeddings, embeddings)

#Find the pairs with the highest cosine similarity scores
pairs = []
for i in range(len(cosine_scores)-1):
    for j in range(i+1, len(cosine_scores)):
        pairs.append({'index': [i, j], 'score': cosine_scores[i][j]})

#Sort scores in decreasing order
pairs = sorted(pairs, key=lambda x: x['score'], reverse=True)

for pair in pairs[0:10]:
    i, j = pair['index']
    print("{} \t\t {} \t\t Score: {:.4f}".format(sentences[i], sentences[j], pair['score']))

