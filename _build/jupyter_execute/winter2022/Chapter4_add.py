#!/usr/bin/env python
# coding: utf-8

# # Chapter 4.5 - New Developments: Topic Modeling with BERTopic! 
# 
# 2022 July 30
# 
# ![bertopic](img/bert_topic.png)

# ## What is BERTopic? 
# * As part of NLP analysis, it's likely that at some point you will be asked, "What topics are most common in these documents?"  
# 
#     * Though related, this question is definitely distinct from a query like "What words or phrases are most common in this corpus?" 
# 
#         * For example, the sentences "I enjoy learning to code." and "Educating myself on new computer programming techniques makes me happy!" contain wholly unique tokens, but encode a similar sentiment. 
# 
#         * If possible, we would like to extract *generalized topics* instead of specific words/phrases to get an idea of what a document is about. 
# 
# * This is where BERTopic comes in! BERTopic is a cutting-edge methodology that leverages the transformers defining the base BERT technique along with other ML tools to provide a flexible and powerful topic modeling module (with great visualization support as well!)
# 
# * In this notebook, we'll go through the operation of BERTopic's key functionalities and present resources for further exploration. 
# 

# ### Required installs:

# In[1]:


# Installs the base bertopic module:
get_ipython().system('pip install bertopic ')

# If you want to use other transformers/language backends, it may require additional installs: 
get_ipython().system("pip install bertopic[flair] # can substitute 'flair' with 'gensim', 'spacy', 'use'")

# bertopic also comes with its own handy visualization suite: 
get_ipython().system('pip install bertopic[visualization]')


# ### Data sourcing 
# 
# * For this exercise, we're going to use a popular data set, '20 Newsgroups,' which contains ~18,000 newsgroups posts on 20 topics. This dataset is readily available to us through Scikit-Learn: 

# In[2]:


from bertopic import BERTopic
from sklearn.datasets import fetch_20newsgroups

documents = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))['data']

print(documents[0]) # Any ice hockey fans? 


# ## Creating a BERTopic model: 
# 
# * Using the BERTopic module requires you to fetch an instance of the model. When doing so, you can specify multiple different parameters including: 
#     * ```language``` -> the language of your documents
#     * ```min_topic_size``` -> the minimum size of a topic; increasing this value will lead to a lower number of topics 
#     * ```embedding_model``` -> what model you want to use to conduct your word embeddings; many are supported!    

# * For a full list of the parameters and their significance, please see https://github.com/MaartenGr/BERTopic/blob/master/bertopic/_bertopic.py. 
# 
# * Of course, you can always use the default parameter values and instantiate your model as ```model = BERTopic()```. Once you've done so, you're ready to fit your model to your documents! 

# #### *Example instantiation:*

# In[3]:


from sklearn.feature_extraction.text import CountVectorizer 

# example parameter: a custom vectorizer model can be used to remove stopwords from the documents: 
stopwords_vectorizer = CountVectorizer(ngram_range=(1, 2), stop_words='english') 

# instantiating the model: 
model = BERTopic(vectorizer_model = stopwords_vectorizer)


# ### Fitting the model: 
# 
# * The first step of topic modeling is to fit the model to the documents: 

# In[4]:


topics, probs = model.fit_transform(documents)


# * ```.fit_transform()``` returns two outputs:
#     
#     * ```topics``` contains mappings of inputs (documents) to their modeled topic (alternatively, cluster)
#     
#     * ```probs``` contains a list of probabilities that an input belongs to their assigned topic 
# 
# * **Note:** ```fit_transform()``` can be substituted with ```fit()```. ```fit_transform()``` allows for the prediction of new documents but demands additional computing power/time.

# ### Viewing topic modeling results: 
# 
# * The BERTopic module has many built-in methods to view and analyze your fitted model topics. Here are some basics:

# In[6]:


# view your topics: 
topics_info = model.get_topic_info()

# get detailed information about the top five most common topics: 
print(topics_info.head(5))


# * When examining topic information, you may see a topic with the assigned number '-1.' Topic -1 refers to all input outliers which do not have a topic assigned and should typically be ignored during analysis. 
# 
# * Forcing documents into a topic could decrease the quality of the topics generated, so it's usually a good idea to allow the model to discard inputs into this 'Topic -1' bin. 

# In[8]:


# access a single topic: 
print(model.get_topic(topic=0)) # .get_topics() accesses all topics


# In[19]:


# get representative documents for a specific topic: 
print(model.get_representative_docs(topic=0)) # omit the 'topic' parameter to get docs for all topics 


# In[17]:


# find topics similar to a key term/phrase: 
topics, similarity_scores = model.find_topics("sports", top_n = 5)
print("Most common topics:" + str(topics)) # view the numbers of the top-5 most similar topics

# print the initial contents of the most similar topics
for topic_num in topics: 
    print('\nContents from topic number: '+ str(topic_num) + '\n')
    print(model.get_topic(topic_num))
    


# ### Saving/loading models: 
# * One of the most obvious drawbacks of using the BERTopic technique is the algorithm's run-time. But, rather than re-running a script every time you want to conduct topic modeling analysis, you can simply save/load models! 

# In[22]:


# save your model: 
model.save("TAML_ex_model")


# In[23]:


# load it later: 
loaded_model = BERTopic.load("TAML_ex_model")


# ## Visualizing topics:
# * Although the prior methods can be used to manually examine the textual contents of topics, visualizations can be an excellent way to succinctly communicate the same information. 
# 
# * Depending on the visualization, it can even reveal patterns that would be much harder/impossible to see through textual analysis - like inter-topic distance! 
# 
# * Let's see some examples!

# In[18]:


# Create a 2D representation of your modeled topics & their pairwise distances: 
model.visualize_topics()


# In[20]:


# Get the words and probabilities of top topics, but in bar chart form! 
model.visualize_barchart()


# In[21]:


# Evaluate topic similarity through a heat map: 
model.visualize_heatmap()


# ## Conclusion
# * Hopefully you're convinced of how accessible but powerful a technique BERTopic topic modeling can be! There's plenty more to learn about BERTopic than what we've covered here, but you should be ready to get started! 
# 
# * During your adventures, you may find the following resources useful: 
#     * *Original BERTopic Github:* https://github.com/MaartenGr/BERTopic
# 
#     * *BERTopic visualization guide:* https://maartengr.github.io/BERTopic/getting_started/visualization/visualization.html#visualize-terms
#     
#     * *How to use BERT to make a custom topic model:* https://towardsdatascience.com/topic-modeling-with-bert-779f7db187e6
# 
# * Recommended things to look into next include: 
#     - how to select the best embedding model for your BERTopic model; 
# 
#     - controlling the number of topics your model generates; and 
# 
#     - other visualizations and deciding which ones are best for what kinds of documents. 
# 
# * Questions? Please reach out! Anthony Weng, SSDS consultant, is happy to help (contact: ad2weng@stanford.edu)
