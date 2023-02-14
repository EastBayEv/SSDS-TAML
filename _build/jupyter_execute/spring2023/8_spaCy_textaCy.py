#!/usr/bin/env python
# coding: utf-8

# # Chapter 8 - spaCy and textaCy
# 
# 2022 August 22 
# > These abridged materials are borrowed from the CIDR Workshop [Text Analysis with Python](https://github.com/sul-cidr/Workshops/tree/master/Text_Analysis_with_Python)

# <a target="_blank" href="https://colab.research.google.com/github/EastBayEv/SSDS-TAML/blob/main/fall2022/8_spaCy_textaCy.ipynb">
#   <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
# </a>

# ## Why spaCy and textacy?
# 
# The language processing features of spaCy and the corpus analysis methods of textacy together offer a wide range of functionality for text analysis in a well-maintained and well-documented software package that incorporates cutting-edge techniques as well as standard approaches.
# 
# The "C" in spaCy (and textacy) stands for Cython, which is Python that is compiled to C code and thus offers some performance advantages over interpreted Python, especially when working with large machine-learning models. The use of machine-learning models, including neural networks, is a key feature of spaCy and textacy. The writers of these libraries also have developed [Prodigy](https://prodi.gy/), a similarly leading-edge but approachable tool for training custom machine-learning models for text analysis, among other uses.
# 
# Check out the [spaCy 101](https://spacy.io/usage/spacy-101) to learn more. 

# ### Topics
# 
# - Document Tokenization
# - Part-of-Speech (POS) Tagging
# - Named-Entity Recognition (NER)
# - Corpus Vectorization
# - Topic Modeling
# - Document Similarity
# - Stylistic Analysis
# 
# **Note:** The examples from this workshop use English texts, but all of the methods are applicable to other languages. The availability of specialized resources (parsing rules, dictionaries, trained models) can vary considerably by language, however.
# 
# ### A brief word about terms
# 
# **Text analysis** involves extraction of information from significant amounts  of free-form text, e.g., literature (prose, poetry), historical records, long-form survey responses, legal documents. Some of the techniques used also are applicable to short-form text data, including documents that are already in tabular format.
# 
# Text analysis methods are built upon techniques for **Natural Language Processing** (NLP), which began as rule-based approaches to parsing human language and eventually incorporated statistical machine learning methods as well as, most recently, neural network/deep learning-based approaches.
# 
# **Text mining** typically refers to the extraction of information from very large corpora of unstructured texts.

# In[1]:


# !pip install textacy
import spacy
import textacy


# # Document-level analysis with `spaCy`
# 
# Let's start by learning how spaCy works and using it to begin analyzing a single text document. We'll work with larger corpora later in the workshop.

# For this workshop we will work with a pre-trained statistical and deep-learning model provided by spaCy to process text. spaCy's [models](https://spacy.io/models) are differentiated by language (21 languages are supported at present), capabilities, training text, and size. Smaller models are more efficient; larger models are more accurate. Here we'll download and use a medium-sized English multi-task model, which supports part of speech tagging, entity recognition, and includes a word vector model.

# In[2]:


get_ipython().system('python -m spacy download en_core_web_md')


# In[3]:


# Once we've installed the model, we can import it like any other Python library
import en_core_web_md


# In[4]:


# This instantiates a spaCy text processor based on the installed model
nlp = en_core_web_md.load()


# In[5]:


# From H.G. Wells's A Short History of the World, Project Gutenberg 
text = """Even under the Assyrian monarchs and especially under
Sardanapalus, Babylon had been a scene of great intellectual
activity.  {111} Sardanapalus, though an Assyrian, had been quite
Babylon-ized.  He made a library, a library not of paper but of
the clay tablets that were used for writing in Mesopotamia since
early Sumerian days.  His collection has been unearthed and is
perhaps the most precious store of historical material in the
world.  The last of the Chaldean line of Babylonian monarchs,
Nabonidus, had even keener literary tastes.  He patronized
antiquarian researches, and when a date was worked out by his
investigators for the accession of Sargon I he commemorated the
fact by inscriptions.  But there were many signs of disunion in
his empire, and he sought to centralize it by bringing a number of
the various local gods to Babylon and setting up temples to them
there.  This device was to be practised quite successfully by the
Romans in later times, but in Babylon it roused the jealousy of
the powerful priesthood of Bel Marduk, the dominant god of the
Babylonians.  They cast about for a possible alternative to
Nabonidus and found it in Cyrus the Persian, the ruler of the
adjacent Median Empire.  Cyrus had already distinguished himself
by conquering Croesus, the rich king of Lydia in Eastern Asia
Minor.  {112} He came up against Babylon, there was a battle
outside the walls, and the gates of the city were opened to him
(538 B.C.).  His soldiers entered the city without fighting.  The
crown prince Belshazzar, the son of Nabonidus, was feasting, the
Bible relates, when a hand appeared and wrote in letters of fire
upon the wall these mystical words: _"Mene, Mene, Tekel,
Upharsin,"_ which was interpreted by the prophet Daniel, whom he
summoned to read the riddle, as "God has numbered thy kingdom and
finished it; thou art weighed in the balance and found wanting and
thy kingdom is given to the Medes and Persians."  Possibly the
priests of Bel Marduk knew something about that writing on the
wall.  Belshazzar was killed that night, says the Bible.
Nabonidus was taken prisoner, and the occupation of the city was
so peaceful that the services of Bel Marduk continued without
intermission."""


# By default, spaCy applies its entire NLP "pipeline" to the text as soon as it is provided to the model and outputs a processed "doc."
# 
# <img src="https://d33wubrfki0l68.cloudfront.net/3ad0582d97663a1272ffc4ccf09f1c5b335b17e9/7f49c/pipeline-fde48da9b43661abcdf62ab70a546d71.svg">

# In[6]:


doc = nlp(text)


# ## Tokenization
# 
# The doc created by spaCy immediately provides access to the word-level tokens of the text.

# In[7]:


for token in doc[:15]:
    print(token)


# Each of these tokens has a number of properties, and we'll look a bit more closely at them in a minute.
# 
# spaCy also automatically provides sentence-level segmenting (senticization).

# In[8]:


import itertools

for sent in itertools.islice(doc.sents, 10):
    print(sent.text + "\n--\n")


# You'll notice that the line breaks in the sample text are making the extracted sentences and also the word-level tokens a bit messy. The simplest way to avoid this is just to replace all single line breaks from the text with spaces before running it throug the spaCy pipeline, i.e., as a **preprocessing** step.
# 
# There are other ways to handle this within the spaCy pipeline; an important feature of spaCy is that every phase of the built-in pipeline can be replaced by a custom module. One could imagine, for example, writing a replacement sentencizer that takes advantage of the presence of two spaces between all sentences in the sample text. But we will leave that as an exercise for the reader.

# In[9]:


text_as_line = text.replace("\n", " ")

doc = nlp(text_as_line)

for sent in itertools.islice(doc.sents, 10):
    print(sent.text + "\n--\n")


# We can collect both words and sentences into standard Python data structures (lists, in this case).

# In[10]:


sentences = [sent.text for sent in doc.sents]
sentences


# In[11]:


words = [token.text for token in doc]
words


# ### Filtering tokens
# 
# After extracting the tokens, we can use some attributes and methods provided by spaCy, along with some vanilla Python methods, to filter the tokens to just the types we're interested in analyzing.

# In[12]:


# If we're only interested in analyzing word tokens, we can remove punctuation:
for token in doc[:20]:
    print(f'TOKEN: {token.text:15} IS_PUNCTUATION: {token.is_punct:}')
no_punct = [token for token in doc if token.is_punct == False]

no_punct[:20]


# In[13]:


# There are still some space tokens; here's how to remove spaces and newlines:
no_punct_or_space = [token for token in doc if token.is_punct == False and token.is_space == False]
for token in no_punct_or_space[:30]:
    print(token.text)


# In[14]:


# Let's say we also want to remove numbers and lowercase everything that remains
lower_alpha = [token.lower_ for token in no_punct_or_space if token.is_alpha == True]
lower_alpha[:30]


# One additional common filtering step is to remove stopwords. In theory, stopwords can be any words we're not interested in analyzing, but in practice, they are often the most common words in a language that do not carry much semantic information (e.g., articles, conjunctions).

# In[15]:


clean = [token.lower_ for token in no_punct_or_space if token.is_alpha == True and token.is_stop == False]
clean[:30]


# We've used spaCy's built-in stopword list; membership in this list determines the property `is_stop` for each token. It's good practice to be wary of any built-in stopword list, however -- there's a good chance you will want to remove some words that aren't on the list and to include some that are, especially if you're working with specialized texts.

# In[16]:


# We'll just pick a couple of words we know are in the example
custom_stopwords = ["assyrian", "babylon"]

custom_clean = [token for token in clean if token not in custom_stopwords]
custom_clean


# At this point, we have a list of lower-cased tokens that doesn't contain punctuation, white-space, numbers, or stopwords. Depending on your analytical goals, you may or may not want to do this much cleaning, but hopefully you have a greater appreciation for the kinds of cleaning that can be done with spaCy.

# ### Counting tokens
# 
# Now that we've used spaCy to tokenize and clean our text, we can begin one of the most fundamental text analysis tasks: counting words!

# In[17]:


print("Number of tokens in document: ", len(doc))
print("Number of tokens in cleaned document: ", len(clean))
print("Number of unique tokens in cleaned document: ", len(set(clean)))


# In[18]:


from collections import Counter

full_counter = Counter([token.lower_ for token in doc])
full_counter.most_common(20)


# In[19]:


cleaned_counter = Counter(clean)
cleaned_counter.most_common(20)


# ## Part-of-speech tagging
# 
# Let's consider some other aspects of the text that spaCy exposes for us. One of the most noteworthy features is part-of-speech tagging.

# In[20]:


# spaCy provides two levels of POS tagging. Here's the more general level.
for token in doc[:30]:
    print(token.text, token.pos_)


# In[21]:


# spaCy also provides the more specific Penn Treenbank tags.
# https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
for token in doc[:30]:
    print(token.text, token.tag_)


# We can count the occurrences of each part of speech in the text, which may be useful for document classification (fiction may have different proportions of parts of speech relative to nonfiction, for example) or stylistic analysis (more on that later).

# In[22]:


nouns = [token for token in doc if token.pos_ == "NOUN"]
verbs = [token for token in doc if token.pos_ == "VERB"]
proper_nouns = [token for token in doc if token.pos_ == "PROPN"]
adjectives = [token for token in doc if token.pos_ == "ADJ"]
adverbs = [token for token in doc if token.pos_ == "ADV"]


# In[23]:


pos_counts = {
    "nouns": len(nouns),
    "verbs": len(verbs),
    "proper_nouns": len(proper_nouns),
    "adjectives": len(adjectives),
    "adverbs": len(adverbs) 
}

pos_counts


# spaCy performs morphosyntactic analysis of individual tokens, including lemmatizing inflected or conjugated forms to their base (dictionary) forms. Reducing words to their lemmatized forms can help to make a large corpus more manageable and is generally more effective than just stemming words (trimming the inflected/conjugated endings of words until just the base portion remains), but should only be done if the inflections are not relevant to your analysis.

# In[24]:


for token in doc:
    if token.pos_ in ["NOUN", "VERB"] and token.orth_ != token.lemma_:
        print(f"{token.text:15} {token.lemma_}")


# ### Parsing
# 
# spaCy's trained models also provide full dependency parsing, tagging word tokens with their syntactic relations to other tokens. This functionality drives spaCy's built-in senticization as well.
# 
# We won't spend much time exploring this feature, but it's useful to see how it enables the extraction of multi-word "noun chunks" from the text. Note also that textacy (discussed below) has a built-in function to extract subject-verb-object triples from sentences.

# In[25]:


for chunk in itertools.islice(doc.noun_chunks, 20):
    print(chunk.text)


# ## Named-entity recognition
# 
# spaCy's models do a pretty good job of identifying and classifying named entities (people, places, organizations).
# 
# It is also fairly easy to customize and fine-tune these models by providing additional training data (e.g., texts with entities labeled according to the desired scheme), but that's out of the scope of this workshop.

# In[26]:


for ent in doc.ents:
    print(f'{ent.text:20} {ent.label_:15} {spacy.explain(ent.label_)}')


# What if we only care about geo-political entities or locations?

# In[27]:


ent_filtered = [(ent.text, ent.label_) for ent in doc.ents if ent.label_ in ["GPE", "LOC"]]
ent_filtered


# ### Visualizing Parses
# 
# The built-in displaCy visualizer can render the results of the named-entity recognition, as well as the dependency parser.

# In[28]:


from spacy import displacy
displacy.render(doc, style="ent", jupyter=True)


# # Corpus-level analysis with `textacy`

# Let's shift to thinking about a whole corpus rather than a single document. We could analyze multiple documents with spaCy and then knit the results together with some extra Python. Instead, though, we're going to take advantage of textacy, a library built on spaCy that adds corpus analysis features.
# 
# For reference, here's the [online documentation for textacy](https://textacy.readthedocs.io/en/stable/api_reference/root.html).

# ## Generating corpora
# 
# We'll use some of the data that is included in textacy as our corpus. It is certainly possible to build your own corpus by importing data from files in plain text, XML, JSON, CSV or other formats, but working with one of textacy's "pre-cooked" datasets simplifies things a bit.

# In[29]:


import textacy.datasets


# In[30]:


# We'll work with a dataset of ~8,400 ("almost all") U.S. Supreme Court
# decisions from November 1946 through June 2016
# https://github.com/bdewilde/textacy-data/releases/tag/supreme_court_py3_v1.0
data = textacy.datasets.SupremeCourt()


# In[31]:


data.download()


# The documentation indicates the metadata that is available with each text.

# In[32]:


# help(textacy.datasets.supreme_court)


# textacy is based on the concept of a corpus, whereas spaCy focuses on single documents. A textacy corpus is instantiated with a spaCy language model (we're using the one from the first half of this workshop) that is used to apply its analytical pipeline to each text in the corpus, and also given a set of records consisting of texts with metadata (if metadata is available).
# 
# Let's go ahead and define a set of records (texts with metadata) that we'll then add to our corpus. To keep the processing time of the data set a bit more manageable, we'll just look at a set of court decisions from a short span of time.

# In[33]:


from IPython.display import display, HTML, clear_output
corpus = textacy.Corpus(nlp)

# There are 79 docs in this range -- they'll take a minute or two to process
recent_decisions = data.records(date_range=('2010-01-01', '2010-12-31'))

for i, record in enumerate(recent_decisions):
    clear_output(wait=True)
    display(HTML(f"<pre>{i+1:>2}/79: Adding {record[1]['case_name']}</pre>"))
    corpus.add_record(record)

# If the three lines above are taking too long to process all 79 docs,
# comment them out and uncomment the two lines below to download and import
# a preprocessed version of the corpus

#!wget https://github.com/sul-cidr/Workshops/raw/master/Text_Analysis_with_Python/data/scotus_2010.bin.gz
#corpus = textacy.Corpus.load(nlp, "scotus_2010.bin.gz")


# In[34]:


print(len(corpus))
[doc._.preview for doc in corpus[:5]]


# We can see that the type of each item in the corpus is a `Doc` - this is a processed spaCy output document, with all of the extracted features. textacy provides some capacity to work with those features via its API, and also exposes new document-level features, such as ngrams and algorithms to determine a document's readability level, among others.

# We can filter this corpus based on metadata attributes.

# In[35]:


corpus[0]._.meta


# In[36]:


# Here we'll find all the cases where the number of justices voting in the majority was greater than 6. 
supermajorities = [doc for doc in corpus.get(lambda doc: doc._.meta["n_maj_votes"] > 6)]
len(supermajorities)


# In[37]:


supermajorities[0]._.preview


# ## Finding important words in the corpus

# In[38]:


print("number of documents: ", corpus.n_docs)
print("number of sentences: ", corpus.n_sents)
print("number of tokens: ", corpus.n_tokens)


# In[39]:


corpus.word_counts(by="orth_", filter_stops=False, filter_punct=False, filter_nums=False)


# In[40]:


def show_doc_counts(input_corpus, weighting, limit=20):
    doc_counts = input_corpus.word_doc_counts(weighting=weighting, filter_stops=True, by="orth_")
    print("\n".join(f"{a:15} {b}" for a, b in sorted(doc_counts.items(), key=lambda x:x[1], reverse=True)[:limit]))


# `word_doc_counts` provides a few ways of quantifying the prevalence of individual words across the corpus: whether a word appears many times in most documents, just a few times in a few documents, many times in a few documents, or just a few times in most documents.

# In[41]:


print("# DOCS APPEARING IN / TOTAL # DOCS", "\n", "-----------", sep="")
show_doc_counts(corpus, "freq")
print("\n", "LOG(TOTAL # DOCS / # DOCS APPEARING IN)", "\n", "-----------", sep="")
show_doc_counts(corpus, "idf")


# textacy provides implementations of algorithms for identifying words and phrases that are representative of a document (aka **keyterm extraction**).

# In[42]:


from textacy.extract import keyterms as ke


# In[56]:


# corpus[0].text


# In[44]:


# Run the Yake algorithim (Campos et al., 2018) on a given document
key_terms_yake = ke.yake(corpus[0])
key_terms_yake


# ## Keyword in context
# 
# Sometimes researchers find it helpful just to see a particular keyword in context.

# In[45]:


for doc in corpus[:5]:
    print("\n", doc._.meta.get('case_name'), "\n", "-" * len(doc._.meta.get('case_name')), "\n")
    for match in textacy.extract.kwic.keyword_in_context(doc.text, "judgment"):
        print(" ".join(match).replace("\n", " "))


# ## Vectorization
# 
# Let's continue with corpus-level analysis by taking advantage of textacy's vectorizer class, which wraps functionality from `scikit-learn` to count the prevalence of certain tokens in each document of the corpus and to apply weights to these counts if desired. We could just work directly in `scikit-learn`, but it can be nice for mental overhead to learn one library and be able to do a great deal with it.
# 
# We'll create a vectorizer, sticking with the normal term frequency defaults but discarding words that appear in fewer than 3 documents or more than 95% of documents. We'll also limit our features to the top 500 words according to document frequency. This means our feature set, or columns, will have a higher degree of representation across the corpus. We could further scale these counts according to document frequency (or inverse document frequency) weights, or normalize the weights so that they add up to 1 for each document row (L1 norm), and so on.

# In[46]:


import textacy.representations

vectorizer = textacy.representations.Vectorizer(min_df=3, max_df=.95, max_n_terms=500)

tokenized_corpus = [[token.orth_ for token in list(textacy.extract.words(doc, filter_nums=True, filter_stops=True, filter_punct=True))] for doc in corpus]

dtm = vectorizer.fit_transform(tokenized_corpus)
dtm


# We have now have a matrix representation of our corpus, where rows are documents, and columns (or features) are words from the corpus. The value at any given point is the number of times that the word appears in that document. Once we have a document-term matrix, we could do several things with it just within textacy, though we also can pass it into different algorithms within `scikit-learn` or other libraries. 

# In[47]:


# Let's look at some of the terms
vectorizer.terms_list[:20]


# We can see that we are still getting a number of terms which might be filtered out, such as symbols and abbreviations. The most straightforward solutions are to filter the terms against a dictionary during vectorization, which carries the risk of inadvertently filtering words that you'd prefer to keep in the dataset, or curating a custom stopword list, which can be inflexible and time consuming. Otherwise, it is often the case that the corpus analysis tools used with the vectorized texts (e.g., topic modeling or stylistic analysis -- see below) have ways of recognizing and sequestering unwanted terms so that they can be excluded from the results if desired.

# ## Exercise - topic modeling
# 
# 1. Read through the below code to quickly look at one example of what we can do with a vectorized corpus. Topic modeling is very popular for semantic exploration of texts, and there are numerous implementations of it. Textacy uses implementations from scikit-learn. Our corpus is rather small for topic modeling, but just to see how it's done here, we'll go ahead. First, though, topic modeling works best when the texts are divided into approximately equal-sized "chunks." A quick word-count of the corpus will show that the decisions are of quite variable lengths, which will skew the topic model.

# In[48]:


for doc in corpus:
    print(f"{len(doc): >5}  {doc._.meta['case_name'][:80]}")


# We'll re-chunk the texts into documents of not more than 500 words and then recompute the document-term matrix.

# In[49]:


chunked_corpus_unflattened = [
    [text[x:x+500] for x in range(0, len(text), 500)] for text in tokenized_corpus
]
chunked_corpus = list(itertools.chain.from_iterable(chunked_corpus_unflattened))
chunked_dtm = vectorizer.fit_transform(chunked_corpus)
chunked_dtm


# In[50]:


import textacy.tm

model = textacy.tm.TopicModel("lda", n_topics=15)
model.fit(chunked_dtm)
doc_topic_matrix = model.transform(chunked_dtm)


# In[51]:


for topic_idx, top_terms in model.top_topic_terms(vectorizer.id_to_term, top_n=10):
  print(f"{topic_idx: >2} {model.topic_weights(doc_topic_matrix)[topic_idx]: >3.0%}", "|", ", ".join(top_terms))


# ## Document similarity with word2vec and clustering
# 
# spaCy and textacy provide several built-in methods for measuring the degree of similarity between two documents, including a `word2vec`-based approach that computes the semantic similarity between documents based on the word vector model included with the spaCy language model. This technique is capable of inferring, for example, that two documents are topically related even if they don't share any words but use synonyms for a shared concept.
# 
# To evaluate this similarity comparison, we'll compute the similarity of each pair of docs in the corpus, and then branch out into `scikit-learn` a bit to look for clusters based on these similarity measurements.

# In[52]:


import numpy as np

dim = corpus.n_docs

distance_matrix = np.zeros((dim,dim))
    
for i, doc_i in enumerate(corpus):
    for j, doc_j in enumerate(corpus):
        if i == j:
            continue # defaults to 0
        if i > j:
            distance_matrix[i,j] = distance_matrix[j,i]
        else:
            distance_matrix[i,j] = 1 - doc_i.similarity(doc_j)
distance_matrix


# The [OPTICS](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.OPTICS.html) hierarchical density-based clustering algorithm only finds one cluster with its default settings, but an examination of the legal issue types coded to each decision indicates that the `word2vec`-based clustering has indeed produced a group of semantically related documents.

# In[53]:


from sklearn.cluster import OPTICS

clustering = OPTICS(metric='precomputed').fit(distance_matrix)
print(clustering.labels_)


# In[54]:


from itertools import groupby
clusters = groupby(sorted(enumerate(clustering.labels_), key=lambda x: x[1]), lambda x: x[1])

for cluster_label, docs in clusters:
    
    if cluster_label == -1:
        continue

    print(f"Cluster {cluster_label}", "\n---------")
    print("\n".join(
        f"{corpus[i]._.meta['us_cite_id']: <12} | {data.issue_area_codes[corpus[i]._.meta['issue_area']]: <18}"
        f" | {data.issue_codes[corpus[i]._.meta['issue']][:60]}"
        for i, _ in docs
    ))
    print("\n\n")


# ## Exercise - spacy101
# 
# 1. Read through the spacy101 guide and begin to apply its principles to your own corpus: https://spacy.io/usage/spacy-101

# ## Topic modeling - going further
# 
# There are many different approaches to modeling abstract topics in text data, such as [top2vec](https://github.com/ddangelov/Top2Vec) and [lda2vec](https://github.com/cemoody/lda2vec). 
# 
# Click ahead to see our coverage of the BERTopic algorithm in Chapter 10! 
