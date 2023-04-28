#!/usr/bin/env python
# coding: utf-8

# # Solutions
# 2023 April 20
# 
# <a target="_blank" href="https://colab.research.google.com/github/EastBayEv/SSDS-TAML/blob/main/spring2023/Solutions.ipynb">
#   <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
# </a>Example solutions for challenge exercises from each chapter in this book.

# ## Chapter 1 - Exercises
# 
# 1. You will find challenge exercises to work on at the end of each chapter. They will require you to write code such as that found in the cell at the top of this notebook. 
# 
# 2. Click the "Colab" badge at the top of this notebook to open it in the Colaboratory environment. Press `shift` and `enter` simultaneously on your keyboard to run the code and draw your lucky card!
# 
# > Remember: Press `shift` and `enter` on your keyboard to run a cell.

# In[1]:


# import necessary librarys to make the code work
import random
import calendar
from datetime import date, datetime


# In[2]:


# define the deck and suits as character strings and split them on the spaces
deck = 'ace two three four five six seven eight nine ten jack queen king'.split()
suit = 'spades clubs hearts diamonds'.split()
print(deck)
print(suit)


# In[3]:


# define today's day and date
today = calendar.day_name[date.today().weekday()]
date = datetime.today().strftime('%Y-%m-%d')
print(today)
print(date)


# In[4]:


# randomly sample the card value and suit
select_value = random.sample(deck, 1)[0]
select_suit = random.sample(suit, 1)[0]
print(select_value)
print(select_suit)


# In[5]:


# combine the character strings and variables into the final statement
print("\nWelcome to TAML at SSDS!")
print("\nYour lucky card for " + today + " " + date + " is: " + select_value + " of " + select_suit)


# ## Chapter 2 - Exercises
# 
# 1. (Required) Set up your Google Colaboratory (Colab) environment following the instructions in #1 listed above. 
# 2. (Optional) Check that you can correctly open these notebooks in Jupyter Lab. 
# 3. (Optional) Install Python Anaconda distribution on your machine.
# 
# > See 2_Python_environments.ipynb for instructions.

# ## Chapter 3 - Exercises
# 
# 1. Define one variablez for each of the four data types introduced above: 1) string, 2) boolean, 3) float, and 4) integer. 
# 2. Define two lists that contain four elements each. 
# 3. Define a dictionary that containts the two lists from #2 above.
# 4. Import the file "dracula.txt". Save it in a variable named `drac`
# 5. Import the file "penguins.csv". Save it in a variable named `pen`
# 6. Figure out how to find help to export just the first 1000 characters of `drac` as a .txt file named "dracula_short.txt"
# 7. Figure out how to export the `pen` dataframe as a file named "penguins_saved.csv"
# 
# If you encounter error messages, which ones? 

# In[6]:


#1 
string1 = "Hello!"
string2 = "This is a sentence."
print(string1)
print(string2)


# In[7]:


bool1 = True
bool2 = False
print(bool1)
print(bool2)


# In[8]:


float1 = 3.14
float2 = 12.345
print(float1)
print(float2)


# In[9]:


integer1 = 8
integer2 = 4356
print(integer1)
print(integer2)


# In[10]:


#2
list1 = [integer2, string2, float1, "My name is:"]
list2 = [3, True, "What?", string1]
print(list1)
print(list2)


# In[11]:


#3
dict_one = {"direction": "up",
           "code": 1234,
           "first_list": list1,
           "second_list": list2}
dict_one


# In[12]:


#4
# !wget -P data/ https://raw.githubusercontent.com/EastBayEv/SSDS-TAML/main/spring2023/data/dracula.txt
drac = open("data/dracula.txt").read()
# print(drac)


# In[13]:


#5
import pandas as pd
# !wget -P data/ https://raw.githubusercontent.com/EastBayEv/SSDS-TAML/main/spring2023/data/penguins.csv
pen = pd.read_csv("data/penguins.csv")
pen


# In[14]:


#6
# first slice the string you want to save
drac_short = drac[:1000]

# second, open in write mode and write the file to the data directory!
with open('data/dracula_short.txt', 'w', encoding='utf-8') as f:
    f.write(drac_short)


# In[15]:


# You can also copy files from Colab to your Google Drive
# Mount your GDrive
# from google.colab import drive
# drive.mount('/content/drive')

# Copy a file from Colab to GDrive
# !cp data/dracula_short.txt /content/drive/MyDrive


# In[16]:


#7
pen.to_csv("data/penguins_saved.csv")

# !cp data/penguins_saved.csv /content/drive/MyDrive


# ## Chapter 4 - Exercises
# 
# 1. Load the file "gapminder-FiveYearData.csv" and save it in a variable named `gap`
# 2. Print the column names
# 3. Compute the mean for one numeric column
# 4. Compute the mean for all numeric columns
# 5. Tabulate frequencies for the "continent" column
# 6. Compute mean lifeExp and dgpPercap by continent
# 7. Create a subset of `gap` that contains only countries with lifeExp greater than 75 and gdpPercap less than 5000.

# In[17]:


#1
import pandas as pd
# !wget -P data/ https://raw.githubusercontent.com/EastBayEv/SSDS-TAML/main/spring2023/data/gapminder-FiveYearData.csv
gap = pd.read_csv("data/gapminder-FiveYearData.csv")
gap


# In[18]:


#2
gap.columns


# In[19]:


#3
gap["lifeExp"].mean()


# In[20]:


# or
gap.describe()


# In[21]:


#4
print(gap.mean())


# In[22]:


#5
gap["continent"].value_counts()


# In[23]:


#6
le_gdp_by_continent = gap.groupby("continent").agg(mean_le = ("lifeExp", "mean"), 
                                                  mean_gdp = ("gdpPercap", "mean"))
le_gdp_by_continent


# In[24]:


#7 
gap_75_1000 = gap[(gap["lifeExp"] > 75) & (gap["gdpPercap"] < 5000)]
gap_75_1000


# ## Chapter 5 - Penguins Exercises
# 
# Learn more about the biological and spatial characteristics of penguins! 
# 
# 1. Use seaborn to make a scatterplot of two continuous variables. Color each point by species. 
# 2. Make the same scatterplot as #1 above. This time, color each point by sex. 
# 3. Make the same scatterplot as #1 above again. This time color each point by island.
# 4. Use the `sns.FacetGrid` method to make faceted plots to examine "flipper_length_mm" on the x-axis, and "body_mass_g" on the y-axis. 

# In[25]:


import pandas as pd
import seaborn as sns
# !wget -P data/ https://raw.githubusercontent.com/EastBayEv/SSDS-TAML/main/spring2023/data/penguins.csv
peng = pd.read_csv("data/penguins.csv")


# In[26]:


# set seaborn figure size, background theme, and axis and tick label size
sns.set(rc={'figure.figsize':(10, 7)})
sns.set(font_scale = 2)
sns.set_theme(style='ticks')


# In[27]:


peng


# In[28]:


#1
sns.scatterplot(data = peng, x = "flipper_length_mm", y = "body_mass_g", 
                hue = "species",
               s = 250, alpha = 0.75);


# In[29]:


#2
sns.scatterplot(data = peng, x = "flipper_length_mm", y = "body_mass_g", 
                hue = "sex",
               s = 250, alpha = 0.75, 
                
               palette = ["red", "green"]).legend(title = "Species",
                                                  fontsize = 20, 
                                                  title_fontsize = 30,
                                                 loc = "best");


# In[30]:


#3
sns.scatterplot(data = peng, x = "flipper_length_mm", y = "body_mass_g", 
                hue = "island").legend(loc = "lower right");


# In[31]:


#4
facet_plot = sns.FacetGrid(data = peng, col = "island",  row = "sex")
facet_plot.map(sns.scatterplot, "flipper_length_mm", "body_mass_g");


# ## Chapter 5 - Gapminder Exercises
# 
# 1. Figure out how to make a line plot that shows gdpPercap through time. 
# 2. Figure out how to make a second line plot that shows lifeExp through time. 
# 3. How can you plot gdpPercap with a different colored line for each continent? 
# 4. Plot lifeExp with a different colored line for each continent. 

# In[32]:


import pandas as pd
import seaborn as sns
# !wget -P data/ https://raw.githubusercontent.com/EastBayEv/SSDS-TAML/main/spring2023/data/gapminder-FiveYearData.csv
gap = pd.read_csv("data/gapminder-FiveYearData.csv")


# In[33]:


#1
sns.lineplot(data = gap, x = "year", y = "gdpPercap", ci = 95);


# In[34]:


#2
sns.lineplot(data = gap, x = "year", y = "lifeExp", ci = False);


# In[35]:


#3
sns.lineplot(data = gap, x = "year", y = "gdpPercap", hue = "continent", ci = False);


# In[36]:


#4
sns.lineplot(data = gap, x = "year", y = "lifeExp", 
             hue = "continent", ci = False);


# In[37]:


#4 with custom colors
sns.lineplot(data = gap, x = "year", y = "lifeExp", 
             hue = "continent", 
             ci = False, 
            palette = ["#00FFFF", "#458B74", "#E3CF57", "#8A2BE2", "#CD3333"]);

# color hex codes: https://www.webucator.com/article/python-color-constants-module/
# seaborn color palettes: https://www.reddit.com/r/visualization/comments/qc0b36/all_seaborn_color_palettes_together_so_you_dont/


# ## Exercise - scikit learn's `LinearRegression()` function
# 
# 1. Compare our "by hand" OLS results to those producd by sklearn's `LinearRegression` function. Are they the same? 
#     * Slope = 4
#     * Intercept = -4
#     * RMSE = 2.82843
#     * y_hat = y_hat = B0 + B1 * data.x

# In[38]:


#1 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# In[39]:


# Recreate dataset
import pandas as pd
data = pd.DataFrame({"x": [1,2,3,4,5],
                     "y": [2,4,6,8,20]})
data


# In[40]:


# Our "by hand" OLS regression information:
B1 = 4
B0 = -4
RMSE = 2.82843
y_hat = B0 + B1 * data.x


# In[41]:


# use scikit-learn to compute R-squared value
lin_mod = LinearRegression().fit(data[['x']], data[['y']])
print("R-squared: " + str(lin_mod.score(data[['x']], data[['y']])))


# In[42]:


# use scikit-learn to compute slope and intercept
print("scikit-learn slope: " + str(lin_mod.coef_))
print("scikit-learn intercept: " + str(lin_mod.intercept_))


# In[43]:


# compare to our by "hand" versions. Both are the same!
print(int(lin_mod.coef_) == B1)
print(int(lin_mod.intercept_) == B0)


# In[44]:


# use scikit-learn to compute RMSE
RMSE_scikit = round(mean_squared_error(data.y, y_hat, squared = False), 5)
print(RMSE_scikit)


# In[45]:


# Does our hand-computed RMSE equal that of scikit-learn at 5 digits?? Yes!
print(round(RMSE, 5) == round(RMSE_scikit, 5))


# ## Chapter 7 - Exercises - redwoods webscraping
# 
# This also works with data scraped from the web. Below is very brief BeautifulSoup example to save the contents of the Sequoioideae (redwood trees) Wikipedia page in a variable named `text`. 
# 
# 1. Read through the code below
# 2. Practice by repeating for a webpage of your choice

# In[46]:


#1 
# See 7_English_preprocessing_basics.ipynb


# In[47]:


#2
from bs4 import BeautifulSoup
import requests
import regex as re
import nltk


# In[48]:


url = "https://en.wikipedia.org/wiki/Observable_universe"
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html')


# In[49]:


text = ""

for paragraph in soup.find_all('p'):
    text += paragraph.text


# In[50]:


text = re.sub(r'\[[0-9]*\]',' ',text)
text = re.sub(r'\s+',' ',text)
text = re.sub(r'\d',' ',text)
text = re.sub(r'[^\w\s]','',text)
text = text.lower()
text = re.sub(r'\s+',' ',text)


# In[51]:


# print(text)


# ## Chapter 7 - Exercise - _Dracula_ versus _Frankenstein_
# 
# 1. Practice your text pre-processing skills on the classic novel Dracula! Here you'll just be performing the standardization operations on a text string instead of a DataFrame, so be sure to adapt the practices you saw with the UN HRC corpus processing appropriately. 
# 
#     Can you:
#     * Remove non-alphanumeric characters & punctuation?
#     * Remove digits?
#     * Remove unicode characters?
#     * Remove extraneous spaces?
#     * Standardize casing?
#     * Lemmatize tokens?
# 
# 2. Investigate classic horror novel vocabulary. Create a single TF-IDF sparse matrix that contains the vocabulary for _Frankenstein_ and _Dracula_. You should only have two rows (one for each of these novels), but potentially thousands of columns to represent the vocabulary across the two texts. What are the 20 most unique words in each? Make a dataframe or visualization to illustrate the differences.
# 
# 3. [Read through this 20 newsgroups dataset example](https://scikit-learn.org/0.19/datasets/twenty_newsgroups.html) to get familiar with newspaper data. Do you best to understand and explain what is happening at each step of the workflow. "The 20 newsgroups dataset comprises around 18000 newsgroups posts on 20 topics split in two subsets: one for training (or development) and the other one for testing (or for performance evaluation). The split between the train and test set is based upon a messages posted before and after a specific date."

# In[52]:


# 1
import regex as re
from string import punctuation
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import pandas as pd
from collections import Counter
import seaborn as sns


# ### Import dracula.txt

# In[53]:


# !wget -P data/ https://raw.githubusercontent.com/EastBayEv/SSDS-TAML/main/spring2023/data/dracula.txt
text = open("data/dracula.txt").read()

# print just the first 100 characters
print(text[:100])


# ### Standardize Text 
# 
# ### Casing and spacing
# 
# Oftentimes in text analysis, identifying occurences of key word(s) is a necessary step. To do so, we may want "apple," "ApPLe," and "apple      " to be treated the same; i.e., as an occurence of the token, 'apple.' To achieve this, we can standardize text casing and spacing: 

# In[54]:


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

# In[55]:


print(punctuation)


# In[56]:


for char in punctuation:
    text = text.lower().replace(char, "")


# In[57]:


print(text[:100])


# ### Tokenize the text
# 
# Split each word on spaces.

# In[58]:


# .split() returns a list of the tokens in a string, separated by the specified delimiter (default: " ")
tokens = text.split()


# In[59]:


# View the first 20
print(tokens[:20])


# ### Remove stop words
# 
# Below is a list comprehension (a sort of shortcut for loop, or chunk of repeating code) that can accomplish this task for us.

# In[60]:


filtered_text = [word for word in tokens if word not in stopwords.words('english')]


# In[61]:


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

# In[62]:


stmer = nltk.PorterStemmer()

lmtzr = nltk.WordNetLemmatizer()


# In[63]:


# do you see any differences?
token_stem  = [ stmer.stem(token) for token in filtered_text]

token_lemma = [ lmtzr.lemmatize(token) for token in filtered_text ]

print(token_stem[:20])

print(token_lemma[:20])


# ### Part of speech tags
# 
# Part of speech tags are labels given to each word in a text such as verbs, adverbs, nouns, pronouns, adjectives, conjunctions, and their various derivations and subcategories. 

# In[64]:


tagged = nltk.pos_tag(token_lemma)

# Let's see a quick example: 
ex_string = 'They refuse to permit us to obtain the refuse permit.'
print(nltk.pos_tag(ex_string.split())) 


# The output of .pos_tag is a list of tuples (immutable pairs), where the first element is a text token and the second is a part of speech. Note that, in our example string, the token 'refuse' shows up twice - once as a verb, and once as a noun. In the output to .pos_tag, the first tuple with 'refuse' has the 'VBP' tag (present tense verb) and the second tuple has the 'NN' tag (noun). Nifty!

# In[65]:


chunked = nltk.chunk.ne_chunk(tagged)


# ## Convert to dataframe

# In[66]:


df = pd.DataFrame(chunked, columns=['word', 'pos'])
df.head(n = 10)


# In[67]:


df.shape


# ## Visualize the 20 most frequent words

# In[68]:


top = df.copy()

count_words = Counter(top['word'])
count_words.most_common()[:20]


# In[69]:


words_df = pd.DataFrame(count_words.items(), columns=['word', 'count']).sort_values(by = 'count', ascending=False)
words_df[:20]


# In[70]:


# What would you need to do to improve an approach to word visualization such as this one?
top_plot = sns.barplot(x = 'word', y = 'count', data = words_df[:20])
top_plot.set_xticklabels(top_plot.get_xticklabels(),rotation = 40);


# In[71]:


#2
import spacy
import regex as re
from sklearn.feature_extraction.text import TfidfVectorizer


# In[72]:


# Create a new directory to house the two novels
get_ipython().system('mkdir data/novels/')

# Download the two novels
# !wget -P data/novels/ https://raw.githubusercontent.com/EastBayEv/SSDS-TAML/main/spring2023/data/dracula.txt
# !wget -P data/novels/ https://raw.githubusercontent.com/EastBayEv/SSDS-TAML/main/spring2023/data/frankenstein.txt


# In[73]:


# See that they are there!
get_ipython().system('ls data/novels')


# In[74]:


import os
corpus = os.listdir('data/novels/')

# View the contents of this directory
corpus


# In[75]:


empty_dictionary = {}

# Loop through the folder of documents to open and read each one
for document in corpus:
    with open('data/novels/' + document, 'r', encoding = 'utf-8') as to_open:
         empty_dictionary[document] = to_open.read()

# Populate the data frame with two columns: file name and document text
novels = (pd.DataFrame.from_dict(empty_dictionary, 
                                       orient = 'index')
                .reset_index().rename(index = str, 
                                      columns = {'index': 'file_name', 0: 'document_text'}))


# In[76]:


novels


# In[77]:


novels['clean_text'] = novels['document_text'].str.replace(r'[^\w\s]', ' ', regex = True)
novels


# In[78]:


novels['clean_text'] = novels['clean_text'].str.replace(r'\d', ' ', regex = True)
novels


# In[79]:


novels['clean_text'] = novels['clean_text'].str.encode('ascii', 'ignore').str.decode('ascii')
novels


# In[80]:


novels['clean_text'] = novels['clean_text'].str.replace(r'\s+', ' ', regex = True)
novels


# In[81]:


novels['clean_text'] = novels['clean_text'].str.lower()
novels


# In[82]:


# !python -m spacy download en_core_web_sm


# In[83]:


nlp = spacy.load('en_core_web_sm')
novels['clean_text'] = novels['clean_text'].apply(lambda row: ' '.join([w.lemma_ for w in nlp(row)]))
novels


# In[84]:


tf_vectorizer = TfidfVectorizer(ngram_range = (1, 3), 
                                stop_words = 'english', 
                                max_df = 0.50
                                )
tf_sparse = tf_vectorizer.fit_transform(novels['clean_text'])


# In[85]:


tf_sparse.shape


# In[86]:


tfidf_df = pd.DataFrame(tf_sparse.todense(), columns = tf_vectorizer.get_feature_names())
tfidf_df


# In[87]:


tfidf_df.max().sort_values(ascending = False).head(n = 20)


# In[88]:


titles = novels['file_name'].str.slice(stop = -4)
titles = list(titles)
titles


# In[89]:


tfidf_df['TITLE'] = titles
tfidf_df


# In[90]:


# dracula top 20 words
title = tfidf_df[tfidf_df['TITLE'] == 'frankenstein']
title.max(numeric_only = True).sort_values(ascending = False).head(20)


# In[91]:


# dracula top 20 words
title = tfidf_df[tfidf_df['TITLE'] == 'dracula']
title.max(numeric_only = True).sort_values(ascending = False).head(20)


# ## Chapter 8 - Exercises
# 
# 1. Filter the tokens from the HG Well's `text` variable to 1) lowercase all text, 2) remove punctuation, 3) remove spaces and line breaks, 4) remove numbers, and 5) remove stopwords - all in one line! 

# In[92]:


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


# In[93]:


# Once we've installed the model, we can import it like any other Python library
import en_core_web_md

# This instantiates a spaCy text processor based on the installed model
nlp = en_core_web_md.load()


# In[94]:


# Apply the pipeline
doc = nlp(text)


# In[95]:


# lowercase all text
clean = [token.lower_ for token in doc if \
# remove punctuation
token.is_punct == False and \
# remove spaces and line breaks
token.is_space == False and \
# remove numbers
token.is_alpha == True and \
# remove (english) stopwords
token.is_stop == False]


# In[96]:


print(clean)


# 2. Read through the spacy101 guide and begin to apply its principles to your own corpus: https://spacy.io/usage/spacy-101

# ## Chapter 9 - Exercise
# 
# 1. Repeat the steps in this notebook with your own data. However, real data does not come with a `fetch` function. What importation steps do you need to consider so your own corpus works?
