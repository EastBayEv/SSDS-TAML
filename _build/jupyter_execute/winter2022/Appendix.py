#!/usr/bin/env python
# coding: utf-8

# # **Appendix**
# 
# * Welcome to the appendix! Here you'll find instructions/guides to various Python techniques, tasks, and operations. If you have any suggestions as to what you'd like a section to be written about, please let us know! ad2weng@stanford.edu would be happy to receive your feedback! 

# ## Appendix A: *Virtual environments in Python*
# 
# When operating in Python, you'll often hear/read the advice that you should set up a "virtual environment" for each project you are working on. What exactly is a virtual environment, and why do you need one for every project? 
# 
# * From Python's official documentation: 
# 
#     * A virtual environment is a Python environment such that the Python interpreter, libraries, and scripts installed into it are isolated from those installed in other virtual environments, and (by default) any libraries installed in a "system" Python; i.e., the version of Python which is installed as part of your operating system.  
# 
# Put another way, activating your project in a virtual environment allows it to become it's own self-contained application. A few advantages of doing this include: 
# 
# * Allows you to download packages into your project without administrator privileges/status. 
# 
# * Compartmentalizes your project materials for easy sharing and replication. 
# 
# * *Avoids inter-project conflicts regarding versions and dependencies for packages.*
# 
# That last point can become especially relevant as you work on multiple projects in Python, as one critical version/dependency for one project can cause your other projects to stop working. And, the process of uninstalling packages and/or switching versions for projects is tedious and time-consuming. 

# With the benefits demonstrated, how exactly do we go about setting up a virtual environment? So glad you asked! 

# ### Appendix A.1: *Virtual environments in Anaconda*
# 
# Anaconda is the preferred distribution for local Python installs, with many functionalities presented in a user-friendly interface and offering a suite of applications to aid in data science projects. You can download it [here](https://www.anaconda.com/products/distribution).
# 
# 
# #### A.1.1: *Setting up a virtual environment*
# * Once you've downloaded Anaconda, you can set up a virtual environment by: 
# 
#     1. Open Anaconda Navigator on your computer. 
# 
#     2. On the left-hand side of the Navigator window, find and click on the button that say 'Environments': 
#     
#         <div>
#         <img src="img/ana_env.png" width="500"/>
#         <div>
# 
#     3. In the 'Environments' page, go to the bottom-left of the page and click the button that says 'Create'.
# 
#         * When you do so, you'll be prompted by a pop-up window to provide a name for your new virtual environment. The location for the virutal environment will be shown to you, and you can install a specific version of Python and/or R: 
#         
#             <div>
#             <img src="img/test_env.png"/>
#             <div>
# 
#         * Click the 'Create' button in the pop-up window, and wait for the virtual environment to finish being created.  

# #### A.1.2: *Activating your virtual environment*
# Now you're ready to use your virtual environment! To work in this environment, anytime you open Anaconda: 
# 
# * Navigate to the 'Environments' page; 
# 
# * Find the environment you want to use; and 
# 
# * Click on it!

# #### A.1.3: *Adding packages to your virutal environment*
# 
# To install packages in this virtual environment, either: 
# 
# * Stay on the 'Environments' page, and on the right-hand side of the page:
#      
#      * Change the field dictating displayed packages to 'Not installed' or 'All': 
# 
#           <div>
#           <img src="img/packages_ana_env.png" width="500"/>
#           <div>
# 
#      * Go to the 'Search Packages' field and type in the name(s) of the package you want to install; 
# 
#      * If your package is available, click the open checkbox to the left of the package name: 
# 
#           <div>
#           <img src="img/install_numpy_base.png" width="500"/>
#           <div>
# 
#      * Once you've selected all of the packages of interest, click the 'Apply' button in the bottom right-hand corner of the page to install them.  
# 
# * Go to the 'Home' page, and (install and then) open the 'CMD.exe Prompt' program:
# 
#      * In your command prompt window, you'll see that you're operating in your previously-selected virtual environment: 
# 
#           <div>
#           <img src="img/my_env_cmd_prompt.png" width="500">
#           <div>
# 
#      * In this window, type ``` pip install [name of package] ``` for each package you want to install/weren't able to install in the 'Environments' page. 

# ## Appendix B: *More on text preprocessing*
# 
# While the exact steps you elect to use for text preprocessing will ultimately depend on applications, there are some more generalizable techniques that you can usually look to apply: 
# 
# * **Expand contractions** - contractions like "don't", "they're", and "it's" all count as unique tokens if punctuation is simply removed (converting them to "dont", "theyre", "its", respectively). Decompose contractions into their constituent words to get more accurate counts of tokens like "is," "they," etc. [pycontractions](https://pypi.org/project/pycontractions/) can be useful here! 
# 
#     * Let's see an example:   
# 

# In[1]:


# required install: 
get_ipython().system('pip install pycontractions')


# In[2]:


from pycontractions import Contractions as ct

# load contractions from a vector model - many models accepted!
contractions = ct('GoogleNews-vectors-negative300.bin')

# optional: load the model before the first .expand_texts call 
ct.load_models() 

example_sentence = "I'd like to know how you're doing! You're her best friend, but I'm your friend too, aren't I?"

# let's see the text, de-contraction-afied!
print(list(ct.expand_texts([example_sentence])))


# * **Remove stopwords** - stopwords are words like "a," "from," and "the" which are typically filtered out from text before analysis as they do not meaningfully contribute to the content of a document. Leaving in stopwords can lead to irrelevant topics in topic modeling, dilute the strength of sentiment in sentiment analysis, etc. 
# 
#     * Here's a quick loop that can help filter out the stopwords from a string: 

# In[ ]:


from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize

example_sentence = "Hi! This is a needlessly wordy sentence with lots of stopwords. My favorite words are: a, the, with, which. You may think that is strange - and it is!"

stop_words = set(stopwords.words('english'))

print("Example stopwords include: " + str(stopwords.words('english')[:5])) # if you want to see what are considered English stopwords by the NLTK package

word_tokens = word_tokenize(example_sentence)

filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]

# let's see the difference!
print(word_tokens)
print(filtered_sentence)


# * Note that different packages have different lists which define stopwords, so make sure you pick a suitable one. Also, feel free to define your own custom stopwords lists!  
# 
# * **Standardize phrases** - oftentimes text preprocessing is carried out as a precursor to a matching exercise (e.g. using company name to merge two databases). In such cases, we may want to standardize key phrases. For example, "My Little Startup, LLC" and "my little startup" clearly refer to the same entity, but will not match currently. 
# 
#     * In such cases, we may need to write a custom script to standardize key phrases, or there may be a packages out there that already do this for us. Let's take a look at one for our example, standardizing company names: 

# In[ ]:


# required install: 
get_ipython().system('pip install cleanco')


# In[ ]:


from cleanco import basename

business_name_one = "My Little Startup, LLC"
cleaned_name_one  = basename(business_name_one) # feel free to print this out! just add: 'print(cleaned_name_one)' below. 

business_name_two = "My Little Startup"
cleaned_name_two  = basename(business_name_two)

# sanity check - are the cleaned company names identical?  
print(cleaned_name_one == cleaned_name_two)


# * How and where you choose to standardize phrases in text will of course depend on your end goal, but there are plenty of resources/examples out there for you to model an approach after if a package doesn't already exist!
# 
# * **Normalize text** - normalization refers to the process of transforming text into a canonical (standard) form. Sometimes, people take this to mean the entire text pre-processing pipeline, but here we're using it to refer to conversions like "2mrrw" to "tomorrow" and "b4" to "before." 
# 
#     * This process is especially useful when using social media comments as your base text for analysis but often requires custom scripting. 
