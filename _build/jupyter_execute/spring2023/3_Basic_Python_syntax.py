#!/usr/bin/env python
# coding: utf-8

# # Chapter 3 - Basic Python syntax
# 2023 April 6

# <a target="_blank" href="https://colab.research.google.com/github/EastBayEv/SSDS-TAML/blob/main/spring2023/3_Basic_Python_syntax.ipynb">
#   <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
# </a>

# ![boiler](img/boiler.jpg)

# This chapter explains foundational Python syntax that you can reuse to accomplish many basic data creation, importing, and exporting tasks. 

# ## Variable assignment
# 
# In Python, data are saved in **variables.** Variable names should be simple and descriptive. 
# 
# The process is saving something in a variable is called **variable assignment.** These variables with our saved data are **objects** that, like everything in Python, can be manipulated. 
# 
# Assign a variable by typing its name to the **left** of the equals sign. Whatever is written to the **right** of the equals sign will be saved in the variable. 
# 
# Save a single number inside of a variable named with a single letter. You could read the below lines as "x is defined as one", "two is assigned to y", or most simply put, "z is three":

# In[1]:


# define one variable
x = 1


# In[2]:


# assign multiple variables in a single code cell
x = 1
y = 2
z = 3


# ## Use `print()` to show it on the screen

# In[3]:


print(x)


# In[4]:


# "call" the variables directly!
x


# In[5]:


y


# In[6]:


print(x / y * z)


# In[7]:


x / y * z


# In[8]:


import pandas as pd


# In[9]:


pd.


# ## Functions, arguments, and methods
# 
# Functions, arguments, and methods form the core user framework for Python programming. 
# 1. **Functions:** Perform self-contained actions on a thing.
# 
# 2. **Argument(s):** The "things" (text, values, expressions, datasets, etc.)
# 
# > Note **"parameters"** are the variables as notated during function definition. Arguments are the values we pass into these placeholders while calling the function.  
# ![params](img/params_vs_args.png)

# In[14]:


# Example custom function
def print_name(first, last):
    return("My name is: " + first + " " + last)


# In[15]:


print_name("Nerd", "Squirrel")


# 3. **Methods:** Type-specific functions (i.e., can only use a specific type of data and not other types). Use "dot" notation to utilize methods on a variable or other object. 
# 
# > For example, you will type `gap = pd.read_csv('data/gapminder-FiveYearData.csv')` to use the `read_csv()` method from the pandas library (imported as the alias `pd`) to load the Gapminder data.

# ## Data types
# 
# Everything in Python has a type which determines how we can use it. Data are no exception! Be careful, it is easy to get confused when trying to complete multiple tasks that use lots of different variables!
# 
# Use the `type` function to get the type of any variable if you're unsure. Below are four core data types: 
# 
# 1. `str`: Character string; text. Always wrapped in quotations (single or double are fine)
# 2. `bool`: Boolean `True`/`False`. `True` is stored under the hood as 1, `False` is stored as 0
# 3. `float`: Decimals (floating-point)
# 4. `int`: Whole numbers (positive and negative, including zero)

# In[16]:


# 1. String data
x1 = "This is string data"
print(x1)
print(type(x1))


# In[17]:


# 2. Boolean data
x2 = True
print(x2)
print(type(x2))


# In[18]:


# 3. float (decimals)
# use a decimal to create floats
pi = 3.14
print(pi)
print(type(pi))


# In[19]:


# 4. integer (whole numbers)
# do not use a decimal for integers
amount = 4
print(amount)
print(type(amount))


# ## String addition versus integer addition

# In[20]:


# character strings
'1' + '1'


# In[21]:


# integers
1 + 1


# ## Data structures
# 
# Data can be stored in a variety of ways. Regardless, we can **index** (positionally reference) a portion of a larger data structure or collection. 
# 
# ### Python is zero-indexed!
# 
# Python is a zero-indexed programming language and means that you start counting from zero. Thus, the first element in a collection is referenced by 0 instead of 1. 
# 
# Four structures are discussed below:
# 1. **Lists** are ordered groups of data that are both created and indexed with square brackets `[]`
# 2. **Dictionaries** are _unordered_ groups of "key`:`value" pairs. Use the key to access the value. Curly braces `{}` are used to create and index dictionaries, while a colon `:` separates a key from its corresponding value. 
# 3. **Character strings** can contain text of virtually any length
# 4. **Data Frames** are tabular data organized into rows and columns. Think of an MS Excel spreadsheet!

# ## 1. List

# In[22]:


# Define a list with with square brackets. This list contains two character strings 'shark' and 'dolphin'
animals = ['shark', 'dolphin']

# Print the first list item to the screen
animals[0]


# In[23]:


# Call the second thing (remember Python is zero-indexed)
animals[1]


# Lists can contain elements of almost any data type, including other lists! 
# 
# When this is the case, we can use multi-indices to extract just the piece of data we want. For example, to return only the element "tree":

# In[24]:


# Lists can contain other structures, such as other lists
# To get an element from a list within a list, double-index the original list!
animals = ['shark', 'dolphin', ['dog', 'cat'], ['tree', 'cactus']]
print(animals[3][0])


# Or, to just return the element "cat":

# In[25]:


# print this 'animals' list
print(animals)


# In[26]:


# print just the 3rd thing - the sublist containing 'dog' and 'cat'
print(animals[2])


# In[27]:


# print only 'cat'
print(animals[2][1])


# Lists can also contain elements of different types:

# In[28]:


# Define a heterogeneous list
chimera = ['lion', 0.5, 'griffin', 0.5]
print(type(chimera[0]))
print(type(chimera[1]))


# ## 2. Dictionary

# In[29]:


# Define two dictionaries - apple and orange
apple = {'name': 'apple', 'color': ['red', 'green'], 'recipes': ['pie', 'salad', 'sauce']}
orange = {'name': 'orange', 'color': 'orange', 'recipes': ['juice', 'marmalade', 'gratin']}
apple


# In[30]:


orange


# Combine two dictionaries into one by placing them in a list value `[apple, orange]`, with a key named `fruits`. Call the key to see the value(s)!

# In[31]:


fruits = {'fruits': [apple, orange]}

fruits


# To index just "juice" - under the 'recipes' key for the `orange` dictionary, combine dictionary key and list techniques to tunnel into the hierarchical structure of the dictionary and extract just what you want:

# In[32]:


# Call the newly combined dictionary
fruits


# In[33]:


# Reference the 'fruits' key
fruits['fruits']


# In[34]:


# Index the second thing (orange)
fruits['fruits'][1]


# In[35]:


# Call the 'recipes' key from 'orange' to see the items list
fruits['fruits'][1]['recipes']


# In[36]:


# Return the first thing from the 'recipes' key of the 'orange' dictionary inside of 'fruits'!
fruits['fruits'][1]['recipes'][0]


# ## Data import
# 
# Python offers a variety of methods for importing data. Thankfully, it is quite straightforward to import data from .csv and .txt files. Other formats, such as .xml and .json, are also supported. 
# 
# Below, let's import: 
# 1. Text from a .txt file
# 2. A dataframe from a .csv file

# ### A note about importing data into Google Colab
# 
# Navigating Google Colab's file system can be challenging since it is slightly different from working on your local machine. 
# 
# Run the code below to import the dataset into a temporary subfolder named "data" inside of the main Colab "/content" directory.
# 
# **Run Steps 1-3 below to save a data file in your Google Drive, which can then be importend to your Colab environment**
# 
# > NOTE: Colab is a temporary environment with an idle timeout of 90 minutes and an absolute timeout of 12 hours.
# >
# >There are other ways to use Colab's file system, such as mounting your Google Drive, but if you are having trouble in Colab refer back to these steps to import data used in this bootcamp. Contact SSDS if you want to learn more. 

# In[37]:


# Step 1. Create the directory
get_ipython().system('mkdir data')

# Step 2. “List” the contents of the current working directory
get_ipython().system('ls')

# Step 3. Use wget to download the data file
# learn more about wget: https://www.gnu.org/software/wget/?
# !wget -P  data/ https://raw.githubusercontent.com/EastBayEv/SSDS-TAML/main/spring2023/data/frankenstein.txt


# ### Import text data as a character string
# 
# Import text as a single string using the `open().read()` Python convention.
# 
# Review your basic building blocks from above: 
# 1. `frank` is the name of the variable we will save the text inside of
# 2. `open` is the function we will use to open the text file
# 3. `'data/frankenstein.txt'` is the argument that we provide to the `open` function. This matches the `file` path argument and needs to contain the location for the Frankenstein book.
# 4. `.read()` reads the file as text. 

# In[38]:


frank = open('data/frankenstein.txt').read()
# print(frank)

# print only the first 1000 characters
print(frank[:1000])


# ### Import data frames with the pandas library
# 
# Data frames are programming speak for tabular spreadsheets organized into rows and columns and often stored in useful formats such as .csv (i.e., a spreadsheet)
# 
# .csv stands for "comma-separated values" and means that these data are stored as text files with a comma used to delineate column breaks. 
# 
# For this part, we will use the [pandas](https://pandas.pydata.org/docs/getting_started/intro_tutorials/02_read_write.html) Python library. Remember how to install user-defined libraries from Chapter 2? This is a two step process.

# In[39]:


# Step 1. Physically download and install the library's files (unhashtab the line below to run)
# !pip install pandas


# In[40]:


# Step 2. link the pandas library to our current notebook
# pd is the alias, or shorthand, way to reference the pandas library
import pandas as pd 


# Now, you can use dot notation (type `pd.`)to access the functions within the pandas library. 
# 
# We want the `read.csv` method. Like the .txt file import above, we must provide the file path of the location of the .csv file we want to import. 
# 
# Save it in the variable named `gap`

# ## What is Gross Domestic Product? (GDP)
# 
# GDP is a general estimate of global societal well-being. Learn more about GDP here: https://en.wikipedia.org/wiki/Gross_domestic_product. 
# 
# Learn more about GDP "per capita" (per person): https://databank.worldbank.org/metadataglossary/statistical-capacity-indicators/series/5.51.01.10.gdp

# In[41]:


# Colab users: grab the data! Unhashtag the line below
# !wget -P data/ https://raw.githubusercontent.com/EastBayEv/SSDS-TAML/main/spring2023/data/gapminder-FiveYearData.csv


# In[42]:


gap = pd.read_csv("data/gapminder-FiveYearData.csv")


# In[43]:


# View the data
print(gap)


# In[44]:


gap


# ## Getting help
# 
# The help pages in Python are generally quite useful and tell you everything you need to know - you just don't know it yet! Type a question mark `?` before a funciton to view its help pages.

# In[45]:


# ?pd.read_csv


# ## Error messages
# 
# Python's learning curve can feel creative and beyond frustrating at the same time. Just remember that everyone encounters errors - lots of them. When you do, start debugging by investigating the type of error message you receive. 
# 
# Scroll to the end of the error message and read the last line to find the type of error.  
# 
# *Helpful debugging tools/strategies:*
# 1. Googling the error text, and referring to a forum like StackOverflow  
# 2. (IDE-dependent) Placing breakpoints in your program and using the debugger tool to step through the program  
# 3. Strategically place print() statements to know where your program is reaching/failing to reach  
# 4. Ask a friend! A fresh set of eyes goes a long way when you're working on code.
# 5. Restart your IDE and/or your machine.  
# 6. [Schedule an SSDS consultation](https://library.stanford.edu/research/software-and-services-data-science/schedule-consulting-appointment-contact-us)
# 
# ## Exercises
# 
# 1. Unhashtag the line of code for each error message below
# 2. Run each cell
# 3. Inspect the error messages. Are they helpful?

# ### Syntax errors
# 
# **Invalid syntax error** 
# 
# You have entered invalid syntax, or something python does not understand.

# In[46]:


# x 89 5


# **Indentation error**
# 
# Your indentation does not conform to the rules. 
# 
# Highlight the code in the cell below and press `command` and `/` (Mac) or `Ctrl` and `/` on Windows to block comment/uncomment multiple lines of code:

# In[47]:


# def example():
#     test = "this is an example function"
#     print(test)
#      return example


# ### Runtime errors
# 
# **Name error** 
# 
# You try to call a variable you have not yet assigned

# In[48]:


# p


# Or, you try to call a function from a library that you have not yet imported

# In[49]:


# example()


# **Type error**
# 
# You write code with incompatible types

# In[50]:


# "5" + 5


# **Index error**
# 
# You try to reference something that is out of range

# In[51]:


my_list = ['green', True, 0.5, 4, ['cat', 'dog', 'pig']]
# my_list[5]


# ### File errors
# 
# **File not found**
# 
# You try to import something that does not exist

# In[52]:


# document = open('fakedtextfile.txt').read()


# ## Exercises
# 
# 1. Define one variable for each of the four data types introduced above: 1) string, 2) boolean, 3) float, and 4) integer. 
# 2. Define two lists that contain four elements each. 
# 3. Define a dictionary that containts the two lists from #2 above.
# 4. Import the file "dracula.txt". Save it in a variable named `drac`
# 5. Import the file "penguins.csv". Save it in a variable named `pen`
# 6. Figure out how to find help to export just the first 1000 characters of `drac` as a .txt file named "dracula_short.txt"
# 7. Figure out how to export the `pen` dataframe as a file named "penguins_saved.csv"
# 
# If you encounter error messages, which ones? 
# 
# > Note: See the Solutions chapter for code to copy files from your Colab environment to your Google Drive! 

# ## What variables do you have saved? 
# 
# It's easy to lose track of what variables you have saved. `%whos` will provide you with details about variables you have saved in memory. `%who` will simply list the variable names. 

# In[53]:


get_ipython().run_line_magic('whos', '')


# In[54]:


get_ipython().run_line_magic('who', '')


# ## Numeric data wrangling
# 
# Importing numeric data from a .csv file is one thing, but wrangling it into a format that suits your needs is another. Read Chapter 4 "Numeric data wrangling" to learn how to use the pandas library to subset numeric data! 
# 
# Chapter 7 contains information about preprocessing text data. 
