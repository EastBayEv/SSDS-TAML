#!/usr/bin/env python
# coding: utf-8

# # Chapter 3 - Basic Python syntax
# 2022 August 25

# <a target="_blank" href="https://colab.research.google.com/github/EastBayEv/SSDS-TAML/blob/main/fall2022/3_Boilerplate_code_review.ipynb">
#   <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
# </a>

# ![boiler](img/boiler.jpg)

# This chapter explains foundational Python syntax that you can reuse to accomplish many basic data creation, importing, and exporting tasks. 

# ## Variable assignment
# 
# In Python, data are saved in **variables.** Variable names should be simple and descriptive. 
# 
# The process is saving something in a variable is called **variable assignment.**
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


# ## Functions, arguments, and methods
# 
# Functions, arguments, and methods form the core user framework for Python programming. 
# 1. **Functions:** Perform self-contained actions on a thing.
# 
# 2. **Argument(s):** The "things" (text, values, expressions, datasets, etc.)
# 
# > Note **"parameters"** are the variables as notated during function definition. Arguments are the values we pass into these placeholders while calling the function.  
# ![params](img/params_vs_args.png)

# In[8]:


# Example custom function
def print_name(first, last):
    return("My name is: " + first + " " + last)


# In[9]:


print_name("Nerd", "Squirrel")


# 3. **Methods:** Type-specific functions (i.e., can only use a specific type of data and not other types). Use "dot" notation to utilize methods on a variable or other object. 
# 
# > For example, you will type `gap = pd.read_csv('data/gapminder-FiveYearData.csv')` to use the `read_csv()` method from the pandas library (imported as the alias `pd`) to load the Gapminder data.

# ## Data types
# 
# Everything in Python has a type which determines how we can use and manipulate it. Data are no exception! Be careful, it is easy to get confused when trying to complete multiple tasks that use lots of different variables!
# 
# Use the `type` function to get the type of any variable if you're unsure. Below are four core data types: 
# 
# 1. `str`: Character string; text. Always wrapped in quotations (single or double are fine)
# 2. `bool`: Boolean `True`/`False`. `True` is stored under the hood as 1, `False` is stored as 0
# 3. `float`: Decimals (floating-point)
# 4. `int`: Whole numbers (positive and negative, including zero)

# In[10]:


# 1. String data
x1 = "This is string data"
print(x1)
print(type(x1))


# In[11]:


# 2. Boolean data
x2 = True
print(x2)
print(type(x2))


# In[12]:


# 3. float (decimals)
# use a decimal to create floats
pi = 3.14
print(pi)
print(type(pi))


# In[13]:


# 4. integer (whole numbers)
# do not use a decimal for integers
amount = 4
print(amount)
print(type(amount))


# ## String addition versus integer addition

# In[14]:


# character strings
'1' + '1'


# In[15]:


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
# 2. **Dictionaries** are _unordered_ groups of "key:value" pairs. Use the key to access the value. Curly braces `{}` are used to create and index dictionaries
# 3. **Character strings** can contain text of virtually any length
# 4. **Data Frames** are tabular data organized into rows and columns. Think of an MS Excel spreadsheet!

# ## 1. List

# In[16]:


# Define a list with with square brackets. This list contains two character strings 'shark' and 'dolphin'
animals = ['shark', 'dolphin']
animals[0]


# In[17]:


# Call the second thing (remember Python is zero-indexed)
animals[1]


# Lists can contain elements of almost any data type, including other lists! 
# 
# When this is the case, we can use multi-indices to extract just the piece of data we want. For example, to return only the element "tree":

# In[18]:


# Lists can contain other structures, such as other lists
# To get an element from a list within a list, double-index the original list!
animals = ['shark', 'dolphin', ['dog', 'cat'], ['tree', 'cactus']]
print(animals[3][0])


# Or, to just return the element "cat":

# In[19]:


# print this 'animals' list
print(animals)


# In[20]:


# print just the 3rd thing - the sublist containing 'dog' and 'cat'
print(animals[2])


# In[21]:


# print only 'cat'
print(animals[2][1])


# Lists can also contain elements of different types:

# In[22]:


# Define a heterogeneous list
chimera = ['lion', 0.5, 'griffin', 0.5]
print(type(chimera[0]))
print(type(chimera[1]))


# ## 2. Dictionary

# In[23]:


# Define two dictionaries - apple and orange
apple = {'name': 'apple', 'color': ['red', 'green'], 'recipes': ['pie', 'salad', 'sauce']}
orange = {'name': 'orange', 'color': 'orange', 'recipes': ['juice', 'marmalade', 'gratin']}
apple


# In[24]:


orange


# Combine two dictionaries into one by placing them in a list value `[apple, orange]`, with a key named `fruits`. Call the key to see the value(s)!

# In[25]:


fruits = {'fruits': [apple, orange]}

fruits


# To index just "juice" - under the 'recipes' key for the `orange` dictionary, combine dictionary key and list techniques to tunnel into the hierarchical structure of the dictionary and extract just what you want:

# In[26]:


# Call the newly combined dictionary
fruits


# In[27]:


# Reference the 'fruits' key
fruits['fruits']


# In[28]:


# Index the second thing (orange)
fruits['fruits'][1]


# In[29]:


# Call the 'recipes' key from 'orange' to see the items list
fruits['fruits'][1]['recipes']


# In[30]:


# Return the first thing from the 'recipes' key of the 'orange' dictionary inside of 'fruits'!
fruits['fruits'][1]['recipes'][0]


# ## Data import
# 
# Python offers a variety of methods for importing data. Thankfully, it is quite straightforward to import data from .csv and .txt files. Other formats, such as .xml and .json, are also supported. 
# 
# Below, let's import: 
# 1. Text from a .txt file
# 2. A dataframe from a .csv file

# ### Import text data as a character string
# 
# Import text as a single string using the `open().read()` Python convention.
# 
# Review your basic building blocks from above: 
# 1. `frank` is the name of the variable we will save the text inside of
# 2. `open` is the function we will use to open the text file
# 3. `'data/frankenstein.txt'` is the argument that we provide to the `open` function. This matches the `file` path argument and needs to contain the location for the Frankenstein book.
# 4. `.read()` reads the file as text. 

# In[31]:


get_ipython().system('wget https://raw.githubusercontent.com/EastBayEv/SSDS-TAML/main/fall2022/data/gapminder-FiveYearData.csv')


# In[32]:


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

# In[33]:


# Step 1. Physically download and install the library's files (unhashtab the line below to run)
# !pip install pandas


# In[34]:


# Step 2. link the pandas library to our current notebook
# pd is the alias, or shorthand, way to reference the pandas library
import pandas as pd 


# Now, you can use dot notation (type `pd.`)to access the functions within the pandas library. 
# 
# We want the `read.csv` method. Like the .txt file import above, we must provide the file path of the location of the .csv file we want to import. 
# 
# Save it in the variable named `gap`

# In[35]:


gap = pd.read_csv("data/gapminder-FiveYearData.csv")


# In[36]:


# View the data
print(gap)


# In[37]:


gap


# ## Getting help
# 
# The help pages in Python are generally quite useful and tell you everything you need to know - you just don't know it yet! Type a question mark `?` before a funciton to view its help pages.

# In[38]:


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

# In[39]:


# x 89 5


# **Indentation error**
# 
# Your indentation does not conform to the rules. 
# 
# Highlight the code in the cell below and press `command` and `/` (Mac) or `Ctrl` and `/` on Windows to block comment/uncomment multiple lines of code:

# In[40]:


# def example():
#     test = "this is an example function"
#     print(test)
#      return example


# ### Runtime errors
# 
# **Name error** 
# 
# You try to call a variable you have not yet assigned

# In[41]:


# p


# Or, you try to call a function from a library that you have not yet imported

# In[42]:


# example()


# **Type error**
# 
# You write code with incompatible types

# In[43]:


# "5" + 5


# **Index error**
# 
# You try to reference something that is out of range

# In[44]:


my_list = ['green', True, 0.5, 4, ['cat', 'dog', 'pig']]
# my_list[5]


# ### File errors
# 
# **File not found**
# 
# You try to import something that does not exist

# In[45]:


# document = open('fakedtextfile.txt').read()


# ## Exercises
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

# ## Numeric data wrangling
# 
# Importing numeric data from a .csv file is one thing, but wrangling it into a format that suits your needs is another. Read Chapter 4 "Numeric data wrangling" to learn how to use the pandas library to subset numeric data! 
# 
# Chapter 7 contains information about preprocessing text data. 
