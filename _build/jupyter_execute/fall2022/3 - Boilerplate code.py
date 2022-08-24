#!/usr/bin/env python
# coding: utf-8

# # Chapter 3 - Boilerplate code

# ![boiler](img/boiler.jpg)

# This chapter explains foundational Python syntax that you can reuse to accomplish many basic data creation, importing, and exporting tasks. 

# ## Variable assignment
# 
# In Python, data are saved in **variables.** Variable names should be simple and descriptive. This process is called **variable assignment.**
# 
# Assign a variable by typing its name to the left of the equals sign. Whatever is written to the right of the equals sign will be saved in the variable. 
# 
# You could read this as "x is defined as four", "five is assigned to y", or "z is six".

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


# In[ ]:





# ## Functions, arguments, and methods
# 
# Functions, arguments, and methods form the core user framework for Python programming. 
# 1. Functions: Perform self-contained actions on a thing.
# 
# 2. Argument(s):  The "things" (text, values, expressions, datasets, etc.)
# 
# > Note "parameters" are the variables as notated during function definition. Arguments are the values we pass into these placeholders while calling the function.  
# ![params](img/params_vs_args.png)
# 
# 3. Methods:   Type-specific functions (i.e., can only use a specific type of data and not other types). Use "dot" notation to utilize methods on a variable or other object. 
# 
# > For example, you could type `gap = pd.read_csv('data/gapminder-FiveYearData.csv')` to use the `read_csv()` method from the pandas library (imported as the alias `pd`) to load the Gapminder data.
# 

# ## Data types
# 
# Everything in Python has a type which determines how we can use and manipulate it. Data are no exception! Be careful, it is easy to get confused when trying to complete multiple tasks that use lots of different variables!
# 
# Use the `type` function to get the type of any variable if you're unsure. Below are four core data types: 
# 
# 1. `str`: character string; text. Always wrapped in quotations (single or double are fine).
# 2. `bool`: Boolean `True`/`False`. `True` is stored under the hood as 1, `False` is stored as 0. 
# 3. `float`: Decimals (floating-point)
# 4. `int`: whole numbers (positive and negative, including zero)

# In[8]:


# 1. String data
x1 = "This is string data"
print(type(x1))


# In[9]:


# 2. Boolean data
x2 = True
print(type(x2))


# In[10]:


# 3. float (decimals)
# use a decimal to create floats
pi = 3.14
print(type(pi))


# In[11]:


# 4. integer (whole numbers)
# do not use a decimal for integers
amount = 4
print(type(amount))


# ## String addition versus integer addition

# In[12]:


# character strings
'1' + '1'


# In[13]:


# integers
1 + 1


# In[14]:


get_ipython().run_line_magic('pinfo', 'dict')


# ## Data structures
# 
# Data can be stored in a variety of ways. Regardless, we use Python to **index** (positionally reference) a portion of a larger data structure or collection. 
# 
# ### Python is zero-indexed!
# 
# Python is a zero-indexed programming language and means that you start counting from zero. Thus, the first element in a collection is referenced by 0 instead of 1. 
# 
# Three structures are discussed below:
# 1. `list`: Lists are ordered groups of data that are both created and indexed with square brackets `[]`
# 2. `dict`: Dictionaries are _unordered_ groups of "key:value" pairs. Use the key to access the value. Curly braces `{}` are used to create and index dictionaries. 

# ## 1. List

# In[15]:


animals = ['shark', 'dolphin']
animals[0]


# In[16]:


animals[1]


# Lists can contain elements of almost any data type, including other lists! 
# 
# When this is the case, we can use multi-indices to extract just the piece of data we want. For example, to return only the element "tree":

# In[17]:


# To get an element from a list within a list, double-index the original list!
animals = ['shark', 'dolphin', ['dog', 'cat'], ['tree', 'cactus']]
print(animals[3][0])


# Or, to just return the element "cat":

# In[18]:


print(animals[2][1])


# Lists can also contain elements of different types:

# In[19]:


chimera = ['lion', 0.5, 'griffin', 0.5]
print(type(chimera[0]))
print(type(chimera[1]))


# ## 2. Dictionary

# In[20]:


apple = {'name': 'apple', 'color': ['red', 'green'], 'recipes': ['pie', 'salad', 'sauce']}
orange = {'name': 'orange', 'color': 'orange', 'recipes': ['juice', 'marmalade', 'gratin']}
apple


# In[21]:


orange


# Combine two dictionaries into one by placing them in a list value `[apple, orange]`. 
# 
# `fruits` is the key. Call the key to see the value(s)!

# In[22]:


fruits = {'fruits': [apple, orange]}

fruits


# To index just "juice", combine dictionary key and list techniques to tunnel into the hierarchical structure of the dictionary and extract just what you want:

# In[23]:


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
# 3. `'data/frankenstein.txt'` is the argument that we provide to the `open` function. This is the file path location for the Frankenstein book.
# 4. `.read()` reads the file as text. 

# In[24]:


frank = open('data/frankenstein.txt').read()

# print only the first 1000 characters
print(frank[:1000])


# ### Import data frames with the pandas library
# 
# Data frames are programming speak for tabular spreadsheets organized into rows and columns and often stored in .csv format. Just think of a spreadsheet in MS Excel!
# 
# .csv stands for "comma-separated values" and means that these data are stored as text files with a comma used to delineate column breaks. 
# 
# For this part, we will use the [Pandas](https://pandas.pydata.org/docs/getting_started/intro_tutorials/02_read_write.html) Python library. Using additional software libraries requires two steps: 

# In[25]:


# Step 1. Physically download and install the library's files (unhashtab the line below to run)
# !pip install pandas


# In[26]:


# Step 2. link the pandas library to our current notebook
# pd is the alias, or shorthand, way to reference the pandas library
import pandas as pd 


# Now, you should be able to use dot notation (type `pd.`)to access the functions within the pandas library. 
# 
# We want the `read.csv` method. Like the .txt file import above, we must provide the file path of the location of the .csv file we want to import. 
# 
# Save it in the variable named `gap`

# In[27]:


gap = pd.read_csv("data/gapminder-FiveYearData.csv")


# In[28]:


# View the data
print(gap)


# In[29]:


gap


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
# 
# ## Exercises
# 
# 1. Unhashtag the line of code for each error message below
# 2. Run each one
# 3. Inspect the error messages

# ### Syntax errors
# 
# **Invalid syntax error** 
# 
# You have entered invalid syntax, or something python does not understand.

# In[30]:


# x 89 5


# **Indentation error**
# 
# Your indentation does not conform to the rules. 
# 
# Highlight the code in the cell below and press `command` and `/` (Mac) or `Ctrl` and `/` on Windows to block comment/uncomment multiple lines of code:

# In[31]:


# def example():
#     test = "this is an example function"
#     print(test)
#      return example


# ### Runtime errors
# 
# **Name error** 
# 
# You try to call a variable you have not yet assigned

# In[32]:


# p


# Or, you try to call a function from a library that you have not yet imported

# In[33]:


# example()


# **Type error**
# 
# You write code with incompatible types

# In[34]:


# "5" + 5


# **Index error**
# 
# You try to reference something that is out of range

# In[35]:


my_list = ['green', True, 0.5, 4, ['cat', 'dog', 'pig']]
# my_list[5]


# ### File errors
# 
# **File not found**
# 
# You try to import something that does not exist

# In[36]:


# document = open('fakedtextfile.txt').read()


# ## Exercises
# 
# 1. Import the file "dracula.txt". Save it in a variable named `drac`
# 2. Import the file "penguins.csv". Save it in a variable named `pen`
# 3. Figure out how to find help to export just the first 1000 characters of `drac` as a .txt file named "dracula_short.txt"
# 4. Figure out how to export the `pen` dataframe as a file named "penguins_saved.csv"
# 
# If you encounter error messages, which ones? 

# ## Data wrangling
# 
# Importing data is one thing, but wrangling it into a format that suits your needs is another. Read Chapter 4 "Numeric data wrangling" to learn how to use the pandas library to subset your data. 
