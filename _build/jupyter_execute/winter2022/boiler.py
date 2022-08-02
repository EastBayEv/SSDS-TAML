#!/usr/bin/env python
# coding: utf-8

# # iii. Boilerplate code review

# ![boiler](img/boiler.jpg)

# Below are just a few examples of basic Python programming to accomplish data saving and importing tasks.

# ## Variable assignment
# 
# In Python, data are saved in variables.
# 
# Variable names should be simple and descriptive. 
# 
# Assign a variable by typing its name to the left of the equals sign. Whatever is written to the right of the equals sign will be saved in the variable. 
# 
# You could read this as "x is defined as four", "five is assigned to y", or "z is six".

# In[1]:


# define one variable
x = 1


# In[2]:


# assign multiple variables
x = 1
y = 2
z = 3


# ## Use `print()` to show it on the screen

# In[3]:


print(x)


# In[4]:


# call the variables directly!
x / y * z


# ## Functions, arguments, and methods
# 
# Functions, arguments, and methods form the core user framework for Python programming. 
# * Functions: Perform self-contained actions on a thing.
# 
# * Argument:  The "things" (text, values, expressions, datasets, etc.)
# 
# > Note "parameters" are the variables as notated during function definition. Arguments are the values we pass into these placeholders while calling the function.  
# ![params](img/params_vs_args.png)
# 
# * Methods:   Type-specific functions (i.e., can only use a specific type of data and not other types). Use "dot" notation to utilize methods on a variable or other object. 
# 
# > For example, you could type `gap = pd.read_csv('data/gapminder-FiveYearData.csv')` to use the `read_csv()` method from the pandas library (imported as the alias `pd`) to load the Gapminder data.
# 

# ## Data types
# 
# Everything in Python has a type which determines how we can manipulate that piece of data. Be careful, it is easy to get confused when trying to complete multiple tasks that use lots of different variables!

# In[5]:


# Use the type() function to get the type of any variable if you're unsure: 
x1 = "Boolean"
print(type(x1))

x2 = True
print(type(x2))


# In[6]:


# float (decimals)
# use a decimal to create floats
pi = 3.14
print(type(pi))


# In[7]:


# integer (whole numbers)
# do not use a decimal for integers
amount = 4
print(type(amount))


# In[8]:


# string (text)
# wrap text data in quotations
welcome = "Welcome to Stanford Libraries"
print(type(welcome))


# In[9]:


# boolean (logical)
# True or False (stored as 1 and 0)
print(type(True))
print(False - True)


# ### Addition examples with strings versus numbers

# In[10]:


# character strings
'1' + '1'


# In[11]:


# integers
1 + 1


# ## Data structures
# 
# Data can be stored in a variety of ways. 

# ### Indexing
# 
# Python is a zero-indexed programming language and means that you start counting from zero. Thus, the first element in a collection is referenced by 0 instead of 1. 
# 
# ### List
# 
# Lists are ordered groups of data that are both created and indexed (positionally referenced) with square brackets `[]`.

# In[12]:


animals = ['shark', 'dolphin']
animals[0]


# Lists can contain elements of almost any data type, including other lists! 

# In[13]:


# To get an element from a list within a list, double-index the original list!
animals = ['shark', 'dolphin', ['dog', 'cat'], ['tree', 'cactus']]
print(animals[3][0])
print(animals[2][1])


# Lists can also contain elements of different types simultaneously! 

# In[14]:


chimera = ['lion', 0.5, 'griffin', 0.5]
print(type(chimera[0]))
print(type(chimera[1]))


# ### Dictionary
# 
# Dictionaries are _unordered_ groups of "key:value" pairs. Use the key to access the value. 

# In[15]:


apple = {'name': 'apple', 'color': ['red', 'green'], 'recipes': ['pie', 'salad', 'sauce']}
orange = {'name': 'orange', 'color': 'orange', 'recipes': ['juice', 'marmalade', 'gratin']}

fruits = {'fruits': [apple, orange]}

fruits


# In[16]:


fruits['fruits'][1]['recipes'][0]


# ### Import text data as a character string
# 
# Import text using the `open().read()` Python convention to import text as a single string.

# In[17]:


frank = open('data/frankenstein.txt').read()

# print only the first 1000 characters
print(frank[:1000])


# ### Import data frames with the pandas library
# 
# Data frames are programming speak for tabular spreadsheets organized into rows and columns and often stored in .csv format. 

# In[18]:


# Step 1. link the pandas library to our current notebook
import pandas as pd


# In[19]:


# Step 2. enter the file path in pandas's read_csv() function  
gap = pd.read_csv("data/gapminder-FiveYearData.csv")


# In[20]:


# Step 3. view the data
print(gap)


# In[21]:


gap


# ## Challenge
# 
# Open JupyterLab. Try to import a: 
# 1. different .txt file
# 2. different .csv file
# 
# If you encounter error messages, which ones? 

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
# 
# ## Challenge
# 
# 1. In JupyterLab, unhashtag the line of code for each error message below
# 2. Run each one
# 3. Inspect the error messages

# ### Syntax errors

# **Invalid syntax**
# 
# You have entered something python does not understand.

# In[22]:


# x 89 5


# **Indentation**
# 
# Your indentation does not conform to the rules

# In[23]:


### indentation
# def example():
#     test = "this is an example function"
#     print(test)
#      return example


# ### Runtime errors

# **Name** 
# 
# You try to call a variable you have not yet assigned

# In[24]:


# x


# Or, you try to call a function from a library that you have not yet imported

# In[25]:


# example()


# **Type**
# 
# You write code with incompatible types

# In[26]:


# "5" + 5


# **Index**
# 
# You try to reference something that is out of range

# In[27]:


my_list = ['green', True, 0.5, 4, ['cat', 'dog', 'pig']]
# my_list[5]


# ### File errors

# **File not found**
# 
# You try to import something that does not exist

# In[28]:


# document = open('fakedocument.txt').read()

