#!/usr/bin/env python
# coding: utf-8

# # Chapter 4 - Numeric data wrangling
# 2023 February 14
# 
# > For text preprocessing, see Chapter 7 "English text preprocessing basics"

# <a target="_blank" href="https://colab.research.google.com/github/EastBayEv/SSDS-TAML/blob/main/spring2023/4_Numeric_data_wrangling.ipynb">
#   <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
# </a>

# ![wrangle](img/wrangle.png)

# ## The pandas library
# 
# Import the pandas library with the alias `pd`. This is just a shortcut to call its methods!
# 
# Use "dot notation" to apply its methods to the gapminder dataset, which is stored in a tabular .csv file. 
# 
# To import the .csv file, use the `.read_csv()` pandas method. The only argument for now is the file path to a .csv file. 
# 
# [Learn more about the Gapminder data](https://www.gapminder.org/data/)

# In[1]:


import pandas as pd


# In[2]:


# import the gapminder dataset
# !wget -P data/ https://raw.githubusercontent.com/EastBayEv/SSDS-TAML/main/spring2023/data/gapminder-FiveYearData.csv
gap = pd.read_csv("data/gapminder-FiveYearData.csv")


# In[3]:


print(type(gap))


# ## Pandas methods
# 
# Just a small handful of pandas methods will help you accomplish several key data wrangling tasks. How might the wrangling process look? 
# 
# 1. First, look at the data
# 2. Compute summary statistics
# 3. Subset rows
#     * first row
#     * first three rows
#     * rows 10 thru 14
# 4. Subset columns: 
#     * one column
#     * multiple columns
# 5. Row and column subset
# 6. Subset by logical condition(s)

# ## First, look at the data
# 
# `print` or call the dataset to view its output. Use pandas methods to learn about the data!

# In[4]:


print(gap)


# In[5]:


gap


# ### `.head()`

# In[6]:


# .head() shows the first five rows by default
gap.head()


# ### `.columns`

# In[7]:


# .columns shows the column names
# this is an attribute instead of a method - note the lack of parentheses ()
gap.columns


# ### `.shape`

# In[8]:


# .shape shows the number of rows by columns
gap.shape


# ## Summary statistics
# 
# Produce summary statistics, including: 
# * Count, mean, sd, quartiles, min/max
# * Tabulate frequencies

# ### `.describe()`

# In[9]:


# produce summary statistics for numeric data
gap.describe()


# ### `.mean()` and `.std()`

# In[10]:


# calculate mean and standard deviation of lifeExp
lifeExp_mean = gap["lifeExp"].mean()
lifeExp_sd = gap["lifeExp"].std()
print("Life expectancy mean:", lifeExp_mean)
print("Life expectancy sd:", lifeExp_sd)


# ### `.groupby()` and `.count()`
# 
# These two methods are useful for tabulating frequencies by a grouping variable!

# In[11]:


# count the number of observations grouped by each continent
gap.groupby("continent").count()["country"]


# ## Subset rows or columns
# 
# Sampling data is necessary for many reasons, including quick sanity checks. 
# 
# Slice a data frame by using bracket notation to specify start and end points `[start : end]`
# 
# The `[start ` index is **included** and the ` end]` index is **excluded**. 
# 
# > Remember that Python is a zero-indexed language, so starts counting from zero, not one.
# 
# Leave the start or end values blank to start from the beginning, or go to the end of a collection. 

# ### Row subset: slice just the first row

# In[12]:


gap[:1]


# ### Row subset: slice first three rows

# In[13]:


gap[:3]


# ### Row subset: slice rows 10 thru 14

# In[14]:


subset1 = gap[10:15]
subset1


# ### Column subset: one column

# In[15]:


# type the column name as a string in square brackets
gap['lifeExp']


# ### Column subset: multiple columns

# In[16]:


# note the double sets of brackets
subset2 = gap[['continent', 'lifeExp', 'gdpPercap']]
subset2


# ### Row and column subset

# In[17]:


# subset more than one column and rows 855 thru 858
subset3 = gap[['continent', 'lifeExp', 'gdpPercap']][855:859]
subset3


# In[18]:


# A column in a pandas data frame Pandas "Series" can be thought of like numpy arrays
# But, beware, they do not function exactly the same!
type(gap["lifeExp"])


# ### Subset by logical condition(s)

# In[19]:


# lifeExp is greater than 80
le2 = gap[gap['lifeExp'] > 80]
le2


# ### logical AND (`&`)
# 
# All conditions must be satisfied to be included in the subset

# In[20]:


# create subset that includes life expectancy greater than 81 AND pop < 500,000.
year2002 = gap[(gap["lifeExp"] > 81) & (gap["pop"] < 500000)]
year2002


# ### logical OR (`|`)
# 
# Just one of multiple conditions must be satisfied to be included in the subset

# In[21]:


# create a subset that includes country equals Ireland OR life expectancy greater than 82. 
ireland82 = gap[(gap["country"] == "Ireland") | (gap["lifeExp"] > 82)]
ireland82


# ## Exercises
# 
# 1. Load the file "gapminder-FiveYearData.csv" and save it in a variable named `gap`
# 2. Print the column names
# 3. Compute the mean for one numeric column
# 4. Compute the mean for all numeric columns
# 5. Tabulate frequencies for the "continent" column
# 6. Compute mean lifeExp and dgpPercap by continent
# 7. Create a subset of `gap` that contains only countries with lifeExp greater than 75 and gdpPercap less than 5000.

# ## Data visualization
# 
# After importing, exploring, and subsetting your data, visualization is a common technique to perform next. Read Chapter 5 "Data visualization essentials" to get started. 
