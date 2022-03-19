#!/usr/bin/env python
# coding: utf-8

# # iv. Numeric data wrangling
# 
# > For text preprocessing, see the January 19 notebook.

# ![wrangle](img/wrangle.png)

# ## The pandas library
# 
# Import the pandas library with the alias pd. 
# 
# Use "dot notation" to apply its methods to the dataset, which is stored in a tabular .csv file. 
# 
# To import the .csv file, use the `.read_csv()` pandas method. The only argument is the file path to a .csv file. 
# 
# Import the data below and investigate it by applying pandas methods to the data frame `gap`. 
# 
# [Learn more about the Gapminder data](https://www.gapminder.org/data/)

# In[1]:


import pandas as pd


# In[2]:


gap = pd.read_csv("data/gapminder-FiveYearData.csv")


# In[4]:


print(type(gap))


# ## Pandas methods
# 
# Just a small handful of pandas methods will help you accomplish several key data wrangling tasks. See the examplea code below.
# 
# * First, look at the data
# * Summary statistics
# * Subset rows: 
#   * first row
#   * first three rows
#   * rows 10 thru 14
# * Subset columns: 
#   * one column
#   * multiple columns
# * Row and column subset
# * Subset by logical condition(s)

# ## First, look at the data
# 
# Use pandas methods to learn about the data!

# ### `.head()`

# In[5]:


# look at first five rows by default
gap.head()


# ### `.columns`

# In[6]:


# View column names
# this is an attribute - note the lack of parentheses ()
gap.columns


# ### `.shape`

# In[7]:


# Show  number of rows by columns
# this is an attribute - note the lack of parentheses ()
gap.shape


# ## Summary statistics
# 
# Produce summary statistics, including: 
# * Count, mean, sd, quartiles, min/max
# * Tabulate frequencies

# ### `.describe()`

# In[8]:


# produce summary statistics for numeric data
gap.describe()


# ### `.mean()` and `.std()`

# In[53]:


# calculate mean and standard deviation of lifeExp
lifeExp_mean = gap["lifeExp"].mean()
lifeExp_sd = gap["lifeExp"].std()
print(lifeExp_mean)
print(lifeExp_sd)


# ### `.groupby()` and `.count()`

# In[9]:


# count the number of countries grouped by each continent
gap.groupby("continent").count()["country"]


# ## Subset rows or columns
# 
# Sampling data is necessary for many reasons, including quick sanity checks. 
# 
# Slice a data frame by using bracket notation to specify start and end points `[start : end]`
# 
# The `[start ` index is _included_ and the ` end]` index is excluded. 
# 
# > Remember that Python is a zero-indexed language, or starts counting from zero, not one.
# 
# Leave the start or end values blank to start from the beginning, or go to the end of a collection. 

# ### Row subset: slice just the first row

# In[85]:


gap[:1]


# ### Row subset: slice first three rows

# In[86]:


gap[:3]


# ### Row subset: slice rows 10 thru 14

# In[87]:


subset1 = gap[10:15]
subset1


# ### Column subset: one column

# In[88]:


# type the column name as a string in square brackets
gap['lifeExp']


# ### Column subset: multiple columns

# In[89]:


# note the double sets of brackets
subset2 = gap[['continent', 'lifeExp', 'gdpPercap']]
subset2


# ### Row and column subset

# In[95]:


# subset more than one column and rows 855 thru 858
subset3 = gap[['continent', 'lifeExp', 'gdpPercap']][855:859]
subset3


# In[19]:


type(gap["lifeExp"])


# ### Subset by logical condition(s)

# In[104]:


# lifeExp is greater than 80
le2 = gap[gap['lifeExp'] > 81]
le2


# In[109]:


# logical AND (all conditions must be satisfied to be included)

# create subset that includes life expectancy greater than 81 AND pop < 500,000.

year2002 = gap[(gap["lifeExp"] > 81) & (gap["pop"] < 500000)]
year2002


# In[129]:


# logical OR (one of multiple conditions must be satisfied to be included)

# create subset that includes country equals Ireland OR life expectancy greater than 82. 

ireland82 = gap[(gap["country"] == "Ireland") | (gap["lifeExp"] > 82)]
ireland82

