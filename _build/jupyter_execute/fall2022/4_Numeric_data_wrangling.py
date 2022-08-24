#!/usr/bin/env python
# coding: utf-8

# # Chapter 4 - Numeric data wrangling
# 
# > For text preprocessing, see Chapter 6 "English text preprocessing basics"

# ![wrangle](img/wrangle.png)

# ## The pandas library
# 
# Import the pandas library with the alias `pd` 
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
# Use pandas methods to learn about the data!

# ### `.head()`

# In[4]:


# look at first five rows by default
gap.head()


# ### `.columns`

# In[5]:


# View column names
# this is an attribute instead of a method - note the lack of parentheses ()
gap.columns


# ### `.shape`

# In[6]:


# Show number of rows by columns
# this is an attribute instead of a method - note the lack of parentheses ()
gap.shape


# ## Summary statistics
# 
# Produce summary statistics, including: 
# * Count, mean, sd, quartiles, min/max
# * Tabulate frequencies

# ### `.describe()`

# In[7]:


# produce summary statistics for numeric data
gap.describe()


# ### `.mean()` and `.std()`

# In[8]:


# calculate mean and standard deviation of lifeExp
lifeExp_mean = gap["lifeExp"].mean()
lifeExp_sd = gap["lifeExp"].std()
print("Life expectancy mean:", lifeExp_mean)
print("Life expectancy sd:", lifeExp_sd)


# ### `.groupby()` and `.count()`
# 
# These two methods are useful for tabulating frequencies by a grouping variable!

# In[9]:


# count the number of countries grouped by each continent
gap.groupby("continent").count()["country"]


# ## Subset rows or columns
# 
# Sampling data is necessary for many reasons, including quick sanity checks. 
# 
# Slice a data frame by using bracket notation to specify start and end points `[start : end]`
# 
# The `[start ` index is _included_ and the ` end]` index is **excluded**. 
# 
# > Remember that Python is a zero-indexed language, so starts counting from zero, not one.
# 
# Leave the start or end values blank to start from the beginning, or go to the end of a collection. 

# ### Row subset: slice just the first row

# In[10]:


gap[:1]


# ### Row subset: slice first three rows

# In[11]:


gap[:3]


# ### Row subset: slice rows 10 thru 14

# In[12]:


subset1 = gap[10:15]
subset1


# ### Column subset: one column

# In[13]:


# type the column name as a string in square brackets
gap['lifeExp']


# ### Column subset: multiple columns

# In[14]:


# note the double sets of brackets
subset2 = gap[['continent', 'lifeExp', 'gdpPercap']]
subset2


# ### Row and column subset

# In[15]:


# subset more than one column and rows 855 thru 858
subset3 = gap[['continent', 'lifeExp', 'gdpPercap']][855:859]
subset3


# In[16]:


# Pandas "Series" can be thought of like numpy arrays - but beware, they do not function the same!
type(gap["lifeExp"])


# ### Subset by logical condition(s)

# In[17]:


# lifeExp is greater than 80
le2 = gap[gap['lifeExp'] > 80]
le2


# ### logical AND (`&`)
# 
# All conditions must be satisfied to be included in the subset

# In[18]:


# create subset that includes life expectancy greater than 81 AND pop < 500,000.
year2002 = gap[(gap["lifeExp"] > 81) & (gap["pop"] < 500000)]
year2002


# ### logical OR (`|`)
# 
# Just one of multiple conditions must be satisfied to be included in the subset

# In[19]:


# create a subset that includes country equals Ireland OR life expectancy greater than 82. 
ireland82 = gap[(gap["country"] == "Ireland") | (gap["lifeExp"] > 82)]
ireland82


# ## Exercises TODO
# 
# 1. Load csv
# 2. Print col names
# 3. Print mean for all columns
# 4. Summary stats for all data
# 5. mean for one col ie gap.groupby('continent')["lifeExp"].mean()
# 6. multi columns ie le_table = gap.groupby('continent')[["lifeExp", "gdpPercap"]].mean()

# ## Data visualization
# 
# After importing, exploring, and potentially subsetting your data, visualization is a common technique to perform next. Read Chapter 5 "Data visualization essentials" to get started. 
