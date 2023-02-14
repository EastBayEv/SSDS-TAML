#!/usr/bin/env python
# coding: utf-8

# # Chapter 5 - Data visualization essentials
# 2022 August 26

# <a target="_blank" href="https://colab.research.google.com/github/EastBayEv/SSDS-TAML/blob/main/spring2023/5_Data_visualization_essentials.ipynb">
#   <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
# </a>

# ![viz](img/viz.png)

# In[1]:


# import libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# make sure plots show in the notebook
get_ipython().run_line_magic('matplotlib', 'inline')


# After importing data, you should examine it closely. 
# 
# 1. Look at the raw data and perform rough checks of your assumptions
# 2. Compute summary statistics
# 3. Produce visualizations to illustrate obvious - or not so obvious - trends in the data

# ## First, a note about matplotlib
# There are many different ways to visualize data in Python but they virtually all rely on matplotlib. You should take some time to read through the tutorial: https://matplotlib.org/stable/tutorials/introductory/pyplot.html. 
# 
# Because many other libraries depend on matplotlib under the hood, you should familiarize yourself with the basics. For example: 

# In[2]:


import matplotlib.pyplot as plt
x = [1,2,3,4,5]
y = [2,4,6,8,20]
plt.scatter(x, y)
plt.title('title')
plt.ylabel('some numbers')
plt.xlabel('x-axis label')
plt.show()


# ## Visualization best practices
# 
# Consult Wilke's _Fundamentals of Data Visualization_ https://clauswilke.com/dataviz/ for discussions of theory and best practices. 
# 
# The **goal of data visualization** is to accurately communicate _something_ about the data. This could be an amount, a distribution, relationship, predictions, or the results of sorted data.
# 
# Utilize characteristics of different data types to manipulate the aesthetics of plot axes and coordinate systems, color scales and gradients, and formatting and arrangements to impress your audience!
# 
# ![wilke](img/wilke.png)

# ![wilke12](img/wilke12.png)

# ## Plotting with seaborn
# 
# ### Basic plots
# 
# 1. Histogram: visualize distribution of one (or more) continuous (i.e., integer or float) variable.
# 
# 2. Boxplot: visualize the distribution of one (or more) continuous variable.
# 
# 3. Scatterplot: visualize the relationship between two continuous variables. 
# 
# Study the seaborn tutorial for more examples and formatting options: https://seaborn.pydata.org/tutorial/function_overview.html

# ## Histogram
# 
# Use a histogram to plot the distribution of one continuous (i.e., integer or float) variable. 

# In[3]:


# load gapminder dataset
# !wget -P data/ https://raw.githubusercontent.com/EastBayEv/SSDS-TAML/main/fall2022/data/gapminder-FiveYearData.csv
gap = pd.read_csv("data/gapminder-FiveYearData.csv")
gap.head()


# In[4]:


# all data
sns.histplot(data = gap,
            x = 'lifeExp'); 


# In[5]:


# by continent
sns.histplot(data = gap, 
            x = 'lifeExp', 
            hue = 'continent');


# ## Boxplot
# 
# Boxplots can be used to visualize one distribution as well, and illustrate different aspects of the table of summary statistics.

# In[6]:


# summary statistics
gap.describe()


# In[7]:


# all data
sns.boxplot(data = gap,
            y = 'lifeExp', 
            color = 'gray');


# In[8]:


gap.groupby('continent').count()['country']


# In[9]:


# Sums to the total number of observations in the dataset
sum(gap.groupby('continent').count()['country'])


# In[10]:


# by continent
sns.boxplot(data = gap,
            x = 'continent', 
            y = 'lifeExp').set_title('Boxplots');


# In[11]:


# custom colors
sns.boxplot(data = gap, 
            x = 'continent', 
            y = 'lifeExp', 
            palette = ['gray', '#8C1515', '#D2C295', '#00505C', 'white']).set_title('Boxplots example');


# ## Scatterplot
# 
# Scatterplots are useful to illustrate the relationship between two continuous variables. Below are several options for you to try.

# In[12]:


### change figure size
sns.set(rc = {'figure.figsize':(12,8)})

### change background
sns.set_style("ticks")

# commented code
ex1 = sns.scatterplot(
    
    # dataset
    data = gap,
    
    # x-axis variable to plot
    x = 'lifeExp', 
    
    # y-axis variable to plot
    y = 'gdpPercap', 
    
    # color points by categorical variable
    hue = 'continent', 
    
    # point transparency
    alpha = 1)

### log scale y-axis
ex1.set(yscale="log")

### set axis labels
ex1.set_xlabel("Life expectancy (Years)", fontsize = 20)
ex1.set_ylabel("GDP per cap (US$)", fontsize = 20);

### unhashtag to save 
### NOTE: this might only work on local Python installation and not JupyterLab - try it!

# plt.savefig('img/scatter_gap.pdf')


# ## Exercises - Penguins dataset
# 
# Learn more about the biological and spatial characteristics of penguins! 
# 
# 1. Use seaborn to make a scatterplot of two continuous variables. Color each point by species. 
# 2. Make the same scatterplot as #1 above. This time, color each point by sex. 
# 3. Make the same scatterplot as #1 above again. This time color each point by island.
# 4. Use the `sns.FacetGrid` method to make faceted plots to examine "flipper_length_mm" on the x-axis, and "body_mass_g" on the y-axis. 
# 
# ![penguins](img/penguins.png)

# ## Visualizations as an inferential tool
# 
# Below is a map of Antarctica past the southernmost tip of the South American continent. 
# 
# The distance from the Biscoe Islands (Renaud) to the Torgersen and Dream Islands is about 140 km. 
# 
# Might you suggest any similarities or differences between the penguins from these three locations? 
# 
# ![antarctica](img/antarctica.png)

# ## Exercises - Gapminder dataset
# 
# 1. Figure out how to make a line plot that shows gdpPercap through time. 
# 2. Figure out how to make a second line plot that shows lifeExp through time. 
# 3. How can you plot gdpPercap with a different colored line for each continent? 
# 4. Plot lifeExp with a different colored line for each continent. 

# ## What does this all mean for machine learning and text data?
# 
# You might be wondering what this all means for machine learning and text data! Oftentimes we are concerned sorting data, predicting something, the amounts of words (and their synonyms) being used, or with calculating scores between words. As you will see in the next chapters, we do not change text to numbers, but we do change the _representation_ of text to numbers. Read Chapter 6 "Core machine learning concepts; building text vocabularies" and Chapter 7 "English text preprocessing basics" to learn more!
