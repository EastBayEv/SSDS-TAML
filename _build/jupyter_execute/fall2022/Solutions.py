#!/usr/bin/env python
# coding: utf-8

# # Solutions
# 
# Example solutions for challenge exercises from each chapter in this book.

# ## Chapter 1 - Exercises
# 
# 1. You will find challenge exercises to work on at the end of each chapter. They will require you to write code such as that found in the cell at the top of this notebook. 
# 
# 2. Click the "Colab" badge at the top of this notebook to open it in the Colaboratory environment. Press `shift` and `enter` simultaneously on your keyboard to run the code and draw your lucky card!

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
drac = open("data/dracula.txt").read()
print(drac)


# In[13]:


#5
import pandas as pd
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


#7
pen.to_csv("data/penguins_saved.csv")


# ## Chapter 4 - Exercises
# 
# 1. Load the file "gapminder-FiveYearData.csv" and save it in a variable named `gap`
# 2. Print the column names
# 3. Compute the mean for one numeric column
# 4. Compute the mean for all numeric columns
# 5. Tabulate frequencies for the "continent" column
# 6. Compute mean lifeExp and dgpPercap by continent
# 7. Create a subset of `gap` that contains only countries with lifeExp greater than 75 and gdpPercap less than 5000.

# In[16]:


#1
import pandas as pd
gap = pd.read_csv("data/gapminder-FiveYearData.csv")
gap


# In[17]:


#2
gap.columns


# In[18]:


#3
gap["lifeExp"].mean()


# In[19]:


# or
gap.describe()


# In[20]:


#4
print(gap.mean())


# In[21]:


#5
gap["continent"].value_counts()


# In[22]:


#6
le_gdp_by_continent = gap.groupby("continent").agg(mean_le = ("lifeExp", "mean"), 
                                                  mean_gdp = ("gdpPercap", "mean"))
le_gdp_by_continent


# In[23]:


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

# In[24]:


import pandas as pd
import seaborn as sns
peng = pd.read_csv("data/penguins.csv")


# In[25]:


# set seaborn figure size, background theme, and axis and tick label size
sns.set(rc={'figure.figsize':(10, 7)})
sns.set(font_scale = 2)
sns.set_theme(style='ticks')


# In[26]:


peng


# In[27]:


#1
sns.scatterplot(data = peng, x = "flipper_length_mm", y = "body_mass_g", 
                hue = "species",
               s = 250, alpha = 0.75);


# In[28]:


#2
sns.scatterplot(data = peng, x = "flipper_length_mm", y = "body_mass_g", 
                hue = "sex",
               s = 250, alpha = 0.75, 
                
               palette = ["red", "green"]).legend(title = "Species",
                                                  fontsize = 20, 
                                                  title_fontsize = 30,
                                                 loc = "best");


# In[29]:


#3
sns.scatterplot(data = peng, x = "flipper_length_mm", y = "body_mass_g", 
                hue = "island").legend(loc = "lower right");


# In[30]:


#4
facet_plot = sns.FacetGrid(data = peng, col = "island",  row = "sex")
facet_plot.map(sns.scatterplot, "flipper_length_mm", "body_mass_g");


# ## Chapter 5 - Gapminder Exercises
# 
# 1. Figure out how to make a line plot that shows gdpPercap through time. 
# 2. Figure out how to make a second line plot that shows lifeExp through time. 
# 3. How can you plot gdpPercap with a different colored line for each continent? 
# 4. Plot lifeExp with a different colored line for each continent. 

# In[31]:


import pandas as pd
import seaborn as sns
gap = pd.read_csv("data/gapminder-FiveYearData.csv")


# In[32]:


#1
sns.lineplot(data = gap, x = "year", y = "gdpPercap", ci = 95);


# In[33]:


#2
sns.lineplot(data = gap, x = "year", y = "lifeExp", ci = False);


# In[34]:


#3
sns.lineplot(data = gap, x = "year", y = "gdpPercap", hue = "continent", ci = False);


# In[35]:


#4
sns.lineplot(data = gap, x = "year", y = "lifeExp", 
             hue = "continent", ci = False);


# In[36]:


#4 with custom colors
sns.lineplot(data = gap, x = "year", y = "lifeExp", 
             hue = "continent", 
             ci = False, 
            palette = ["#00FFFF", "#458B74", "#E3CF57", "#8A2BE2", "#CD3333"]);

# color hex codes: https://www.webucator.com/article/python-color-constants-module/
# seaborn color palettes: https://www.reddit.com/r/visualization/comments/qc0b36/all_seaborn_color_palettes_together_so_you_dont/


# # Chapter 6

# In[ ]:





# # Chapter 7

# 1. Compare our "by hand" OLS results to those producd by sklearn's `LinearRegression` function. Are they the same? 
#     * Slope = 4
#     * Intercept = -4
#     * RMSE = 2.82843
#     * y_hat = y_hat = B0 + B1 * data.x

# In[37]:


# Recreate dataset
import pandas as pd
data = pd.DataFrame({"x": [1,2,3,4,5],
                     "y": [2,4,6,8,20]})
data


# In[38]:


# Our "by hand" OLS regression information:
B1 = 4
B0 = -4
RMSE = 2.82843
y_hat = B0 + B1 * data.x


# In[39]:


# use scikit-learn to compute R-squared value
from sklearn.linear_model import LinearRegression

lin_mod = LinearRegression().fit(data[['x']], data[['y']])
print("R-squared: " + str(lin_mod.score(data[['x']], data[['y']])))


# In[40]:


# use scikit-learn to compute slope and intercept
print("scikit-learn slope: " + str(lin_mod.coef_))
print("scikit-learn intercept: " + str(lin_mod.intercept_))


# In[41]:


# compare to our by "hand" versions. Both are the same!
print(int(lin_mod.coef_) == B1)
print(int(lin_mod.intercept_) == B0)


# In[42]:


# use scikit-learn to compute RMSE
from sklearn.metrics import mean_squared_error

RMSE_scikit = round(mean_squared_error(data.y, y_hat, squared = False), 5)
print(RMSE_scikit)


# In[43]:


# Does our hand-computed RMSE equal that of scikit-learn at 5 digits?? Yes!
print(round(RMSE, 5) == round(RMSE_scikit, 5))


# 2. 

# In[ ]:





# In[ ]:





# In[ ]:





# 3.

# In[ ]:





# In[ ]:





# In[ ]:





# # Chapter 8

# In[ ]:





# # Chapter 9

# In[ ]:





# In[ ]:




