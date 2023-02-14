#!/usr/bin/env python
# coding: utf-8

# # i. How to use this book

# ## Organization
# 
# The interface is divided into three sections. 
# 1. **The center portion** contains the main text. Here, goals, concepts, objectives, and code will be presented and explained. 

# In[1]:


# card trick

import random
import calendar
from datetime import date, datetime
deck = 'ace two three four five six seven eight nine ten jack queen king'.split()
suit = 'spades clubs hearts diamonds'.split()
today = calendar.day_name[date.today().weekday()]
date = datetime.today().strftime('%Y-%m-%d')
select_value = random.sample(deck, 1)[0]
select_suit = random.sample(suit, 1)[0]
print("\nWelcome to Stanford Libraries!")
print("\nYour lucky card for " + today + " " + date + " is: " + select_value + " of " + select_suit)


# 2. **The left sidebar** shows the search field and shortcuts to session materials.
#     * Use the `left and right arrow keys` to browse the different chapters. 
# 3. **The right sidebar** contains the section header shortcuts. 
#     * Use the `up and down arrow keys` to scroll the page.

# ## Jupyter icons
# Note the buttons at the top of each page. 
# ![icons](img/icons.png)
# * The rocketship icon appears only on pages with executable code. 
# Hover your mouse over it and click "Binder" to launch all of the materials there. Or, click "Live Code" to run code on this webpage. 

# > **Card trick**  
# > * Click "Live Code" underneath the rocketship icon way up top. 
# > * Scroll to the "card trick" Python code above. 
# > * Click "run" to draw your lucky card. 

# * Click the square icon to enter fullscreen mode. Press the escape key to exit. 
# * Move your mouse over the Octocat icon. Click "Repository" to visit the GitHub site. Or, click "Open Issue" to contribute to this book. 
# * The download icon can be used to export the contents of this book. 
# * Click the left facing arrow `<-` to hide the left sidebar.

# ## Two parts to each coding walkthrough: 
# 1. The "Code" lesson that contains exercises we will walk through together. 
# 2. One "Quiz", designed as a multi-part mini research project. This is accompanied by a solution notebook if you need a hint or to check your work.
# 
# Any extra time will be reserved for consulting, talking about projects, etc.

# ## Copy/paste - Boilerplate code
# 
# The first three lessons of this book contain: 
# 1. The schedule
# 2. Book organization
# 3. How to start coding
# 
# The next three contain copy/paste code for applying to your own projct:
# 
# 4. Data types/structures and error messages  
# 5. Importing and wrangling data  
# 6. Visualizing data
