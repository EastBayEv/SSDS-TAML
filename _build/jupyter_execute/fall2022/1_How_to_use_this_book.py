#!/usr/bin/env python
# coding: utf-8

# # Chapter 1 - How to use this book

# This book provides a reference for the code we present and discuss in the Text Analysis and Machine Learning (TAML) group. While you can actually run the code in the book itself (although it might take awhile to load! see below), think of this book as the single point to access the content, conceptual introductions, and vocabulary terms. 
# 
# Additionally, each chapter with runnable code has a link to open the materials in a Google Colab Python environment. Simply click the Colab badge to open the notebook for a specific chapter. 
# 
# * [Instructions for setting up Google Colaboratory]()
# 
# This book is divided into three sections. 
# 
# 1. **The center portion** contains the main text. Here, goals, concepts, vocabulary, objectives, and code will be presented and explained. 
# 
# Below is a crude program that randomly draws a lucky card by randomly sampling a card and suit, joins them together to form the welcome message, and shows the output on the screen. 

# In[1]:


# Lucky card

import random
import calendar
from datetime import date, datetime
deck = 'ace two three four five six seven eight nine ten jack queen king'.split()
suit = 'spades clubs hearts diamonds'.split()
today = calendar.day_name[date.today().weekday()]
date = datetime.today().strftime('%Y-%m-%d')
select_value = random.sample(deck, 1)[0]
select_suit = random.sample(suit, 1)[0]
print("\nWelcome to TAML at SSDS!")
print("\nYour lucky card for " + today + " " + date + " is: " + select_value + " of " + select_suit)


# 2. **The left sidebar** shows the search field and shortcuts to session materials.
#     * Use the `left and right arrow keys` to browse the different chapters. 
# 3. **The right sidebar** contains the section header shortcuts. 
#     * Use the `up and down arrow keys` to scroll the page.

# ## Jupyter icons
# Note the buttons at the top of each page. 
# ![icons](img/icons.png)
# * The rocketship icon appears only on pages with executable code. 
# Hover your mouse over it and click "Binder" to launch all of the materials there. Or, click "Live Code" to run code on this webpage, although it might take a long time to load! 

# * Click the square icon to enter fullscreen mode. Press the escape key to exit. 
# * Move your mouse over the Octocat icon. Click "Repository" to visit the GitHub site. Or, click "Open Issue" to contribute to this book. 
# * The download icon can be used to export the contents of this book. 
# * Click the left facing arrow `<-` to hide the left sidebar.

# ## Format
# 
# TAML's format consists of 2-hour sessions, divided into the following sections:
# 
# * 55 minute lecture overview of concepts, vocabulary, and workflows
# * 5 minute break
# * 40 minutes of challenge exercises/question asking
# * 20 minutes of solutions discussion
# 
# > Be aware that we won't always go through all of the exercises due to time constraints (especially during the Fall Quarter 2022 Bootcamp), but that they are there along with solutions for you to practice on your own. 
# 
# Any extra time will be reserved for consulting, talking about projects, etc.

# ## Exercises
# 
# 1. You will find challenge exercises to work on at the end of each chapter. They will frequently require you to write code such as that found in the cell at the top of this notebook. 
# 
# 2. Click the "Colab" badge at the top of this notebook to open it in the Colaboratory environment. Press `shift` and `enter` simultaneously on your keyboard to run the code and draw your lucky card!

# ## Solutions
# 
# Solutions to the challenge exercises can be found in the "Solutions" chapter.

# ## Start coding
# 
# Read Chapter 2 "Start Coding" to learn various ways to access the materials and start running code!
