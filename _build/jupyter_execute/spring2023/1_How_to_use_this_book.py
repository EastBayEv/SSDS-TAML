#!/usr/bin/env python
# coding: utf-8

# # Chapter 1 - How to use this book
# 2023 April 5

# <a target="_blank" href="https://colab.research.google.com/github/EastBayEv/SSDS-TAML/blob/main/spring2023/1_How_to_use_this_book.ipynb">
#   <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
# </a>

# This is the textbook for the Stanford Libraries Text Analysis and Machine Learning (TAML) group.
# 
# It is divided into three sections: **1) center portion, 2) left sidebar, and 3) right sidebar.**
# 
# **1. The center portion** contains the main text. Here the information is presented and explained. **2. The left sidebar** shows the table of contents. Use the `left` and `right` arrow keys to browse the different chapters. **The right sidebar** contains clickable section headers for the chapter you are currently viewing. Use the `up` and `down` arrow keys to scroll the page.

# ## Jupyter icons
# Notice the icons at the top of each page. 
# ![icons](img/icons.png)
# * The **rocketship** icon appears on pages with executable code. 
# Hover your mouse over it and click "Binder" to launch all of the materials there in a Jupyter Lab on Binder. This is a neat option for running code directly in a web browser - not ideal here, but fun for smaller projects!
# * Click the **square** icon to enter fullscreen mode. Press the escape key to exit. 
# * Move your mouse over the **Octocat** icon. Click "Repository" to visit the GitHub site. Or, click "Open Issue" to contribute to this book. 
# * The **download** icon can be used to download this book. 
# * Click the **three lines** (toggle navigation) to hide the left sidebar.

# ## Exercises
# 
# 1. You will find challenge exercises to work on at the end of each chapter. They will require you to interpret research tasks and write code to provide a solution. 
# 
# 
# * At the top of each chaper you will see a badge that says "Open in Colab" to open these materials in a Google Colab Python environment. Simply click the Colab badge to open that notebook for a given chapter. 
# 
# 2. Click the "Ope in Colab" badge at the top of this notebook to open it in the Colaboratory environment. Press the `shift` and `enter` keys simultaneously on your keyboard to run the code in the section "Lucky Card" below. 
# 
# > NOTE: Colab will prompt you with "Warning: This notebook was not authored by Google."
# >
# > Click "Run anyway"
# 
# 3. When you are done, save this notebook by clicking the disk icon and then "File" --> "Close and Halt" to close it. 

# ## Lucky card
# 
# For example, below is a program that displays a welcome message. It creates a deck of playing cards and draws a card by randomly sampling a card and suit, joins them together, and then shows the output on the screen. 
# 
# You will learn how to understand and write code like this throughout the book!

# In[1]:


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


# ## Solutions
# 
# Solutions to the challenge exercises can be found in the "Solutions" chapter.

# ## Python environments
# 
# Read Chapter 2 "Python environments" to learn a few different ways you can start coding in Python. The TAML Group currently uses the Google Colab environment, so be sure to follow the instructions for setting up your Colab account!
