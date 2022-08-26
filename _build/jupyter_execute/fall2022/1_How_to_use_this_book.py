#!/usr/bin/env python
# coding: utf-8

# # Chapter 1 - How to use this book

# <a target="_blank" href="https://colab.research.google.com/github/EastBayEv/SSDS-TAML/blob/main/fall2022/1_How_to_use_this_book.ipynb">
#   <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
# </a>

# This book provides a reference for the code we present and discuss in the Text Analysis and Machine Learning (TAML) group. While you can actually run the code in the book itself (although it might take awhile to load! see below), think of this book as the single point to access the content, conceptual introductions, vocabulary terms, and code. 
# 
# At the top of each chaper you will see a badge that says "Open in Colab" to open these materials in a Google Colab Python environment. Simply click the Colab badge to open that notebook for a given chapter. 
# 
# This book is divided into three sections: **1) center portion, 2) left sidebar, and 3) right sidebar.**
# 
# 1. **The center portion** contains the main text. Here, goals, concepts, vocabulary, objectives, and code will be presented and explained. 
# 
# ## Lucky card
# 
# Below is a crude program that displays a welcome message. It draws a card by randomly sampling a card and suit, joins them together to form the welcome message, and then shows the output on the screen. You will how to understand and write code similar to this throughout the book!

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


# 2. **The left sidebar** shows the table of contents. 
#     * Use the `left and right arrow keys` to browse the different chapters. 
# 
# <br/>
# 
# 3. **The right sidebar** contains clickable section headers for the chapter you are currently viewing. 
#     * Use the `up and down arrow keys` to scroll the page.

# ## Jupyter icons
# Notice the icons at the top of each page. 
# ![icons](img/icons.png)
# * The **rocketship** icon appears on pages with executable code. 
# Hover your mouse over it and click "Binder" to launch all of the materials there in a Jupyter Lab on Binder. Or, click **"Live Code"** to be able to run the code on the webpage, although this might take a long time to load! 

# * Click the **square** icon to enter fullscreen mode. Press the escape key to exit. 
# * Move your mouse over the **Octocat** icon. Click "Repository" to visit the GitHub site. Or, click "Open Issue" to contribute to this book. 
# * The **download** icon can be used to download this book. 
# * Click the **three lines** (toggle navigation) to hide the left sidebar.

# ## Format
# 
# TAML's format consists of 2-hour sessions, divided into the following sections:
# 
# * Lecture overviews of various lengths that introduce concepts, vocabulary, and workflows
# * Short break
# * Time to work on challenge exercises and ask questions
# * Discussion of solutions to challenge exercises

# ## Exercises
# 
# 1. You will find challenge exercises to work on at the end of each chapter. They will require you to write code such as that found in the cell at the top of this notebook. 
# 
# 2. Click the "Colab" badge at the top of this notebook to open it in the Colaboratory environment. Press `shift` and `enter` simultaneously on your keyboard to run the code and draw your lucky card!

# ## Solutions
# 
# Solutions to the challenge exercises can be found in the "Solutions" chapter.
# 
# * Be aware that we won't always go through all of the challenge exercises in-class due to time constraints! 
# * Any extra time will be reserved for consulting, talking about projects, etc.

# ## Python environments
# 
# Read Chapter 2 "Python environments" to learn a few different ways you can start coding in Python. The TAML Group currently uses the Google Colab environment, so be sure to follow the instructions for setting up your Colab account!
