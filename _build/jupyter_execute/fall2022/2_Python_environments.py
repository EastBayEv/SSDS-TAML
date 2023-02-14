#!/usr/bin/env python
# coding: utf-8

# # Chapter 2 - Python environments
# 
# 2022 August 25

# <a target="_blank" href="https://colab.research.google.com/github/EastBayEv/SSDS-TAML/blob/main/fall2022/2_Python_environments.ipynb">
#   <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
# </a>

# ## What is a programming language?
# 
# For all intents and purposes, a programming language is a means to give instructions to a computer. [Read Python's executive summary here.](https://www.python.org/doc/essays/blurb/)
# 
# You can write your python code in the below environments (and more!), but no matter which option you choose the content is virtually identical:
# 
# 1. Google Colab
# 2. Jupyter Lab via Binder
# 3. Local Anaconda environment
# 4. Text editor
# 
# Furthermore, it is hard to say which is "better" or "best", since this is dependent on the task at hand. Remember that TAML sessions use Google Colab:
# 
# ## Run the code
# 
# If you are using Google Colab, Jupyter Lab, or a local Python Anaconda installation, press `shift` and `enter` simultaneously on your keyboard to:
# * Run a code cell
# * Render a text/markdown cell

# ## 1. Google Colaboratory
# 
# ## REQUIRED: Set up your Colab environment before the bootcamp! 
# 
# Google Colab is the easiest way to run the TAML materials, and is what we will use to teach you. Instructions for Jupyter Lab or a local Python installation are also below. While they function similarly, keep in mind that there are some slight differences between the environments. 
# 
# Start by installing the extension to your Stanford Google Drive account. 
# 
# 1. [Click here to visit the Google Colab sign in page](https://accounts.google.com/ServiceLogin/signinchooser?service=wise&passive=true&continue=http%3A%2F%2Fdrive.google.com%2F%3Futm_source%3Den&utm_medium=button&utm_campaign=web&utm_content=gotodrive&usp=gtd&ltmpl=drive&flowName=GlifWebSignIn&flowEntry=ServiceLogin)
# 2. Enter your SUNet email address on the “Choose an account” screen
# 3. Click “My Drive”
# 4. Click the “+New” button and select “Google Colaboratory”
# 
# ![](img/colab_new.png)

# 5. If you do not see this option, click “+ Connect more apps” and search for Colaboratory and the icon will appear.
# 
# ![](img/colab_search.png)
# 
# 6. Click the brown and orange Colaboratory icon and click “Install”
# 
# ![](img/colab_install.png)
# 
# 7. Click “Continue” and select your **stanford.edu account** if prompted
# 
# ![](img/colab_continue.png)
# 
# 8. You will see that Colaboratory was successfully installed/connected to Good Drive when finished. Click OK and Done to complete and close any remaining pop-up windows.
# 
# ![](img/colab_success.png)

# ## Code versus markdown cells
# 
# Click the Colab badge at the top of each chapter in the book.
# 
# Click the `Run` button in the toolbar, or press `shift` and `enter` on your keyboard to execute the cell. 
# 
# ### Make note of the top menu toolbar
# 
# * Click the **+ Code** button to add a code cell
# * Click the **+ Text** button to add a text/markdown cell
# * Click the **Menu** to view the table of contents
# * Click the **File folder** to view your available files
# 
# ![](img/colab_toolbar.png)
# 
# ### Save your work
# 
# * Click the **Copy to Drive** toolbar button, or
# * Click "File" --> "Download" --> "Download.ipynb"
# 
# ![](img/colab_save_notebook.png)
# 
# ### Learn more
# 
# Check out [Google Colab Tips for Power Users ](https://amitness.com/2020/06/google-colaboratory-tips/) for shortcuts. 

# ## 2. Jupyter Lab 
# 
# JupyterLab is another way to access these materials. It is a web-based interactive development environment (IDE) which allows you to write and run code in your browser. 
# 
# Cick the "Launch Binder" button below and open it in a new browser tab. 
# 
# [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/EastBayEv/SSDS-TAML/HEAD)

# ### Click "Python 3 (ipykernel)" 
# 
# * Click "Python 3 (ipykernel) under the "Notebook" header to **open a blank notebook.**
# * Otherwise, navigate the file structure in the top left corner to **open an existing notebook.**
# ![setkernel](img/kernel.png)

# ### Cells are code by default
# 
# Open a blank notebook and in the first cell, type:
#     
# > print("Your Name Here") 

# ### Insert new cells
# 
# Click the "plus" button to insert a new cell. It can be found between the "save disk" and "scissors" icons. 
# ![addcell](img/addcell.png)  
# 
# Alternatively, click on an area outside of any cells in the notebook, and press the `a` or `b` key to create a new cell above or below your current one. 

# ### Switch between code and text cells
# 
# Switch between code and text cells by clicking the dropdown menu. 
# ![switch](img/switch.png)

# ### Edit a markdown cell
# 
# Double-click a markdown cell to edit it. Press `shift + enter` to render the text. 
# 
# Go through the cells on this page and check out [Markdown Cheatsheet](https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet) (direct links to different sections below) to explore formatting rules and shortcuts. 
# 
# **Table of Contents**  
# [Headers](https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet#headers)  
# [Emphasis](https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet#emphasis)  
# [Lists](https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet#lists)  
# [Links](https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet#links)  
# [Images](https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet#images)  
# [Code and Syntax Highlighting](https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet#code)  
# [Footnotes](https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet#footnotes)  
# [Tables](https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet#tables)  
# [Blockquotes](https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet#blockquotes)  
# [Inline HTML](https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet#html)  
# [Horizontal Rule](https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet#hr)  
# [Line Breaks](https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet#lines)  
# [YouTube Videos](https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet#videos)

# ### Save your work
# 
# Click the "disk" icon to save your work. Right-click the tab and click "Rename Notebook" to give your notebook a name:
# 
# ![rename](img/rename.png)

# * Click "File" --> "Download" to save your Jupyter Notebook (extension `.ipynb`) before exiting a Jupyter Lab session. 
# * There are many other options, take some time to explore the top menu and buttons.  
# 
# ![download](img/dl.png)

# ### Close and halt
# 
# Safely close the notebook by clicking "File" --> "Close and Halt" before closing your browser.

# ![closehalt](img/close.png)

# ## 3. Local Python installation
# 
# Despite the increasing number of cloud-based solutions, it is always good to have a local installation on your own computer. Point your web browswer here to install Python Anaconda distribution 3.9 (as of August 2022): https://www.anaconda.com/products/individual
# 
# * Download the installation file for your operating system, open it, and follow the instructions to install. 
# * Once installation completes, open the application named "Anaconda Navigator". It looks like this: 
# 
# ![navigator](img/navigator.png)
# 
# * Click the "Launch" button under either Jupyter Notebook or Jupyter Lab. 
# * Open the `.ipynb` notebook you just downloaded from JupyterLab in your local installation and repeat the steps above.

# ### Download the workshop materials
# 
# You can also download the TAML materials for use on your local installation. 
# 
# 1. Visit: https://github.com/EastBayEv/SSDS-TAML
# 2. Click the green "Code" button
# 3. Click "Download ZIP"
# ![zip](img/zip.png)
# 4. Extract this folder someplace familiar (we recommend your Desktop). 
# 5. Open Anaconda Navigator, launch JupyterLab, navigate your your file structure, and click the notebook file to launch it. 
# 
# ### git clone
# Git users open a Terminal and type: `git clone https://github.com/EastBayEv/SSDS-TAML.git`

# ### Install external libraries
# 
# Install user-defined software libraries to enhance Python's functionality. In a new notebook cell type `!pip install <library name>`, e.g.: 
# * `!pip install pandas` 
# * `!pip install seaborn`

# ### Dead kernel? 
# 
# The notebook kernel will fail from time to time, which is normal. Simply click "Kernel" from the File menu and one of the "Restart" options.

# ![restart](img/restart.png)

# > NOTE: about the terminal, you will see it run processes in the background like this - you can ignore this, but don't close it!
# 
# ![terminal](img/terminal.png)

# ### If you accidentally delete a Jupyter Notebook ...
# 
# Check your operating system's Recycle Bin!

# ## 4. Text editors
# 
# You might find that Jupyter environments are too limiting for your task. Using a text editor to write your scripts and run them via the command line is another common option. Contact SSDS if you want to learn more!

# ## Exercises
# 
# 1. (Required) Set up your Google Colaboratory (Colab) environment following the instructions in #1 listed above. 
# 2. (Optional) Check that you can correctly open these notebooks in Jupyter Lab. 
# 3. (Optional) Install Python Anaconda distribution on your machine.

# ## Basic Python syntax
# 
# Open Chapter 3 "Basic Python syntax" to start coding!
