#!/usr/bin/env python
# coding: utf-8

# # Chapter 2 - Start coding

# <a target="_blank" href="https://colab.research.google.com/github/EastBayEv/SSDS-TAML/blob/main/fall2022/2_Start_coding.ipynb">
#   <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
# </a>

# You can program Python in several different environments: 
# 
# 1. Google Colab
# 2. Jupyter Lab via Binder
# 3. Local Anaconda environment
# 4. Text editor
# 
# No matter which option you choose, the content is identical, although there are a few slight differences in the interfaces. Furthermore, one interface might be better than the others for a given task. 

# ## 1. Google Colaboratory
# 
# Right-click the Colab badge at the top of each chapter that contains runnable code. 
# 
# Click the `Run` button in the toolbar, or press `shift` and `enter` on your keyboard to execute the cell. 
# 
# * Although there are some differences (keyboard shortcuts and runtimes, for example), Google Colab functions similarly to Jupyter Lab and Notebooks. 
# 
# **Save your work** by clicking "File" --> "Download"
# ![](img/colab_save_notebook.png)

# ## 2. Jupyter Lab 
# 
# JupyterLab is another way to access these materials. It is a web-based interactive development environment (IDE) which allows you to write and run code in your browser. 
# 
# Right click the "Launch Binder" button below and open it in a new browser tab. 
# 
# [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/EastBayEv/SSDS-TAML/HEAD)

# ### Click "Python 3 (ipykernel)" 
# 
# * Click "Python 3 (ipykernel) under the "Notebook" header to launch a blank notebook. 
# * Otherwise, navigate the file structure in the top left corner to open an existing notewook.
# ![setkernel](img/kernel.png)

# ### Cells are code by default
# 
# In the first cell, type:
#     
# > print("Your Name Here") 

# ### Insert new cells
# 
# Click the "plus" button to insert a new cell. It can be found between the "save disk" and "scissors" icons. 
# ![addcell](img/addcell.png)  
# 
# Alternatively, click on an area outside of any cells in the notebook, and press 'A' or 'B' to create new cell above or below your current one, respectively. 

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
# Click the "disk" icon to save your work. Right-click the tab to give your notebook a name:
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
# Despite the increasing number of cloud-based solutions, it is always good to have a local installation on your own computer. Point your web browswer here to install Python Anaconda distribution 3.9 (as of January 2022): https://www.anaconda.com/products/individual
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
# Copy these materials for use on your local installation. 
# 
# 1. Visit: https://github.com/EastBayEv/SSDS-TAML
# 2. Click the green "Code" button
# ![code](img/code.png)
# 3. Click "Download ZIP"
# ![zip](img/zip.png)
# 4. Extract this folder someplace familiar such as your Desktop.
# 5. Open Anaconda Navigator, launch JupyterLab, and navigate your directories to launch these notebooks.
# 
# #### git clone
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

# > NOTE: about the terminal, you will see it run processes in the background like this:
# 
# ![terminal](img/terminal.png)

# ### If you accidentally delete a Jupyter Notebook ...
# 
# Check your operating system's Recycle Bin!

# ## 4. Text editors
# 
# You might find that Jupyter environments are too limiting for your task. Using a text editor to write your scripts and run them via the command line is another common option. Contact SSDS if you want to learn more!

# ## Python basics
# 
# Open Chapter 3 "Boilerplate code" to start learning basic Python syntax.
