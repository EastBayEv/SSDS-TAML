#!/usr/bin/env python
# coding: utf-8

# # Chapter 5 - Ensemble machine learning, deep learning

# 2022 February 23

# ![kandc](img/kandc.jpg)
# 
# [Texas Monthly, Music Monday: Uncovering The Mystery Of The King & Carter Jazzing Orchestra](https://www.texasmonthly.com/the-daily-post/music-monday-uncovering-the-mystery-of-the-king-carter-jazzing-orchestra/)

# ## Ensemble machine learning

# "Ensemble machine learning methods use multiple learning algorithms to obtain better predictive performance than could be obtained from any of the constituent learning algorithms." [H2O.ai ensemble example](https://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/stacked-ensembles.html)
# 
# In this manner, SuperLearner ensembles are powerful tools because they: 
# * elucidate issues of algorithmic bias and variance
# * circumvent bias introduced by selecting single models
# * offer a means to optimize prediction through the stacking/blending of weaker models
# * allow for comparison of multiple algorithms, and/or comparison of the same model but tuned in many different ways
# * utilize a second-level algorithm that produces an ideal weighted prediction that is suitable for data of virtually all distributions and uses cross-validation to prevent overfitting
# 
# The below example utilizes the h2o package, and requires Java to be installed on your machine.
# * install Java: https://www.java.com/en/download/help/mac_install.html
# * h2o SuperLearner example: https://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/stacked-ensembles.html
# 
# Check out some other great tutorials: 
# * Python mlens library: https://mlens.readthedocs.io/en/0.1.x/install/
# * Machine Learning Mastery: https://machinelearningmastery.com/super-learner-ensemble-in-python/
# * KDNuggets: https://www.kdnuggets.com/2018/02/introduction-python-ensembles.html/2#comments
# 
# The quintessential R guide: 
# * Guide to SuperLearner: https://cran.r-project.org/web/packages/SuperLearner/vignettes/Guide-to-SuperLearner.html
# 
# Read the papers: 
# * [Van der Laan, M.J.; Polley, E.C.; Hubbard, A.E. Super Learner. Stat. Appl. Genet. Mol. Biol. 2007, 6, 1–21.](https://www.degruyter.com/document/doi/10.2202/1544-6115.1309/html)
# * [Polley, E.C.; van der Laan, M.J. Super Learner in Prediction, UC Berkeley Division of Biostatistics Working Paper Series Paper 266.](https://biostats.bepress.com/ucbbiostat/paper266)

# ## H2O SuperLearner ensemble

# In[1]:


# !pip install h2o

# Requires install of Java
# https://www.java.com/en/download/help/mac_install.html


# In[2]:


import h2o
from h2o.estimators.random_forest import H2ORandomForestEstimator
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.estimators.stackedensemble import H2OStackedEnsembleEstimator
from h2o.grid.grid_search import H2OGridSearch
from __future__ import print_function
h2o.init()


# In[3]:


get_ipython().run_cell_magic('capture', '', '# Import a sample binary outcome train/test set into H2O\n# Learn about subset of Higgs Boson dataset: https://www.kaggle.com/c/higgs-boson\ntrain = h2o.import_file("https://s3.amazonaws.com/erin-data/higgs/higgs_train_10k.csv")\ntest = h2o.import_file("https://s3.amazonaws.com/erin-data/higgs/higgs_test_5k.csv")')


# In[4]:


train


# In[5]:


print(train.shape)
print(test.shape)


# In[6]:


# Identify predictors and response
x = train.columns
y = "response"
x.remove(y)


# In[7]:


# For binary classification, response should be a factor
train[y] = train[y].asfactor()
test[y] = test[y].asfactor()


# In[8]:


# Number of CV folds (to generate level-one data for stacking)
nfolds = 5


# In[9]:


get_ipython().run_cell_magic('capture', '', '# There are a few ways to assemble a list of models to stack together:\n# 1. Train individual models and put them in a list\n# 2. Train a grid of models\n# 3. Train several grids of models\n# Note: All base models must have the same cross-validation folds and\n# the cross-validated predicted values must be kept.\n\n\n# 1. Generate a 2-model ensemble (GBM + RF)\n\n# Train and cross-validate a GBM\nmy_gbm = H2OGradientBoostingEstimator(distribution="bernoulli",\n                                      ntrees=10,\n                                      max_depth=3,\n                                      min_rows=2,\n                                      learn_rate=0.2,\n                                      nfolds=nfolds,\n                                      fold_assignment="Modulo",\n                                      keep_cross_validation_predictions=True,\n                                      seed=1)\nmy_gbm.train(x=x, y=y, training_frame=train)')


# In[10]:


get_ipython().run_cell_magic('capture', '', '# Train and cross-validate a RF\nmy_rf = H2ORandomForestEstimator(ntrees=50,\n                                 nfolds=nfolds,\n                                 fold_assignment="Modulo",\n                                 keep_cross_validation_predictions=True,\n                                 seed=1)\nmy_rf.train(x=x, y=y, training_frame=train)')


# In[11]:


get_ipython().run_cell_magic('capture', '', '# Train a stacked ensemble using the GBM and GLM above\nensemble = H2OStackedEnsembleEstimator(model_id="my_ensemble_binomial",\n                                       base_models=[my_gbm, my_rf])\nensemble.train(x=x, y=y, training_frame=train)\n\n# Eval ensemble performance on the test data\nperf_stack_test = ensemble.model_performance(test)')


# In[12]:


get_ipython().run_cell_magic('capture', '', '# Compare to base learner performance on the test set\nperf_gbm_test = my_gbm.model_performance(test)\nperf_rf_test = my_rf.model_performance(test)\nbaselearner_best_auc_test = max(perf_gbm_test.auc(), perf_rf_test.auc())\nstack_auc_test = perf_stack_test.auc()\nprint("Best Base-learner Test AUC:  {0}".format(baselearner_best_auc_test))\nprint("Ensemble Test AUC:  {0}".format(stack_auc_test))')


# In[13]:


get_ipython().run_cell_magic('capture', '', '# Generate predictions on a test set (if neccessary)\npred = ensemble.predict(test)\n\n\n# 2. Generate a random grid of models and stack them together\n\n# Specify GBM hyperparameters for the grid\nhyper_params = {"learn_rate": [0.01, 0.03],\n                "max_depth": [3, 4, 5, 6, 9],\n                "sample_rate": [0.7, 0.8, 0.9, 1.0],\n                "col_sample_rate": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]}\nsearch_criteria = {"strategy": "RandomDiscrete", "max_models": 3, "seed": 1}')


# In[14]:


get_ipython().run_cell_magic('capture', '', '# Train the grid\ngrid = H2OGridSearch(model=H2OGradientBoostingEstimator(ntrees=10,\n                                                        seed=1,\n                                                        nfolds=nfolds,\n                                                        fold_assignment="Modulo",\n                                                        keep_cross_validation_predictions=True),\n                     hyper_params=hyper_params,\n                     search_criteria=search_criteria,\n                     grid_id="gbm_grid_binomial")\ngrid.train(x=x, y=y, training_frame=train)')


# In[15]:


get_ipython().run_cell_magic('capture', '', '# Train a stacked ensemble using the GBM grid\nensemble = H2OStackedEnsembleEstimator(model_id="my_ensemble_gbm_grid_binomial",\n                                       base_models=grid.model_ids)\nensemble.train(x=x, y=y, training_frame=train)\n\n# Eval ensemble performance on the test data\nperf_stack_test = ensemble.model_performance(test)\n\n# Compare to base learner performance on the test set\nbaselearner_best_auc_test = max([h2o.get_model(model).model_performance(test_data=test).auc() for model in grid.model_ids])\nstack_auc_test = perf_stack_test.auc()\nprint("Best Base-learner Test AUC:  {0}".format(baselearner_best_auc_test))\nprint("Ensemble Test AUC:  {0}".format(stack_auc_test))\n\n# Generate predictions on a test set (if neccessary)\npred = ensemble.predict(test)')


# ## Deep learning basics
# 
# Deep learning is a subfield of machine learning that uses a variety of multi-layered artificial neural networks to model datasets and predict outcomes. Deep learning is ideal for numeric, text, image, video, and sound data because deep representative networks store these data as large matrices and recycle error to make better predictions during the next epoch. 
# 
# To understand deep networks, let's start with a toy example of a single feed forward neural network - a perceptron.
# 
# Read Goodfellow et al's Deep Learning Book to learn more: https://www.deeplearningbook.org/

# In[16]:


import pandas as pd

# generate toy dataset
example = {'x1': [1, 0, 1, 1, 0], 
           'x2': [1, 1, 1, 1, 0], 
           'xm': [1, 0, 1, 1, 0],
           'y': ['yes', 'no', 'yes', 'yes', 'no']
           }
example_df = pd.DataFrame(data = example)
example_df


# ![perceptron](img/perceptron.png)
# 
# Perceptron figure modified from [Sebastian Raschka's Single-Layer Neural Networks and Gradient Descent](https://sebastianraschka.com/Articles/2015_singlelayer_neurons.html)

# Perceptron key terms: 
# * **Layer:** the network typology of a deep learning model, usually divided into variations of input, hidden, preprocessing, encoder/decoder, and output. 
# * **Inputs:** the features/covariates for a single observation. These are just the individual cells in a dataframe (the 1's and 0s from `example_df` above), but they could be words from a text or pixels from an image. 
# * **Weights:** the learnable parameters of a model that connect the input layer to the output via the net input (summation) and activation functions. 
# * **Bias term:** A placeholder "1" assures that we do not receive 0 outputs by default. 
# * **Net input function:** computes the weighted sum of the input layer. 
# * **Activation function:** determine if a neuron should be fired or not. In binary classification for example, this means should a 1 or 0 be output?
# * Output: one node that contains the y prediction
# * **Error:** how far off an output prediction was. The weights can be updated by adjusting the learning rate based on the error to reduce it for the next epoch

# ## What makes a network "deep"?
# 
# A "deep" network is just network with multiple/many hidden layers for handling potential nonlinear transformations.
# 
# * Fully connected layer: a layer where all nodes are connected to every node in the next layer (as indicated by the purple arrows 

# ![deep](img/deep.png)
# 
# Example of "deep" network with two hidden layers modified from [DevSkrol's Artificial Neural Network Explained with an Regression Example](https://devskrol.com/2020/11/22/388/)
# 
# >NOTE: Bias term not shown for some reason!

# Let's go through François Chollet's "Image classification from scratch" [tutorial](https://keras.io/examples/vision/image_classification_from_scratch/) to examine this architecture to predict images of cats versus dogs. 
# 
# [Click here to open the Colab notebook](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/vision/ipynb/image_classification_from_scratch.ipynb)
# 
# You should also check out his deep learning book! https://www.manning.com/books/deep-learning-with-python-second-edition

# ![dogcat](img/dogcat.jpg)
