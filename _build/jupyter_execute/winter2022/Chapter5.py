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
# In this manner, stacking/SuperLearner ensembles are powerful tools because they: 
# 
# 1. Eliminates bias of single algorithm selection for framing a research problem.
# 
# 2. Allows for comparison of multiple algorithms, and/or comparison of the same model but tuned in many different ways.
# 
# 3. Utilizes a second-level algorithm that produces an ideal weighted prediction that is suitable for data of virtually all distributions and uses external cross-validation to prevent overfitting.
# 
# The below example utilizes the h2o package, and requires Java to be installed on your machine.
# * install Java: https://www.java.com/en/download/help/mac_install.html
# * h2o SuperLearner example: https://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/stacked-ensembles.html
# 
# Check out some other tutorials: 
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

# Use machine learning ensembles to detect whether or not (simulated) particle collisions produce the Higgs Boson particle or not. Learn more about the data: https://www.kaggle.com/c/higgs/overview
# 
# ## What is the Higgs Boson?
# 
# "The Higgs boson is the fundamental particle associated with the Higgs field, a field that gives mass to other fundamental particles such as electrons and quarks. A particle’s mass determines how much it resists changing its speed or position when it encounters a force. Not all fundamental particles have mass. The photon, which is the particle of light and carries the electromagnetic force, has no mass at all." (https://www.energy.gov/science/doe-explainsthe-higgs-boson)
# 
# 
# ![hb](img/hb.png)

# ### Install h2o and Java

# In[1]:


# !pip install h2o

# Requires install of Java
# https://www.java.com/en/download/help/mac_install.html


# ### Import

# In[2]:


import h2o
from h2o.estimators.random_forest import H2ORandomForestEstimator
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
from h2o.estimators.stackedensemble import H2OStackedEnsembleEstimator
from h2o.grid.grid_search import H2OGridSearch
from __future__ import print_function

# turn off progress bars
h2o.no_progress()


# ### Initialize an h2o cluster

# In[3]:


h2o.init(nthreads=-1, max_mem_size='2G')


# ### Import a sample binary outcome train/test set into H2O

# In[4]:


train = h2o.import_file("https://s3.amazonaws.com/erin-data/higgs/higgs_train_10k.csv")
test = h2o.import_file("https://s3.amazonaws.com/erin-data/higgs/higgs_test_5k.csv")


# In[5]:


train


# In[6]:


print(train.shape)
print(test.shape)


# ### Identify predictors and response

# In[7]:


x = train.columns
y = "response"
x.remove(y)


# ### For binary classification, response should be a factor

# In[8]:


train[y] = train[y].asfactor()
test[y] = test[y].asfactor()


# ### Number of CV folds (to generate level-one data for stacking)

# In[9]:


nfolds = 5


# ### How to stack
# 
# There are a few ways to assemble a list of models to stack together:
# 1. Train individual models and put them in a list
# 2. Train a grid of models
# 3. Train several grids of models
# 
# >Note: All base models must have the same cross-validation folds and the cross-validated predicted values must be kept.

# ## 1. Generate a 2-model ensemble 
# 
# Use three algorithms: 
# 1. random forest
# 2. gradient boosted machine
# 3. lasso

# TODO: add RF and GBM defining characteristics
# 
# TODO: show how changing hyperparamters randomly can lead to overfitting (specifically # trees)

# ### Train and cross-validate a random forest

# In[10]:


rf = H2ORandomForestEstimator(ntrees = 100,
                              nfolds = nfolds,
                              fold_assignment = 'Modulo',
                              keep_cross_validation_predictions = True,
                              seed = 1)
rf.train(x = x, y = y, training_frame = train)


# ### Random forest test set performance

# In[11]:


rf.model_performance(test)


# ### Train and cross-validate a gradient boosted machine

# In[12]:


gbm = H2OGradientBoostingEstimator(distribution = "bernoulli",
                                   ntrees = 10,
                                   max_depth = 3,
                                   min_rows = 2,
                                   learn_rate = 0.2,
                                   nfolds = nfolds,
                                   fold_assignment = "Modulo",
                                   keep_cross_validation_predictions = True,
                                   seed = 1)
gbm.train(x = x, y = y, training_frame = train)


# ### Gradient boosted machine test set performance

# In[13]:


gbm.model_performance(test)


# ## 3. Train a stacked ensemble using the GBM and RF above
# 
# What's going on here - anything suspicious?

# In[14]:


ensemble = H2OStackedEnsembleEstimator(model_id = "my_ensemble_binomial",
                                       base_models = [rf, gbm])
ensemble.train(x = x, y = y, training_frame = train)


# ### Ensemble performance on test set

# In[15]:


perf_stack_test = ensemble.model_performance(test)
perf_stack_test


# ### Compare to base learner performance on the test set
# 
# The ensemble is a little better, but it is still pretty close...

# In[16]:


perf_gbm_test = gbm.model_performance(test)
perf_rf_test = rf.model_performance(test)
baselearner_best_auc_test = max(perf_gbm_test.auc(), perf_rf_test.auc())
stack_auc_test = perf_stack_test.auc()
print("Best Base-learner Test AUC:  {0}".format(baselearner_best_auc_test))
print("Ensemble Test AUC:  {0}".format(stack_auc_test))


# ### Generate predictions on a test set (if neccessary)

# In[17]:


predictions = ensemble.predict(test)
predictions


# ## 4. Generate a random grid of models and stack them together

# ### Specify GBM hyperparameters for the grid
# 
# Keep in mind it might be easier to define sequences of numbers for your various hyperparameter tunings. 
# 
# Also, exponential and logarithmic scales are probably preferred to linear ones.

# In[18]:


hyper_params = {"learn_rate": [0.01, 0.03, 0.05, 0.2, 0.3, 0.4, 0.7, 0.8],
                "max_depth": [3, 4, 5, 6, 9],
                "sample_rate": [0.7, 0.8, 0.9, 1.0],
                "col_sample_rate": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]}
search_criteria = {"strategy": "RandomDiscrete", "max_models": 3, "seed": 1}


# In[19]:


# Train the grid
grid = H2OGridSearch(model=H2OGradientBoostingEstimator(ntrees = 10,
                                                        seed = 1,
                                                        nfolds = nfolds,
                                                        fold_assignment = 'Modulo',
                                                        keep_cross_validation_predictions = True),
                     hyper_params=hyper_params,
                     search_criteria=search_criteria,
                     grid_id="gbm_grid_binomial")
grid.train(x=x, y=y, training_frame=train)


# ## 5. Train a stacked ensemble using the GBM grid

# In[20]:


ensemble = H2OStackedEnsembleEstimator(model_id = "my_ensemble_gbm_grid_binomial",
                                       base_models = grid.model_ids)
ensemble.train(x = x, y = y, training_frame = train)


# ### Eval ensemble performance on the test data

# In[21]:


perf_stack_test = ensemble.model_performance(test)
perf_stack_test


# ## 6. Compare to base learner performance on the test set

# In[22]:


baselearner_best_auc_test = max([h2o.get_model(model).model_performance(test_data=test).auc() for model in grid.model_ids])
stack_auc_test = perf_stack_test.auc()
print("Best Base-learner Test AUC:  {0}".format(baselearner_best_auc_test))
print("Ensemble Test AUC:  {0}".format(stack_auc_test))


# ### Generate predictions on a test set (if neccessary)

# In[23]:


predictions2 = ensemble.predict(test)
predictions2


# ## Deep learning basics
# 
# Deep learning is a subfield of machine learning that uses a variety of multi-layered artificial neural networks to model datasets and predict outcomes. 
# 
# ### The human brain model key terms
# 
# The idea was originally modelled on the human brain, which has ~100 billion neurons. 
# * The **soma** (neuron body) holds the architecture for cell function and energy processing. 
# * **Dendrites** receive information from other neurons and transfer it towards the soma. 
# * **Axons** send information from the soma towards other dendrites/soma. 
# * Dendrites and axons are connected by a **synapse**. 
# 
# ### Information transfer in the human brain
# 
# 1. An outbound neuron produces an electrical signal called a **spike** that travel's to the synapse, where chemicals called **neurotransmitters**. 
# 2. Receptors on the inbound neuron receive the neurotransmitter, to generate another electrical signal to send the original signal to the soma. 
# 3. Whether or not neurons are fired in simultaneously or in success depend on the strength/amount of the spikes. If a certain threshold is crossed, the next neuron will be activated. 
# 
# ![neuron](img/neuron.png)
# 
# [Wikipedia](https://en.wikipedia.org/wiki/Neuron)

# ## Are deep neural networks really like the human brain? 
# 
# Deep artificial networks were originally modelled on the human brain, and many argue for and against their likeness. See for yourself by reading the below posts!
# 
# * [Neural Networks](https://medium.com/nerd-for-tech/neural-networks-68531432fb5)
# * [Do neural networks really work like neurons?](https://medium.com/swlh/do-neural-networks-really-work-like-neurons-667859dbfb4f)
# * [Neural Networks Do Not Work Like Human Brains – Let’s Debunk The Myth](https://analyticsindiamag.com/neural-networks-not-work-like-human-brains-lets-debunk-myth/)
# * [Artificial neural networks are more similar to the brain than they get credit for](https://bdtechtalks.com/2020/06/22/direct-fit-artificial-neural-networks/)
# * [Here’s Why We May Need to Rethink Artificial Neural Networks](https://towardsdatascience.com/heres-why-we-may-need-to-rethink-artificial-neural-networks-c7492f51b7bc)
# * [Artificial Neural Nets Finally Yield Clues to How Brains Learn](https://www.quantamagazine.org/artificial-neural-nets-finally-yield-clues-to-how-brains-learn-20210218/)

# ## Why deep learning?
# 
# Deep learning is ideal for all data types, but especially text, image, video, and sound because deep representative networks store these data as large matrices. Also, error is recycled (backpropagated) to update the model weights and make better predictions during the next epoch. . 
# 
# To understand deep networks, let's start with a toy example of a single feed forward neural network - a perceptron.
# 
# Read Goodfellow et al's Deep Learning Book to learn more: https://www.deeplearningbook.org/

# In[24]:


import pandas as pd

# generate toy dataset
example = {'x1': [1, 0, 1, 1, 0], 
           'x2': [1, 1, 1, 1, 0], 
           'xm': [1, 0, 1, 1, 0],
           'output': ['yes', 'no', 'yes', 'yes', 'no']
           }
example_df = pd.DataFrame(data = example)
example_df


# ![perceptron](img/perceptron.png)
# 
# Perceptron figure modified from [Sebastian Raschka's Single-Layer Neural Networks and Gradient Descent](https://sebastianraschka.com/Articles/2015_singlelayer_neurons.html)

# Perceptron key terms: 
# * **Layer:** the neural architecture or the network typology of a deep learning model, usually divided into variations of input, hidden, preprocessing, encoder/decoder, and output. 
# * **Inputs/Nodes:** features/covariates/predictors/independent variables (the columns of 1's and 0s from `example_df` above), but they could be words from a text or pixels from an image. 
# * **Weights:** the learnable parameters of a model that connect the input layer to the output via the net input (summation) and activation functions. Weights are often randomly initialized.
# * **Bias term:** A placeholder "1" assures that we do not receive only 0 predictions of our features are zero or close to 0. 
# * **Net input function:** computes the weighted sum of the input layer. 
# * **Activation function:** determines if a neuron should be fired or not. In binary classification for example, this defines a threshold (0.5 for example) for determining if a 1 or 0 should be predicted. 
# * **Output:** a node that contains the y prediction.
# * **Error:** how far off an output prediction was. The weights are updated by adjusting the learning rate based on the error to reduce it for the next epoch. 
# * **Epoch:** full pass of the training data. 
# * **Backpropagation:** 
# * **Hyperparameters:** our definition of the neural architecture, including but not limited to: number of hidden units, weight initialization, learning rate, batch size, dropout, etc.

# ## What makes a network "deep"?
# 
# A "deep" network is just network with multiple/many hidden layers for handling potential nonlinear transformations.
# 
# * Fully connected layer: a layer where all nodes are connected to every node in the next layer (as indicated by the purple arrows 

# ![deep](img/deep.png)
# 
# Example of "deep" network with two hidden layers modified from [DevSkrol's Artificial Neural Network Explained with an Regression Example](https://devskrol.com/2020/11/22/388/)
# 
# >NOTE: Bias term not shown for some reason?

# ## Classify images of cats and dogs
# 
# Let's go through François Chollet's "Image classification from scratch" [tutorial](https://keras.io/examples/vision/image_classification_from_scratch/) to examine this architecture and predict images of cats versus dogs. 
# 
# [Click here to open the Colab notebook](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/vision/ipynb/image_classification_from_scratch.ipynb)
# 
# > NOTE: One pain point for working with your own images is importing them correctly. Schedule a consultation with SSDS if you need help! https://ssds.stanford.edu/
# 
# You should also check out his deep learning book Deep Learning with Python (R version also available): https://www.manning.com/books/deep-learning-with-python-second-edition

# ![dogcat](img/dogcat.jpg)
