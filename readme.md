MAARS: Mobile AR Recommender System
----------------------------

Contains the datasets, models structure files, and training outcomes for the paper "Personalised Recommendations for Daily Automations in a Mobile Augmented Reality Application" submitted to the EICS 2025 conference.

The code is organized into three main directories containing the part of the experiment performed with Tensorflow-Keras (tested with Python 3.8.9) and with Scikit-Learn (tested with Python 3.12.4). Each has a requirements.txt file with its own dependencies. The "datasets" directory contains the "multiple rule elements" dataset in different versions, and the script to recreate the "IFTTT silding window" dataset starting from the original one from Mi and colleagues. 

F1 scores are not present in the outcomes of the Keras-Tensorflow evironment because they were calculated using the precision and recall obtained from the validation data. 

To setup an environment to reproduce the training, enter the Keras or the Scikit subdirectories, create a virtual environment and activate it, then run pip install -r requirements.txt. 
