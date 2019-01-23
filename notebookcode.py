
# coding: utf-8

# $\newcommand{\xv}{\mathbf{x}}
# \newcommand{\Xv}{\mathbf{X}}
# \newcommand{\yv}{\mathbf{y}}
# \newcommand{\zv}{\mathbf{z}}
# \newcommand{\av}{\mathbf{a}}
# \newcommand{\Wv}{\mathbf{W}}
# \newcommand{\wv}{\mathbf{w}}
# \newcommand{\tv}{\mathbf{t}}
# \newcommand{\Tv}{\mathbf{T}}
# \newcommand{\muv}{\boldsymbol{\mu}}
# \newcommand{\sigmav}{\boldsymbol{\sigma}}
# \newcommand{\phiv}{\boldsymbol{\phi}}
# \newcommand{\Phiv}{\boldsymbol{\Phi}}
# \newcommand{\Sigmav}{\boldsymbol{\Sigma}}
# \newcommand{\Lambdav}{\boldsymbol{\Lambda}}
# \newcommand{\half}{\frac{1}{2}}
# \newcommand{\argmax}[1]{\underset{#1}{\operatorname{argmax}}}
# \newcommand{\argmin}[1]{\underset{#1}{\operatorname{argmin}}}$

# # Assignment 5: Neural Networks

# *Type your name here and rewrite all of the following sections.  Add more sections to present your code, results, and discussions.*

# ## Overview

# You will write and apply code that trains neural networks of various numbers of hidden layers and units in each hidden layer and returns results as specified below.  You will do this once for a regression problem and once for a classification problem. 

# ## Required Code

# Download [nn2.tar](http://www.cs.colostate.edu/~anderson/cs440/notebooks/nn2.tar) that was used in lecture and extract its contents, which are
# 
# * `neuralnetworks.py`
# * `scaledconjugategradient.py`
# * `mlutils.py`

# Write the following functions that train and evaluate neural network models.
# 
# * `results = trainNNs(X, T, trainFraction, hiddenLayerStructures, numberRepetitions, numberIterations, classify)`
# 
# The arguments to `trainNNs` are
# 
# * `X` is a matrix of input data of shape `nSamples x nFeatures`
# * `T` is a matrix of target data of shape `nSamples x nOutputs`
# * `trainFraction` is fraction of samples to use as training data. 1-`trainFraction` is number of samples for testing data
# * `hiddenLayerStructures` is list of network architectures. For example, to test two networks, one with one hidden layer of 20 units, and one with 3 hidden layers with 5, 10, and 20 units in each layer, this argument would be `[[20], [5, 10, 20]]`.
# * `numberRepetitions` is number of times to train a neural network.  Calculate training and testing average performance (two separate averages) of this many training runs.
# * `numberIterations` is the number of iterations to run the scaled conjugate gradient algorithm when a neural network is trained.
# * `classify` is set to `True` if you are doing a classification problem, in which case `T` must be a single column of target class integers.
# 
# This function returns `results` which is list with one element for each network structure tested.  Each element is a list containing 
# 
# * the hidden layer structure (as a list),
# * a list of training data performance for each repetition, 
# * a list of testing data performance for each repetition, and
# * the number of seconds it took to run this many repetitions for this network structure.
# 
# This function should follow these steps:
# 
#   * For each network structure given in `hiddenLayerStructures`
#     * For numberRepetitions
#       * Use `ml.partition` to randomly partition X and T into training and testing sets.
#       * Create a neural network of the given structure
#       * Train it for numberIterations
#       * Use the trained network to produce outputs for the training and for the testing sets
#       * If classifying, calculate the fraction of samples incorrectly classified for training and testing sets.
#        Otherwise, calculate the RMSE of training and testing sets.
#       * Add the training and testing performance to a collection (such as a list) for this network structure
#     * Add to a collection of all results the hidden layer structure, lists of training performance and testing performance, and seconds taken to do these repetitions.
#   * return the collection of all results

# Also write the following two functions. `summarize(results)` returns a list of lists like `results` but with the list of training performances replaced by their mean and the list of testing performances replaced by their mean.   
# `bestNetwork(summary)` takes the output of `summarize(results)` and returns the best element of `results`, determined by the element that has the smallest test performance.
# 
# * `summary = summarize(results)` where `results` is returned by `trainNNs` and `summary` is like `results` with the training and testing performance lists replaced by their means
# * `best = bestNetwork(summary)` where `summary` is returned by `summarize` and `best` is the best element of `summary`

# In[53]:

# Replace this cell with several cells defining the above functions.


# ## Examples
# 

# In[54]:

import neuralnetworks as nn
import numpy as np
import math
import random
import time
import sys
import copy
from statistics import mean
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# I have a few helper methods that help my trainNNs functions. Cases for classifier and regular network are handeled seperately.

# In[55]:

def findRMSE(targets, tested):
    #print(tested)
    result = 0;
    
    
    #if a.shape[1] == 1:
    #    for x in range(0,len(targets)):
    #       result +=  (targets[x]-tested[x])**2 
    cols = len(targets[0])
    
    for c in range(0,cols):
        for x in range(0,len(targets)):
            result +=  (targets[x][c]-tested[x][c])**2 
    
    
    result = result / len(targets)
    result = math.sqrt(result) 
    return result
    


# In[56]:

def findPercentIncorrect(targets, tested):
    numIncorrect = 0;
    for x in range(0,len(targets)):
        if round( float(targets[x])) != round( float(tested[x])) : 
            #print(str(targets[x]) + " " + str(tested[x]) + "\n")
            numIncorrect += 1
    
    result = float(numIncorrect)/float(len(targets))
    
    return result
    


# In[57]:

def splitIntoSets(X,T,b):
    
    
    
    dataSize = len(X)
    split = dataSize*b
    split = int(split)
    testX = X[split:]
    trainX = X[:split]
    testT = T[split:]
    trainT = T[:split]
    
    
    
    
    return trainX, testX, trainT, testT
    


# In[107]:

def trainONE_NN(X, T, trainFraction, hiddenLayerStructure, numberRepetitions, numberIterations, classify):
    
    trainingPerformances = []
    testingPerformances = []
    
    start_time = time.time()
    
    trainX, testX, trainT, testT = splitIntoSets(X,T,trainFraction)
    
    for k in range(0,numberRepetitions):
        #X,T = shuffle(X,T)
        nnet = 'hi'
        """
        trainX = np.asanyarray(trainX)
        trainT = np.asanyarray(trainT)
        testX = np.asanyarray(testX)
        testT = np.asanyarray(testT)
        """
        if not classify:
            nnet = nn.NeuralNetwork(trainX.shape[1], hiddenLayerStructure, trainT.shape[1])
            nnet.train(trainX, trainT, numberIterations)
        else:
            nnet = nn.NeuralNetworkClassifier(trainX.shape[1], hiddenLayerStructure, len(np.unique(trainT)))
            nnet.train(trainX, trainT, numberIterations)
        #rint(trainX.shape)
        #print(trainT.shape)
        
        
        trainRes = nnet.use(trainX)
        testRes = nnet.use(testX)
        
        if classify: 
            trainingPerformances.append(findPercentIncorrect(trainT,trainRes))
            testingPerformances.append(findPercentIncorrect(testT,testRes))
        else: 
            trainingPerformances.append(findRMSE(trainT,trainRes))
            testingPerformances.append(findRMSE(testT,testRes))
    end_time = time.time()
    
    exec_time = float(end_time - start_time)

    result = [hiddenLayerStructure, trainingPerformances, testingPerformances, exec_time]
    
    return result
    


# In[108]:

def trainNNs(X, T, trainFraction, hiddenLayerStructures, numberRepetitions, numberIterations, classify=False):
    results = []
    
    for hl in hiddenLayerStructures:
        results.append(trainONE_NN(X, T, trainFraction, hl, numberRepetitions, numberIterations, classify))
        
    return results


# In[109]:

def summarize(data):
    results = copy.deepcopy(data)
    for x in range(0,len(results)):
        results[x][1] = mean(np.asarray(results[x][1]))
        results[x][2] = mean(np.asarray(results[x][2]))
    return results


# In[110]:

def bestNetwork(summary):
    result = summary[0]
    for x in range(0,len(summary)):
        if result[2] > summary[x][2]: result = summary[x]
    return result
    
    


# In[111]:

X = np.arange(10).reshape((-1,1))
T = X + 1 + np.random.uniform(-1, 1, ((10,1)))
findRMSE(X,T)
trX, teX, trT, teT = splitIntoSets(X, T, .8)
#trX, teX, trT, teT
x = findPercentIncorrect(X,T)
x


# In[112]:

plt.plot(X, T, 'o-');


# In[113]:

nnet = nn.NeuralNetwork(X.shape[1], 2, T.shape[1])
nnet.train(X, T, 100)
nnet.getErrorTrace()


# In[114]:

nnet = nn.NeuralNetwork(X.shape[1], [50, 50, 50], T.shape[1])
nnet.train(X, T, 1000)
nnet.getErrorTrace()
y = nnet.use(X)
y


# In[115]:

T


# In[117]:

results = trainONE_NN(X, T, 0.8, [10,10,10], 5, 100, classify=True)
results


# In[118]:

results = trainNNs(X, T, 0.8, [2, 10, [10, 10]], 5, 100, classify=False)
results


# In[38]:

results = trainNNs(X, T, 0.8, [0, 1, 2, 10, [10, 10], [5, 5, 5, 5], [2]*5], 50, 100, classify=False)


# In[39]:

suma = summarize(results)
suma


# In[40]:

best = bestNetwork(summarize(results))
print(best)
print('Hidden Layers {} Average RMSE Training {:.2f} Testing {:.2f} Took {:.2f} seconds'.format(*best))


# Hummm...neural nets with no hidden layers did best on this simple data set.  Why?  Remember what "best" means.

# Linear regression

# In[41]:

X = np.random.uniform(-1, 1, (100, 3))
T = np.hstack(((X**2 - 0.2*X**3).sum(axis=1,keepdims=True),
               (np.sin(X)).sum(axis=1,keepdims=True)))
result = trainNNs(X, T, 0.7, [0, 5, 10, [20, 20]], 10, 100, False)


# In[42]:

result


# 

# In[ ]:




# ## Data for Regression Experiment
# 
# From the UCI Machine Learning Repository, download the [Appliances energy prediction](http://archive.ics.uci.edu/ml/datasets/Appliances+energy+prediction) data.  You can do this by visiting the Data Folder for this data set, or just do this:
# 
#      !wget http://archive.ics.uci.edu/ml/machine-learning-databases/00374/energydata_complete.csv
# 
# 

# Read this data into python.  One suggestion is to use the `loadtxt` function in the `numpy` package.  You may ignore the first column of each row which contains a data and time.  Also ignore the last two columns of random variables.  We will not use that in our modeling of this data.  You will also have to deal with the double quotes that surround every value in every field.  Read the first line of this file to get the names of the features.
# 
# Once you have read this in correctly, you should see values like this:

# In[24]:

get_ipython().system('wget http://archive.ics.uci.edu/ml/machine-learning-databases/00374/energydata_complete.csv')
    


# In[25]:

import csv
T = [ [], []]
X = []

for a in range(0,26):
    X.append([])

    
with open('energydata_complete.csv') as csvfile:
    reader = csv.reader(csvfile)
    
    
    names = reader.__next__()
    #print(reader.__next__())
    for row in reader:
        T[0].append(float(row[1]))
        T[1].append(float(row[2]))
    
        for y in range(3,len(row)):
            X[y-3].append(float(row[y]))
    #print(reader.__next__())
    #print(reader.__next__())
print((T))


# In[26]:

names


# I already formatted the data by target and input when I read it in, so skip the two statements that were written here to look at the data, replace them with the inputs X

# In[27]:

print(T)


# In[28]:


X[0] = np.asanyarray(X[0])
X[1] = np.asanyarray(X[1])

for j in range(0,len(T)):
    T[j] = np.asanyarray(T[j])

X = np.asanyarray(X)
T = np.asanyarray(T)
X


# In[29]:

print(X.shape)
T.shape


# In[30]:

print(X[:,:])
T[:2,:]


# Use the first two columns, labelled `Appliances` and `lights` as the target variables, and the remaining 24 columns as the input features.  So

# In[31]:

Xenergy, Tenergy = X.transpose(),T.transpose()


# In[32]:

Xenergy.shape, Tenergy.shape


# Train several neural networks on all of this data for 100 iterations.  Plot the error trace (nnet.getErrorTrace()) to help you decide now many iterations might be needed.  100 may not be enough.  If for your larger networks the error is still decreasing after 100 iterations you should train all nets for more than 100 iterations.
# 
# Now use your `trainNNs`, `summarize`, and `bestNetwork` functions on this data to investigate various network sizes.

# In[33]:

import matplotlib.pyplot as plt


# In[34]:

nnet = nn.NeuralNetwork(Xenergy.shape[1], 0, Tenergy.shape[1])
nnet.train(Xenergy, Tenergy, 500)
errorTrace = nnet.getErrorTrace()
plt.plot(errorTrace)
plt.show()


# In[35]:

nnet = nn.NeuralNetwork(Xenergy.shape[1], 10, Tenergy.shape[1])
nnet.train(Xenergy, Tenergy, 500)
errorTrace = nnet.getErrorTrace()
plt.plot(errorTrace)
plt.show()


# In[36]:

nnet = nn.NeuralNetwork(Xenergy.shape[1], [10,10], Tenergy.shape[1])
nnet.train(Xenergy, Tenergy, 500)
errorTrace = nnet.getErrorTrace()
plt.plot(errorTrace)
plt.show()


# In[37]:

nnet = nn.NeuralNetwork(Xenergy.shape[1], [25,25], Tenergy.shape[1])
nnet.train(Xenergy, Tenergy, 500)
errorTrace = nnet.getErrorTrace()
plt.plot(errorTrace)
plt.show()


# In[38]:

nnet = nn.NeuralNetwork(Xenergy.shape[1], [10,10,10], Tenergy.shape[1])
nnet.train(Xenergy, Tenergy, 500)
errorTrace = nnet.getErrorTrace()
plt.plot(errorTrace)
plt.show()


# The error traces of all the networks here seem to be pretty slowly decreasing around 200-300 iterations. Therefore I will test my networks with 200 iterations. Note also that some of the large networks may keep decreasing with more iterations without actually improving performance on the testing data, because they start to overfit the data after being run for too long.

# In[46]:

results = trainNNs(Xenergy, Tenergy, 0.8, [0, 5, 10, 20, [5, 5], [10, 10], [15,15], [2,2], [5,2,2], [10,5,5], [5,5,5]], 3, 200)


# In[47]:

summarize(results)


# In[48]:

bestNetwork(summarize(results))


# It seems to be that the really big networks are overfitting the training data too much and doing badly on the testing data because of that.

# In[49]:

trainX, testX, trainT, testT = splitIntoSets(Xenergy,Tenergy,.8)

nnet = nn.NeuralNetwork(trainX.shape[1], [0], trainT.shape[1])
nnet.train(trainX, trainT, 500)
res = nnet.use(testX)


# In[55]:


results = res.transpose()
TT = testT.transpose()
print(results[1][:10])
print(TT[1][:10])


plt.plot(TT[0], 'bo')
plt.show()

plt.plot(TT[0], 'bo')
plt.plot(results[0], 'ro')
plt.show()

plt.plot(TT[1], 'bo')
plt.show()

plt.plot(results[1], 'ro')
plt.plot(TT[1], 'bo')
plt.show()





# The prediction graph for the appliances energy use seem to be pretty good, it was able to match some of the spikes in the data. For the lights energy use however, the prediction was really bad. I think that is because the lights energy use only has 4 different set values, and the neural network was having trouble approximating that with a continous function. 

#   

# Test at least 10 different hidden layer structures.  Larger numbers of layers and units may do the best on training data, but not on testing data. Why?
# 
# Now train another network with your best hidden layer structure on 0.8 of the data and use the trained network on the testing data (the remaining 0.2 of the date).  As before use `ml.partition` to produce the training and testing sets.
# 
# For the testing data, plot the predicted and actual `Appliances` energy use, and the predicted and actual `lights` energy use, in two separate plots.  Discuss what you see.

# In[ ]:




# ## Data for Classification Experiment
# 
# From the UCI Machine Learning Repository, download the [Anuran Calls (MFCCs)](http://archive.ics.uci.edu/ml/datasets/Anuran+Calls+%28MFCCs%29) data.  You can do this by visiting the Data Folder for this data set, or just do this:
# 
#      !wget 'http://archive.ics.uci.edu/ml/machine-learning-databases/00406/Anuran Calls (MFCCs).zip'
#      !unzip Anuran*zip
#      
# Read the data in the file `Frogs_MFCCs.csv` into python.  This will be a little tricky. Each line of the file is a sample of audio features plus three columns that label the sample by family, genus, and species. We will try to predict the species.  The tricky part is that the species is given as text.  We need to convert this to a target class, as an integer. The `numpy` function `unique` will come in handy here.

# In[42]:

get_ipython().system("wget 'http://archive.ics.uci.edu/ml/machine-learning-databases/00406/Anuran Calls (MFCCs).zip' ")
get_ipython().system('unzip Anuran*zip')


# In[138]:

import csv
T = []
X = []



for a in range(0,22):
    X.append([])

    
with open('Frogs_MFCCs.csv') as csvfile:
    reader = csv.reader(csvfile)
    
    
    names = reader.__next__()
    #print(reader.__next__())
    for row in reader:
        #print(len(row))
        T.append(row[24])
    
        for y in range(0,22):
            #print(y)
            X[y].append(float(row[y]))
            

species = np.unique(np.asanyarray(T))

#print(species)

speConv = {}
num2spec = {}
for x in range(0,len(species)):
    speConv[species[x]] = x
    num2spec[x] = species[x]


# In[139]:

for x in range(0,len(T)):
    T[x] = speConv[T[x]]
    


# In[140]:

T = [T]
Xanuran = np.asanyarray(X)
Tanuran = np.asanyarray(T)


# In[141]:

Xanuran = np.transpose(Xanuran)
Tanuran = np.transpose(Tanuran)


# Everything above is pretty much just to get the input into the right form for the network training.

# In[142]:

Tanuran


# In[143]:

Xanuran.shape, Tanuran.shape


# In[144]:

Xanuran[:2,:]


# In[145]:

Tanuran


# In[146]:

for i in range(10):
    print('{} samples in class {}'.format(np.sum(Tanuran==i), i))


# In[147]:

results = trainNNs(Xanuran, Tanuran, 0.8, [0, 5, [5, 5]], 5, 100, classify=False)


# In[148]:

summarize(results)


# In[149]:

bestNetwork(summarize(results))


# Now do an investigation like you did for the regression data. 
# 
# Test at least 10 different hidden layer structures. Then train another network with your best hidden layer structure on 0.8 of the data and use the trained network on the testing data (the remaining 0.2 of the date). 
# 
# Plot the predicted and actual `Species` for the testing data as an integer.  Discuss what you see.

# In[151]:

nnet = nn.NeuralNetwork(Xanuran.shape[1], 0, Tanuran.shape[1])
nnet.train(Xanuran, Tanuran, 500)
errorTrace = nnet.getErrorTrace()
plt.plot(errorTrace)
plt.show()


# In[152]:

nnet = nn.NeuralNetwork(Xanuran.shape[1], 5, Tanuran.shape[1])
nnet.train(Xanuran, Tanuran, 500)
errorTrace = nnet.getErrorTrace()
plt.plot(errorTrace)
plt.show()


# In[153]:

nnet = nn.NeuralNetwork(Xanuran.shape[1], 15, Tanuran.shape[1])
nnet.train(Xanuran, Tanuran, 500)
errorTrace = nnet.getErrorTrace()
plt.plot(errorTrace)
plt.show()


# In[150]:

nnet = nn.NeuralNetwork(Xanuran.shape[1], [25,25], Tanuran.shape[1])
nnet.train(Xanuran, Tanuran, 500)
errorTrace = nnet.getErrorTrace()
plt.plot(errorTrace)
plt.show()


# Networks seem to have fallen of pretty well by 100 iterations

# In[154]:

results = trainNNs(Xanuran, Tanuran, 0.8, [0, 2, 5, 10, [2,2], [5, 5], [10,10], [15,15], [2,2,2], [5,5,5]], 5, 100, classify=False)


# In[155]:

summarize(results)


# In[156]:

bestNetwork(summarize(results))


#   

#   

# In[158]:

trainX, testX, trainT, testT = splitIntoSets(Xanuran,Tanuran,.8)


# In[160]:


nnet = nn.NeuralNetwork(trainX.shape[1], [2,2,2], trainT.shape[1])
nnet.train(trainX, trainT, 100)
res = nnet.use(testX)


# In[164]:


results = res.transpose()
TT = testT.transpose()
#print(results[1][:10])
#print(TT[1][:10])


plt.plot(TT[0], 'bo')
plt.show()

plt.plot(TT[0], 'bo')
plt.plot(results[0], 'ro')
plt.show()




# Its totally messed up. I tried using the NN classifier class, but I ran into a lot of tuple errors inside some other file. So these are the graphs for the classifier trained as if it was not a classifier.

# ## Grading
# 
# Download [A5grader.tar](http://www.cs.colostate.edu/~anderson/cs440/notebooks/A5grader.tar) and extract `A5grader.py` from it.

# In[131]:

#%run -i "A5grader.py"


# ## Check-in

# Do not include this section in your notebook.
# 
# Name your notebook ```Lastname-A5.ipynb```.  So, for me it would be ```Anderson-A5.ipynb```.  Submit the file using the ```Assignment 5``` link on [Canvas](https://colostate.instructure.com/courses/68135).

# ## Extra Credit
# 
#   2. Repeat the above regression and classification experiments with a second regression data set and a second classification data set.
#   
#   2. Since you are collecting the performance of all repetitions for each network structure, you can calculate a confidence interval about the mean, to help judge significant differences. Do this for either the regression or the classification data and plot the mean test performance with confidence intervals for each network structure tested.  Discuss the statistical significance of the differences among the means.  One website I found to help with this is the site [Correct way to obtain confidence interval with scipy](https://stackoverflow.com/questions/28242593/correct-way-to-obtain-confidence-interval-with-scipy).
#   
# 

# In[ ]:



