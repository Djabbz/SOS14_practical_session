
# coding: utf-8

## Starting kit for the Higgs boson machine learning challenge

# This notebook contains a starting kit for the <a href="https://www.kaggle.com/c/higgs-boson">
# Higgs boson machine learning challenge</a>. Download the training set (called <code>trainHiggsKegl.csv</code>) and the qualifying set <code>qualifyingHiggsKegl.csv</code>), then execute cells in order.

# In[58]:

import random,string,math,csv
import numpy as np
import matplotlib.pyplot as plt


### Reading an formatting training data

# In[3]:

all = list(csv.reader(open("trainHiggsKegl.csv","rb"), delimiter=','))


# Slicing off header row and weight and label columns.

# In[15]:

xs = np.array([map(float, row[:-2]) for row in all[1:]])
(numPoints,numFeatures) = xs.shape


# Perturbing features to avoid ties. It's far from optimal but makes life easier in this simple example.

# In[14]:

xs = np.add(xs, np.random.normal(0.0, 0.0001, xs.shape))


# Label selectors.

# In[18]:

sSelector = np.array([row[-1] == 's' for row in all[1:]])
bSelector = np.array([row[-1] == 'b' for row in all[1:]])


# Weights and weight sums.

# In[19]:

weights = np.array([float(row[-2]) for row in all[1:]])
sumWeights = np.sum(weights)
sumSWeights = np.sum(weights[sSelector])
sumBWeights = np.sum(weights[bSelector])


### Training and validation cuts

# We will train a classifier on the training set for minimizing weighted error, then we will maximize the AMS on the held out validation set.

# In[26]:

randomPermutation = random.sample(range(len(xs)), len(xs))
numPointsTrain = int(numPoints*0.9)
numPointsValidation = numPoints - numPointsTrain

xsTrain = xs[randomPermutation[:numPointsTrain]]
xsValidation = xs[randomPermutation[numPointsTrain:]]

sSelectorTrain = sSelector[randomPermutation[:numPointsTrain]]
bSelectorTrain = bSelector[randomPermutation[:numPointsTrain]]
sSelectorValidation = sSelector[randomPermutation[numPointsTrain:]]
bSelectorValidation = bSelector[randomPermutation[numPointsTrain:]]

weightsTrain = weights[randomPermutation[:numPointsTrain]]
weightsValidation = weights[randomPermutation[numPointsTrain:]]

sumWeightsTrain = np.sum(weightsTrain)
sumSWeightsTrain = np.sum(weightsTrain[sSelectorTrain])
sumBWeightsTrain = np.sum(weightsTrain[bSelectorTrain])


# In[27]:

xsTrainTranspose = xsTrain.transpose()


# Making signal and background weights sum to $1/2$ each to emulate uniform priors $p(s)=p(b)=1/2$.

# In[28]:

weightsBalancedTrain = np.array([0.5 * weightsTrain[i]/sumSWeightsTrain
                                 if sSelectorTrain[i]
                                 else 0.5 * weightsTrain[i]/sumBWeightsTrain\
                                 for i in range(numPointsTrain)])


### Training naive Bayes and defining the score function

# Number of bins per dimension for binned naive Bayes.

# In[29]:

numBins = 10


# <code>logPs[fI,bI]</code> will be the log probability of a data point <code>x</code> with <code>binMaxs[bI - 1] < x[fI] <= binMaxs[bI]</code> (with <code>binMaxs[-1] = -</code>$\infty$ by convention) being a signal under uniform priors $p(\text{s}) = p(\text{b}) = 1/2$.

# In[31]:

logPs = np.empty([numFeatures, numBins])
binMaxs = np.empty([numFeatures, numBins])
binIndexes = np.array(range(0, numPointsTrain+1, numPointsTrain/numBins))


# In[32]:

for fI in range(numFeatures):
    # index permutation of sorted feature column
    indexes = xsTrainTranspose[fI].argsort()

    for bI in range(numBins):
        # upper bin limits
        binMaxs[fI, bI] = xsTrainTranspose[fI, indexes[binIndexes[bI+1]-1]]
        # training indices of points in a bin
        indexesInBin = indexes[binIndexes[bI]:binIndexes[bI+1]]
        # sum of signal weights in bin
        wS = np.sum(weightsBalancedTrain[indexesInBin]
                    [sSelectorTrain[indexesInBin]])
        # sum of background weights in bin
        wB = np.sum(weightsBalancedTrain[indexesInBin]
                    [bSelectorTrain[indexesInBin]])
        # log probability of being a signal in the bin
        logPs[fI, bI] = math.log(wS/(wS+wB))


# The score function we will use to sort the test examples. For readability it is shifted so negative means likely background (under uniform prior) and positive means likely signal. <code>x</code> is an input vector.

# In[33]:

def score(x):
    logP = 0
    for fI in range(numFeatures):
        bI = 0
        # linear search for the bin index of the fIth feature
        # of the signal
        while bI < len(binMaxs[fI]) - 1 and x[fI] > binMaxs[fI, bI]:
            bI += 1
        logP += logPs[fI, bI] - math.log(0.5)
    return logP


### Optimizing the AMS on the held out validation set

# The Approximate Median Significance
# \begin{equation*}
# \text{AMS} = \sqrt{ 2 \left( (s + b + 10) \ln \left( 1 + \frac{s}{b +
#     10} \right) - s \right) }
# \end{equation*}
# <code>s</code> and <code>b</code> are the sum of signal and background weights, respectively, in the selection region.

# In[34]:

def AMS(s,b):
    assert s >= 0
    assert b >= 0
    bReg = 10.
    return math.sqrt(2 * ((s + b + bReg) * 
                          math.log(1 + s / (b + bReg)) - s))


# Computing the scores on the validation set

# In[35]:

validationScores = np.array([score(x) for x in xsValidation])


# Sorting the indices in increasing order of the scores.

# In[37]:

tIIs = validationScores.argsort()


# Weights have to be normalized to the same sum as in the full set.

# In[38]:

wFactor = 1.* numPoints / numPointsValidation


# Initializing $s$ and $b$ to the full sum of weights, we start having all point being in the selectiom region.

# In[39]:

s = np.sum(weightsValidation[sSelectorValidation])
b = np.sum(weightsValidation[bSelectorValidation])


# <code>amss</code> will contain AMSs after each point moved out of the selection region in the sorted validation set.

# In[40]:

amss = np.empty([len(tIIs)])


# <code>amsMax</code> will contain the best test AMS, and <code>threshold</code> will be the index of the first point of the selection region (that is, the <b>index of the point with the lowest score in the selection region</b>).

# In[41]:

amsMax = 0
threshold = -1


# We will do <code>len(tIIs)</code> iterations, which means that <code>amss[-1]</code> is the AMS when only the point with the highest score is selected.

# In[42]:

for tI in range(len(tIIs)):
    # don't forget to renormalize the weights to the same sum 
    # as in the complete training set
    amss[tI] = AMS(s * wFactor,b * wFactor)
    if amss[tI] > amsMax:
        amsMax = amss[tI]
        threshold = tI
    if sSelectorValidation[tIIs[tI]]:
        s -= weightsValidation[tIIs[tI]]
    else:
        b -= weightsValidation[tIIs[tI]]


# In[56]:

amsMax


# In[59]:

plt.plot(amss)


### Computing the permutation on the qualifying set

# Reading qualifying file and converting the data into float.

# In[62]:

qualifying = list(csv.reader(open("qualifyingHiggsKegl.csv", "rb"),
                                 delimiter=','))
xsQualifying = np.array([map(float, row) for row in qualifying[1:]])


# Computing the scores.

# In[51]:

qualifyingScores = np.array([score(x) for x in xsQualifying])


# Computing the permutation with the convention of starting counting at 1.

# In[52]:

qualifyingPermutation = np.add(qualifyingScores.argsort(), 1)


# Computing the threshold index, with the convention of starting counting at 1. Linear transformation according to the ratio of file sizes.

# In[61]:

qualifyingThreshold = round(len(xsQualifying)/len(xsValidation)
                            *(threshold+1))


# Saving the permutation and the threshold, to be submitted to Kaggle.

# In[63]:

np.savetxt("submission.txt",
           np.append(qualifyingPermutation, qualifyingThreshold),
           fmt='%d', delimiter='\n')

