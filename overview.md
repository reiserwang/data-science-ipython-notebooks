# Aritificial Intelligence, Machine Learning, and Deep Learning
<p><img src= "https://blogs.nvidia.com/wp-content/uploads/2016/07/Deep_Learning_Icons_R5_PNG.jpg.png" /></p>

# Fundamentals on Machine Leanring

## Linear Regression
<p><img src="https://en.wikipedia.org/wiki/Logistic_regression#/media/File:Exam_pass_logistic_curve.jpeg"/>

[scikit - logistic regression](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)

> Sample codes from http://www.blopig.com/blog/2017/07/using-random-forests-in-python-with-scikit-learn/ 
``` python
from numpy import *
from sklearn.datasets import load_iris # import datasets

# load the dataset: iris
# Sklearn comes with several nicely formatted real-world toy data sets which we can use to experiment with the tools at our disposal. We’ll be using the venerable iris dataset for classification and the Boston housing set for regression. 

iris = load_iris()
samples = iris.data
 #print samples
target = iris.target

 # import the LogisticRegression
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression() #all use default parameters
classifier.fit(samples, target) 

x = classifier.predict([5, 3, 5, 2.5]) 

print x
```

## Random Forest

[scikit - random forest classifier](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
``` Python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import datasets
iris = datasets.load_iris()

df = pd.DataFrame(iris.data, columns=iris.feature_names)

# sklearn provides the iris species as integer values since this is required for classification
# here we're just adding a column with the species names to the dataframe for visualisation
df['species'] = np.array([iris.target_names[i] for i in iris.target])

sns.pairplot(df, hue='species')
```


``` Python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df[iris.feature_names], iris.target, test_size=0.5, stratify=iris.target, random_state=123456)


```
> Fit a random forest classifier to our training set. For the most part we’ll use the default settings since they’re quite robust. One exception is the out-of-bag estimate: by default an out-of-bag error estimate is not computed, so we need to tell the classifier object that we want this.

``` Python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=123456)
rf.fit(X_train, y_train)
```
> Let’s see how well our model performs when classifying our unseen test data. For a random forest classifier, the out-of-bag score computed by sklearn is an estimate of the classification accuracy we might expect to observe on new data. We’ll compare this to the actual score obtained on our test data.

``` python
from sklearn.metrics import accuracy_score
predicted = rf.predict(X_test)
accuracy = accuracy_score(y_test, predicted)
print(f'Out-of-bag score estimate: {rf.oob_score_:.3}')
print(f'Mean accuracy score: {accuracy:.3}')
```
> Out-of-bag score estimate: 0.973
Mean accuracy score: 0.933

## Gradient Boosted Machines (GBM)
[scikit - Gradient Boosting Classifier](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)


## Convolutional neuro network (CNN)
<p><img src="http://api.ning.com/files/0gGC4ZQuxjPQZ*7CfZBPKZM7mP-Zfs7mU4MeRsxVnjfhumeFIbr5M1CtJcMmdXjoWl22QlmarTJ2BgMF2ha*2N9jkqfeHUZQ/DeepConvolutionalNeuralNetworks.jpg" /></p>

* Types of Neural Networks
<p><img src="https://pbs.twimg.com/media/DNk_8dDW0AAJaWr.jpg" /></p>


* [Convolutional Neural Networks for Visual Recognition](http://cs231n.github.io/convolutional-networks/)

## RNN (Recurrent Neural Network)

# Deep Learning Frameworks

## 1. TensorFlow
* Low-level core (C++/CUDA)
* Simple Python API to define he computational graph
* High-level API's (TF-Learn, TF-Slim, Keras)
* (+)  Auto differentiation - easy multi-GPU/multi-node
* (+) Active on Github. Lot's of new APIs
> [Tensorflow Tutorials](README.md#tensor-flow-tutorials)
> [Tensorflow Playground](htttp://playground.tensorflow.org)

## 2. Theano
## 3. Keras
* (+) Easy-to-use Python library that wraps Theano and Tensorflow.
* (+) Libraries, tools and Google official support
* (-) Less flexible and less projects (as of current than caffe).
* (-) Multi-GPU not working 100%
* (-) No RBM
> [Keras tutorials](README.md#keras-tutorials)

## 4. Torch
## 5. Caffe
* Applications in machine learning, vision, speech, and multimedia.
* Good for feed-forward networks and image processing.
* Widely acceptable from research communities.
* Good Python and MATLAB interfaces
* Excellent ConvNet implementation
* (-) Not intended for applications such as text, sound, or time series data
* (-) No auto-differentiation
* Not good for RNN, mainly CNN
* (-) Cubersome for big networks (GooLeNet, ResNet)

## Conclusions
* You're a PhD student on DL --> **TensorFlow**, **Theano**, **Torch**
* You awant to use DL only to get deatures --> **Keras**, **Caffe**
* You work in industry --> **TensorFlow**, **Caffe**
* You want to give pracrice works to your students --> **Keras**, **Caffe**
* You're curuous about deep learning --> **Caffe**
* You don't even know python --> **Keras**, **Torch**

## Applications
* **Computer Vision** - distill information from images ot video, face recognition, content moderator, emotion detection, etc.



## References
* [Deep learning framework](https://project.inria.fr/deeplearning/files/2016/05/DLFrameworks.pdf), Vucky K, Stephane L., et al.
* [NVIDIA Deep Learning](https://developer.nvidia.com/deep-learning)
* [Microsoft Cognittive Toolkit](https://www.microsoft.com/en-us/cognitive-toolkit/)
* [Deep Learning on AWS](https://aws.amazon.com/tw/deep-learning/)

