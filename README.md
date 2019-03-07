# Face detection with logistic regression using Python

This repository consists of implementation of logistic regression classifier with testing module and visualization.
Model is trained on dataset of images of people to be able to detect faces.
Image included for testing was aquired from pexels.com

## Prerequisites

This project uses numpy for computation, pickle for data managment and matplotlib for visualization of results.

List of all packages used:
* numpy
* functools
* warnings
* matplotlib
* pickle
* time
* sys

## Content.py

File consisting of needed methods:

* sigmoid
* logistic cost function
* gradient descent
* stochastic gradient descent
* regularized logistic cost function
* prediction function
* f-measure
* model selection - based on f-measure

Inputs and outputs of all methods are described in the comments in the code.

## Utils.py
 
 File consisting of HOG feature extraction function

## Test.py

File consisting of unittests for methods in content.py. Expected result for all tests is "ok".

## Main.py

From main the tests are run and traing and prediction are performed.

Training is performed for following parameters:

```
eta = 0.1
theta = 0.65
lambdas = [0, 0.00001, 0.0001, 0.001, 0.01, 0.1]
EPOCHS = 100
MINIBATCH_SIZE = 50
```
Model is optimized with regards to lambda parameter.
Results are presented on the graphs using matplotlib.
Face detection is performed on sample image attached to the project.

## Acknowledgments

This project was prepared as a part of Machine Learning course in Wrocław University of Technology.


