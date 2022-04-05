---
title: Comparing Image Clustering/Classification performance between Supervised and Unsupervised Learning models using Dog Breed Image dataset
---

## Overview
### Background

### Problem Definition


## Data Exploration and Cleaning


## Supervised Learning Models
Methods 
describe cnns and do calculations for number of parameters

Results and Discussion 

All groups should have their dataset cleaned at this point (justification for why we didnt need cleaning)

We expect to see data pre-processing in your project such as feature selection (Forward or backward feature selection, dimensionality reduction methods such as PCA, Lasso, LDA, .. ), taking care of missing features in your dataset, ... (data augmentation to prevent overfitting, balance dataset to get even results across breeds)

We expect to see at least one supervised or unsupervised method implemented and the results need to be studied in details. For example evaluating your predictive model performance using different metrics (take a look at ML Metrics) 

## Unsupervised Learning Models
### Dealing with the data
For an unsupervised model, we need to somehow "standardize" the data. "Standardize" here refers to transforming all the images in a standard format, where all the images are of the same size so that we can compare each image with each other. The problem with many image dataset (even here with Tsinghua dataset and Stanford dataset of dog images) is that images are not of a standard size. In order to convert them into a standardized size, we will need to perform a `reshape` operation on each of the images. The operation in itself is very easy, BUT we must remember throughout our experiment that due to this operation, we will lose some of the knowledge that we can obtain from the original image - for example, if we are doing this reshape operation on an image that is of a larger size and is not of the aspect ratio of the target size, we will lose a lot of knowledge about the shape of the features of dog from the image, as the image may be distorted. Hence, when we compare our results, we must keep these transformations and the reduction in the data from our datset in our mind.

### Model 1 - The Naive Way (KMeans on unfiltered data)
The simple way to apply an unsupervised learning algorithm on a dataset is to simply perform a "KMeans on the reshaped images using sum-squared errors as the loss function". As simple as it sounded in our heads at the start, when we started on this model, we came across a lot of issues why this is the wrong way.
#### Finding if this model is feasible
In order to first analyze whether just calculating a simple sum-squared error will work or not, we tried finding the errors between images of the same breed, and then images of the different breed, and seeing whether these errors have a certain pattern. The motivation behind this exercise was to see if there are certain thresholds OR some patterns with the sum-squared loss that we can utilize in order to concretely perform KMeans on dog images between different breeds.

So, we started by calculating the errors within images of the same breed, and seeing what errors do they throw. The method was slow, but there was no other way. Here is the algorithm that we used -
1. Get all the reshaped images (224x224x3) of the same breed into a one big Numpy array.
2. Perform a `O(N^2)` for-loop for finding errors between each images.
3. Plot a boxplot of these errors in order to visualize them.

The results of these boxplots of the 130 breeds in the Stanford dataset can be seen here -

But before we could move onto capturing patterns from this data OR calculating errors between images of different breeds, we ran into some of the problems, which we explain in the next sub-section.
#### Problems
- The above algorithm gave us our first big problem - Memory Overflow. Instead of the for-loop in the 2nd step of the algorithm, we tried a faster way by using Numpy broadcasting. But the data was so huge for our code, the broadcasted array could not fit into memory. Hence, we had to utilize a for-loop instead of broadcasting.
- Because of using a for-loop, this step became excruciatingly slow. To calculate intra-breed errors of 130 breeds, the algorithm deployed on a Google Colab instance took 4.5 hours to complete. With this, we realized that performing such an analysis for inter-breed error calculation will be very time consuming.

With the above 2 problems, it was clear that there is a desperate need of feature engineering OR feature reduction that will need to be performed on the datset before we can continue on our journey to image clustering goal.
