---
title: Comparing Image Clustering/Classification performance between Supervised and Unsupervised Learning models using Dog Breed Image dataset
youtubeId: aQi4CUArJYM
---
- [Proposal Report](https://unkemptArc99.github.io/cs7641-project/proposal-report)
- [Midterm Report](https://unkemptArc99.github.io/cs7641-project/midterm-report)
## Introduction and Background

Our project revolves around identifying the breeds of dogs in images. We selected a dataset that contains images of multiple dog breeds, along with their corresponding breed and if they were originally used as training data, testing data, or validation data. To classify dog breeds, we used multiple methods: firstly, simply classifying dog breeds with an input image using a convolutional neural network; secondly, using clustering algorithms to sort dogs based on extracted image features. Using a convolutional neural network had already been done well many times provided there is sufficient training data, but a main goal of ours was to create a unique algorithm in the latter part that is comparable to the supervised learning algorithm. To achieve this, we wound up actually using a neural network to obtain relevant features from the images. After this, we considered using PCA to decrease the number of features used, but given the correlation results, we went against it. Finally, we ran KMeans as our unsupervised learning algorithm to sort the dog breed images.
## Problem Definition

The purpose of this project is to investigate the effectiveness of methods we are learning in class with classifying dog breeds. In particular, we wish to compare standard supervised learning methods in their accuracy and efficiency to our own methods utilizing unsupervised learning algorithms, which rely on feature creation using the input images.

## Datasets
1. [70 Dog Breeds Image dataset from Kaggle](https://www.kaggle.com/gpiosenka/70-dog-breedsimage-data-set)
2. [Stanford Dogs Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/)

## Supervised Learning Models

## Unsupervised Learning Models
When we proposed the idea in our proposal report, we had a very small and noble goal of clustering images and comparing performance against supervised classification. But as we went along with our idea, the goal became harder and harder because of the enormity of the data involved in the clustering. We started with brief experiments of using the whole image as the input data for clustering algorithms (which did not turn out to be a good idea for obvious reasons) and then tried to use traditional feature extraction algorithms from Computer Vision (which did not have good performance metrics). However, at last we are confident that we have come up with something comparable to supervised classification models.

**Motivation -** It is very important to first understand the motivation behind such an exercise. Why is there a need to cluster images, when does classification work well? We must understand that every problem about images is not just classifying images into particular categories. Sometimes, you might just want to group similar images together in order to form a rudimentary grouping for your problem. Hence, image clustering is also an essential technique in data processing. There are traditional Computer Vision algorithms, which will provide you key descriptors and matches between images (like SIFT, SURF, BRIEF, etc.). But when the data is large, these algorithms are not as fast or memory-efficient (as one might infer from our experiments in the midterm report). Hence, a new approach is required, and here is our exercise in finding one.

### Feature Extraction
Images are the **worst** dataset elements to deal with. There's too much information: 3 color channels, information about each pixel, etc. There's too much inconsistency - different sizes, different resolutions, etc. And finally, a ton of memory is required when processing images. The worst news is that there aren't many batch processing unsupervised algorithms, as unsupervised algorithms are designed to extract information out of the data that is presented. Batch algorithms are necessarily not that great at doing that. So, we need something that can compress/convert/summarize the image into a compact, informative vector.

It is almost paradoxical that we would be comparing our results against a supervised model, and our inspiration to get features out of an image is based on a supervised algorithm. So, here goes - imagine instead of using Convolutional Neural Networks (CNNs) to classify images, we use them to give us a bunch of features! We remembered how Prof. Mahdi told off-handedly in the class on how we could simply modify the last layer of any Neural Network, converting the output to the needs of the problem. This idea struck us, and hence, instead of using a sigmoid activation function in the last layer for classification, we replaced it with an identity function to provide me the features (or neuron outputs) that form the basis of the classification decision. The selection of these neuron outputs is essential as they make up a huge chunk of the decision of classification. These features will definitely be able to form a basis for the similarity analysis that goes into unsupervised clustering algorithms.

We chose ResNet models to be our base for the feature extraction. We have trialed three different ResNet models - ResNet18, ResNet34, ResNet50. We first chose a pre-trained model of all the above three models (which is readily available from the PyTorch library). Then we went on to train our set of images on the model to update the parameters. We chose a train-validate-test split of 70%,10%,20% out of the whole Tsinghua Dog Dataset. The train and validation datasets were used to train the ResNet models. The test dataset was used for clustering algorithms, which we will discuss in a future section. The trained models were then modified to serve as feature extraction models, by modifying the last layer of each model to an identity layer. The resulting models are available here for use - https://gtvault-my.sharepoint.com/:f:/g/personal/asharma756_gatech_edu/ErVYpyAQsPhCiIGdQe6BQdwBxVrM8eJm0PFNkeSW7b751Q?e=B5CGoC

Each of our ResNet models gave us 512 features for each image. We first started thinking about whether we could reduce a few features from the given set. Principal Component Analysis (PCA) is the first algorithm that came to our mind. PCA tends to work better where variance between features are maximised. To determine whether we could do that, we plotted a correlation matrix between all our 512 features. Here is the result that we got for our models -

{% include image.html url="img/Corr_Resnet18.png" description="Correlation Matrix of Features derived from ResNet18 model" %}
{% include image.html url="img/Corr_Resnet34.png" description="Correlation Matrix of Features derived from ResNet34 model" %}
{% include image.html url="img/Corr_Resnet50.png" description="Correlation Matrix of Features derived from ResNet50 model" %}

Please feel free to open the images in a new tab, or go directly to the Github repository. The images are high-resolution, but can't be displayed well on the Github pages.

As you might observe, there is not much correlation between the features. Reducing features may lose information. Hence, we chose the decision to move forward with the untouched features.

### Applying clustering on the extracted features


## Results and Discussion

## Final Presentation Video

{% include youtubePlayer.html id=page.youtubeId %}

## References
[1] K. Lai, X. Tu and S. Yanushkevich, "Dog Identification using Soft Biometrics and Neural Networks," 2019 International Joint Conference on Neural Networks (IJCNN), 2019, pp. 1-8, doi: 10.1109/IJCNN.2019.8851971.

[2] W. Gansbeke, S. Vandenhende, S. Georgoulis, M. Proesmans, and L. Gool, "Scan: Learning to classify images without labels," ECCV, 2020, [Paper link](https://arxiv.org/pdf/2005.12320v2.pdf)

[3] N. Manohar, Y. H. Sharath Kumar and G. H. Kumar, "Supervised and unsupervised learning in animal classification," 2016 International Conference on Advances in Computing, Communications and Informatics (ICACCI), 2016, pp. 156-161, doi: 10.1109/ICACCI.2016.7732040.

[4] Hsu, Alexander. “Unsupervised Learning for Investigating Animal Behaviors?” Medium, Medium, 15 Apr. 2020, https://medium.com/@ahsu2/unsupervised-learning-for-investigating-animal-behaviors-90ab645e8098. 

[5] Huang, Jiabo, Qi Dong, Shaogang Gong, and Xiatian Zhu. "Unsupervised deep learning by neighbourhood discovery." In International Conference on Machine Learning, pp. 2849-2858. PMLR, 2019., [Paper link](https://arxiv.org/abs/1904.11567)
