# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.png "Visualization"
[image2]: ./examples/image_normalization.png "Normalization"
[image3]: ./examples/image_CLAHE.png "CLAHE"
[image4]: ./examples/image_CLAHE_norm.png "CLAHE_Norm"
[image5]: ./examples/image_rotate.png "Rotated"
[image6]: ./examples/image_translate.png "Translated"
[image7]: ./examples/image_scaled.png "Scaled"
[image8]: ./test_images/Original_RoadAbout.jpg "Traffic Sign 1"
[image9]: ./test_images/Original_RoadWork.jpg "Traffic Sign 2"
[image10]: ./test_images/Original_Speed_Limit_50.jpg "Traffic Sign 3"
[image11]: ./test_images/Original_Stop.jpg "Traffic Sign 4"
[image12]: ./test_images/Original_Yield.jpg "Traffic Sign 5"
[image13]: ./examples/softmax.png "Softmax"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

You're reading it! and here is a link to my [project code](https://github.com/jnsagai/traffic-sign-classifier-project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Basic summary of the data set.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of the training set is 34799 samples
* The size of the validation set is 4410 samples
* The size of the test set is 12630 samples
* The shape of a traffic sign image is 32 x 32 pixels
* The number of unique classes/labels in the data set is 43

#### 2. Exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is distributed among the training, validation and test sets

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Pre-processing the image data

Most of my decision regarding the image data pre-processing was inspired by these two papers:
[Traffic Sign Recognition with Multi-Scale Convolutional Networks](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf)
[Multi-Column Deep Neural Network for Traffic Sign Classification](http://people.idsia.ch/~juergen/nn2012traffic.pdf)

As a first step, I decided to contrast limited adaptive histogram equalization (CLAHE) then normalization in order to get a more equalized pixel histogram for the samples and also get a better contrast.
Here is an example of a traffic sign image before and after CLAHE and the normalization.

![alt text][image2]

![alt text][image3]

![alt text][image4]

Then I decided to generate additional data because was clear that in the training samples distribution some classes had much more sample than another, and my DNN would eventually be overtrained for just those classes.

To add more data to the data set, I used calculated the difference (delta) between the class with the highest amount of sample and the current number of samples for each other class. Then I simply created a "delta" number of replicas of the existing image class randomly. Moreover, I decided also to randomly apply a disturbance of the new samples, where 10% of the new images would be rotated between -15 and 15 degrees, 10% of the new images would be translated between 1 and 4 pixels, and 10% of the new images would be scaled between 90% and 110%. The goal for this would be to turn my DNN more generalist and robust for the test phase.

Here is an example of an original image and the disturbing image:

![alt text][image5]

![alt text][image6]

![alt text][image7]
#### 2. Final model architecture

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, same padding, outputs 28x28x100 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x100				|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 12x12x150 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 6x6x150				|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 4x4x250 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 2x2x250			|
| Fully connected		| 1000 neurons        									|
| Dropout		| 60%       									|
| RELU					|												|
| Fully connected		| 500 neurons        									|
| Dropout		| 60%       									|
| RELU					|												|
| Fully connected		| 250 neurons        									|
| Dropout		| 60%       									|
| RELU					|												|
| Softmax				| Softwax Cross Entropy        									|

#### 3. Model training

To train the model, I used the following parameters:
* Adam Optimizer
* 10 Epochs ( Not so much to avoid overfitting )
* Batch size: 128
* Learning rate: 0.001
* Mean of distribution (mu) for initial weights: 0
* Standard deviation (sigma): 0.03

#### 4. Training approach

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation, and test sets and wherein the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well-known implementation or architecture. In this case, discuss why you think architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of 0.975 
* test set accuracy of 0.952

First of all, I decided to use the same LeNet architecture provided by the module in order to have a starting point, but the accuracy was never greater than 0.85.
Then I decided to move to some well-known architecture and most of my decision in order to bring up the new network architecture was based on two articles described before, where both of them had the greatest performance in the GTSRB ( German Traffic Sign Recognition Benchmark) competition. The final architecture is really close regarding the one defined by IDSIA team, with the difference that I didn't adopt a multi-column DNN with multiple complementary networks, I've simply adopted one single DNN. In order to get a better performance in terms of accuracy and generalization, I decided to add an additional Fully Connected layer in my DNN and also apply some dropout between the FC layers.
The pre-processing techniques were inspired by both articles as well.

### Model testing on New Images

#### 1.Five German traffic signs found on the web

Here are five German traffic signs that I found on the web:

![alt text][image8] ![alt text][image9] ![alt text][image10] 
![alt text][image11] ![alt text][image12]

The biggest difficulty was that the original images have a high resolution, then after the resizing for the network input (32x32) the image quality was really degraded, and some features that really characterize the image might be lost, as can be seen in the Speed_Limit_50, where the number "5" is not clear anymore.

#### 2. Model's predictions on the new traffic signs

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Yield      		| Yield   									| 
| Roundabout mandatory     			| Roundabout mandatory 										|
| Road work and the prediction					| Priority road										|
| Speed limit (50km/h)	      		| Speed limit (30km/h)		 				|
| Stop		| Stop      							|

The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%.
In my opinion, the high image degradation after the resizing was really a key feature for the misestimation of some signs.
Those signs in which features are well defined without so many details ( Yield, Roundabout mandatory, and Stop ) were correctly predicted. It is also important to notice that even that the "Speed limit (50km/h)" was not correctly identified, most of its features were corrected learned by the network seen that the prediction was a similar class (Speed limit (30km/h))

#### 3. Top 5 softmax probabilities for each new image

#### 3. Describe how certain the model is when predicting each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 30th cell of the Ipython notebook.

 * For the first image, the model is totally sure that this is a Yield sign (probability of 1.0).
 * For the second image, the model is also totally sure that this is a Roundabout mandatory sign (probability of 1.0).
 * For the third image, the model wrongly predicted with a probability of 0.8 that this is a Priority Road sign. However, the second guess was correct (Road work) with a probability of 0.2
 * For the fourth image, the model wrongly predicted with a probability of 0.49 that this is a Speed Limit (30km/h) sign. However, the second guess was correct (Speed Limit (50km/h)) with a probability of 0.22. It is interesting to notice that all the top 5 guesses were about Speed Limit signs, which demonstrated that the network got to learn the main features.
 * For the fifth image, the model is totally sure that this is a Stop sign (probability of 1.0).

![alt text][image13]
