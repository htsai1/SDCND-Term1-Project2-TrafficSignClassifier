#**Traffic Sign Recognition** 

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

[image1]: ./examples/writeup_image_visualization_of_dataset.JPG "Visualization"
[image2]: ./examples/grayscale.JPG "Grayscaling"
[image4]: ./downloaded_images/0_20.JPG "Traffic Sign 1"
[image5]: ./downloaded_images/13_yield.JPG "Traffic Sign 2"
[image6]: ./downloaded_images/14_stop.JPG "Traffic Sign 3"
[image7]: ./downloaded_images/27_pedestrian.JPG "Traffic Sign 4"
[image8]: ./downloaded_images/28_child.JPG "Traffic Sign 5"
[image9]: ./downloaded_images/4_70.JPG "Traffic Sign 6"
[image10]: ./downloaded_images/40_roundabout.JPG "Traffic Sign 7"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code implementation](https://github.com/htsai1/SDCND-Term1-Project2-TrafficSignClassifier/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.
*I used the numpy to calculate summary statistics of the traffic signs data set, below answers are outputs from the code.

* The size of training set is ?
As output from the node: Number of training examples = 34799

* The size of the validation set is ?
As output from the node: Number of validation examples = 4410

* The size of test set is ?
As output from the node: Number of testing examples = 12630

* The shape of a traffic sign image is ?
Image data shape = 34799

* The number of unique classes/labels in the data set is ?
Number of classes = 43



####2. Include an exploratory visualization of the dataset.

Gaol is to create an exploratory visualization of the data set. I used matplotlib to plot out few images from the training dataset. And used histogram to create the sample distribution list with counts of each label.

![alt text][image1]


###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

Two steps of preprocessing techniques are used and the reason of these techniques were chosen are listed below.

Step1: I decided to convert the images to grayscale (rgb2gray) because the classification doesnt necessarily need color, i.e. all three RGB, to do the traffic sign classification, sigle channel gray scale should be sufficient. This also help improve the processing time.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

Step2: I normalized the image data because to be close to orgin, between -1.0 and 0.1 because of mathematical reasons, i.e. for the better numerical stability, it is suggested in the class that to keep value you compute roughly around a mean of zero and have equal variance when you do the optimization. This make it easier for optimizer for doing searching and find a good solution.


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

With reference to the LeNet model showed in the lesson, my final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Preprocessed Grayscale image   							| 
| Convolution 5x5     	|  1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|		Activation- activate the output from previous layer										|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5     	|  1x1 stride, valid padding, outputs 10x10x6 	|
| RELU					|		Activation- activate the output from previous layer										|
| Max pooling	      	| 2x2 stride,  outputs 5x5x6 				|
| Flatten	    | flatten the Conv2 into a vector. Input= 5x5x16, Output = 400       									|
| Fully connected	Layer	| Input = 400. Output = 120        									|
| RELU			  | 	Activation- activate the output from previous layer						|
|	Droupout			|			The regularization technique for reducing overfitting using keep probability.						|
| Fully connected	Layer	| Input = 84. Output = 43        									|
 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the AdamOptimizer(which is more sophisticated than SGD)that suggested in the class. 
For the hyperparameters, I used batch size=128, epochs =20, mu =0, sigma=0.1, learning rate=0.001, dropout keep probability=0.65.

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of 0.956
* test set accuracy of 0.932

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?

I referenced LeNet as it was taught in the class. 
* What were some problems with the initial architecture?

The initial unchanged LeNet architecture didnt provide good validation accuracy. 
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

I end up had to tune several parameters (like epoch) for improving validation accuracy to be above 0.93.
Added dropout layer (tf.nn.dropout).
* Which parameters were tuned? How were they adjusted and why?

Epoch was tuned: Increased epoch from 10 to 20 improved the accuracy from ~0.93 to ~0.95.
Dropout keep probability was tuned: Initially I used 0.5 and the outcome accuracy was not as good as using 0.65.
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
Dropout layer definitely helped due to it preventing overfitting. However the keep probability also has a certain level of effect.

If a well known architecture was chosen:
* What architecture was chosen?
I referenced LeNet architecture and modified the parameters.
* Why did you believe it would be relevant to the traffic sign application?
Suggested in class that is a good starting architecture for traffic sign classification.
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 After the modification mentioned above, the validation accuracy could reach above 0.95 and test accuracy could reach above 0.93. 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

I picked 7 German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8] ![alt text][image10]
![alt text][image9]

The children crossing sign is difficult to classified due to low resolution of the image, and the graph is much complicated than numerical numbers.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (20km/h)      		| ID0: Speed limit (20km/h)   									| 
| Yield					| ID13: Yield											|
| Stop	      		| ID14: Stop					 				|
| Pedestrians			| ID27: Pedestrians     							|
| Childern Crossing			| ID11: Right-of-way at the next intersection    							|
| Roundabout mandatory			| ID40: Roundabout mandatory     							|
| Speed limit (70km/h) 		| ID25: Road work     							|

The model was able to correctly guess 5 of the 7 traffic signs, which gives an accuracy of 71%. This compares favorably to the accuracy on the test set of 93% it is lower but I believe it is because I only test 7 images so far, the accuracy will increase if I test out more new images.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0        			| ID0: Speed limit (20km/h)   	  									| 
| 1.0     				| ID13: Yield 										|
| 1.0					| ID14: Stop										|
| 0.999	      			| ID27: Pedestrians 					 				|
| 0.992				    | ID11: Right-of-way at the next intersection    							|
| 0.992				    | ID40: Roundabout mandatory							|
| 0.997				    | ID25: Road work    							|

Looking into the softmax probabilities for each prediction, I found that all correct predictions have almost 1.0 probability on the correct answer. The two wrong predictions show 0.992 & 0.997, which still too high but show the relatively low confidence level. especially on the Child Crossing sign, the right answer is on the 2nd place (0.07) among the 5 softmax probabilities. 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


