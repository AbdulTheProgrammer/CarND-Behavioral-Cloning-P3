# **Behavioral Cloning**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road



[//]: # (Image References)

[image1]: ./img/nvidia_net.png "Model Visualization"
[image2]: ./img/hist1.png "Histogram With Preprocessing"
[image3]: ./img/preprocessed_images.png "Preprocessed Images"
[image4]: ./img/hist2.png "Histogram Without Preprocessing"
---

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolutional neural network
* writeup_report.md or writeup_report.pdf summarizing the results

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing the below command. Note that the latest version of Keras is required (2.0.6) to successfully run the model.

```sh
python drive.py model.h5
```

The Behavioural Cloning Network.ipynb file contains the code for training and saving the convolutional neural network. The file shows the pipeline I used for training and validating the model, and it contains comments and other embedded documentation to explain how the code works.

## Model Architecture

I used the Nvidia deep neural network architecture described in the End to End Learning for Self-Driving Cars Research paper (https://arxiv.org/abs/1604.07316). A picture of this architecture can be seen below:

![alt text][image1]

This model was implemented in Keras with some slight modifications for this project. These modifications are listed below.

- I removed one dense fully connected layer (the one with 1164 neurons) from the model to reduce the number of neurons. I figured that the scope of this project was smaller than what the team at Nvidia were accomplish and such an modification would prevent overfitting. This worked out well as it meant marginally faster training times and a smaller model.h5 file during testing on the simulator.
- I added dropout layers (set at 20%) after almost every layer in the model. This was again to prevent overfitting.
- I added a lambda layer to engage in on the fly normalization and resizing to 66 by 200 for any input images
- I used the adam optimizer to train the model as it proved to have a good adaptive learning rate in my past projects

The code for the entire model can be seen in the fourth cell in the Jupyter Notebook.


## Data Preprocessing

One of the largest challenges with this project was to determine how to appropriately preprocess the data before feeding it into the network.

### Exploratory Data Analysis

I first recorded three sets of training data using my mouse on the simulator. I then decided to analyze the range of values of the center angles by plotting a histogram.
#### Histogram Without Preprocessing
![alt text][image4]

Immediately I noticed that there was an issue with this dataset; there was simply too much low angle turning data from the car driving straight. To remedy this I had to get creative and engage in some preprocessing the input images.

For details about how I created the training data, see the next section.

### Data Preprocessing Pipeline

At the start, I knew the preprocessing steps were going to be a potential bottle neck in training my neural network as they ran on the CPU. Initially, I looked into using python generators to modify images on the fly to change the distribution of my data. This prevented any out of memory errors from using up too much CPU memory. However, with a little more reasearch I discovered that I could create a custom object that inherited from the Sequence class and do all my image preprocessing/augmentation using that with support from the python multiprocessing library. This allowed me to use all the CPU cores available on my AWS instance by specifiying the number of workers (4) and not having to worry about any synchronization issues. Ultimately, this resulted in faster prototyping of different preprocessing techniques.

### Data Preprocessing Techniques

Most of the preprocessing techniques were inspired by the Nvidia Research paper in conjuction with Vivek Yadav's Blog (https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9). Some of the preprocessing techniques I used for this project are:
- Random brightness variations so that the network is immune to the lighting conditions of the road.
- Random image flipping with associated steering angle reversal for more data diversity. This is an alternative to collecting more data on the track in reverse.
- Random image translations to allow for more variation in steering angles
- Randomly selecting the right or left camera images and compensating with a 0.25 offset appropriately. This allowed me to include some recovery images in the dataset for use during times when the car strayed from the center of the road. I think this technique was major factor to my model's success.
- Cropping the image to remove unnessecary background scenery allowed the model to focus soley on the road and curb markings.
- Converting the image to the HSV color space from the RGB/BGR color space

The associated functions that performed these image manipulations + the aforementioned Sequence classes can be seen in cell 2 of the Jupyter Notebook. A sample of the final preprocessed images can be seen bellow with their associated steering angle.

#### Preprocessed Images
![alt text][image3]

#### Histogram After Preprocessing

![alt text][image2]

### Solution Design Approach and Training

The overall strategy for deriving a model architecture was to incrementally build up my model. I first started off with a simple 3 layer fully connected classifier and verfied that it had learning ability by testing it on the simulator. Then I proceeded to do the same with the Nvidia Model. Then I slightly modified that model to reduce overfitting and speed up the training process (this is further described in the Model Architecture section). Then I proceeded to try out a whole variety of preprocessing techniques, which took the majority of my time.

Initially I was using my own collected data (approximately 14882 data entries) as the training set and the udacity data as the validation set. However, I found that training with the udacity data resulted in better car performance on the simulator (likely due to it being a more accurate representation of driving at the centre of the road). Hence, I switched my training and validation data sets when training my final model.

In terms of the number of epochs for training, I was initially using Keras callbacks with a patience of 3 for validation loss. However, I found that just training for around 20 epochs gave me better simulator performance.

### Final Thoughts

If I had more time I would have trained the model a little better to handle the test track. This would likely involve tuning some hyper parameters for translations and so forth. But aside from that, I'm generally satified with the end result. I learned a lot about the intricacies of training a real convolutional neural network, the powerful Keras API and the importance of data manipulation. Ideally, I would like to try other neural network architectures such as Comma.ai and possibly my own custom one in the future.