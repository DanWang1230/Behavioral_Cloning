# Behavioral_Cloning

The goals/steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road

[//]: # (Image References)

[image1]: ./images_behaviorla_cloning/1.jpg "center"
[image2]: ./images_behaviorla_cloning/2.jpg "center_flip"
[image3]: ./images_behaviorla_cloning/3.jpg "left"
[image4]: ./images_behaviorla_cloning/5.jpg "right"
[image5]: ./images_behaviorla_cloning/loss_20.png "loss"

## Run the simulator on a local machine

Here are the steps to run this project on **local machine**. I have tested them on Mac and Unbuntu.

1. Download the car simulator from [here](https://github.com/udacity/self-driving-car-sim#available-game-builds-precompiled-builds-of-the-simulator) in Term 1. Both version 1 and version 2 will work.

2. If you would like to change the design of the car simulator, you can make it happen by using Unity. Here is [the tutorial](https://github.com/udacity/self-driving-car-sim#available-game-builds-precompiled-builds-of-the-simulator). I find [this blog post](https://kaigo.medium.com/how-to-install-udacitys-self-driving-car-simulator-on-ubuntu-20-04-14331806d6dd) very helpful.

3. Create a conda environment and install dependencies following instructions in [this repo](https://github.com/udacity/CarND-Term1-Starter-Kit). There might be some errors with keras versions. Just keep the version you installed in the envrionment and the version the same.

4. Using the car simulator and the drive.py file, the car can be driven autonomously around the track by executing 
```
python drive.py model.h5
```

5. In debugging, my car didn't move in autonomous control mode. A **solution** to this issue is to downgrade the python-socketio to 4.2.1. I found this solution [here](https://github.com/udacity/self-driving-car-sim/issues/131).


## Pipeline

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it includes comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 and 5x5 filter sizes and depths between 24 and 64 (model.py lines 48-52) 

The model includes RELU layers to introduce nonlinearity (code lines 48-52), and the data is normalized in the model using a Keras lambda layer (code line 46). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 55, 57, and 59). 

The model was trained and validated on the provided data set. The model was tested by running it through the simulator and ensuring that the vehicle could stay on track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 62).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, images from the left and right cameras, and flipped driving images. 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to use convolutional layers followed by fully connected layers.

My first step was to use a convolution neural network model similar to [this NVIDIA CNN](https://developer.nvidia.com/blog/deep-learning-self-driving-cars/). I thought this model might be appropriate because it is designed for self-driving cars.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that three dropout layers are added.

In the end, the vehicle can drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 45-60) consisted of a convolution neural network with the following layers and layer sizes.

```python
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25), (0,0))))
model.add(Conv2D(24, (5,5), strides=(2,2), activation='relu'))
model.add(Conv2D(36, (5,5), strides=(2,2), activation='relu'))
model.add(Conv2D(48, (5,5), strides=(2,2), activation='relu'))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Dropout(0.5))
model.add(Dense(1))
```

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I used the provided data set. Here is an example image of center lane driving:

![alt text][image1]

I then took advantage of the left and right cameras. A correction factor of 0.2 is used so that the vehicle would learn how to recover when being off the center. These images show the left, center, and right images:

![alt text][image3]
![alt text][image1]
![alt text][image4]

To augment the data sat, I also flipped images and angles, thinking that this would generalize the data set. For example, here is an image that has then been flipped:

![alt text][image1]
![alt text][image2]

After the collection process, I had 48216 data points. I then preprocessed this data by normalizing the image data to between -0.5 and 0.5 and cropping the unuseful top and bottom of the images.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or underfitting. The number of epochs was set as 20. The loss function is shown below. I used an adam optimizer so that manually training the learning rate wasn't necessary.

![alt text][image5]
