#**Behavioral Cloning** 

**Behavioral Cloning Project**

The goals / steps of this project are the following:

* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./report/fell_left_example.jpg "Fell of the track to left side"
[image2]: ./report/fell_right_example.jpg "Fell of the track to right side"
[image3]: ./report/model_visualization.png "Model Visualization"
[image4]: ./report/center.jpg "Center Image"
[image5]: ./report/fell_left.jpg "Fell to left side"
[image6]: ./report/fell_left_recovery_center.jpg "Recovery from left side"
[image7]: ./report/fell_right.jpg "Fell to right side"
[image8]: ./report/fell_right_recovery_center.jpg "Recovery from right side"
[image9]: ./report/curve1.jpg "Curve Image 1"
[image10]: ./report/curve2.jpg "Curve Image 2"
[image11]: ./report/curve3.jpg "Curve Image 3"
[image12]: ./report/curve4.jpg "Curve Image 4"
[image13]: ./report/curve5.jpg "Curve Image 5"
[image14]: ./report/curve6.jpg "Curve Image 6"
[image15]: ./report/center_2017_05_30_23_35_56_808.jpg "Center Camera"
[image16]: ./report/left_2017_05_30_23_35_56_808.jpg "Left Camera"
[image17]: ./report/right_2017_05_30_23_35_56_808.jpg "Right Camera"
[image18]: ./report/OriginalImage.png "Original Image"
[image19]: ./report/FlipImage.png "Flip Image"
[image20]: ./report/training_visualization_data_sample.png "Sample from Udacity"
[image21]: ./report/training_visualization_data_clockwise.png "Center Driving Clockwise"
[image22]: ./report/training_visualization_data_anti_clockwise.png "Center Driving Anti Clockwise"
[image23]: ./report/training_visualization_data_recovery.png "Vehicle Recovering"
[image24]: ./report/training_visualization_data_curve_clockwise.png "Smoothly Around Curves Clockwise"
[image25]: ./report/training_visualization_data_curve_anti_clockwise.png "Smoothly Around Curves Anti Clockwise"
[image26]: ./report/training_visualization_all.png "All Samples"
[image27]: ./report/OriginalImage.png "BGR Image"
[image28]: ./report/HSVImage.png "HSV Image"
[image29]: ./report/CroppedImage.png "Cropped Image"
[image30]: ./report/ScaledImage.png "Scaled Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:

| File name                 | Content                                               |
|:-------------------------:|:-----------------------------------------------------:|
| model.py                  | containing the script to create and train the model   |
| drive.py                  | for driving the car in autonomous mode                |
| video.py                  | creates a video based on images                       |
| model.h5                  | containing a trained convolution neural network       |
| video_output/video.mp4    | recording of my vehicle driving autonomously          |
| writeup_report.md         | summarizing the results                               |

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

To output images for video recording, using the below script
```sh
python drive.py model.h5 img_output/
```

To make the video recording, using the below script
```sh
python video.py img_output/ video_output/ --fps 48
```

####3. Submission code is usable and readable

The **model.py** file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a convolution neural network with:

* 5x5 filter sizes and depths between 24, 36 and 48 (model.py lines 180, 185 and 190)
* 3x3 filter sizes and depths 64 (model.py lines 195 and 200)

The model includes ELU layers to introduce nonlinearity with all layers, and the data is normalized in the model using a Keras lambda layer (code line 172).

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 182, 187, 192, 197, 202, 244, 249 and 254).

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 281 and 282).

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 261).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of:

* sample data training from Udacity
* center lane driving clockwise/anti-clockwise
* recovering from the left and right sides of the road
* driving smoothly around curves clockwise/anti-clockwise

For details about how I created the training data, see the next section.

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to control the Car to keep lane in Simulator.

My first step was to use a convolution neural network model similar to the NVIDIA's network. I thought this model might be appropriate because it's a large network with more than 250 thousand parameters.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting.

To combat the overfitting, I modified the model by adding ELU and Dropout with each convolution and full connected layer.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track:

![alt text][image1]
![alt text][image2]

To improve the driving behavior in these cases, I try to collect more datas for these case.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 170-261) consisted of a convolution neural network with the following layers and layer sizes.

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image3]

####3. Creation of the Training Set & Training Process

####3.1 Center lane driving

To capture good driving behavior, I first recorded three laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image4]

####3.2 Vehicle Recovering

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn how to return center lane in fell of the track cases. These images show what a recovery looks like starting from the center:

* Recovery from the left side:

![alt text][image5]
![alt text][image6]

*  Recovery from the right side:

![alt text][image7]
![alt text][image8]

####3.3 Driving Smoothly Around Curves

After that, I recorded the vehicle keep lane smoothly around curves:

![alt text][image9]
![alt text][image10]
![alt text][image11]
![alt text][image12]
![alt text][image13]
![alt text][image14]

####3.4  Using Mutiple Cameras

To help my model how to steer if the car drifts off to the left or the right easier, I use 3 camera's data to input to my model and create adjusted steering measurements for the side camera images with correction is 0.24, adding for the left side and minute for the right side.

* Center Camera:

![alt text][image15]

* Left Camera:

![alt text][image16]

* Right Camera:

![alt text][image17]

####3.5 Data Augmentation

To augment the data set, I also flipped images and angles. This would be better for the network training when have double training data. For example, here is an image that has then been flipped:

* Original

![alt text][image18]

* Flipped

![alt text][image19]

After the collection process, I had these datas:

* Data sample from Udacity: 8036 samples

![alt text][image20]

* Data Center Lane Driving: 7869 clockwise + 11028 anti-clockwise samples

![alt text][image21]
![alt text][image22]

* Data Vehicle Recovering: 3357 samples

![alt text][image23]

* Data Driving Smoothly Around Curves: 4296 clockwise + 2443 anti-clockwise samples

![alt text][image24]
![alt text][image25]

Total is 37056 x 3 x 2 = 222336 samples

* 3 by Camera
* 2 by Flipped

![alt text][image26]

####3.6 Preprocessed Data

I then preprocessed this data by these step:

* Convert from BGR to HSV (160x320x3)

![alt text][image27]
![alt text][image28]

* Crop image to remove redundant region in the image: the sky, trees, mountain ... and the front of the Car (78x320x3)

![alt text][image29]

* Scale image to compatible with NVIDIA's network model (66x200x3)

![alt text][image30]

####3.7 Create Training Data Set and Validation Data Set

I finally randomly shuffled the data set and put 30% of the data into a validation set.

####3.8 Training Process

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 and batch size was 132 (=22\*(3\*2)). I used an adam optimizer so that manually training the learning rate wasn't necessary.

## Note
I have just used NVIDIA's architecture model and a little bit data agumentation techniques for this project. I think it is enough at this time for track 1. When I have free time, I will try with track 2 and I also will try another architecture (LeNet, VGG, GoogLeNet, etc) for better performance (training time, accuracy).

*Another ideas*: I will try to modify Simulator to improve user's ability to driving Car by keyboard, etc. I also want to get 3 camera data (center/left/right) from Car in Autonomous mode instead just only 1 camera data (center). Like some games, if user can see their map they can driving better. And a latest idea, I want to add feature switching camera from 3rd view of point to 1st view of point. Maybe some one will professional in this.

## Reference
* https://slack-files.com/T2HQV035L-F50B85JSX-7d8737aeeb
* https://discussions.udacity.com/t/behavioral-cloning-non-spoiler-hints/233194
* https://medium.com/@ValipourMojtaba/my-approach-for-project-3-2545578a9319
* https://becominghuman.ai/end-to-end-self-driving-car-using-behavioral-cloning-5cad2610522c