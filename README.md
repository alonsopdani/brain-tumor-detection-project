# **Brain tumor detection project**

This project consists on a program which receives a brain Magnetic Resonance Image (MRI) and gives a diagnosis that can be the presence or not of a tumor in that brain.

## **Goals**

The result when we give an image to the program is a probability that the brain contains a tumor, so we could prioritize the patients which magnetic resonance have higher probabilities to have one, and treat them first.

Another goal could be to transfer the duty of seeing these images from the doctors to the machine, which eventually could have more capability of detection, as it have learnt by watching a large quantity of images knowing their real diagnosis. This would be a clear example of cooperation between humans and machines.
    
## **The dataset**

The dataset used in the project is a bunch of images with and without tumors, from which we know the real diagnosis. You can find it here:

https://www.kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection

## **Technical procedure**

First I had to do some image processing, and then pass 80% of these images to a neural network, make it learn and be capable of making an accurate diagnosis of a new image.

The other 20% of the images are used to test the model. We will compare their real diagnosis to the one that the model gives, to see how it performs.

## **Steps**

Eventually, images are data, since they contain pixels which also contain information about their color. We can manipulate this information to achieve our goals which, in this case, are making the images better for the model to learn. The objective here is to make all the images as similar as possible, so that the actual discriminating feature is the presence or not of a tumor, and not the difference of shape, size, color… of the images.

Original images

![0-raw](https://github.com/alonsopdani/brain-tumor-detection-project/blob/master/images/0-raw.png)

Squaring

![1-squared](https://github.com/alonsopdani/brain-tumor-detection-project/blob/master/images/1-squared.png)

Resizing

![2-resized](https://github.com/alonsopdani/brain-tumor-detection-project/blob/master/images/2-resized.png)

Grayscale

![3-grayscale](https://github.com/alonsopdani/brain-tumor-detection-project/blob/master/images/3-grayscale.png)

Median filter

![4-filtered](https://github.com/alonsopdani/brain-tumor-detection-project/blob/master/images/4-filtered.png)

Black and white

![5-b&w](https://github.com/alonsopdani/brain-tumor-detection-project/blob/master/images/5-b&w.png)

At this point, each image is a 2-dimension Numpy array with 128 x 128 shape. In order to put all the images together in a Pandas dataframe, I had to flatten each array so the dimension is 1 and its shape is 16.384.

Once we have these dataframe, we can train the model with 80% of the images, and test it with the other 20%. The result of testing is the following Confusion Matrix:

![30-2-6-13](https://github.com/alonsopdani/brain-tumor-detection-project/blob/master/images/30-2-6-13.png)

We can see the actual diagnosis on y-axis and the predicted diagnosis on the x-axis. We can see that the model has learnt better to detect the tumors than the not tumors. It could be because the dataset contained more tumor images than not tumor images.

## **Next steps**

Related to the previous section, we could carry out some oversampling techniques to make our dataset have the same information about tumors and not tumors.

We could also use more advanced image processing techniques to make the images even better for the model to train.

Finally, I would like to research about more complex neural networks than the one I used (the one proposed by Keras), to try to make our accuracy better.

