# libraries to use
import pandas as pd
import numpy as np
import random
import warnings

import cv2
import glob
from PIL import Image

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

# function to show images
def show_images(img1, img2):
    plt.subplot(1,2,1)
    plt.imshow(img1)
    plt.title('Brain with tumor')
    plt.subplot(1,2,2)
    plt.imshow(img2)
    plt.title('Brain without tumor')
    plt.show()

images_to_show = (0, 0)

# import images
def import_data():    
    def read_images(path):
        return [Image.open(file) for file in glob.glob(path)]

    path_y = './brain-mri-images-for-brain-tumor-detection/yes/*'
    path_n = './brain-mri-images-for-brain-tumor-detection/no/*'

    images_y = read_images(path_y)
    images_n = read_images(path_n)
    print('images imported')
    show_images(images_y[images_to_show[0]], images_n[images_to_show[1]])
    return images_y, images_n
    
# image preparation
def preparation(images_y, images_n):
    # squaring images function:
    # desired_size: a square which side is the max between original base and height
    # creates a black image with desired size
    # pastes the original image in the 'center' of the new image
    def square_image(list_of_images):
        res = []
        for img in list_of_images:
            desired_size = (max(img.size), max(img.size))
            position = int(max(img.size)/2) - int(min(img.size)/2) 
            sq_img = Image.new("RGB", desired_size, color='black')
            if img.size[0] < img.size[1]:
                sq_img.paste(img, (0, position))
            else:
                sq_img.paste(img, (position, 0))
            res.append(sq_img)
        return res

    images_y, images_n = square_image(images_y), square_image(images_n)
    print('images squared')
    show_images(images_y[images_to_show[0]], images_n[images_to_show[1]])

    # now we want to reshape all the images to 128x128
    def resize_images(list_of_images, size=128):
        return [img.resize((size,size)) for img in list_of_images]

    images_y, images_n = resize_images(images_y), resize_images(images_n)
    print('images resized')
    show_images(images_y[images_to_show[0]], images_n[images_to_show[1]])

    # now we start to use open cv library, that works with numpy arrays instead of images
    def image_to_nparray(list_of_images):
        return [np.array(img) for img in list_of_images]

    images_y, images_n = image_to_nparray(images_y), image_to_nparray(images_n)
    print('images as np arrays')

    # function to get gray scale images
    def img_to_gray_scale(list_of_images):
        return [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in list_of_images]

    images_y, images_n = img_to_gray_scale(images_y), img_to_gray_scale(images_n)
    print('images in gray scale')
    show_images(images_y[images_to_show[0]], images_n[images_to_show[1]])

    '''
    # convert all images from gray to color
    def gray2bgr(list_of_images):
        return [cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) for img in list_of_images]

    images_y, images_n = img_to_gray_scale(images_y), img_to_gray_scale(images_n)

    # skull removal
    def skull_removal(pic):
        gray = cv2.cvtColor(pic,cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_OTSU)
        ret, markers = cv2.connectedComponents(thresh)
        #Get the area taken by each component. Ignore label 0 since this is the background.
        marker_area = [np.sum(markers==m) for m in range(np.max(markers)) if m!=0]
        
        #Get label of largest component by area
        largest_component = np.argmax(marker_area)+1 #Add 1 since we dropped zero above                        

        #Get pixels which correspond to the brain
        brain_mask = markers==largest_component
        brain_out = pic.copy()
        #In a copy of the original image, clear those pixels that don't correspond to the brain
        brain_out[brain_mask==False] = (0,0,0)
        brain_mask = np.uint8(brain_mask)
        kernel = np.ones((8,8),np.uint8)
        closing = cv2.morphologyEx(brain_mask, cv2.MORPH_CLOSE, kernel)

        brain_out = pic.copy()
        #In a copy of the original image, clear those pixels that don't correspond to the brain
        brain_out[closing==False] = (0,0,0)
        return brain_out

    # function to apply the previous one in the whole list
    def skull_removal_list(list_of_images):
        return [skull_removal(img) for img in list_of_images]

    images_y, images_n = skull_removal_list(images_y), skull_removal_list(images_n)
    '''
    
    # function to apply a filter to soften the images
    def median_filter(list_of_images):
        return [cv2.medianBlur(img,1) for img in list_of_images]

    images_y, images_n = median_filter(images_y), median_filter(images_n)
    print('median filter applied')
    show_images(images_y[images_to_show[0]], images_n[images_to_show[1]])

    '''
    # function to transform list of images to b&w
    # I get the element 1 because it returs a tuple (threshold, image)
    # meaning of the threshold: 0 is black, 255 is white, so I set that the pixels which are below 127
    # turn black, and the ones that are above 127 turn white
    def images_to_bw(list_of_images):
        return [cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1] for img in list_of_images]

    images_y, images_n = images_to_bw(images_y), images_to_bw(images_n)
    '''

    return images_y, images_n

def dataframe_preparation(images_y, images_n):
    # function to convert the images (np arrays 256x256) in 1d arrays
    # and then put them all together in a pandas dataframe
    def list_np_to_pd(list_of_images):
        return pd.DataFrame([img.flatten() for img in list_of_images])

    images_y, images_n = list_np_to_pd(images_y), list_np_to_pd(images_n)
    print('dataframe ready')

    # preparation of the dataframes for the neural network
    # input image dimensions
    img_rows, img_cols = 128, 128

    df_y = images_y.T
    df_n = images_n.T

    num_y = df_y.values.shape[1]
    num_n = df_n.values.shape[1]

    yes = df_y.values.reshape((img_rows, img_cols, num_y))
    no = df_n.values.reshape((img_rows, img_cols, num_n))
    X = np.concatenate((yes,no), axis=2).swapaxes(2,0)
    y = np.concatenate((np.ones(num_y),np.zeros(num_n)))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, img_rows, img_cols

def neural_network_architecture(X_train, X_test, y_train, y_test, img_rows, img_cols):

    # Prepare data to feed the NN
    num_classes = 2

    # Ask keras which format to use depending on used backend and arrange data as expected
    if K.image_data_format() == 'channels_first':
        X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
        X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    # Incoming data is in uint8. Cast the input data images to be floats in range [0.0-1.0]  
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    # This is the neural network proposed architecture
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                    activation='relu',
                    input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=keras.optimizers.Adadelta(),
                metrics=['accuracy'])
    
    print('model created')
    return model, X_train, X_test, y_train, y_test

def fit_neural_network(model, X_train, X_test, y_train, y_test):
    # Fit the NN
    batch_size = 30
    epochs = 5

    model.fit(X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(X_test, y_test))
    print('training model')
    return model

def evaluate_model(model, X_test, y_test):
    score = model.evaluate(X_test, y_test, verbose=0)
    result = 'Test loss: {}, Test accuracy: {}'.format(score[0], score[1])
    print(result)

    y_pred = [int(round(model.predict(np.expand_dims(e,axis=0))[0][0])) for e in X_test]
    y_true = [int(e[0]) for e in y_test]
    
    # confusion matrix
    def cm(y_true, y_pred):
        return confusion_matrix(y_true, y_pred)

    cm = cm(y_true, y_pred)

    plt.clf()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    classNames = ['Negative','Positive']
    plt.title('Tumor or Not Tumor Confusion Matrix - Test Data')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    tick_marks = np.arange(len(classNames))
    plt.xticks(tick_marks, classNames, rotation=45)
    plt.yticks(tick_marks, classNames)
    s = [['TN','FP'], ['FN', 'TP']]
    for i in range(2):
        for j in range(2):
            plt.text(j,i, str(s[i][j])+" = "+str(cm[i][j]))
    plt.show()

    return None