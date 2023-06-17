# PixBlock - Image Classification with Pixel Blocking
PixBlock is an image classification project that explores the effects of pixel blocking on the performance of convolutional neural networks (CNNs). The project uses the Intel Image Classification dataset and trains a CNN model to classify images into various classes.

## Dataset
The dataset used in this project is the Intel Image Classification dataset, which consists of images of natural scenes categorized into six classes: buildings, forest, glacier, mountain, sea, and street. The dataset is divided into a training set and a test set.

## Model Architecture
The CNN model used in this project is a simple architecture consisting of convolutional layers, ReLU activation functions, and max pooling layers. The output of the convolutional layers is flattened and fed into a fully connected layer, followed by a softmax activation function to produce the final class probabilities.

## Training Process
- The training process starts by loading the training dataset and applying data transformations such as resizing, normalization, and tensor conversion.
- A subset of (2000-5000) images is randomly selected from the training set for faster training.
- The model is initialized and moved to the appropriate device (CPU or GPU).
- The model is trained for a specified number of epochs. In each epoch, the training data is loaded in batches, and pixel blocking is applied to a percentage of pixels in each image (5%, 10%, or 15%).
- The model parameters are optimized using the Adam optimizer and the cross-entropy loss function.
- After each epoch, the model is evaluated on the test set to calculate the loss and accuracy.
- The training and test loss, as well as the training and test accuracy, are recorded for each epoch.

## Results
The project explores the impact of pixel blocking on the model's performance. The training and test loss and accuracy are plotted over the course of training for each blocking percentage (0%, 5%, 10%, and 15%).

The confusion matrix is generated to evaluate the model's performance in classifying the test set.
