# Dog Breed Identification
This work is part of Udacity’s MLND capstone project. 

## Project overview
The goal of this project is to build a pipeline that can be used for dog breed identification. The model will take any real-world, user-supplied images as input and make predictions accordingly. Specifically, two main functionalities come with the model:
1) Given an image of a dog, the model will identify an estimate of the canine’s breed;
2) Given an image of a human, the model will identify the resembling dog breed.

#### Results preview

![alt text](https://github.com/Jianming-Han/dog_breed_classifier_CNN/blob/master/output/o1.png)
![alt text](https://github.com/Jianming-Han/dog_breed_classifier_CNN/blob/master/output/o2.png)
![alt text](https://github.com/Jianming-Han/dog_breed_classifier_CNN/blob/master/output/o3.png)

## Data
You can download the [dog datasets](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip) and [human datasets](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip).
Totally we get 13233 and 8351 images for human and dogs respectively. The human images are sorted by names. All human images are in a standard size of 250x250, most images contain only one human face, but some contain multiple human faces. The dog images are classified into 33 dog categories and split into train (6680 images), validation (835 images) and test (836 images) subsets. The average image quantity for each breed is about 63 (≈8351/133). And some breeds contain more and some contain less, it’s slightly imbalanced among all the breeds. Dog images are with different image sizes, resolutions and lighting conditions, some contain complete dog body while some just contain a dog head. Further, some images contain multiple dogs and even human.

## Dog and human detector
OpenCV model will be used to detect human face in an image because OpenCV provides many pre-trained face detectors. To detect dog in an image, a pre-trained VGG16 model will be applied. Given an image, this VGG16 model will return a prediction for the object that is contained in the image.

## Dog classifier
I constructed a 3-layer CNN architecture with ReLu activation function from scratch in PyTorch to classify dog breeds. The model is not performing well with an accuracy of 18% (156/836) on the test dataset. So I chose to use the pre-trained ResNet152 model as fixed feature extractors that identify general features in the images, and add one fully-connected layer at the end to act as a final classifier for dog breed. The model using transfer learning improve the accuracy significantly from 18% to 86% (727/836)

## Libraries 
    opencv-python==4.2.0.34
    matplotlib==3.2.2
    numpy==1.18.0
    PIL==7.2.0
    pytorh==1.5.1
    tqdm==4.47.0
