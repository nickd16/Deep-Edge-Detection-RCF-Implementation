# Deep-Edge-Detection---RCF-Implementation

This project is an implementation of the deep edge detection method known as Richer Convolutional Features for Edge Detection by Yun Liu, Ming-Ming Cheng, Xiaowei Hu, Jia-Wang Bian, Le Zhang, Xiang Bai, and Jinhui Tang

To read the original paper and see other implementations, visit https://paperswithcode.com/paper/richer-convolutional-features-for-edge

Below are examples from the trained model. The images from from the BSDS500 dataset (link at the bottom). 

LEFT IMAGE: Real Image

MIDDLE IMAGE: RCF Edge Map

RIGHT IMAGE: Canny


![image](https://github.com/nickd16/Deep-Edge-Detection---RCF-Implementation/assets/108239710/72d2a0ef-73f3-4df8-9f08-440ea7012b4d)


![image](https://github.com/nickd16/Deep-Edge-Detection---RCF-Implementation/assets/108239710/45bd96e0-1aa9-409b-92de-c055847105c5)


![image](https://github.com/nickd16/Deep-Edge-Detection---RCF-Implementation/assets/108239710/6241d1ff-49b1-4546-8411-b12d73b5c565)


![image](https://github.com/nickd16/Deep-Edge-Detection---RCF-Implementation/assets/108239710/0a7c6810-c0c5-4460-af31-84b1515bad09)


![image](https://github.com/nickd16/Deep-Edge-Detection---RCF-Implementation/assets/108239710/3f2a1dfd-8750-4cbc-bcd8-acce3a083bd7)


Canny edge detection is one of the most popular original edge detection methods due to its speed and effectiveness. However, canny fails in most cases in getting rid of noise in an image. Due to this problem, deep edge detection methods like RCF have become popular recently as they are fast and effective in extracting important features and getting rid of noise. 


RCF Model -> Built on top of pretrained VGG16 

![image](https://github.com/nickd16/Deep-Edge-Detection---RCF-Implementation/assets/108239710/c933d490-b650-45d4-8ffa-5d4c089e642e)


Implementation Details:
Optimization -> Adam with LR = 5E-4
LR Scheduler -> StepLR with step_size =4000 and gamma = 0.1
Data Augmentations -> Normalization by mean and std for VGG16 and random horizontal and vertical flips

Dataset: https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html
