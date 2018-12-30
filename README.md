# unet-imageprocessing

该项目是在学习数字图像处理课程时最后的大作业。在基于Tensorflow的框架上用Keras进行深度学习对细胞图像分割并利用分割后的图像进行细胞计数。

# Overview

复现该项目需要先运行data_strength.py进行数据增强，再运行unet.py进行机器学习。

data_set/result是测试图像的结果图。data_set/image是训练数据。data_set/merge是经过数据增强后的训练数据。

## data

原始数据集来自[isbi挑战](http://brainiac2.mit.edu/isbi_challenge/)。他提供了tif格式的train，label，test图。但是由于是打包好的30张tif图片，需要先用python的TIFF库对文件进行处理。

## data_segmentation

由于训练图片是30张512×512的数据集，这对于机器学习来说是远远不够的，而且由于本人的硬件环境不太友好。我利用了keras.preprocessing.image一个名为ImageDataGenerator的模块进行图像扭曲来数据扩充。并且将图像转换成256×256的图像后批处理输入网络模型。

## model

![u-net-architecture](G:\tensor\u-net-architecture.png)

这个Unet网络模型是一种全卷积神经网络模型结构，详细资料可阅读：https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/

网络实现是利用Keras的API进行构建的，相当方便。

## trainning

训练次数epochs=10次。

训练样本数batch_size=2，由于硬件设备有限，只能跑2。需要注意是batch_size是与最后的loss值有关系的，batch_size太小，网络收敛不稳定，收敛慢。当batch_size太大，计算量太大，内存消耗多，前期收敛可能快，训练次数减少。

## Cell-count

利用了OpenCv中的Canny函数和findContours函数，先用Canny对图像进行边缘检测，再用findContours进行区域连同量计数。

