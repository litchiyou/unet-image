 # -*- coding: UTF-8 -*-
from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator, array_to_img
import numpy as np
import os
import skimage.io as io
import cv2
import skimage.transform as trans


class Augmentation(object):
	'''
	图像增强类
	'''
	def __init__(self, train_path="./data_set/image/train", label_path="./data_set/image/label",
	             merge_path="./data_set/merge", result_path='./data_set/result', img_type="tif"):
		#用glob模块实现简单的文件检索,初始化训练集和标签集的路径
		self.train_path = train_path
		self.label_path = label_path
		self.merge_path = merge_path
		self.result_path = result_path
		self.img_type = img_type
		self.datagen = ImageDataGenerator(      #调用数据增强的一些参数
			rotation_range=0.2,                 #图片随机转动的角度
			width_shift_range=0.05,             #随机水平偏移的幅度
			height_shift_range=0.05,            #随机竖直偏移的幅度
			shear_range=0.05,                   #剪切变换的程度
			zoom_range=0.05,                    #随机缩放的程度
			horizontal_flip=True,               #是否水平反转
			fill_mode='nearest')

	def adjustData(self, img, mask, flag_multi_class, num_class):
		if flag_multi_class:
			img = img / 255
			mask = mask[:, :, :, 0] if (len(mask.shape) == 4) else mask[:, :, 0]
			new_mask = np.zeros(mask.shape + (num_class,))
			for i in range(num_class):
				# for one pixel in the image, find the class in mask and convert it into one-hot vector
				# index = np.where(mask == i)
				# index_mask = (index[0],index[1],index[2],np.zeros(len(index[0]),dtype = np.int64) + i) if (len(mask.shape) == 4) else (index[0],index[1],np.zeros(len(index[0]),dtype = np.int64) + i)
				# new_mask[index_mask] = 1
				new_mask[mask == i, i] = 1
			new_mask = np.reshape(new_mask, (new_mask.shape[0], new_mask.shape[1] * new_mask.shape[2], new_mask.shape[3])) if flag_multi_class else np.reshape(new_mask, (new_mask.shape[0] * new_mask.shape[1], new_mask.shape[2]))
			mask = new_mask
			#归一化处理
		elif np.max(img) > 1:
			img = img / 255
			mask = mask / 255
			mask[mask > 0.5] = 1
			mask[mask <= 0.5] = 0
		return img, mask

	def trainGenerator(self, batch_size, image_path, image_folder, mask_folder, color_mode="grayscale",
	                   image_save_prefix="image", mask_save_prefix="mask",save_to_dir='./data_set/merge',save_format='tif',
	                   flag_multi_class=False, num_class=2,
	                   target_size=(256, 256), seed=1):
		'''
		利用flow_from_directory方法批量处理train图和label图进行数据增强，img
		'''
		datagen = self.datagen
		image_generator = datagen.flow_from_directory(
			image_path,
			classes=[image_folder],
			class_mode=None,
			color_mode=color_mode,
			target_size=target_size,
			batch_size=batch_size,
			save_to_dir=save_to_dir,
			save_prefix=image_save_prefix,
			save_format=save_format,
			shuffle=True,
			seed=seed)
		mask_generator = datagen.flow_from_directory(
			image_path,
			classes=[mask_folder],
			class_mode=None,
			color_mode=color_mode,
			target_size=target_size,
			batch_size=batch_size,
			save_to_dir=save_to_dir,
			save_prefix=mask_save_prefix,
			save_format=save_format,
			shuffle=True,
			seed=seed)
		train_generator = zip(image_generator, mask_generator)
		for img, mask in train_generator:
			img, mask = self.adjustData(img, mask, flag_multi_class, num_class)
			yield img, mask

	def testGenerator(self, test_path, num_image=30, target_size=(256, 256), flag_multi_class=False, as_gray=True):
		for i in range(num_image):
			img = io.imread(os.path.join(test_path, "%d.tif" % i), as_gray=as_gray)
			img = img / 255
			img = trans.resize(img, target_size)
			img = np.reshape(img, img.shape + (1,)) if (not flag_multi_class) else img
			img = np.reshape(img, (1,) + img.shape)
			yield img

	def saveResult(self):
		imgs = np.load('./data_set/test/imgs_mask_test_1.npy')
		for i in range(imgs.shape[0]):
			img = imgs[i]
			img = array_to_img(img)
			img.save("./data_set/result/%d.tif" % (i))