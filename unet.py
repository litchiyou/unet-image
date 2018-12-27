import cv2
import os
from keras.callbacks import ModelCheckpoint
from keras.layers import *
from keras.models import *
from keras.optimizers import *
import keras
from data_strength import Augmentation

os.environ["CUDA_VISIBLE_DEVICES"] = "0"      #指定运行在编号为0的GPU，若没有GPU加速，可以注释掉，其计算会自动运行在CPU上

class myUnet(keras.Model):

	def __init__(self, img_rows=256, img_cols=256):
		self.img_rows = img_rows
		self.img_cols = img_cols

	# 载入数据
	def unet(self, pretrained_weights=None):
		'''
		unet网络，卷积核是3*3
		:return:
		'''
		inputs = Input((self.img_rows, self.img_cols, 1))

		conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
		conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
		pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

		conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
		conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
		pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

		conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
		conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
		pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

		conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
		conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
		drop4 = Dropout(0.5)(conv4)
		pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

		conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
		conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
		drop5 = Dropout(0.5)(conv5)

		up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
			UpSampling2D(size=(2, 2))(drop5))
		merge6 = concatenate([drop4, up6], axis=3)
		conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
		conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

		up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
			UpSampling2D(size=(2, 2))(conv6))
		merge7 = concatenate([conv3, up7], axis=3)
		conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
		conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

		up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
			UpSampling2D(size=(2, 2))(conv7))
		merge8 = concatenate([conv2, up8], axis=3)
		conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
		conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

		up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
			UpSampling2D(size=(2, 2))(conv8))
		merge9 = concatenate([conv1, up9], axis=3)
		conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
		conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
		conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
		conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

		model = Model(inputs=[inputs], outputs=[conv10])

		model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
		print('model compile')

		if (pretrained_weights):
			model.load_weights(pretrained_weights)

		return model

	def train(self):
		print("got unet")
		model = self.unet()
		data = Augmentation()
		model_checkpoint = ModelCheckpoint('my_unet.hdf5', monitor='loss', verbose=1, save_best_only=True)

		print('Fitting model...')
		myunet= data.trainGenerator(batch_size=2, image_path='./data_set/image', image_folder='train',
		                            mask_folder='label')
		model.fit_generator(
			myunet,
			steps_per_epoch=300,
			epochs=10,
			verbose=1,
			callbacks=[model_checkpoint])                 #训练集训练网络模型

		print('predict test data')
		testGene = data.testGenerator("./data_set/test")
		model = load_model('my_unet.hdf5')
		imgs_mask_test = model.predict_generator(testGene, 30, verbose=1)              #测试集加载已训练好的模型
		np.save('./data_set/test/imgs_mask_test_1.npy', imgs_mask_test)
		data.saveResult()
		print("预测概率：", imgs_mask_test)


	def Cell_conut(self):
		'''
		计算图像中的细胞个数
		:return:
		'''
		print("图像中的细胞个数：")
		img = cv2.imread('./data_set/result/0.tif')
		binaryimg = cv2.Canny(img, 50, 200)             #边缘检测
		h = cv2.findContours(binaryimg, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)       #连同区域计数
		contours = h[1]
		cv2.imshow('img',binaryimg)
		cv2.waitKey(0)
		print(len(contours))

if __name__ == '__main__':
	myunet = myUnet()
	myunet.train()
	myunet.Cell_conut()
