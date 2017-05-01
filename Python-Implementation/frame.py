import cv2
import numpy as np

class Frame(object):
	def __init__(self, path):
		self.path = path

		#Getting the gloss
		splitted_path = self.path.split('-') 
		gloss_and_extension = splitted_path[-1]
		self.gloss = gloss_and_extension.split('.')[0]
		
		#Getting the image
		self.img = cv2.imread(path)
		self.img = cv2.resize(self.img, (324/5,242/5))
	
	def get_path(self):
		return self.path
	
	def get_img(self):
		return self.img
	
	def set_path(self, path):
		self.path = path
	
	def set_img(self, img):
		self.img = img
	
	def get_gloss(self):
		return self.gloss
	
	def __str__(self):
		return self.path+'\n'+self.gloss+'\n'+str(self.img)
