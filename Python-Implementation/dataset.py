from sign import *
import glob
from random import shuffle
import numpy as np

class Dataset(object):
	FRONT_VIEW = 'Front'
	FACE_VIEW = 'Face'
	SIDE_VIEW = 'Side'
	
	def __init__(self, path, view_point=FRONT_VIEW):
		self.path = path
		self.view_point = view_point
		self.gloss = []
		self.signs = []
		self.gloss_to_number = {}
		self.read_signs()
	
	def read_signs(self):
		i = 0
		sings_paths = glob.glob(self.path+'/*')
		for sign_path in sings_paths:
			#sign = Sign(sign_path)
			#self.signs.append(sign)
			#self.gloss.append(sign.get_gloss())
			sign_path = sign_path + '/'+self.view_point
			#print sign_path
			sign_versions_paths = glob.glob(sign_path+'/*')
			#print sign_versions_paths
			for sign_version_path in sign_versions_paths:
				sign = Sign(sign_version_path)
				self.signs.append(sign)
				self.gloss.append(sign.get_gloss())
				if len(sign) != 36:
					print sign
				if sign.get_gloss() not in self.gloss_to_number:
					self.gloss_to_number[sign.get_gloss()] = i
					i += 1
					
			#if i == 3:
				#break	
				
	def get_path(self):
		return self.path
	
	def get_gloss_to_numb(self):
		return self.gloss_to_number
		
	def get_view_point(self):
		return self.view_point
	
	def get_glosses(self):
		return self.glosses
	
	def get_gloss_at(self, pos):
		return self.gloss[pos]
	
	def get_signs(self):
		return self.signs
	
	def get_sign_at(self, pos):
		return self.signs[pos]
	
	def get_signs_matrix(self):
		matrix = []
		for sign in self.signs:
			sign_matrix = []
			for frame in sign.get_frames_matrices():
				sign_matrix.append(frame)
			matrix.append(sign_matrix)
		return matrix
	
	def shuffle_dataset(self):
		shuffle(self.signs)
		self.glosses = []
		
		for sign in self.signs:
			self.glosses.append(sign.get_gloss())
	
	def organize_signs_by_gloss(self):
		map_gloss_sign = {}
		for i in range(len(self.gloss)):
			cur_gloss = self.gloss[i]
			cur_sign = self.signs[i]
			
			if cur_gloss in map_gloss_sign:
				map_gloss_sign[cur_gloss].append(cur_sign)
			else:
				map_gloss_sign[cur_gloss] = [cur_sign]
		return map_gloss_sign
	
	def get_data_split(self, training_fraction):
		X_train = []
		y_train = []
		
		X_test = []
		y_test = []
	
		signs_matrix = self.get_signs_matrix()
		#organize dataset by gloss
		map_gloss_sign = self.organize_signs_by_gloss()
		
		#Setting up initial train fractions to 0
		train_count = {}
		#train_fractions = {}
		for gloss in map_gloss_sign:
			train_count[gloss] = 0.0
		
		for i in range(len(self.gloss)):
			cur_gloss = self.gloss[i]
			cur_sign = signs_matrix[i]
			
			#Training
			#print (train_count[cur_gloss]/(1.0*len(map_gloss_sign[cur_gloss])))
			if (train_count[cur_gloss]/(1.0*len(map_gloss_sign[cur_gloss]))) < training_fraction:
				X_train.append(cur_sign)
				y_train.append(self.gloss_to_number[cur_gloss])
			else:
				X_test.append(cur_sign)
				y_test.append(self.gloss_to_number[cur_gloss])
				
			train_count[cur_gloss] += 1.0

		#print '============================='
		#print X_train
		X_train = np.array(X_train)
		y_train = np.array(y_train)
		X_test = np.array(X_test)
		y_test = np.array(y_test)
		return (X_train, y_train), (X_test, y_test)
	
	def get_numb_classes(self):
		return len(self.gloss_to_number)
		
	def __str__(self):
		seigns_str = ''
		for sign in self.signs:
			seigns_str += str(sign)
			seigns_str += '\n'
		return self.path+'\n'+str(len(self.signs)) + seigns_str
		
			
		
		
				
			
				