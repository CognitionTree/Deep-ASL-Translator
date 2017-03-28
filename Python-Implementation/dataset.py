from sign import *
import glob

class Dataset(object):
	FRONT_VIEW = 'Font'
	FACE_VIEW = 'Face'
	SIDE_VIEW = 'Side'
	
	def __init__(self, path, view_point=FRONT_VIEW):
		self.path = path
		self.view_point = view_point
		self.gloss = []
		self.signs = []
		self.read_signs()
	
	def read_signs(self):
		sings_paths = glob.glob(path+'/*')
		for sign_path in sings_paths:
			sign = Sign(sign_path)
			self.signs.append(sign)
			self.gloss.append(sign.get_gloss())
	
	def get_path(self):
		return self.path
	
	def get_view_point(self):
		return self.view_point
	
	def get_glosses(self):
		return array(self.glosses)
	
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
			for frame in sign.get_frames():
				sign_matrix.append(frame)
			matrix.append(array(sign_matrix))
		return array(matrix)
	
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
				map_gloss_sign[gloss].append(cur_sign)
			else:
				map_gloss_sign[gloss] = [cur_sign]
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
		for gloss in map_gloss_sign:
			train_fractions[gloss] = 0.0
		
		for i in range(len(self.gloss)):
			cur_gloss = self.gloss[i]
			cur_sign = signs_matrix[i]
			
			#Training
			if (train_count[cur_gloss]/(1.0*len(map_gloss_sign[cur_gloss]))) < training_fraction:
				X_train.append(cur_sign)
				y_train.append(cur_gloss)
			else:
				X_test.append(cur_sign)
				y_test.append(cur_gloss)
				
			train_count[cur_gloss] += 1.0
		
		return (array(X_train), array(y_train)), (array(X_test), array(y_test))
		
			
		
		
				
			
				