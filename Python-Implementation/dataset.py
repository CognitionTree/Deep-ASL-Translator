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
		return self.glosses
	
	def get_gloss_at(self, pos):
		return self.gloss[pos]
	
	def get_signs(self):
		return self.signs
	
	def get_sign_at(self, pos):
		return self.signs[pos]
	
	def get_signs_matrix(self):
		matrix = []
		#TODO: Read everithing with this structure [sign1[frame1[],frame2[]], sign2[f1[],f2[]]]
			
				