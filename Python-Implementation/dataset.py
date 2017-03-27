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
		print 'Implement This'