from action import *
import glob
from random import shuffle
import numpy as np

class Dataset(object):
	def __init__(self, path): #the path received should be '/Frames'
		self.path = path
		self.numb_frames = 22 #this is the smallest number of frames
		self.actions = []
		self.action_labels = []
		self.action_video_lengths = []
		self.action_list = self.set_action_list()
		self.action_to_number = self.set_action_to_number()
		self.read_actions()
		#self.min_length = min(self.action_video_lengths)
		
	def set_action_list(self):
		action_list = []
		action_paths = glob.glob(self.path + '/*') #all paths for all action categories
		for action_path in action_paths:
			action = action_path.split('/')[-1]
			action_list.append(action)
		return action_list
	
	def get_action_list(self):
		return self.action_list
	
	def set_action_to_number(self):
		action_to_number = {}
		for action in self.action_list:
			action_to_number[action] = self.action_list.index(action)
		return action_to_number
	
	def get_action_to_number(self):
		return self.action_to_number
		
	def read_actions(self):
		action_paths = [] #this will contain all paths to each video folder such as '/Frames/Diving-Side/001'
		for action_name in self.action_list:
			action_paths += glob.glob(self.path + '/' + action_name + '/*')
		
		for action_path in action_paths:
			action_video = Action(action_path, self.numb_frames)
			self.actions.append(action_video)
			self.action_labels.append(action_video.get_action_label())
			self.action_video_lengths.append(action_video.get_action_length())
	
	def get_path(self):
		return self.path
	
	def get_action_labels(self):
		return self.action_labels
	
	def get_action_label_at(self, pos):
		return self.action_labels[pos]
	
	def get_actions(self):
		return self.actions
	
	def get_action_at(self, pos):
		return self.actions[pos]
	
	def get_action_matrix(self):
		matrix = []
		for action in self.actions:
			action_matrix = []
			for frame in action.get_frame_matrices():
				action_matrix.append(frame)
			matrix.append(action_matrix)
		return matrix
	
	def shuffle_dataset(self):
		shuffle(self.actions)
		self.action_labels = []
		
		for action in self.actions:
			self.action_labels.append(action.get_action_label())
		
	def organize_actions_by_action_label(self):
		map_label_action = {}
		for i in range(len(self.action_labels)):
			cur_label = self.action_labels[i]
			cur_action = self.actions[i]
			
			if cur_label in map_label_action:
				map_label_action[cur_label].append(cur_action)
			else:
				map_label_action[cur_label] = [cur_action]
		return map_label_action
	
	def get_data_split(self, training_fraction):
		X_train = []
		y_train = []
		
		X_test = []
		y_test = []
	
		action_matrix = self.get_action_matrix()
		#organize dataset by gloss
		map_label_action = self.organize_actions_by_action_label()
		
		#Setting up initial train fractions to 0
		train_count = {}
		#train_fractions = {}
		for label in map_label_action:
			print "label: ", label
			print map_label_action[label]
			train_count[label] = 0.0
		
		for i in range(len(self.actions)):
			cur_label = self.action_labels[i]
			cur_action = action_matrix[i]
			
			#Training
			if (train_count[cur_label]/(1.0*len(map_label_action[cur_label]))) < training_fraction:
				X_train.append(cur_action)
				y_train.append(self.action_to_number[cur_label])
			else:
				X_test.append(cur_action)
				y_test.append(self.action_to_number[cur_label])
				
			train_count[cur_label] += 1.0

		#print '============================='
		#print X_train
		X_train = np.array(X_train)
		y_train = np.array(y_train)
		X_test = np.array(X_test)
		y_test = np.array(y_test)
		return (X_train, y_train), (X_test, y_test)

	def get_numb_classes(self):
		return len(self.action_to_number)
	
	def __str__(self):
		seigns_str = ''
		for action in self.actions:
			seigns_str += str(action)
			seigns_str += '\n'
		return self.path+'\n'+str(len(self.actions)) + seigns_str