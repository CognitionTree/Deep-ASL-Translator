import cv2
import numpy as np
import glob
from frame import *
from numpy import *

class Action(object):
	def __init__(self, path, numb_frames = None): #path received should be something like '/Frames/Diving-Side/001'
		self.path = path
		self.numb_frames = numb_frames
		
		#getting labels
		split_directory = path.split('/')
		self.action_video_name = split_directory[-1]
		self.action_video_label = split_directory[-2]
		
		#getting frames:
		self.frames = []
		self.read_frames()
	
	def read_frames(self):
		frames_paths = glob.glob(self.path + '/*.jpg')
		frames_paths.reverse()
		
		if self.numb_frames == None:
			self.numb_frames = len(frames_paths)
		
		for i in range(self.numb_frames):
			frame = Frame(frames_paths[i])
			height, width = frame.get_shape()
			self.frames.append(frame)
			
		self.frames.reverse()
	
	def get_path(self):
		return self.path
		
	def get_action_label(self):
		return self.action_video_label
	
	def get_action_length(self):
		return self.numb_frames
	
	def get_frames(self):
		return self.frames
	
	def get_frame_at(self, pos):
		return self.frames[pos]
		
	def get_frame_matrices(self):
		frame_matrices = []
		for frame in self.frames:
			frame_matrices.append(frame.get_frame())
		return frame_matrices
		
	def get_frame_matrix_at(self, pos):
		return self.frames[pos].get_frame()
	
	def set_path(self, path):
		self.path = path
	
	def set_frames(self, frames):
		self.frames = frames
	
	def set_frame_at(self, frame):
		self.frames[pos] = frame
	
	def set_action_label(self, label):
		self.action_video_label = label
	
	def __str__(self):
		return self.action_video_label + '\n' +self.path + '\n' + str(len(self.frames))+' frames'
	
	def __len__(self):
		print "HEy they called me!!!!"
		print len(self.frames)
		return len(self.frames)