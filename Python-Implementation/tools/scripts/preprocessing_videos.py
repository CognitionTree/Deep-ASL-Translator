import cv2
import glob
import os
import shutil
from numpy import *

def save_frame(path, frame):
	cv2.imwrite(path, frame)

def print_frame_col1(frame):
	count = 1
	for row in frame:
		print count, row[0]
		count +=1

def crop_frame(frame, left, top, width, height):
    return frame[top:(top+height), left:(left+width)]
    
def get_words(dataset_directory):
	words = []
	words_paths = glob.glob(dataset_directory + '/*')
	for word_path in words_paths:
		words.append(word_path.split('/')[-1])
	
	return words

def get_all_videos_paths(dataset_directory, words, camera_views):
	#List containing all path for all videos in the dataset
	all_videos_paths = []
	
	for word in words:
		print word
		for view in camera_views:
			print view
			all_videos_paths += glob.glob(dataset_directory + '/' + word + '/' + view + '/*.mp4')
	
	print all_videos_paths
	return all_videos_paths

def read_and_save_all_frames(all_videos_paths, frames_directory, crop_height):
	video_information_matrix = []
	video_number = 1
	for video_path in all_videos_paths:
		print "video: ", video_number, "/", len(all_videos_paths)
		video_number += 1
		directory_name = video_path.split('.')[0]
		
		if video_number == 100:
			break
		
		split_directory_name = directory_name.split('/')
		video_name = split_directory_name[-1]
		view = split_directory_name[-2]
		word = split_directory_name[-3]

		directory_name_to_save = frames_directory + '/' + word + '/' + view + '/' + video_name
		if not os.path.exists(directory_name_to_save):
			os.makedirs(directory_name_to_save)
		else:
			shutil.rmtree(directory_name_to_save)
			os.makedirs(directory_name_to_save)
		
		cap = cv2.VideoCapture(video_path)
		frame_count = 1
		
		while(cap.isOpened()):
			ret, frame = cap.read()
			
			if frame == None:
				break
			
			cropped_frame = crop_frame(frame, 0, 0, len(frame[0]), crop_height)
			save_frame(directory_name_to_save + '/' + str(frame_count) + '.jpg', cropped_frame)
			frame_count += 1
		
		length = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
		width  = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
		height = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
		fps    = cap.get(cv2.cv.CV_CAP_PROP_FPS)
		
		video_info = [video_name, word, view, length, width, height, fps]
		video_information_matrix.append(video_info)
		
		cap.release()
		cv2.destroyAllWindows()
		
		
	
	video_information_file = open("video_information.txt", 'w')
	for row in video_information_matrix:
		video_information_file.write(str(row) + '\n')
		
def finding_crop_size(dataset_directory):
	frame_path = '/Users/danielaflorit/Github/ASL_Dataset/Dataset/Book/Front/ncslgr10a-002-04-dom-BOOK/1.jpg'
	frame = cv2.imread(frame_path)
	print_frame_col1(frame)
	cropped_frame = crop_frame(frame, 0, 0, len(frame[0]), 242)
	save_frame(dataset_directory + "cropped.jpg", cropped_frame)
	
	
	'''This piece of code was done just to find the pixel at which the bottom black box starts.
	This was done by inspection, by printing the first pixels of each column of the frame
	it was determined that it was at pixel 242'''
	
		
#-----------------main-----------------
videos_directory = '/Users/danielaflorit/Github/ASL_Dataset/Videos'
frames_directory = '/Users/danielaflorit/Github/ASL_Dataset/Frames'
testing_directory = '/Users/danielaflorit/Github/ASL_Dataset/Testing'
augmented_frames_directory = '/Users/danielaflorit/Github/ASL_Dataset/Frames_augmented'

camera_views = ['Face', 'Front', 'Side']
words_list = get_words(videos_directory)
all_videos_paths = get_all_videos_paths(videos_directory, words_list, camera_views)

########################################
#Uncomment to call finding_crop_size()
#finding_crop_size(videos_directory)
crop_height = 242 #this was found using the function above

########################################
#Uncomment to split all videos by frame and crop the bottom section out
#read_and_save_all_frames(all_videos_paths, frames_directory, crop_height)


	