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

def create_empty_folders(dir, words_list, camera_views):
	for word in words_list:
		for view in camera_views:
			if not os.path.exists(dir + '/' + word + '/' + view):
				os.makedirs(dir + '/' + word + '/' + view)

def crop_frame(frame, left, top, width, height):
    return frame[top:(top+height), left:(left+width)]
    
def get_words(videos_directory):
	words = []
	words_paths = glob.glob(videos_directory + '/*')
	for word_path in words_paths:
		words.append(word_path.split('/')[-1])
	
	return words

def get_all_videos_paths(videos_directory, words, camera_views):
	all_videos_paths = []
	
	for word in words:
		for view in camera_views:
			all_videos_paths += glob.glob(videos_directory + '/' + word + '/' + view + '/*.mp4')
	
	return all_videos_paths

def get_all_frames_paths(frames_directory, words, camera_views):
	all_frames_paths = []
	
	for word in words:
		for view in camera_views:
			directory_names = glob.glob(frames_directory + '/' + word + '/' + view + '/*')
			for directory in directory_names:
				all_frames_paths += glob.glob(directory + '/*.jpg')
	return all_frames_paths

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
		
def finding_crop_size(videos_directory):
	frame_path = '/Users/danielaflorit/Github/ASL_Dataset/Dataset/Book/Front/ncslgr10a-002-04-dom-BOOK/1.jpg'
	frame = cv2.imread(frame_path)
	print_frame_col1(frame)
	cropped_frame = crop_frame(frame, 0, 0, len(frame[0]), 242)
	save_frame(videos_directory + "cropped.jpg", cropped_frame)
	
	
	'''This piece of code was done just to find the pixel at which the bottom black box starts.
	This was done by inspection, by printing the first pixels of each column of the frame
	it was determined that it was at pixel 242'''
	
def compute_optical_flow(input_video_path, output_image_path, crop_height):
	cap = cv2.VideoCapture(input_video_path)
	ret, frame1 = cap.read()
	prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
	hsv = zeros_like(frame1)
	hsv[...,1] = 255
	of_acc = None
	
	while(1):
		ret, frame2 = cap.read()
		if frame2 == None:
			break
		next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
		flow = cv2.calcOpticalFlowFarneback(prvs,next, 0.5, 3, 4, 3, 5, 1.2, 0)
		mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
		hsv[...,0] = ang*180/pi/2
		hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
		bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
		
		if of_acc == None:
			of_acc = bgr
		else:
			of_acc += bgr
			
		k = cv2.waitKey(30) & 0xff
		if k == 27:
			break
		elif k == ord('s'):
			cv2.imwrite('opticalfb.png',frame2)
			cv2.imwrite('opticalhsv.png',bgr)
		prvs = next
	
	cropped_frame = crop_frame(of_acc, 0, 0, len(of_acc[0]), crop_height)
	cv2.imwrite(output_image_path,cropped_frame)
	#cv2.imwrite('opticalhsv.png',of_acc)
	cap.release()
	cv2.destroyAllWindows()

def get_optical_flow_images(all_videos_paths, optical_flow_directory, crop_height):
	video_number = 1
	for video_path in all_videos_paths:
		
		print "video: ", video_number, "/", len(all_videos_paths)
		video_number += 1
		
		directory_name = video_path.split('.')[0]
		split_directory_name = directory_name.split('/')
		video_name = split_directory_name[-1]
		view = split_directory_name[-2]
		word = split_directory_name[-3]
		
		output_image_path = optical_flow_directory + '/' + word + '/' + view + '/' + video_name + '.png'
		compute_optical_flow(video_path,output_image_path, crop_height)

def compute_SIFT_image(frame_path, output_image_path):
	img = cv2.imread(frame_path)
	grayscale= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	sift = cv2.SIFT() #.SIFT_create()
	kp, des = sift.detectAndCompute(grayscale,None)
	cv2.imwrite(output_image_path, des)
	
def get_SIFT_all_images(all_frames_paths, frames_SIFT_directory):
	frame_number = 1
	for frame_path in all_frames_paths:
		print "frame_path"
		print frame_path
		print "frame: ", frame_number, "/", len(all_frames_paths)
		frame_number += 1
		
		directory_name = frame_path.split('.')[0]
		split_directory_name = directory_name.split('/')
		frame_name = split_directory_name[-1]
		video_name = split_directory_name[-2]
		view = split_directory_name[-3]
		word = split_directory_name[-4]
		
		output_directory = frames_SIFT_directory + '/' + word + '/' + view + '/' + video_name
		output_image_path = output_directory + '/' + frame_name + '.png'
		
		if not os.path.exists(output_directory):
			os.makedirs(output_directory)
		
		compute_SIFT_image(frame_path, output_image_path) #This saves the descriptor as an image
	
#-----------------main-----------------
videos_directory = '/Users/danielaflorit/Github/ASL_Dataset/Videos'
frames_directory = '/Users/danielaflorit/Github/ASL_Dataset/Frames'
testing_directory = '/Users/danielaflorit/Github/ASL_Dataset/Testing'
augmented_frames_directory = '/Users/danielaflorit/Github/ASL_Dataset/Frames_augmented'
optical_flow_directory = '/Users/danielaflorit/Github/ASL_Dataset/Optical_flow'
frames_SIFT_directory = '/Users/danielaflorit/Github/ASL_Dataset/Frames_SIFT'

camera_views = ['Face', 'Front', 'Side']
camera_views_single = ['Front'] 
words_list = get_words(videos_directory)

all_videos_paths = get_all_videos_paths(videos_directory, words_list, camera_views)
all_frames_paths = get_all_frames_paths(frames_directory, words_list, camera_views)

########################################
#Uncomment to call finding_crop_size()
#finding_crop_size(videos_directory)
crop_height = 242 #this was found using the function above

########################################
#Split all videos by frame and crop the bottom section out
read_and_save_all_frames(all_videos_paths, frames_directory, crop_height)

########################################
#Compute all optical flow images:
create_empty_folders(optical_flow_directory, words_list, camera_views)
get_optical_flow_images(all_videos_paths, optical_flow_directory, crop_height)

########################################
#Compute all SIFT images:
get_SIFT_all_images(all_frames_paths, frames_SIFT_directory)


