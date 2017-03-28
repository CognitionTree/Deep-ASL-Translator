'''
File name: count_words.py
Objective: To count the number of times words repeat themselfs on the dataset
Author: Andy D. Martinez & Daniela Florit
Date created: 02/18/2017
Python Version: 2.7.12
'''
import glob

def build_count_dictionary(d, file_name):
	f = open(file_name)

	for line in f:
		word = line.split(',')[2]
		if word in d:
			d[word] += 1
		else:
			d[word] = 1

	f.close()


#main
all_file_names = glob.glob("*.csv")
d = {}

for file_name in all_file_names:
	build_count_dictionary(d, file_name)	


count = {}
for key in d:
	if d[key] in count:
		count[d[key]] += 1
	else:
		count[d[key]] = 1
		

print count
