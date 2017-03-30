from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt; plt.rcdefaults()
import matplotlib.pyplot as pl
import numpy as np
import itertools
from sklearn.manifold import TSNE

#-----------------------Confusion Matrix-------------------------
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def build_confusion_matrix(y_true, y_pred, classes, title='Confusion Matrix'):
	cnf_matrix = confusion_matrix(y_true, y_pred)
	plt.figure()
	plot_confusion_matrix(cnf_matrix, classes=classes, title=title)
	#plt.show()
	plt.savefig(title+'.png')
	plt.close()
#--------------------------Confusion Matrix------------------------

#--------------------------Distributions---------------------------
def display_distribution(distribution, title='From Current Camera To Next Camera'):
	y_pos = np.arange(len(distribution))
 
	plt.bar(y_pos, distribution, align='center', alpha=0.5)
	plt.xticks(y_pos, y_pos)
	plt.ylabel('Number Of People')
	plt.title('Next Camera')
 	plt.title(title)
	#plt.show()
	plt.savefig(title+'.png')
	plt.close()
#--------------------------Distributions---------------------------
#It is assumin the 8 classes
#TODO: Generalize tool for multi purpose
def display_feature_vectors(feature_vectors, y_true, y_pred, title='Feature Vectors'):
	model = TSNE(n_components=9, random_state=0)
	#The feature vectors in 2D
	points_2d = model.fit_transform(feature_vectors)
	print('LENGTH = ' + str(len(points_2d)))
	print(points_2d)

	#Correct Next camera/prediction color
	colors = {0:'indigo', 1:'b', 2:'g', 3:'c', 4:'y', 5:'m', 6:'r', 7:'k', 8:'pink'}
	
	#Markers for matching labels and for non matching labels between predictions and real label
	markers = {'W':'x', 'R':'o'}

	#Data Structure to parse the data
	#TODO: Get  rid of this	
	data = {(0,'R'):([],[]), (0,'W'):([],[]), (1,'R'):([],[]), (1,'W'):([],[]), (2,'R'):([],[]), (2,'W'):([],[]), (3,'R'):([],[]), (3,'W'):([],[]), (4,'R'):([],[]), (4,'W'):([],[]), (5,'R'):([],[]), (5,'W'):([],[]), (6,'R'):([],[]), (6,'W'):([],[]), (7,'R'):([],[]), (7,'W'):([],[]), (8,'R'):([],[]), (8,'W'):([],[])}

	for i in range(len(points_2d)):
		if y_true[i] == y_pred[i]:
			data[(y_true[i], 'R')][0].append(points_2d[0])
			data[(y_true[i], 'R')][1].append(points_2d[1])
		else:
			data[(y_true[i], 'W')][0].append(points_2d[0])
			data[(y_true[i], 'W')][1].append(points_2d[1])
	
	print(data)

	r0 = plt.scatter(data[(0, 'R')][0], data[(0, 'R')][1], marker=markers['R'], color=colors[0])
	w0 = plt.scatter(data[(0, 'W')][0], data[(0, 'W')][1], marker=markers['W'], color=colors[0])
	r1 = plt.scatter(data[(1, 'R')][0], data[(1, 'R')][1], marker=markers['R'], color=colors[1])
	w1 = plt.scatter(data[(1, 'W')][0], data[(1, 'W')][1], marker=markers['W'], color=colors[1])
	r2 = plt.scatter(data[(2, 'R')][0], data[(2, 'R')][1], marker=markers['R'], color=colors[2])
	w2 = plt.scatter(data[(2, 'W')][0], data[(2, 'W')][1], marker=markers['W'], color=colors[2])
	r3 = plt.scatter(data[(3, 'R')][0], data[(3, 'R')][1], marker=markers['R'], color=colors[3])
	w3 = plt.scatter(data[(3, 'W')][0], data[(3, 'W')][1], marker=markers['W'], color=colors[3])
	r4 = plt.scatter(data[(4, 'R')][0], data[(4, 'R')][1], marker=markers['R'], color=colors[4])
	w4 = plt.scatter(data[(4, 'W')][0], data[(4, 'W')][1], marker=markers['W'], color=colors[4])
	r5 = plt.scatter(data[(5, 'R')][0], data[(5, 'R')][1], marker=markers['R'], color=colors[5])
	w5 = plt.scatter(data[(5, 'W')][0], data[(5, 'W')][1], marker=markers['W'], color=colors[5])
	r6 = plt.scatter(data[(6, 'R')][0], data[(6, 'R')][1], marker=markers['R'], color=colors[6])
	w6 = plt.scatter(data[(6, 'W')][0], data[(6, 'W')][1], marker=markers['W'], color=colors[6])
	r7 = plt.scatter(data[(7, 'R')][0], data[(7, 'R')][1], marker=markers['R'], color=colors[7])
	w7 = plt.scatter(data[(7, 'W')][0], data[(7, 'W')][1], marker=markers['W'], color=colors[7])
	r8 = plt.scatter(data[(8, 'R')][0], data[(8, 'R')][1], marker=markers['R'], color=colors[8])
	w8 = plt.scatter(data[(8, 'W')][0], data[(8, 'W')][1], marker=markers['W'], color=colors[8])	

	plt.savefig(title+'.png')
	plt.close()
	
#build_confusion_matrix(np.array([1,1,1,2,2,2,4,4,4]), np.array([1,1,2,2,2,4,4,4,4]), [1,2,4])
#display_distribution([1,2,3,4,5,6,7,8,9])
