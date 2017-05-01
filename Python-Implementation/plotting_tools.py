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
	results_dir = 'results/'
	plt.savefig(results_dir+title+'.png')
	plt.close()
#--------------------------Confusion Matrix------------------------

#build_confusion_matrix(np.array([1,1,1,2,2,2,4,4,4]), np.array([1,1,2,2,2,4,4,4,4]), [1,2,4])
#display_distribution([1,2,3,4,5,6,7,8,9])
