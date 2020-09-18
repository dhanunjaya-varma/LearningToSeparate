import os
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
import matplotlib.pyplot as plt

np.random.seed(3)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    #cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    #classes = classes[y_true]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    fig, ax = plt.subplots(figsize=(10,10))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    #ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

classes=['beach', 'bus', 'cafe/restaurant', 'car', 'city_center',
         'forest_path', 'grocery_store', 'home', 'library', 'metro_station',
         'office', 'park', 'residential_area', 'train', 'tram']

cvscores = []

p = 0
q = 150
e = 0
f = 150
x_train = np.load('../data/x_audio_fold1_train.npy')
y_train = np.load('../data/y_audio_fold1_train.npy')
x_test = np.load('../data/x_audio_fold1_test.npy')
y_test = np.load('../data/y_audio_fold1_test.npy')
basis_fg = np.load('../basis/basis_single_fold1_fg.npy')
basis_bg = np.load('../basis/basis_single_fold1_bg.npy')
m = np.mean(x_train,axis=0)
x_train = x_train - m
x_test = x_test - m

basis_fg = np.transpose(basis_fg)
mat_fg = np.dot(basis_fg[:, p:q], np.transpose(basis_fg[:, p:q]))
x_train_fg = np.transpose(np.transpose(x_train) - np.dot(mat_fg,np.transpose(x_train)))
x_test_fg = np.transpose(np.transpose(x_test) - np.dot(mat_fg,np.transpose(x_test)))

basis_bg = np.transpose(basis_bg)
mat_bg = np.dot(basis_bg[:, e:f], np.transpose(basis_bg[:, e:f]))
x_train_bg = np.transpose(np.transpose(x_train) - np.dot(mat_bg,np.transpose(x_train)))
x_test_bg = np.transpose(np.transpose(x_test) - np.dot(mat_bg,np.transpose(x_test)))

#x_train = np.column_stack((x_train_bg, x_train_fg))
#x_test = np.column_stack((x_test_bg, x_test_fg))

svm_bg = svm.SVC(kernel='linear', C=0.01, probability=True)

svm_bg.fit(x_train_bg, y_train) 

svm_fg = svm.SVC(kernel='linear', C=0.01, probability=True)

svm_fg.fit(x_train_fg, y_train) 
y_pred = []

for i in range(x_test_bg.shape[0]):
	pred_bg = svm_bg.predict_proba(x_test_bg[i].reshape(1,-1))
	pred_fg = svm_fg.predict_proba(x_test_fg[i].reshape(1,-1))
	#print(np.argmax(np.mean([pred_bg, pred_fg], axis=0)))
	y_pred.append(np.argmax(np.mean([pred_bg, pred_fg], axis=0)))

y_pred = np.array(y_pred)
print(y_pred.shape)
print(y_test.shape)
cvscores.append(metrics.accuracy_score(y_test, y_pred)*100)
print("Accuracy Fold1:", metrics.accuracy_score(y_test, y_pred)*100)
cm = confusion_matrix(y_test, y_pred)

#plot_confusion_matrix(y_test, y_pred, classes, normalize=True,
#                  title='Normalized fold1 late fusion')
#plt.savefig('../data/fold1_late_fusion.png')

x_train = np.load('../data/x_audio_fold2_train.npy')
y_train = np.load('../data/y_audio_fold2_train.npy')
x_test = np.load('../data/x_audio_fold2_test.npy')
y_test = np.load('../data/y_audio_fold2_test.npy')
basis_fg = np.load('../basis/basis_single_fold2_fg.npy')
basis_bg = np.load('../basis/basis_single_fold2_bg.npy')
m = np.mean(x_train,axis=0)
x_train = x_train - m
x_test = x_test - m

basis_fg = np.transpose(basis_fg)
mat_fg = np.dot(basis_fg[:, p:q], np.transpose(basis_fg[:, p:q]))
x_train_fg = np.transpose(np.transpose(x_train) - np.dot(mat_fg,np.transpose(x_train)))
x_test_fg = np.transpose(np.transpose(x_test) - np.dot(mat_fg,np.transpose(x_test)))

basis_bg = np.transpose(basis_bg)
mat_bg = np.dot(basis_bg[:, e:f], np.transpose(basis_bg[:, e:f]))
x_train_bg = np.transpose(np.transpose(x_train) - np.dot(mat_bg,np.transpose(x_train)))
x_test_bg = np.transpose(np.transpose(x_test) - np.dot(mat_bg,np.transpose(x_test)))


svm_bg = svm.SVC(kernel='linear', C=0.01, probability=True)

svm_bg.fit(x_train_bg, y_train) 

svm_fg = svm.SVC(kernel='linear', C=0.01, probability=True)

svm_fg.fit(x_train_fg, y_train) 
y_pred = []

for i in range(x_test_bg.shape[0]):
	pred_bg = svm_bg.predict_proba(x_test_bg[i].reshape(1,-1))
	pred_fg = svm_fg.predict_proba(x_test_fg[i].reshape(1,-1))
	y_pred.append(np.argmax(np.mean([pred_bg, pred_fg], axis=0)))

y_pred = np.array(y_pred)
cvscores.append(metrics.accuracy_score(y_test, y_pred)*100)
print("Accuracy Fold2:", metrics.accuracy_score(y_test, y_pred)*100)
cm = cm + confusion_matrix(y_test, y_pred)

#plot_confusion_matrix(y_test, y_pred, classes, normalize=True,
#                  title='Normalized fold2 late fusion')
#plt.savefig('../data/fold2_late_fusion.png')

x_train = np.load('../data/x_audio_fold3_train.npy')
y_train = np.load('../data/y_audio_fold3_train.npy')
x_test = np.load('../data/x_audio_fold3_test.npy')
y_test = np.load('../data/y_audio_fold3_test.npy')
basis_fg = np.load('../basis/basis_single_fold3_fg.npy')
basis_bg = np.load('../basis/basis_single_fold3_bg.npy')
m = np.mean(x_train,axis=0)
x_train = x_train - m
x_test = x_test - m

basis_fg = np.transpose(basis_fg)
mat_fg = np.dot(basis_fg[:, p:q], np.transpose(basis_fg[:, p:q]))
x_train_fg = np.transpose(np.transpose(x_train) - np.dot(mat_fg,np.transpose(x_train)))
x_test_fg = np.transpose(np.transpose(x_test) - np.dot(mat_fg,np.transpose(x_test)))

basis_bg = np.transpose(basis_bg)
mat_bg = np.dot(basis_bg[:, e:f], np.transpose(basis_bg[:, e:f]))
x_train_bg = np.transpose(np.transpose(x_train) - np.dot(mat_bg,np.transpose(x_train)))
x_test_bg = np.transpose(np.transpose(x_test) - np.dot(mat_bg,np.transpose(x_test)))

svm_bg = svm.SVC(kernel='linear', C=0.01, probability=True)

svm_bg.fit(x_train_bg, y_train) 

svm_fg = svm.SVC(kernel='linear', C=0.01, probability=True)

svm_fg.fit(x_train_fg, y_train) 
y_pred = []

for i in range(x_test_bg.shape[0]):
	pred_bg = svm_bg.predict_proba(x_test_bg[i].reshape(1,-1))
	pred_fg = svm_fg.predict_proba(x_test_fg[i].reshape(1,-1))
	y_pred.append(np.argmax(np.mean([pred_bg, pred_fg], axis=0)))

y_pred = np.array(y_pred)
cvscores.append(metrics.accuracy_score(y_test, y_pred)*100)
print("Accuracy Fold3:", metrics.accuracy_score(y_test, y_pred)*100)
cm = cm + confusion_matrix(y_test, y_pred)

#plot_confusion_matrix(y_test, y_pred, classes, normalize=True,
#                  title='Normalized fold3 late fusion')
#plt.savefig('../data/fold3_late_fusion.png')

x_train = np.load('../data/x_audio_fold4_train.npy')
y_train = np.load('../data/y_audio_fold4_train.npy')
x_test = np.load('../data/x_audio_fold4_test.npy')
y_test = np.load('../data/y_audio_fold4_test.npy')
basis_fg = np.load('../basis/basis_single_fold4_fg.npy')
basis_bg = np.load('../basis/basis_single_fold4_bg.npy')
m = np.mean(x_train,axis=0)
x_train = x_train - m
x_test = x_test - m

basis_fg = np.transpose(basis_fg)
mat_fg = np.dot(basis_fg[:, p:q], np.transpose(basis_fg[:, p:q]))
x_train_fg = np.transpose(np.transpose(x_train) - np.dot(mat_fg,np.transpose(x_train)))
x_test_fg = np.transpose(np.transpose(x_test) - np.dot(mat_fg,np.transpose(x_test)))

basis_bg = np.transpose(basis_bg)
mat_bg = np.dot(basis_bg[:, e:f], np.transpose(basis_bg[:, e:f]))
x_train_bg = np.transpose(np.transpose(x_train) - np.dot(mat_bg,np.transpose(x_train)))
x_test_bg = np.transpose(np.transpose(x_test) - np.dot(mat_bg,np.transpose(x_test)))

svm_bg = svm.SVC(kernel='linear', C=0.01, probability=True)

svm_bg.fit(x_train_bg, y_train) 

svm_fg = svm.SVC(kernel='linear', C=0.01, probability=True)

svm_fg.fit(x_train_fg, y_train) 
y_pred = []

for i in range(x_test_bg.shape[0]):
	pred_bg = svm_bg.predict_proba(x_test_bg[i].reshape(1,-1))
	pred_fg = svm_fg.predict_proba(x_test_fg[i].reshape(1,-1))
	y_pred.append(np.argmax(np.mean([pred_bg, pred_fg], axis=0)))

y_pred = np.array(y_pred)
cvscores.append(metrics.accuracy_score(y_test, y_pred)*100)
print("Accuracy Fold4:", metrics.accuracy_score(y_test, y_pred)*100)
cm = cm + confusion_matrix(y_test, y_pred)

#plot_confusion_matrix(cm, classes, normalize=True,
#                  title='Normalized confusion matrix late fusion')
#plt.savefig('../data/late_fusion.png')

print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
