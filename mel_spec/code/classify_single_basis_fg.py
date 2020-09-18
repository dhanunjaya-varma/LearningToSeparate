import os
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
import matplotlib.pyplot as plt

def plot_confusion_matrix(y_true, y_pred, classes,
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
    cm = confusion_matrix(y_true, y_pred)
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
q = 200
x_train = np.load('../data/x_audio_fold1_train.npy')
y_train = np.load('../data/y_audio_fold1_train.npy')
x_test = np.load('../data/x_audio_fold1_test.npy')
y_test = np.load('../data/y_audio_fold1_test.npy')
basis = np.load('../basis/basis_single_fold1_fg.npy')
print(x_train.shape)
print(basis.shape)
basis = np.transpose(basis)
m = np.mean(x_train,axis=0)
x_train = x_train - m
x_test = x_test - m

mat = np.dot(basis[:, p:q], np.transpose(basis[:, p:q]))
x_train = np.transpose(np.transpose(x_train) - np.dot(mat,np.transpose(x_train)))
x_test = np.transpose(np.transpose(x_test) - np.dot(mat,np.transpose(x_test)))

svm_clas = svm.SVC(kernel='linear', C=0.01)

svm_clas.fit(x_train, y_train) 
#joblib.dump(svm_clas1,'svm_error_fold1_weights.pkl')

y_pred = svm_clas.predict(x_test)
cvscores.append(metrics.accuracy_score(y_test, y_pred)*100)
print("Accuracy Fold1:", metrics.accuracy_score(y_test, y_pred)*100)

#plot_confusion_matrix(y_test, y_pred, classes, normalize=True,
#                  title='Normalized fold1 single basis')
#plt.savefig('../data/fold1_single_basis.png')

x_train = np.load('../data/x_audio_fold2_train.npy')
y_train = np.load('../data/y_audio_fold2_train.npy')
x_test = np.load('../data/x_audio_fold2_test.npy')
y_test = np.load('../data/y_audio_fold2_test.npy')
basis = np.load('../basis/basis_single_fold2_fg.npy')
print(x_train.shape)
print(basis.shape)
basis = np.transpose(basis)
m = np.mean(x_train,axis=0)
x_train = x_train - m
x_test = x_test - m
mat = np.dot(basis[:, p:q], np.transpose(basis[:, p:q]))
x_train = np.transpose(np.transpose(x_train) - np.dot(mat,np.transpose(x_train)))
x_test = np.transpose(np.transpose(x_test) - np.dot(mat,np.transpose(x_test)))

svm_clas = svm.SVC(kernel='linear', C=0.01)

svm_clas.fit(x_train, y_train) 
#joblib.dump(svm_clas1,'svm_error_fold1_weights.pkl')

y_pred = svm_clas.predict(x_test)
cvscores.append(metrics.accuracy_score(y_test, y_pred)*100)
print("Accuracy Fold2:", metrics.accuracy_score(y_test, y_pred)*100)

#plot_confusion_matrix(y_test, y_pred, classes, normalize=True,
#                  title='Normalized fold2 single basis')
#plt.savefig('../data/fold2_single_basis.png')

x_train = np.load('../data/x_audio_fold3_train.npy')
y_train = np.load('../data/y_audio_fold3_train.npy')
x_test = np.load('../data/x_audio_fold3_test.npy')
y_test = np.load('../data/y_audio_fold3_test.npy')
basis = np.load('../basis/basis_single_fold3_fg.npy')
print(x_train.shape)
print(basis.shape)
basis = np.transpose(basis)
m = np.mean(x_train,axis=0)
x_train = x_train - m
x_test = x_test - m
mat = np.dot(basis[:, p:q], np.transpose(basis[:, p:q]))
x_train = np.transpose(np.transpose(x_train) - np.dot(mat,np.transpose(x_train)))
x_test = np.transpose(np.transpose(x_test) - np.dot(mat,np.transpose(x_test)))

svm_clas = svm.SVC(kernel='linear', C=0.01)

svm_clas.fit(x_train, y_train) 
#joblib.dump(svm_clas1,'svm_error_fold1_weights.pkl')

y_pred = svm_clas.predict(x_test)
cvscores.append(metrics.accuracy_score(y_test, y_pred)*100)
print("Accuracy Fold3:", metrics.accuracy_score(y_test, y_pred)*100)

#plot_confusion_matrix(y_test, y_pred, classes, normalize=True,
#                  title='Normalized fold3 single basis')
#plt.savefig('../data/fold3_single_basis.png')

x_train = np.load('../data/x_audio_fold4_train.npy')
y_train = np.load('../data/y_audio_fold4_train.npy')
x_test = np.load('../data/x_audio_fold4_test.npy')
y_test = np.load('../data/y_audio_fold4_test.npy')
basis = np.load('../basis/basis_single_fold4_fg.npy')
print(x_train.shape)
print(basis.shape)
basis = np.transpose(basis)
m = np.mean(x_train,axis=0)
x_train = x_train - m
x_test = x_test - m
mat = np.dot(basis[:, p:q], np.transpose(basis[:, p:q]))
x_train = np.transpose(np.transpose(x_train) - np.dot(mat,np.transpose(x_train)))
x_test = np.transpose(np.transpose(x_test) - np.dot(mat,np.transpose(x_test)))

svm_clas = svm.SVC(kernel='linear', C=0.01)

svm_clas.fit(x_train, y_train) 
#joblib.dump(svm_clas1,'svm_error_fold1_weights.pkl')

y_pred = svm_clas.predict(x_test)
cvscores.append(metrics.accuracy_score(y_test, y_pred)*100)
print("Accuracy Fold4:", metrics.accuracy_score(y_test, y_pred)*100)

#plot_confusion_matrix(y_test, y_pred, classes, normalize=True,
#                  title='Normalized fold4 single basis')
#plt.savefig('../data/fold4_single_basis.png')

print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))