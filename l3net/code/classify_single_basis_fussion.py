import os
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics

cvscores = []

p = 0
q = 200
e = 0
f = 50
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

x_train = np.column_stack((x_train_bg, x_train_fg))
x_test = np.column_stack((x_test_bg, x_test_fg))

svm_clas = svm.SVC(kernel='linear', C=0.01)

svm_clas.fit(x_train, y_train) 
#joblib.dump(svm_clas1,'svm_error_fold1_weights.pkl')

y_pred = svm_clas.predict(x_test)
cvscores.append(metrics.accuracy_score(y_test, y_pred)*100)
print("Accuracy Fold1:", metrics.accuracy_score(y_test, y_pred)*100)


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

x_train = np.column_stack((x_train_bg, x_train_fg))
x_test = np.column_stack((x_test_bg, x_test_fg))

svm_clas = svm.SVC(kernel='linear', C=0.01)

svm_clas.fit(x_train, y_train) 
#joblib.dump(svm_clas1,'svm_error_fold1_weights.pkl')

y_pred = svm_clas.predict(x_test)
cvscores.append(metrics.accuracy_score(y_test, y_pred)*100)
print("Accuracy Fold2:", metrics.accuracy_score(y_test, y_pred)*100)

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

x_train = np.column_stack((x_train_bg, x_train_fg))
x_test = np.column_stack((x_test_bg, x_test_fg))

svm_clas = svm.SVC(kernel='linear', C=0.01)

svm_clas.fit(x_train, y_train) 
#joblib.dump(svm_clas1,'svm_error_fold1_weights.pkl')

y_pred = svm_clas.predict(x_test)
cvscores.append(metrics.accuracy_score(y_test, y_pred)*100)
print("Accuracy Fold3:", metrics.accuracy_score(y_test, y_pred)*100)

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

x_train = np.column_stack((x_train_bg, x_train_fg))
x_test = np.column_stack((x_test_bg, x_test_fg))

svm_clas = svm.SVC(kernel='linear', C=0.01)

svm_clas.fit(x_train, y_train) 
#joblib.dump(svm_clas1,'svm_error_fold1_weights.pkl')

y_pred = svm_clas.predict(x_test)
cvscores.append(metrics.accuracy_score(y_test, y_pred)*100)
print("Accuracy Fold4:", metrics.accuracy_score(y_test, y_pred)*100)
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))