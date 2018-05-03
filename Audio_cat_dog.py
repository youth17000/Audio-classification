#Libraries used
import os
import re
import numpy as np
import pandas as pd
import scipy.io.wavfile as sw
import python_speech_features as psf
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from scipy import interp
from sklearn.metrics import confusion_matrix
from sklearn import svm
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
#import scikitplot as skplt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers import Dense
from keras.layers.normalization import BatchNormalization
from keras.constraints import max_norm
from keras .models import Sequential 
from keras.callbacks import ModelCheckpoint
from keras.layers import Dropout
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from sklearn.naive_bayes import GaussianNB

#from sklearn.model_selection import cross_val_score
#from sklearn.model_selection import StratifiedKFold
#from sklearn.pipeline import Pipeline
#from sklearn.preprocessing import LabelEncoder
#%matplotlib inline

#Importing directory
RootDir=os.chdir("E:\\Mission Data Scientist\\INSOFE\\Internship\\Cats Dogs Audio dataset\\cats_dogs")

file_names = os.listdir(RootDir)#list of file names
final_dataset = pd.DataFrame()#Blank initial dataset 

#Feature Extraction
for i in file_names:
    rate,signal = sw.read(i)
    features = psf.base.mfcc(signal)
    features = psf.base.fbank(features)
    features = psf.base.logfbank(features[1])
    features = psf.base.lifter(features,L=22)
    features = psf.base.delta(features,N=13)
    features = pd.DataFrame(features)
    features["Target"] = i
    final_dataset = final_dataset.append(features)#rbind(final_dataset,features)

"""for i in file_names:
    rate,signal = sw.read(i)
    features = pd.DataFrame(signal)
    features["Target"] = i
    final_dataset = final_dataset.append(features)#rbind(final_dataset,features)"""

#Correcting indexing
index = 26
for i in range(0,len(final_dataset)):
    final_dataset.iloc[i,index] = final_dataset.iloc[i,index].replace('.wav', '')
    final_dataset.iloc[i,index] = re.sub(r'[0-9]+', '',final_dataset.iloc[i,index])
    final_dataset.iloc[i,index] = final_dataset.iloc[i,index].replace('_', '')
    final_dataset.iloc[i,index] = final_dataset.iloc[i,index].replace('barking', '0')
    final_dataset.iloc[i,index] = final_dataset.iloc[i,index].replace('cat', '1')
    final_dataset.iloc[i,index] = final_dataset.iloc[i,index].replace('dog', '0')
    final_dataset.iloc[i,index] = final_dataset.iloc[i,index].replace('00', '0')

#Finalize dataset with the attributes and target
fd=final_dataset
fd = fd.rename(columns = {'y' : 'target'})
y=fd.iloc[:,-1]
X=fd.iloc[:,0:26]

fd.to_csv('Raw_structured_data.csv', encoding = 'utf-8')
#test.shape
#train = fd.iloc[0:200,0:26]
#test = fd.iloc[201:267,0:26]

"""p_fd=pd.DataFrame(p_fd)
y=p_fd.iloc[:,-1]
X=p_fd.iloc[:,0:26]
p_fd.to_csv('Preprocessed_data.csv', encoding = 'utf-8')"""


#Feature Scaling
sc = StandardScaler()
X = sc.fit_transform(X)
P_DF = pd.DataFrame(X,y)
P_DF.to_csv('Preprocessed_data.csv', encoding = 'utf-8')


#Spliting into test and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 1)
#del y_train

#Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

type(y_train)
X_train=pd.DataFrame(X_train)
y_train=pd.DataFrame(y_train)
X_test=pd.DataFrame(X_test)

preprocessed_dataset = pd.DataFrame()
preprocessed_dataset = preprocessed_dataset.append(X_train,y_train)
preprocessed_dataset = preprocessed_dataset.append(X_train,X_train)


#Dimensionality Resuction
pcatrain = PCA(n_components = 26)
pcatest = PCA(n_components = 26)
pca_train = pcatrain.fit(X_train)
pca_test = pcatest.fit(X_test)
#The amount of variance that each PC explains
varTrain = pcatrain.explained_variance_ratio_
varTest = pcatest.explained_variance_ratio_
#Cumulative Variance explains
var1train=np.cumsum(np.round(pcatrain.explained_variance_ratio_, decimals=4)*100)
var1test=np.cumsum(np.round(pcatest.explained_variance_ratio_, decimals=4)*100)
print(var1train)
print(var1test)
plt.plot(var1train)
plt.plot(var1test)


#Model Building

#SVM
model = svm.SVC(kernel = 'rbf', C = 1)
model1 = model.fit(X_train,y_train)
model1.score(X_train,y_train)
predicted=model.predict(X_test)
accuracy_score(y_test,predicted)

X_train = X_train.as_matrix(columns = None)
y_train = y_train.as_matrix(columns = None)
X_test = X_test.as_matrix(columns = None)
y_test = y_test.as_matrix(columns = None)


learning_rate = 0.2
momentum = 0.95


#ANN model structure
model2 = Sequential()#sequential is like a container in which
model2.add(Dense(8, activation='relu', input_shape = (26,)))
#model2.add(BatchNormalization(axis = -1))
#model2.add(Dropout(0.2))
model2.add(Dense(4, kernel_initializer='normal', activation='relu'))
#model2.add(BatchNormalization(axis = -1))
#model2.add(Dropout(0.2))
model2.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
#sgd = SGD(lr=learning_rate, momentum = momentum)
model2.compile(optimizer='Adam',
      loss='binary_crossentropy',
      metrics=['accuracy'])
#filepath = "weights_improvement.hdf5"
#checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
#callbacks_list = [checkpoint]
model2.fit(X_train, y_train, epochs = 25, batch_size = 26, 
           validation_data = (X_test, y_test), shuffle = True, verbose = 1)
model2.summary()

#Checking accuracy score
score, acc = model2.evaluate(X_test,y_test,batch_size = 32)
print('score: ', score)
print('accu: ', acc)


#del y_pred

#y_pred=model2.predict(X_test)
#y_pred1 = (y_pred > 0.5)
#type(y_test)

#sklearn.metrics.roc_curve(y_true, y_score, pos_label=None, sample_weight=None, drop_intermediate=True)
#cm = confusion_matrix(y_test, y_pred)

#type(y_test)
#print(y_test)
#X_train=np.reshape(X_train,(1,np.product(X_train.shape)))
#y_train=np.reshape(y_train,(1,np.product(y_train.shape)))


#Naive bayes model
model = GaussianNB()
m1=model.fit(X_train, y_train).predict(X_test)
conf_m=confusion_matrix(y_test, m1)
s1 = conf_m[0,0]+conf_m[0,1]
s2 = conf_m[1,0]+conf_m[1,1]
s=s1+s2
Accu_nb=((conf_m[0,0]+conf_m[1,1])/(s))*100
print(Accu_nb) #Accuracy 57 on test



#CNN  model
"""learning_rate = 0.0025
momentum = 0.85
model = Sequential()
model.add(Convolution2D(filters = 2, kernel_size = (2,2), 
                        input_shape = (207,26,1), activation='relu'))
model.add(MaxPooling2D(pool_size = (1,1)))
model.add(Dropout(0.25))
model.add(Convolution2D(filters = 2, kernel_size = (2,2), activation='relu'))
model.add(MaxPooling2D(pool_size = (1,1)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(2, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='sigmoid'))
model.summary()
X_train = X_train.reshape(X_train.shape, (1,X_train.shape,1))
X_test = X_test.reshape(1,70,26,1)
sgd = SGD(lr=learning_rate, momentum=momentum)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['acc'])
model.fit(X_train, y_train, epochs = 25, batch_size = 32)"""



max_review_length = 26
model = Sequential()
model.add(Embedding(30, 32, input_length=max_review_length))
model.add(LSTM(10))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3, batch_size=32)