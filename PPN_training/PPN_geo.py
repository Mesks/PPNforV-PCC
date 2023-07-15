#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
from random import shuffle
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation, Dropout, Dense, Flatten, Input
from tensorflow.keras.models import Model
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import concatenate
from keras.utils.vis_utils import plot_model
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import numpy.random
import argparse
import locale
import os

seed = 246

# model-compile parameter sets
model_metrics = 'acc'
epochs = 300
batchs = 128
splits = 0.2
lr        = 1e-5
input_dim = 5
opt = Adam(learning_rate=lr,weight_decay=1e-5/128)

concatenated_df=pd.read_csv("extraFeatures_Geo.csv", header=None)
XY = concatenated_df.values
for i in range(10):
    np.random.shuffle(XY)
X = XY[:,[0,1,3,5,6,8,9]]## 'MPD','CBF','CUD','OEF','CUC','FLM','PPS','Label','tempRDCost','bestRDCost'
Y = XY[:,[7]]
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=splits, random_state=seed)
cost=x_train[:,[input_dim,input_dim+1]]
x_train=x_train[:,0:input_dim]
x_test=x_test[:,0:input_dim]

model = Sequential()
inputShape=(input_dim,)
model.add(Input(shape=inputShape))
x = Dense(10,activation="sigmoid", kernel_initializer="RandomNormal", bias_initializer="RandomNormal")(model.output)
x = Dense(1,activation ="sigmoid", kernel_initializer="RandomNormal", bias_initializer="RandomNormal")(x)
model = Model(inputs=[model.input],outputs=x)
model.compile(loss="mse",optimizer=opt,metrics=['acc'])

y_train_flatten = y_train.flatten()
class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train_flatten), y=y_train_flatten)
class_weights = dict(zip(np.unique(y_train_flatten),class_weights))
# cost_max = np.max(cost[:,0])
# cost_min = np.min(cost[:,0])
# cost_average = np.average(cost[:,0])
# sample_weightss = np.array((cost[:,0]-cost_min)/(cost_max-cost_min))
# sample_weightss = np.array(cost[:,0]/cost_average)
sample_num=np.size(y_train,0)
cost_sum=0
cost_num=0
cost_difference = []
for sample in np.concatenate([cost,y_train],axis=1):
    cost_difference_value = sample[0]-sample[1]
    if (sample[2]==0)&(cost_difference_value!=0):
        cost_difference.append(0)
    elif (sample[2]==0)&(cost_difference_value==0):
        cost_difference.append(1)
    elif (sample[2]==1)&(cost_difference_value<=0):
        cost_difference.append(0)
    else:
        cost_difference.append(cost_difference_value)
        cost_sum+=cost_difference_value
        cost_num+=1
sample_weights = np.array(cost_difference)
cost_average=cost_sum/cost_num
for i in range(sample_num):
    if (y_train[i]==1)&(sample_weights[i]!=0):
        sample_weights[i]=sample_weights[i]/cost_average
    if sample_weights[i]>1:
        sample_weights[i]=1
    elif sample_weights[i]<0:
        sample_weights[i]=0

plot_model(model,to_file='FeaturesPlots/model.png',show_shapes=True)


# In[2]:


history = model.fit(x=[x_train],y=y_train, validation_data=([x_test], y_test), 
                    epochs=epochs, batch_size=batchs, class_weight=class_weights, sample_weight=sample_weights)

model.save_weights(r'revision/geo_model_allFeatures_withsamplewight.h5')
eval_model=[]
eval_model.append(model.evaluate([x_test], y_test)[1])
print("\nTest Accuracy: %.4f" % eval_model[0])


# In[3]:


plt.plot(history.history['loss'],color='r')
plt.plot(history.history['val_loss'],color='g')
plt.plot(history.history['acc'],color='b')
plt.plot(history.history['val_acc'],color='k')
plt.title('Learning curve (Geometry)')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train_loss', 'test_loss','train_acc', 'test_acc'], loc='upper left',bbox_to_anchor=(0,-0.3))
plt.savefig('FeaturesPlots/P_GeoTrainingCurve.jpg', bbox_inches='tight', dpi=1280)
plt.show()

import pickle
with open('revision/geo_model_allFeatures_withsamplewight.txt', 'wb') as file_txt:
    pickle.dump(history.history, file_txt)


# In[4]:


np.set_printoptions(suppress=True)

a_weight1=model.get_weights()[0]
a_bias1=model.get_weights()[1]
a_weight2=model.get_weights()[2]
a_bias2=model.get_weights()[3]
# a_weight3=model.get_weights()[4]
# a_bias3=model.get_weights()[5]


print("\na_weight1: ")
for a in a_weight1:
    for b in a:
        print(b,end=",")
        
print("\n\na_bias1: ")
for a in a_bias1:
        print(a,end=",")
        
print("\n\na_weight2: ")
for a in a_weight2:
    for b in a:
        print(b,end=",")
        
print("\n\na_bias2: ")
for a in a_bias2:
        print(a,end=",")

# print("\n\na_weight3: ")
# for a in a_weight3:
#     for b in a:
#         print(b,end=",")
        
# print("\n\na_bias3: ")
# for a in a_bias3:
#         print(a,end=",")
        
# g_weight1=model.get_layer(index=0).get_weights()
# g_weight2=model.get_layer(index=1).get_weights()
        
# print(g_weight1)
# print(g_weight2)


# In[5]:


# import numpy as np
# from keras.layers import Dense, Dropout
# from keras.models import Sequential, load_model
# import tensorflow.compat.v1 as tf
# inp_num = 6

# mmodel = Sequential()
# mmodel.add(Dense(10, input_dim=inp_num, activation='sigmoid'))
# mmodel.add(Dense(1,activation='sigmoid'))
# mmodel.compile(loss='mean_squared_error', optimizer='Adam', metrics=['acc'])
# # mmodel.add(Dense(3, activation='softmax'))
# # mmodel.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits = False), optimizer='Adam', metrics=['acc'])
# mmodel.load_weights(r'weightANDlearningcurve/geo_model.h5')

# data=np.array([0,0,1,1,0,0]).reshape(1,-1)
# print(mmodel.predict(data))


# In[6]:


import pandas as pd
import numpy as np
import os
from random import shuffle
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation, Dropout, Dense, Flatten, Input
from tensorflow.keras.models import Model
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import concatenate
from keras.utils.vis_utils import plot_model
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import numpy.random
import argparse
import locale
import os

seed = 246

# model-compile parameter sets
model_metrics = 'acc'
epochs = 300
batchs = 128
splits = 0.2
lr        = 1e-5
input_dim = 4
opt = Adam(learning_rate=lr,weight_decay=1e-5/128)

concatenated_df=pd.read_csv("extraFeatures_Geo.csv", header=None)
XY = concatenated_df.values
for i in range(10):
    np.random.shuffle(XY)
X = XY[:,[1,3,5,6,8,9]]## 'MPD','CBF','CUD','OEF','CUC','FLM','PPS','Label','tempRDCost','bestRDCost'
Y = XY[:,[7]]
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=splits, random_state=seed)
cost=x_train[:,[input_dim,input_dim+1]]
x_train=x_train[:,0:input_dim]
x_test=x_test[:,0:input_dim]

model = Sequential()
inputShape=(input_dim,)
model.add(Input(shape=inputShape))
x = Dense(10,activation="sigmoid", kernel_initializer="RandomNormal", bias_initializer="RandomNormal")(model.output)
x = Dense(1,activation ="sigmoid", kernel_initializer="RandomNormal", bias_initializer="RandomNormal")(x)
model = Model(inputs=[model.input],outputs=x)
model.compile(loss="mse",optimizer=opt,metrics=['acc'])

y_train_flatten = y_train.flatten()
class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train_flatten), y=y_train_flatten)
class_weights = dict(zip(np.unique(y_train_flatten),class_weights))
# cost_max = np.max(cost[:,0])
# cost_min = np.min(cost[:,0])
# cost_average = np.average(cost[:,0])
# sample_weightss = np.array((cost[:,0]-cost_min)/(cost_max-cost_min))
# sample_weightss = np.array(cost[:,0]/cost_average)
sample_num=np.size(y_train,0)
cost_sum=0
cost_num=0
cost_difference = []
for sample in np.concatenate([cost,y_train],axis=1):
    cost_difference_value = sample[0]-sample[1]
    if (sample[2]==0)&(cost_difference_value!=0):
        cost_difference.append(0)
    elif (sample[2]==0)&(cost_difference_value==0):
        cost_difference.append(1)
    elif (sample[2]==1)&(cost_difference_value<=0):
        cost_difference.append(0)
    else:
        cost_difference.append(cost_difference_value)
        cost_sum+=cost_difference_value
        cost_num+=1
sample_weights = np.array(cost_difference)
cost_average=cost_sum/cost_num
for i in range(sample_num):
    if (y_train[i]==1)&(sample_weights[i]!=0):
        sample_weights[i]=sample_weights[i]/cost_average
    if sample_weights[i]>1:
        sample_weights[i]=1
    elif sample_weights[i]<0:
        sample_weights[i]=0

history = model.fit(x=[x_train],y=y_train, validation_data=([x_test], y_test), 
                    epochs=epochs, batch_size=batchs, class_weight=class_weights, sample_weight=sample_weights)

model.save_weights(r'revision/geo_model_noMPD_withsamplewight.h5')
eval_model=[]
eval_model.append(model.evaluate([x_test], y_test)[1])
print("\nTest Accuracy: %.4f" % eval_model[0])

plt.plot(history.history['loss'],color='r')
plt.plot(history.history['val_loss'],color='g')
plt.plot(history.history['acc'],color='b')
plt.plot(history.history['val_acc'],color='k')
plt.title('Learning curve (Geometry)')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train_loss', 'test_loss','train_acc', 'test_acc'], loc='upper left',bbox_to_anchor=(0,-0.3))
plt.savefig('FeaturesPlots/P_GeoTrainingCurve.jpg', bbox_inches='tight', dpi=1280)
plt.show()

import pickle
with open('revision/geo_model_noMPD_withsamplewight.txt', 'wb') as file_txt:
    pickle.dump(history.history, file_txt)
    
np.set_printoptions(suppress=True)

a_weight1=model.get_weights()[0]
a_bias1=model.get_weights()[1]
a_weight2=model.get_weights()[2]
a_bias2=model.get_weights()[3]
# a_weight3=model.get_weights()[4]
# a_bias3=model.get_weights()[5]


print("\na_weight1: ")
for a in a_weight1:
    for b in a:
        print(b,end=",")
        
print("\n\na_bias1: ")
for a in a_bias1:
        print(a,end=",")
        
print("\n\na_weight2: ")
for a in a_weight2:
    for b in a:
        print(b,end=",")
        
print("\n\na_bias2: ")
for a in a_bias2:
        print(a,end=",")


# In[7]:


import pandas as pd
import numpy as np
import os
from random import shuffle
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation, Dropout, Dense, Flatten, Input
from tensorflow.keras.models import Model
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import concatenate
from keras.utils.vis_utils import plot_model
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import numpy.random
import argparse
import locale
import os

seed = 246

# model-compile parameter sets
model_metrics = 'acc'
epochs = 300
batchs = 128
splits = 0.2
lr        = 1e-5
input_dim = 4
opt = Adam(learning_rate=lr,weight_decay=1e-5/128)

concatenated_df=pd.read_csv("extraFeatures_Geo.csv", header=None)
XY = concatenated_df.values
for i in range(10):
    np.random.shuffle(XY)
X = XY[:,[0,3,5,6,8,9]]## 'MPD','CBF','CUD','OEF','CUC','FLM','PPS','Label','tempRDCost','bestRDCost'
Y = XY[:,[7]]
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=splits, random_state=seed)
cost=x_train[:,[input_dim,input_dim+1]]
x_train=x_train[:,0:input_dim]
x_test=x_test[:,0:input_dim]

model = Sequential()
inputShape=(input_dim,)
model.add(Input(shape=inputShape))
x = Dense(10,activation="sigmoid", kernel_initializer="RandomNormal", bias_initializer="RandomNormal")(model.output)
x = Dense(1,activation ="sigmoid", kernel_initializer="RandomNormal", bias_initializer="RandomNormal")(x)
model = Model(inputs=[model.input],outputs=x)
model.compile(loss="mse",optimizer=opt,metrics=['acc'])

y_train_flatten = y_train.flatten()
class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train_flatten), y=y_train_flatten)
class_weights = dict(zip(np.unique(y_train_flatten),class_weights))
# cost_max = np.max(cost[:,0])
# cost_min = np.min(cost[:,0])
# cost_average = np.average(cost[:,0])
# sample_weightss = np.array((cost[:,0]-cost_min)/(cost_max-cost_min))
# sample_weightss = np.array(cost[:,0]/cost_average)
sample_num=np.size(y_train,0)
cost_sum=0
cost_num=0
cost_difference = []
for sample in np.concatenate([cost,y_train],axis=1):
    cost_difference_value = sample[0]-sample[1]
    if (sample[2]==0)&(cost_difference_value!=0):
        cost_difference.append(0)
    elif (sample[2]==0)&(cost_difference_value==0):
        cost_difference.append(1)
    elif (sample[2]==1)&(cost_difference_value<=0):
        cost_difference.append(0)
    else:
        cost_difference.append(cost_difference_value)
        cost_sum+=cost_difference_value
        cost_num+=1
sample_weights = np.array(cost_difference)
cost_average=cost_sum/cost_num
for i in range(sample_num):
    if (y_train[i]==1)&(sample_weights[i]!=0):
        sample_weights[i]=sample_weights[i]/cost_average
    if sample_weights[i]>1:
        sample_weights[i]=1
    elif sample_weights[i]<0:
        sample_weights[i]=0

history = model.fit(x=[x_train],y=y_train, validation_data=([x_test], y_test), 
                    epochs=epochs, batch_size=batchs, class_weight=class_weights, sample_weight=sample_weights)

model.save_weights(r'revision/geo_model_noCBF_withsamplewight.h5')
eval_model=[]
eval_model.append(model.evaluate([x_test], y_test)[1])
print("\nTest Accuracy: %.4f" % eval_model[0])

plt.plot(history.history['loss'],color='r')
plt.plot(history.history['val_loss'],color='g')
plt.plot(history.history['acc'],color='b')
plt.plot(history.history['val_acc'],color='k')
plt.title('Learning curve (Geometry)')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train_loss', 'test_loss','train_acc', 'test_acc'], loc='upper left',bbox_to_anchor=(0,-0.3))
plt.savefig('FeaturesPlots/P_GeoTrainingCurve.jpg', bbox_inches='tight', dpi=1280)
plt.show()

import pickle
with open('revision/geo_model_noCBF_withsamplewight.txt', 'wb') as file_txt:
    pickle.dump(history.history, file_txt)
    
np.set_printoptions(suppress=True)

a_weight1=model.get_weights()[0]
a_bias1=model.get_weights()[1]
a_weight2=model.get_weights()[2]
a_bias2=model.get_weights()[3]
# a_weight3=model.get_weights()[4]
# a_bias3=model.get_weights()[5]


print("\na_weight1: ")
for a in a_weight1:
    for b in a:
        print(b,end=",")
        
print("\n\na_bias1: ")
for a in a_bias1:
        print(a,end=",")
        
print("\n\na_weight2: ")
for a in a_weight2:
    for b in a:
        print(b,end=",")
        
print("\n\na_bias2: ")
for a in a_bias2:
        print(a,end=",")


# In[1]:


import pandas as pd
import numpy as np
import os
from random import shuffle
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation, Dropout, Dense, Flatten, Input
from tensorflow.keras.models import Model
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import concatenate
from keras.utils.vis_utils import plot_model
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import numpy.random
import argparse
import locale
import os

seed = 246

# model-compile parameter sets
model_metrics = 'acc'
epochs = 300
batchs = 128
splits = 0.2
lr        = 1e-5
input_dim = 4
opt = Adam(learning_rate=lr,weight_decay=1e-5/128)

concatenated_df=pd.read_csv("extraFeatures_Geo.csv", header=None)
XY = concatenated_df.values
for i in range(10):
    np.random.shuffle(XY)
X = XY[:,[0,1,5,6,8,9]]## 'MPD','CBF','CUD','OEF','CUC','FLM','PPS','Label','tempRDCost','bestRDCost'
Y = XY[:,[7]]
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=splits, random_state=seed)
cost=x_train[:,[input_dim,input_dim+1]]
x_train=x_train[:,0:input_dim]
x_test=x_test[:,0:input_dim]

model = Sequential()
inputShape=(input_dim,)
model.add(Input(shape=inputShape))
x = Dense(10,activation="sigmoid", kernel_initializer="RandomNormal", bias_initializer="RandomNormal")(model.output)
x = Dense(1,activation ="sigmoid", kernel_initializer="RandomNormal", bias_initializer="RandomNormal")(x)
model = Model(inputs=[model.input],outputs=x)
model.compile(loss="mse",optimizer=opt,metrics=['acc'])

y_train_flatten = y_train.flatten()
class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train_flatten), y=y_train_flatten)
class_weights = dict(zip(np.unique(y_train_flatten),class_weights))
# cost_max = np.max(cost[:,0])
# cost_min = np.min(cost[:,0])
# cost_average = np.average(cost[:,0])
# sample_weightss = np.array((cost[:,0]-cost_min)/(cost_max-cost_min))
# sample_weightss = np.array(cost[:,0]/cost_average)
sample_num=np.size(y_train,0)
cost_sum=0
cost_num=0
cost_difference = []
for sample in np.concatenate([cost,y_train],axis=1):
    cost_difference_value = sample[0]-sample[1]
    if (sample[2]==0)&(cost_difference_value!=0):
        cost_difference.append(0)
    elif (sample[2]==0)&(cost_difference_value==0):
        cost_difference.append(1)
    elif (sample[2]==1)&(cost_difference_value<=0):
        cost_difference.append(0)
    else:
        cost_difference.append(cost_difference_value)
        cost_sum+=cost_difference_value
        cost_num+=1
sample_weights = np.array(cost_difference)
cost_average=cost_sum/cost_num
for i in range(sample_num):
    if (y_train[i]==1)&(sample_weights[i]!=0):
        sample_weights[i]=sample_weights[i]/cost_average
    if sample_weights[i]>1:
        sample_weights[i]=1
    elif sample_weights[i]<0:
        sample_weights[i]=0

history = model.fit(x=[x_train],y=y_train, validation_data=([x_test], y_test), 
                    epochs=epochs, batch_size=batchs, class_weight=class_weights, sample_weight=sample_weights)

model.save_weights(r'revision/geo_model_noOEF_withsamplewight.h5')
eval_model=[]
eval_model.append(model.evaluate([x_test], y_test)[1])
print("\nTest Accuracy: %.4f" % eval_model[0])

plt.plot(history.history['loss'],color='r')
plt.plot(history.history['val_loss'],color='g')
plt.plot(history.history['acc'],color='b')
plt.plot(history.history['val_acc'],color='k')
plt.title('Learning curve (Geometry)')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train_loss', 'test_loss','train_acc', 'test_acc'], loc='upper left',bbox_to_anchor=(0,-0.3))
plt.savefig('FeaturesPlots/P_GeoTrainingCurve.jpg', bbox_inches='tight', dpi=1280)
plt.show()

import pickle
with open('revision/geo_model_noOEF_withsamplewight.txt', 'wb') as file_txt:
    pickle.dump(history.history, file_txt)
    
np.set_printoptions(suppress=True)

a_weight1=model.get_weights()[0]
a_bias1=model.get_weights()[1]
a_weight2=model.get_weights()[2]
a_bias2=model.get_weights()[3]
# a_weight3=model.get_weights()[4]
# a_bias3=model.get_weights()[5]


print("\na_weight1: ")
for a in a_weight1:
    for b in a:
        print(b,end=",")
        
print("\n\na_bias1: ")
for a in a_bias1:
        print(a,end=",")
        
print("\n\na_weight2: ")
for a in a_weight2:
    for b in a:
        print(b,end=",")
        
print("\n\na_bias2: ")
for a in a_bias2:
        print(a,end=",")


# In[2]:


import pandas as pd
import numpy as np
import os
from random import shuffle
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation, Dropout, Dense, Flatten, Input
from tensorflow.keras.models import Model
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import concatenate
from keras.utils.vis_utils import plot_model
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import numpy.random
import argparse
import locale
import os

seed = 246

# model-compile parameter sets
model_metrics = 'acc'
epochs = 300
batchs = 128
splits = 0.2
lr        = 1e-5
input_dim = 4
opt = Adam(learning_rate=lr,weight_decay=1e-5/128)

concatenated_df=pd.read_csv("extraFeatures_Geo.csv", header=None)
XY = concatenated_df.values
for i in range(10):
    np.random.shuffle(XY)
X = XY[:,[0,1,3,6,8,9]]## 'MPD','CBF','CUD','OEF','CUC','FLM','PPS','Label','tempRDCost','bestRDCost'
Y = XY[:,[7]]
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=splits, random_state=seed)
cost=x_train[:,[input_dim,input_dim+1]]
x_train=x_train[:,0:input_dim]
x_test=x_test[:,0:input_dim]

model = Sequential()
inputShape=(input_dim,)
model.add(Input(shape=inputShape))
x = Dense(10,activation="sigmoid", kernel_initializer="RandomNormal", bias_initializer="RandomNormal")(model.output)
x = Dense(1,activation ="sigmoid", kernel_initializer="RandomNormal", bias_initializer="RandomNormal")(x)
model = Model(inputs=[model.input],outputs=x)
model.compile(loss="mse",optimizer=opt,metrics=['acc'])

y_train_flatten = y_train.flatten()
class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train_flatten), y=y_train_flatten)
class_weights = dict(zip(np.unique(y_train_flatten),class_weights))
# cost_max = np.max(cost[:,0])
# cost_min = np.min(cost[:,0])
# cost_average = np.average(cost[:,0])
# sample_weightss = np.array((cost[:,0]-cost_min)/(cost_max-cost_min))
# sample_weightss = np.array(cost[:,0]/cost_average)
sample_num=np.size(y_train,0)
cost_sum=0
cost_num=0
cost_difference = []
for sample in np.concatenate([cost,y_train],axis=1):
    cost_difference_value = sample[0]-sample[1]
    if (sample[2]==0)&(cost_difference_value!=0):
        cost_difference.append(0)
    elif (sample[2]==0)&(cost_difference_value==0):
        cost_difference.append(1)
    elif (sample[2]==1)&(cost_difference_value<=0):
        cost_difference.append(0)
    else:
        cost_difference.append(cost_difference_value)
        cost_sum+=cost_difference_value
        cost_num+=1
sample_weights = np.array(cost_difference)
cost_average=cost_sum/cost_num
for i in range(sample_num):
    if (y_train[i]==1)&(sample_weights[i]!=0):
        sample_weights[i]=sample_weights[i]/cost_average
    if sample_weights[i]>1:
        sample_weights[i]=1
    elif sample_weights[i]<0:
        sample_weights[i]=0

history = model.fit(x=[x_train],y=y_train, validation_data=([x_test], y_test), 
                    epochs=epochs, batch_size=batchs, class_weight=class_weights, sample_weight=sample_weights)

model.save_weights(r'revision/geo_model_noFLM_withsamplewight.h5')
eval_model=[]
eval_model.append(model.evaluate([x_test], y_test)[1])
print("\nTest Accuracy: %.4f" % eval_model[0])

plt.plot(history.history['loss'],color='r')
plt.plot(history.history['val_loss'],color='g')
plt.plot(history.history['acc'],color='b')
plt.plot(history.history['val_acc'],color='k')
plt.title('Learning curve (Geometry)')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train_loss', 'test_loss','train_acc', 'test_acc'], loc='upper left',bbox_to_anchor=(0,-0.3))
plt.savefig('FeaturesPlots/P_GeoTrainingCurve.jpg', bbox_inches='tight', dpi=1280)
plt.show()

import pickle
with open('revision/geo_model_noFLM_withsamplewight.txt', 'wb') as file_txt:
    pickle.dump(history.history, file_txt)
    
np.set_printoptions(suppress=True)

a_weight1=model.get_weights()[0]
a_bias1=model.get_weights()[1]
a_weight2=model.get_weights()[2]
a_bias2=model.get_weights()[3]
# a_weight3=model.get_weights()[4]
# a_bias3=model.get_weights()[5]


print("\na_weight1: ")
for a in a_weight1:
    for b in a:
        print(b,end=",")
        
print("\n\na_bias1: ")
for a in a_bias1:
        print(a,end=",")
        
print("\n\na_weight2: ")
for a in a_weight2:
    for b in a:
        print(b,end=",")
        
print("\n\na_bias2: ")
for a in a_bias2:
        print(a,end=",")


# In[3]:


import pandas as pd
import numpy as np
import os
from random import shuffle
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation, Dropout, Dense, Flatten, Input
from tensorflow.keras.models import Model
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import concatenate
from keras.utils.vis_utils import plot_model
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import numpy.random
import argparse
import locale
import os

seed = 246

# model-compile parameter sets
model_metrics = 'acc'
epochs = 300
batchs = 128
splits = 0.2
lr        = 1e-5
input_dim = 4
opt = Adam(learning_rate=lr,weight_decay=1e-5/128)

concatenated_df=pd.read_csv("extraFeatures_Geo.csv", header=None)
XY = concatenated_df.values
for i in range(10):
    np.random.shuffle(XY)
X = XY[:,[0,1,3,5,8,9]]## 'MPD','CBF','CUD','OEF','CUC','FLM','PPS','Label','tempRDCost','bestRDCost'
Y = XY[:,[7]]
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=splits, random_state=seed)
cost=x_train[:,[input_dim,input_dim+1]]
x_train=x_train[:,0:input_dim]
x_test=x_test[:,0:input_dim]

model = Sequential()
inputShape=(input_dim,)
model.add(Input(shape=inputShape))
x = Dense(10,activation="sigmoid", kernel_initializer="RandomNormal", bias_initializer="RandomNormal")(model.output)
x = Dense(1,activation ="sigmoid", kernel_initializer="RandomNormal", bias_initializer="RandomNormal")(x)
model = Model(inputs=[model.input],outputs=x)
model.compile(loss="mse",optimizer=opt,metrics=['acc'])

y_train_flatten = y_train.flatten()
class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train_flatten), y=y_train_flatten)
class_weights = dict(zip(np.unique(y_train_flatten),class_weights))
# cost_max = np.max(cost[:,0])
# cost_min = np.min(cost[:,0])
# cost_average = np.average(cost[:,0])
# sample_weightss = np.array((cost[:,0]-cost_min)/(cost_max-cost_min))
# sample_weightss = np.array(cost[:,0]/cost_average)
sample_num=np.size(y_train,0)
cost_sum=0
cost_num=0
cost_difference = []
for sample in np.concatenate([cost,y_train],axis=1):
    cost_difference_value = sample[0]-sample[1]
    if (sample[2]==0)&(cost_difference_value!=0):
        cost_difference.append(0)
    elif (sample[2]==0)&(cost_difference_value==0):
        cost_difference.append(1)
    elif (sample[2]==1)&(cost_difference_value<=0):
        cost_difference.append(0)
    else:
        cost_difference.append(cost_difference_value)
        cost_sum+=cost_difference_value
        cost_num+=1
sample_weights = np.array(cost_difference)
cost_average=cost_sum/cost_num
for i in range(sample_num):
    if (y_train[i]==1)&(sample_weights[i]!=0):
        sample_weights[i]=sample_weights[i]/cost_average
    if sample_weights[i]>1:
        sample_weights[i]=1
    elif sample_weights[i]<0:
        sample_weights[i]=0

history = model.fit(x=[x_train],y=y_train, validation_data=([x_test], y_test), 
                    epochs=epochs, batch_size=batchs, class_weight=class_weights, sample_weight=sample_weights)

model.save_weights(r'revision/geo_model_noPPS_withsamplewight.h5')
eval_model=[]
eval_model.append(model.evaluate([x_test], y_test)[1])
print("\nTest Accuracy: %.4f" % eval_model[0])

plt.plot(history.history['loss'],color='r')
plt.plot(history.history['val_loss'],color='g')
plt.plot(history.history['acc'],color='b')
plt.plot(history.history['val_acc'],color='k')
plt.title('Learning curve (Geometry)')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train_loss', 'test_loss','train_acc', 'test_acc'], loc='upper left',bbox_to_anchor=(0,-0.3))
plt.savefig('FeaturesPlots/P_GeoTrainingCurve.jpg', bbox_inches='tight', dpi=1280)
plt.show()

import pickle
with open('revision/geo_model_noPPS_withsamplewight.txt', 'wb') as file_txt:
    pickle.dump(history.history, file_txt)
    
np.set_printoptions(suppress=True)

a_weight1=model.get_weights()[0]
a_bias1=model.get_weights()[1]
a_weight2=model.get_weights()[2]
a_bias2=model.get_weights()[3]
# a_weight3=model.get_weights()[4]
# a_bias3=model.get_weights()[5]


print("\na_weight1: ")
for a in a_weight1:
    for b in a:
        print(b,end=",")
        
print("\n\na_bias1: ")
for a in a_bias1:
        print(a,end=",")
        
print("\n\na_weight2: ")
for a in a_weight2:
    for b in a:
        print(b,end=",")
        
print("\n\na_bias2: ")
for a in a_bias2:
        print(a,end=",")


# In[4]:


import pandas as pd
import numpy as np
import os
from random import shuffle
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation, Dropout, Dense, Flatten, Input
from tensorflow.keras.models import Model
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import concatenate
from keras.utils.vis_utils import plot_model
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import numpy.random
import argparse
import locale
import os

seed = 246

# model-compile parameter sets
model_metrics = 'acc'
epochs = 300
batchs = 128
splits = 0.2
lr        = 1e-5
input_dim = 5
opt = Adam(learning_rate=lr,weight_decay=1e-5/128)

concatenated_df=pd.read_csv("extraFeatures_Geo.csv", header=None)
XY = concatenated_df.values
for i in range(10):
    np.random.shuffle(XY)
X = XY[:,[0,1,3,5,6,8,9]]## 'MPD','CBF','CUD','OEF','CUC','FLM','PPS','Label','tempRDCost','bestRDCost'
Y = XY[:,[7]]
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=splits, random_state=seed)
cost=x_train[:,[input_dim,input_dim+1]]
x_train=x_train[:,0:input_dim]
x_test=x_test[:,0:input_dim]

model = Sequential()
inputShape=(input_dim,)
model.add(Input(shape=inputShape))
x = Dense(10,activation="sigmoid", kernel_initializer="RandomNormal", bias_initializer="RandomNormal")(model.output)
x = Dense(1,activation ="sigmoid", kernel_initializer="RandomNormal", bias_initializer="RandomNormal")(x)
model = Model(inputs=[model.input],outputs=x)
model.compile(loss="mse",optimizer=opt,metrics=['acc'])

y_train_flatten = y_train.flatten()
class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train_flatten), y=y_train_flatten)
class_weights = dict(zip(np.unique(y_train_flatten),class_weights))
# cost_max = np.max(cost[:,0])
# cost_min = np.min(cost[:,0])
# cost_average = np.average(cost[:,0])
# sample_weightss = np.array((cost[:,0]-cost_min)/(cost_max-cost_min))
# sample_weightss = np.array(cost[:,0]/cost_average)
sample_num=np.size(y_train,0)
cost_sum=0
cost_num=0
cost_difference = []
for sample in np.concatenate([cost,y_train],axis=1):
    cost_difference_value = sample[0]-sample[1]
    if (sample[2]==0)&(cost_difference_value!=0):
        cost_difference.append(0)
    elif (sample[2]==0)&(cost_difference_value==0):
        cost_difference.append(1)
    elif (sample[2]==1)&(cost_difference_value<=0):
        cost_difference.append(0)
    else:
        cost_difference.append(cost_difference_value)
        cost_sum+=cost_difference_value
        cost_num+=1
sample_weights = np.array(cost_difference)
cost_average=cost_sum/cost_num
for i in range(sample_num):
    if (y_train[i]==1)&(sample_weights[i]!=0):
        sample_weights[i]=sample_weights[i]/cost_average
    if sample_weights[i]>1:
        sample_weights[i]=1
    elif sample_weights[i]<0:
        sample_weights[i]=0

history = model.fit(x=[x_train],y=y_train, validation_data=([x_test], y_test), 
                    epochs=epochs, batch_size=batchs, class_weight=class_weights)

model.save_weights(r'revision/geo_model_allFeatures_nosamplewight.h5')
eval_model=[]
eval_model.append(model.evaluate([x_test], y_test)[1])
print("\nTest Accuracy: %.4f" % eval_model[0])

plt.plot(history.history['loss'],color='r')
plt.plot(history.history['val_loss'],color='g')
plt.plot(history.history['acc'],color='b')
plt.plot(history.history['val_acc'],color='k')
plt.title('Learning curve (Geometry)')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train_loss', 'test_loss','train_acc', 'test_acc'], loc='upper left',bbox_to_anchor=(0,-0.3))
plt.savefig('FeaturesPlots/P_GeoTrainingCurve.jpg', bbox_inches='tight', dpi=1280)
plt.show()

import pickle
with open('revision/geo_model_allFeatures_nosamplewight.txt', 'wb') as file_txt:
    pickle.dump(history.history, file_txt)
    
np.set_printoptions(suppress=True)

a_weight1=model.get_weights()[0]
a_bias1=model.get_weights()[1]
a_weight2=model.get_weights()[2]
a_bias2=model.get_weights()[3]
# a_weight3=model.get_weights()[4]
# a_bias3=model.get_weights()[5]


print("\na_weight1: ")
for a in a_weight1:
    for b in a:
        print(b,end=",")
        
print("\n\na_bias1: ")
for a in a_bias1:
        print(a,end=",")
        
print("\n\na_weight2: ")
for a in a_weight2:
    for b in a:
        print(b,end=",")
        
print("\n\na_bias2: ")
for a in a_bias2:
        print(a,end=",")


# In[ ]:




