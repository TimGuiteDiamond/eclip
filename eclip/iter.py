import numpy as np
import matplotlib.pyplot as plt
import os
import random
np.random.seed(123)

from keras import activations 
from keras.models import model_from_json
import keras
from keras.models import Sequential

def normalarray(n_array):
  maxi=np.amax(n_array)
  mini=np.amin(n_array)
  norm_array=(n_array-mini)/(maxi-mini)
  return norm_array

filenames=['/dls/mx-scratch/ycc62267/imgfdr/blur2_5_maxminbox/3MST/3MST/X/3MST_078_154X.png','/dls/mx-scratch/ycc62267/imgfdr/blur2_5_maxminbox/3MDP/3MDP_i/X/3MDP_i_001_0X.png']

x_t=np.array([normalarray(np.array(plt.imread(filename))).flatten()for filename in filenames])
x_test = x_t.reshape(2,201,201,3)
#model=load_model('/dls/science/users/ycc62267/eclip/eclip/model.h5')
json_file=open('/dls/science/users/ycc62267/eclip/eclip/model.json','r')
loaded_model_json=json_file.read()
json_file.close()
model=model_from_json(loaded_model_json,custom_objects={"mapModel":Sequential})
model.load_weights('/dls/science/users/ycc62267/eclip/eclip/model.h5')
print('loaded model')


pred=model.predict(x_test)

print(pred)

def plot_confusion_matrix(cm,classes,normalize=False,title=
