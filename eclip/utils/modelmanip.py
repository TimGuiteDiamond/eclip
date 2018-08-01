###############
'''funtions and classes to build the keras model'''
#############
import keras
from keras import initializers, regularizers
from keras.models import Sequential#, load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, BatchNormalization
from keras.utils import np_utils
#from keras.utils import to_catagorical, plot_model
#from keras.applications import vgg16
#from keras.utils import mulit_gup_model


from keras.models import model_from_json




################
#creating a model
class mapModel(Sequential):
  #have chosed a sequential model

  def __init__(self):
    Sequential.__init__(self)

  #def creatCustom1(self,input_shape1,logfile):
    #fill this one in once you are using it...

  def createCustom2(self,input_shape2,logfile):
    text=open(logfile,'a')
    text.write('\nModel setup:\n')

    self.add(Convolution2D(32,(3,3),kernel_initializer=initializers.he_normal(),activation='relu',padding='same',input_shape=input_shape2,name='Conv_1'))
    text.write("Convolution2D(32,(3,3),kernel_initializer=initializers.he_normal(),activation='relu',padding='same',input_shape=input_shape2,name='Conv_1')\n")
    self.add(MaxPooling2D(pool_size=(2,2),name='Pool_1'))
    text.write("MaxPooling2D(pool_size=(2,2),name='Pool_1')\n")
    self.add(BatchNormalization())
    text.write("BatchNormalization()\n")
    self.add(Convolution2D(32,(3,3),activation='relu',padding='same',name='Conv_2'))
    text.write("Convolution2D(32,(3,3),activation='relu',padding='same',name='Conv_2')\n")
    self.add(MaxPooling2D(pool_size=(2,2),name='Pool_2'))
    text.write("MaxPooling2D(pool_size=(2,2),name='Pool_2')\n")
    self.add(BatchNormalization())
    text.write("BatchNormalization()\n")
    self.add(Flatten())
    text.write("Flatten()\n")
    self.add(Dropout(0.75))
    text.write("Dropout(0.75)\n")
    self.add(BatchNormalization())
    text.write("BatchNormalization()\n")
    self.add(Dense(128,activation='relu',name='Dense_1',kernel_regularizer=regularizers.l2(0.15)))
    text.write("Dense(128,activation='relu',name='Dense_1',kernel_regularizer=regularizers.l2(0.15))\n")
    self.add(Dense(2,activation='softmax',name='predictions'))
    text.write("Dense(2,activation='softmax',name='predictions')\n")
    text.close()


##This needs some work, i dont really understand it and have not checked it yet
#class WeightsCheck(keras.callbacks.Callback):
#  def on_epoch_begin(self,batch,logs={}):
#    for layer in model.layers[16:17]:
#      print(layer.get_weights())
#  def on_epoch_end(self,batch,logs={}):
#    for layer in model.layers[16:17]):
#      print(layer.get_weights())

#export as json
def expjson(outfile,model):
  json_string=model.to_json()
  with open(outfile,'w') as outfile:
    outfile.write(json_string)
  outfile.close()

#export as yaml
def expyaml(outfile,model):
  yaml_string=model.to_yaml()
  with open(outfile,'w') as outfile:
    outfile.write(yaml_string)
  outfile.close()


#load model as json and fill with weights
def loadjson(jsonfile,weights_file):
  json_file=open(jsonfile,'r')
  loaded_model_json=json_file.read()
  json_file.close()
  model=model_from_json(loaded_model_json,custom_objects={"mapModel":Sequential})
  model.load_weights(weights_file)
  print('loaded model')
  return model



