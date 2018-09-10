###############
'''
Functions and classes to build the keras model

|

'''
#############
import keras
from keras import initializers, regularizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, BatchNormalization
from keras.utils import np_utils
from keras.models import model_from_json

import logging
################
#creating a model
class MapModel(Sequential):

  '''
  MapModel is a class to build a Sequential keras model


  
  '''

  def __init__(self):
    Sequential.__init__(self)

  def create_custom1(self,input_shape1):
    '''
    
    create_custom1 builds up the layers in the model for raw data.

    **Arguments for createCustom1:**

    * **self:**
    * **input_shape1:** The shape of the input images as arrays
   

    **Output for createCustom1:**

    * **self:** A sequential model is built as a class



    '''

    logging.info('\nModel setup:')
    self.add(Convolution2D(32,(3,3),kernel_initializer=initializers.he_normal(),
                            activation='relu',padding='same',input_shape=input_shape1,name='Conv_1'))
    logging.info('''Convolution2D(32,(3,3),kernel_initializer=initializers.he_normal(), 
                    activation='relu',padding='same',input_shape=input_shape1,name='Conv_1')''')
    
    self.add(MaxPooling2D(pool_size=(2,2),name='Pool_1'))
    logging.info("MaxPooling2D(pool_size=(2,2),name='Pool_1')")
    
    self.add(BatchNormalization())
    logging.info("BatchNormalization()\n")
    
    self.add(Convolution2D(32,(3,3),activation='relu',padding='same',name='Conv_2'))
    logging.info("Convolution2D(32,(3,3),activation='relu',padding='same',name='Conv_2')")
    
    self.add(MaxPooling2D(pool_size=(2,2),name='Pool_2'))
    logging.info("MaxPooling2D(pool_size=(2,2),name='Pool_2')")
    
    self.add(BatchNormalization())
    logging.info("BatchNormalization()")
    
    self.add(Flatten())
    logging.info("Flatten()")
    
    self.add(Dropout(0.75))
    logging.info("Dropout(0.75)")
    
    self.add(BatchNormalization())
    logging.info("BatchNormalization()")
    
    self.add(Dense(64,activation='relu',name='Dense_1',kernel_regularizer=regularizers.l2(0.15)))
    logging.info("Dense(64,activation='relu',name='Dense_1',kernel_regularizer=regularizers.l2(0.15))")
    
    self.add(Dense(2,activation='softmax',name='predictions'))
    logging.info("Dense(2,activation='softmax',name='predictions')\n")
    


  def create_custom2(self,input_shape2):
    '''
    
    create_custom2 builds up the layers in the model for processed data.

    **Arguments for createCustom2:**

    * **self:**
    * **input_shape2:** The shape of the input images as arrays
   
    **Output for createCustom2:**

    * **self:** A sequential model is built as a class

    |

    '''

    logging.info('\n Model setup:')
    self.add(Convolution2D(32,(3,3),kernel_initializer=initializers.he_normal(),
                            activation='relu',padding='same',input_shape=input_shape2,name='Conv_1'))
    logging.info('''Convolution2D(32,(3,3),kernel_initializer=initializers.he_normal(),
                   activation='relu',padding='same',input_shape=input_shape2,name='Conv_1')''')
    
    self.add(MaxPooling2D(pool_size=(2,2),name='Pool_1'))
    logging.info("MaxPooling2D(pool_size=(2,2),name='Pool_1')")
    
    self.add(BatchNormalization())
    logging.info("BatchNormalization()")
    
    self.add(Convolution2D(64,(3,3),activation='relu',padding='same',name='Conv_2'))
    logging.info("Convolution2D(64,(3,3),activation='relu',padding='same',name='Conv_2')")
    
    self.add(MaxPooling2D(pool_size=(2,2),name='Pool_2'))
    logging.info("MaxPooling2D(pool_size=(2,2),name='Pool_2')")
    
    self.add(BatchNormalization())
    logging.info("BatchNormalization()")
    
    self.add(Flatten())
    logging.info("Flatten()")
    
    self.add(Dropout(0.75))
    logging.info("Dropout(0.75)")
    
    self.add(BatchNormalization())
    logging.info("BatchNormalization()")
    
    self.add(Dense(128,activation='relu',name='Dense_1',kernel_regularizer=regularizers.l2(0.15)))
    logging.info("Dense(128,activation='relu',name='Dense_1',kernel_regularizer=regularizers.l2(0.15))")
    
    self.add(Dense(2,activation='softmax',name='predictions'))
    logging.info("Dense(2,activation='softmax',name='predictions')")
    


def exp_json(outfile,model):
  '''
  exp_json is a function that exports a model to a .json format file

  **Arguments for expjson:**

  * **outfile:** File location for the model to be saved to
  * **model:** The model to be saved

  |

  '''
  json_string=model.to_json()
  with open(outfile,'w') as outfile:
    outfile.write(json_string)
  outfile.close()

def exp_yaml(outfile,model):
  '''
  exp_yaml is a function that exports a model to a .yaml format file

  **Arguments for expjson:**

  * **outfile:** File location for the model to be saved to.
  * **model:** The model to be saved 
  
  |

  '''
  yaml_string=model.to_yaml()
  with open(outfile,'w') as outfile:
    outfile.write(yaml_string)
  outfile.close()


def load_json(jsonfile,weights_file):
  '''
  load_json is a function that loads a model from a .json file and populates it
  with weights from a .h5 file

  **Arguments for loadjson:**

  * **jsonfile:** File location for the .json file
  * **weights_file:** File location for the .h5 file

  **Outputs of loadjson:**

  * **model:**

  |
  
  '''
  json_file=open(jsonfile,'r')
  loaded_model_json=json_file.read()
  json_file.close()
  model=model_from_json(loaded_model_json,custom_objects={"MapModel":Sequential})
  model.load_weights(weights_file)
  print('loaded model')
  return model



