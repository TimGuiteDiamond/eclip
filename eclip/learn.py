#one idea is to use the package Keras. this works in a modular manner- and is
#was used for the EM ones
'''
Learn is the module that creates and trains a new CNN model using keras. The
module compiles the model  and fits it, using a set of parameters and keras functions.


Other classes and funcitons called in the module: 

  From eclip.utils.datamanip
   * input_training_image
   * str2bool
  From eclip.utils.visu
   * ConfusionMatrix
   * fit_plot
   * plot_test_results
  From eclip.utils.modelmanip
   * MapModel
   * exp_json
   * exp_yaml

Calling learn results in the creation of: 

 * model.json
 * model.yaml
 * model.h5
 * predict------_-.txt
 * log------_-.txt
 * Plot------_-.png
 * CnfM------_-.png
 * Complot------_-.png

|

Arguments
^^^^^^^^^^^^

The command line arguments for learn are as follows. 

* **--batchsize:** The batchsize for learning. Default: 64 
* **--epochs:** The epochs for learning. Default: 300
* **--loss:** The loss equation for learning. Default: categorical_crossentropy
* **--lrt:** The starting learning rate. Default: 0.0001
* **--para:** Boolean, whether using data parallisation. Default: True
* **--trial:** The starter for trial number. Default: 1
* **--date:** Date to appear on saved names. Default: current date
* **--drt:** Rate of decay of learning rate. Default: 1e-8
* **--raw:** Boolean, if learning on just heavy atom positions. Default: False
* **--insh:** The input shape of the images, as a list of dimensions. Default: [201,201,3]
* **--out:** Directory location for the states etc to be saved. Default: /dls/science/users/ycc62267/eclip/eclip/paratry1
* **--th:** Threshold to round predictions up from. Default: 0.5
* **--nmb:** The number of images per protein per axis. Default: 10
* **--ning:** Boolean, true if adding name to date. Default: False
* **--name:** Name to add to date if ning is true. Default: ''
* **--db:** The sqlite database. Default: /dls/science/users/ycc62267/metrix_db/metrix_db.sqlite'

|

Functions in module
^^^^^^^^^^^^^^^^^^^^^
|

'''

import numpy as np
import matplotlib.pyplot  as plt
plt.switch_backend('agg')
np.random.seed(123) # for reproduciblility
import os

import logging

from sklearn.utils import class_weight

import keras
from keras.utils import multi_gpu_model

from eclip.utils.datamanip import input_training_images, str2bool
from eclip.utils.visu import ConfusionMatrix, fit_plot, plot_test_results
from eclip.utils.modelmanip import MapModel, exp_json, exp_yaml


################################################################################################

def main(batchsize = 64, 
        epochs = 300, 
        lossequation = 'categorical_crossentropy', 
        learningrate= 0.0001,
        parallel= True, 
        trialnum=1,
        date='150818',
        decayrt=1e-8,
        Raw=False,
        inputshape=[201,201,3],
        outputdirectory  = '/dls/science/users/ycc62267/eclip/eclip/paratry1',
        sqlitedb= '/dls/science/users/ycc62267/metrix_db/metrix_db.sqlite',
        number = 10,
        thresh = 0.5):

  '''
  Main is the overall function of learn.

  |

  '''

  
  #: A log file is opened to record the starting parameters and the results of the learning.
  logging.info('\n'.join(['Running learn.py for parameters: '
                          'batchsize: %s', 
                          'epochs : %s',
                          'loss: %s',
                          'learning rate: %s',
                          'parallelization: %s',
                          'decay rate: %s',
                          'Raw: %s',
                          'thresh: %s',
                          'fraction train: 0.7']) %(batchsize,
                                        epochs,
                                        lossequation,
                                        learningrate,
                                        parallel,
                                        decayrt,
                                        Raw,
                                        thresh))
 
 

  #: learn calls inputTrainingImages to select image data and change into the correct form.
  x_train,y_train,x_test,y_test=input_training_images(
                            database=sqlitedb,
                            input_shape=inputshape,
                            fractionTrain=0.7,
                            raw=Raw, 
                            number=number,
                            trialsplitloc = outputdirectory)


  #Printing the data dimensions to logfile
  onestrain=sum(y_train)
  onestest=sum(y_test)
  logging.info('''Train: %s \n Test: %s'''%(str(onestrain),str(onestest)))


  #: Class weights are used to counter the effect of an imbalanced input dataset
  y_ints=[y.argmax() for y in y_train]
  classweights=dict(enumerate(
                          class_weight.compute_class_weight('balanced',
                                                            np.unique(y_ints),y_ints)))
  logging.info('Class weights: %s \ninput shape: %s'%(classweights,inputshape))


  #building model
  model = MapModel()
  if Raw:
    model.create_custom1(input_shape1=inputshape)
  else:
    model.create_custom2(input_shape2=inputshape)
  

  #optimiser could be opt=keras.optimizers.SGD(lr=0.0001), but should check
  opt = keras.optimizers.Adam(lr=learningrate,decay=decayrt)
  

  #: learn can be run on a 4 GPU cluster by setting parallel to True
  if parallel:
    model=multi_gpu_model(model,gpus=4)
  else:
    model=model


  #compiling the model
  model.compile(loss = lossequation,
                optimizer = opt,
                metrics = ['accuracy'])
  print('model compiled')
    

  #fit the model on the training data- this is the bit that trains the model -
  #look at the need for augmenting the data- we have a lot so it may not be
  #necessary
  history=model.fit(x_train,
                    y_train,
                    class_weight=classweights,
                    batch_size =batchsize, 
                    epochs=epochs, 
                    verbose=1,
                    validation_data=(x_test,y_test),
                    callbacks=[])
  
  #: The validation loss and training loss is plotted against epochs 
  #: likewise with the validation accuracy and training accuracy
  outputplot=os.path.join(outputdirectory,'Plot'+date+'_'+str(trialnum)+'.png')
  fit_plot(history.history['loss'],
          history.history['val_loss'],
          history.history['acc'],
          history.history['val_acc'],
          outputplot)
 

  #: Predicting the score of the test data
  prediction=model.predict(x_test,batch_size=batchsize,verbose=1)


  #:saveing model as .json file
  outfilejson= os.path.join(outputdirectory,'model.json')
  exp_json(outfilejson,model)


  #:saving model as .yaml file 
  outfileyaml = os.path.join(outputdirectory,'model.yaml')
  exp_yaml(outfileyaml,model)


  #:saving weigths to .h5 file
  weightsout = os.path.join(outputdirectory,'model.h5')
  model.save_weights(weightsout)


  #Evaluating the success of the model
  loss,acc = model.evaluate(x_test,y_test,verbose=1)
  print('loss is: %s'%loss)
  print('accuracy is: %s'%acc)
 

  logging.info('''loss is: %s \naccuracy is: %s \nPredictions: \n'''%(loss,acc))
  logging.info(prediction)
  
   
  #writing predictions
  outpred=os.path.join(outputdirectory,'predict'+date+'_'+str(trialnum)+'.txt')
  y_pred=ConfusionMatrix.print_convstats(prediction,outpred,y_test)
   
  
  model.summary(print_fn=lambda x: logging.info(x+'\n'))

  
  #: A confusion matrix is plotted to visually assess the successes of the model
  cnfout=os.path.join(outputdirectory,'CnfM'+date+'_'+str(trialnum)+'.png')
  ConfusionMatrix(y_test,y_pred,cnfout)


  #:A plot of predictions and true values of test set
  splotout=os.path.join(outputdirectory,'Compplot'+date+'_'+str(trialnum)+'.png')
  y_test=list(y_test[:,1])



  y_p=list(prediction[:,1])
  plot_test_results(y_test,y_p,splotout, thresh)

########################################################################################  

def run():

  '''
  run allows learn to be called from the command line. 

  '''

  import argparse
  import time

  start_time = time.time()
  date = str(time.strftime("%d%m%y"))

  parser = argparse.ArgumentParser(description = 'command line argument')
  parser.add_argument('--batchsize',
                      dest = 'batchs',
                      type = int,
                      help = 'The batch size to run',
                      default = 64)
  parser.add_argument('--epochs', 
                      dest = 'epochs', 
                      type = int,
                      help = 'Number of epochs to run',
                      default = 300)
  parser.add_argument('--loss',
                      dest = 'loss', 
                      type = str,
                      help = 'The name of loss equation to use',
                      default = 'categorical_crossentropy')
  parser.add_argument('--lrt', 
                      dest = 'learnrt',
                      type = float,
                      help = 'The starting learning rate',
                      default = 0.0001)
  parser.add_argument('--para',
                      dest='para',
                      type = str2bool,
                      help = 'boolean: True if data parallisation.',
                      default = True)
  parser.add_argument('--trial', 
                      dest = 'trial',
                      type = int,
                      help = 'The starting trial number',
                      default = 1)
  parser.add_argument('--date',
                      dest = 'date',
                      type = str,
                      help = 'Date to appear on saved names',
                      default = date)
  parser.add_argument('--drt',
                      dest = 'decayrt',
                      type = float,
                      help = 'Rate of decay of learning rate',
                      default = 1e-8)
  parser.add_argument('--raw',
                      dest = 'raw',
                      type = str2bool,
                      help = 'boolean: True if using .pha files',
                      default = False)
  parser.add_argument('--th',
                      dest = 'thresh',
                      type = float,
                      help = 'Value to round from',
                      default = 0.5)
  parser.add_argument('--insh',
                      dest = 'inshape',
                      type = list,
                      help = 'list of image dimensions',
                      default = [201,201,3])
  parser.add_argument('--out', 
                      dest = 'out',
                      type = str,
                      help = 'output directory for saved files',
                      default =
                      '/dls/science/users/ycc62267/eclip/eclip/paratry1')
  parser.add_argument('--nmb',
                      dest = 'nmb',
                      type = int,
                      help = 'number of images per axis per protein',
                      default = 10)
  parser.add_argument('--ning',
                      dest = 'ning',
                      type = str2bool,
                      help = 'boolean, true if name should be added',
                      default = False)
  parser.add_argument('--name',
                      dest = 'name',
                      type = str,
                      help = ' name to add to date',
                      default = '')
  parser.add_argument('--db',
                      dest = 'db',
                      type = str,
                      help = 'location of sqlite database',
                      default = '/dls/science/users/ycc62267/metrix_db/metrix_db.sqlite')
                      

  args=parser.parse_args()

  raw = args.raw
  trialnum = args.trial
  date = args.date
  ning = args.ning
  name=args.name

  if raw:
    date = date + 'raw'
  if ning:
    date = date + name
  outdir = args.out

  while os.path.exists(os.path.join(outdir,'log'+date+'_'+str(trialnum)+'.txt')):
    trialnum+=1
  
  logfile = os.path.join(outdir, 'log'+date+'_'+str(trialnum)+'.txt')
  logging.basicConfig(filename = logfile, level = logging.DEBUG)
  logging.info('Running learn.py')

  main(args.batchs, 
       args.epochs, 
       args.loss,
       args.learnrt,
       args.para,
       trialnum,
       date,
       args.decayrt,
       raw,
       args.inshape,
       outdir,
       args.db,
       args.nmb,
       args.thresh)

  logging.info('Finished -- %s seconds --'%(time.time()-start_time))


if __name__=="__main__":
  run()


