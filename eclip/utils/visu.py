#######################
'''
Functions for visualising the progess of the network

|

'''
#########################
import numpy as np
import matplotlib as mpl
mpl.rc("savefig", dpi = 200)
import matplotlib.pyplot as plt
plt.switch_backend('agg')

from sklearn.metrics import confusion_matrix
import itertools
import logging
#########################

class ConfusionMatrix():
  '''
  This class creates a confusion matrix plot for y_test and y_pred and saves it
  in outputname 
  
  **Arguments for ConfusionMatrix class:**
  
  * **y_test:** A list of the true scores
  * **y_pred:** A list of the predicted scores
  * **outputname:** File location for the Confusion matrix image
  
  '''
  
  def __init__(self,y_test,y_pred,outputname):

      if type(y_test)!= list:
        y_t=y_test[:,1].tolist()
      else:
        y_t=y_test

      cnf_matrix=confusion_matrix(y_t,y_pred)
      np.set_printoptions(precision=2)

      class_names=['Bad','Good']

      fig=plt.figure()
      self.plot_confusion_matrix(cnf_matrix,classes=class_names,title='Confusion Matrix')
      fig.savefig(outputname)
      plt.close()

    
  def plot_confusion_matrix(self,cm,classes,normalize=False,title='Confusion Matrix',cmap=plt.cm.Blues):
    '''
    plot_confusion_matrix produces the confusion matrix image

    **Arguments for plot_confusion_matrix:**

    * **self:** 
    * **cm:** confusion matrix as an array
    * **classes:** a list of the class names
    * **normalize:** a boolean stating whether to produce a normalised plot or not
    * **title:** Title for the plot
    * **cmap:** controls the color mape used


    '''
    
    if normalize:
      cm=cm.astype('float')/cm.sum(axis=1)[:,np.newaxis]
    plt.imshow(cm,interpolation='nearest',cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks=np.arange(len(classes))
    plt.xticks(tick_marks,classes,rotation=45)
    plt.yticks(tick_marks,classes)

    fmt='.2f' if normalize else 'd'

    thresh = cm.max()/2
    for i,j in itertools.product(range(cm.shape[0]),range(cm.shape[1])):
      plt.text(j,i,format(cm[i,j],fmt),
                horizontalalignment="center",
                color = "white" if cm[i,j]>thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

  def print_convstats(prediction,outfile,y_test):
    '''
    print_convstats writes to a logfile the values for true negatives, true
    positives, false negatives and false positives and prints these values to
    the terminal. It also converts the predictions into a list
    to be used to produce the confusion matrix. 

    **Arguments for printConvstats:**

    * **prediction:** The prediction produced through keras predict
    * **outfile:** File location to write prediction into
    * **y_test:** True scores for test data

    **Outputs for printConvstats:**

    * **outfile:** Saved txt file of predictions
    * **y_pred:** A list of the predicted scores
    
    |

    '''
    outfile=open(outfile,'w')
    n_tn=0
    n_tp=0
    n_fn=0
    n_fp=0
    y_pred=[]

    for i in range(len(prediction)):
      outfile.write(str(prediction[i])+";"+str(y_test[i])+'\n')
      if prediction[i][0] > 0.5 and y_test[i][0]>0.5:
        n_tn+=1
        y_pred.append(0)
      elif prediction[i][0]>0.5 and y_test[i][1]>0.5:
        n_fn+=1
        y_pred.append(0)
      elif prediction[i][1]>0.5 and y_test[i][1]>0.5:
        n_tp+=1
        y_pred.append(1)
      elif prediction[i][1]>0.5 and y_test[i][0]>0.5:
        n_fp+=1 
        y_pred.append(1)
    outfile.close()
    print('Number of ture negatives = ',n_tn)
    print('Number of false negatives = ',n_fn)
    print('Number of true positives = ', n_tp)
    print('Number of flase positives = ',n_fp)
   
    logging.info('Number of true negatives = %s\n'%n_tn)
    logging.info('Number of false negatives = %s\n'%n_fn)
    logging.info('Number of true positives = %s\n'%n_tp)
    logging.info('Number of false positives = %s\n'%n_fp)

    return y_pred


  
def fit_plot(loss,val_loss,acc,val_acc,outputfile):
  '''
  fit_plot is a function that plots a figure of two graphs. One graph of
  validation and training loss against epochs and one graph of validation and
  training accuracy against epochs.

  **Arguments for fitplot:**
  * **loss:** The training loss from keras.fit
  * **val_loss:** The validation loss from keras.fit
  * **acc:** The trainging accuracy from keras.fit
  * **val_acc:** The validation accuracy from keras.fit
  * **outputfile:** File location for the saved image

  |
  
  '''
  fig=plt.figure()
  plt.subplot(2,2,1)
  plt.plot(loss)
  plt.plot(val_loss)
  plt.title('a) Model Loss')
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend(['Train','Validation'],loc='upper right')
  plt.subplot(2,2,2)
  plt.plot(acc)
  plt.plot(val_acc)
  plt.title('b) Model Accuracy')
  plt.ylabel('Accuracy')
  plt.xlabel('Epoch')
  plt.legend(['Train','Validation'],loc='lower right')
  plt.tight_layout()
  fig.savefig(outputfile)
  plt.close(fig)



def plot_test_results(y_test, y_pred,outputfile, thresh):

  '''
  plot_test_results takes the predicted and expected scores for data and outputs
  a plot of the comparison between these

  **Arguments for plot_test_results:***

  * **y_test:** This is a list of the true scores for the data
  * **y_pred:** This is a list of the predicted (percentage) scores of the data
  * **outputfile:** This is the location to save the plot as a .png file
  * **thresh:** The value to round up from

  **Outputs of plot_test_results:**

  * **image:** The image of the plot saved in the location specified by outputfile
  
  |
  
  '''

  y_test = [ y_test for _,y_test in sorted(zip(y_pred,y_test))]
  y_pred= sorted(y_pred)
  x=np.arange(len(y_test))




  truex=[]
  falsex=[]
  
  for i in range(0,len(y_test)):
    if y_pred[i] > thresh:
      y_round = 1
    else:
      y_round = 0
    if y_test[i] == y_round:
      truex.append(i)
    else:
      falsex.append(i)

  truey=[y_test[i] for i in truex]
  falsey=[y_test[i] for i in falsex]

  fig=plt.figure()
  plt.plot(x,y_pred,truex,truey,'g.',falsex,falsey,'r.')
  plt.title('Prediction and True Value against Index')
  plt.xlabel('Index')
  plt.ylabel('Score')
  plt.legend(['Predictions','True Value Correct','True Value Incorrect'],loc=6)
  plt.tight_layout()
  fig.savefig(outputfile)
  plt.close(fig)
