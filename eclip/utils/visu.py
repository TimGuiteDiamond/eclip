#######################
'''Functions for visualising the progess of the network'''
#########################
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')

from sklearn.metrics import confusion_matrix
import itertools
#########################

class ConfusionMatrix():
  '''This class creates a confusion matrix plot for y_test and y_pred and saves it
  in outputname '''
  
  def __init__(self,y_test,y_pred,outputname):
      y_t=y_test[:,1].tolist()

      cnf_matrix=confusion_matrix(y_t,y_pred)
      np.set_printoptions(precision=2)

      class_names=[0,1]
      #fig=plt.figure()
      #self.plot_confusion_matrix(cnf_matrix,classes=class_names,normalize=True,title='Normalized Confusion Matrix')
      #fig.savefig(outputname)
      #plt.close()

      fig=plt.figure()
      self.plot_confusion_matrix(cnf_matrix,classes=class_names,title='Confusion Matrix')
      fig.savefig(outputname)
      plt.close()

    
  def plot_confusion_matrix(self,cm,classes,normalize=False,title='Confusion Matrix',cmap=plt.cm.Blues):
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

  def printConvstats(prediction,outfile,logfile,y_test):
    outfile=open(outfile,'w')
    text=open(logfile,'a')
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
    text.write('Number of true negatives = %s\n'%n_tn)
    text.write('Number of false negatives = %s\n'%n_fn)
    text.write('Number of true positives = %s\n'%n_tp)
    text.write('Number of false positives = %s\n'%n_fp)
    
    text.close()
    return y_pred


  
def fitplot(loss,val_loss,acc,val_acc,outputfile):
  fig=plt.figure()
  plt.subplot(2,2,1)
  plt.plot(loss)
  plt.plot(val_loss)
  plt.title('model train')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['train','validation'],loc='upper right')
  plt.subplot(2,2,2)
  plt.plot(acc)
  plt.plot(val_acc)
  plt.title('model train')
  plt.ylabel('acc')
  plt.xlabel('epoch')
  plt.legend(['train','validation'],loc='lower right')
  fig.savefig(outputfile)
  plt.close(fig)

