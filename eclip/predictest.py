################
''' Code to predict new scores for full maps and check this against the true
scores'''
##############
#from keras.models import predict
import sqlite3
import os

from utils.modelmanip import loadjson
from utils.datamanip import importData, avefirstscore, roundfirstscore,trialsplitlist
from utils.visu import plot_test_results, ConfusionMatrix
################

def main(jsonfile ='/dls/science/users/ycc62267/eclip/eclip/paratry/model.json',
        weights_file ='/dls/science/users/ycc62267/eclip/eclip/paratry/model.h5', 
        fileloc = '/dls/mx-scratch/ycc62267/ingfdr/blur2_5_maxminbox/', 
        protein_split_list ='/dls/science/users/ycc62267/eclip/eclip/trialsplit.txt',
        input_shape =[201,201,3],
        method = 'average first',
        outdir ='/dls/science/users/ycc62267/eclip/eclip/paratry/',
        date ='150818',
        trial_num=1,
        threshould = 0.5):
  '''
  predic loads a model from a json file and weights from a weights file and uses
  this to predict the score for a map - already in image form. The
  predictions are then saved to a txt file and added to the sqlite database. 

  These predictions are then compared to the true scores and a set of statistics
  is produced in a txt file alongside a plot of these results. 

  **Arguments:**

  * **jsonfile:** The location to find model. default: jsonfile='/dls/science/users/ycc62267/eclip/eclip/paratry/model.json'
  * **weights_file:** The loctaion to find model weights. default: weights_file ='/dls/science/users/ycc62267/eclip/eclip/paratry/model.h5'
  * **fileloc:** location to find files. default: fileloc = '/dls/mx-scratch/ycc62267/imgfdr/blur2_5_maxminbox/'
  * **protein_split_list:** text file with list of proteins to use. default: protein_split_list= '/dls/science/users/ycc62267/eclip/eclip/trialsplit.txt'
  * **inputshape:** The input shape of the images. default: inputshape = [201,201,3]
  * **method:** option parameter for how to round. Can be either 'average first' or 'round first'. default: method = 'average first'
  * **outdir:** output predictions file. default: '/dls/science/users/ycc62267/eclip/eclip/paratry/'
  * **date:** date. default: date = '100818'
  * **trial_num:** default: trial_num = 5
  * **threshold:** default: threshold= 0.5

  '''

  while os.path.exists(os.path.join(outdir,'map_results_plot'+date+'_'+str(trial_num)+'.png')):
    trial_num+=1
  
  protein_split = trialsplitlist(protein_split_list)
  outfile = os.path.join(outdir,'newpredic'+date+'_'+str(trial_num)+'.txt')
  resoutfile= os.path.join(outdir,'map_results_plot'+date+'_'+str(trial_num)+'.png')
  cnfout=os.path.join(outdir,'map_cnf'+date+'_'+str(trial_num)+'.png')
    #######################################################################
  
  x, name, proteins = importData(datafileloc=fileloc,proteinlist=protein_split,
  input_shape=inputshape)
  
  model = loadjson(jsonfile,weights_file)
  prediction = model.predict(x)
   
  
  #round first
  if method == 'round first':
    scores, preds, ones, zeros = roundfirstscore(proteins,name, prediction,outfile,threshold)
  if method == 'average first':
    scores, preds, ones, zeros = avefirstscore(proteins,name, prediction,outfile,threshold)
  else:
    RuntimeError('Not a valid method')
  

  # Selecting true scores from database and creating a list of various
  # parameters
  conn = sqlite3.connect('/dls/science/users/ycc62267/metrix_db/metrix_db.sqlite')
  cur=conn.cursor()

  n_tp=0
  n_tn=0
  n_fp=0
  n_fn=0 
  errors =[]
  y_true = []
  
  for i  in range(0,len(proteins)):
    protein = proteins[i]
    score = scores[i]
    pred = preds[i]

    if protein.endswith('_i'):
      protein_name = protein[:-2]
    else:
      protein_name = protein

    cur.execute('''
      SELECT id FROM PDB_id WHERE PDB_id.pdb_id="%s" ''' %(protein_name))
    protein_id = cur.fetchone()[0] 
 
    if protein.endswith('_i'):
      cur.execute('''
        UPDATE Phasing SET (ep_score_i, ep_confidence_i)=(%s,%s) WHERE
        Phasing.pdb_id_id = "%s"''' %(score,pred,protein_id))
      cur.execute('''
        SELECT ep_success_i FROM Phasing WHERE Phasing.pdb_id_id =
        "%s"'''%(protein_id))
      truescore= list(cur.fetchall()[0])[0]
    else:
      cur.execute('''
        UPDATE Phasing SET (ep_score_o,ep_confidence_o)=(%s,%s) WHERE
        Phasing.pdb_id_id = "%s"''' %(score,pred,protein_id))
      cur.execute('''
        SELECT ep_success_o FROM Phasing WHERE Phasing.pdb_id_id =
        "%s"'''%(protein_id))
      truescore= list(cur.fetchall()[0])[0]
      
    if truescore ==None:
      continue
    truescore=int(truescore)
    print('score: %s truescore: %s'%(score,truescore))
    y_true.append(truescore)
   
    if truescore == 1 and score ==1:
      n_tp+=1
    elif truescore ==1 and score ==0:
      n_fn+=1
      errors.append(str(protein_id)+': '+protein)
    elif truescore ==0 and score ==1:
      n_fp+=1
      errors.append(str(protein_id)+': '+protein)
    else:
      n_tn+=1
  conn.commit()
  conn.close()
  print('update EP_success successful')
  acc = (n_tp+n_tn)/(n_fp+n_fn+n_tp+n_tn)
   
  #writing to text file 
  text= open(outfile,'a')
  text.write('''\nNumber of true positives:%s\n
              Number of true negatives: %s\n
              Number of false positives: %s\n
              Number of false negatives: %s\n'''
              %(n_tp,n_tn,n_fp,n_fn))
  text.write('accuracy = %s'%acc)
  for i in errors:
    text.write(i+'\n')

  # plot test results
  plot_test_results(y_true,preds,resoutfile)
  
  # plot Confusion matrix
  ConfusionMatrix(y_true,scores,cnfout)
  
if __name__=='__main__':
  main()
