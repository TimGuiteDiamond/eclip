################
''' Code to predict new scores '''
##############
import keras
import sqlite3

from utils.modelmanip import loadjson
from utils.datamanip import importData, avefirstscore, roundfirstscore, trialsplitlist
################

def main():

  '''
  predic loads a model from a json file and weights from a weights file and uses
  this to predict the score for a new map - already in image form. the
  predictions are then saved to a txt file and added to the sqlite database. 
  '''

  jsonfile='/dls/science/users/ycc62267/eclip/eclip/paratry/model.json'
  weights_file ='/dls/science/users/ycc62267/eclip/eclip/paratry/model.h5'
  fileloc = '/dls/mx-scratch/ycc62267/imgfdr/blur2_5_maxminbox/'
  protein_split = trialsplitlist('/dls/science/users/ycc62267/eclip/eclip/trialsplit.txt')
  inputshape = [201,201,3]
  #method can be either 'average first' or 'round first'
  method = 'average first'
  #output predictions file
  outfile = '/dls/science/users/ycc62267/eclip/eclip//paratry/newpredic22.txt'
  
  x, name, proteins = importData(datafileloc=fileloc,proteinlist=protein_split,
  input_shape=inputshape)
  
  model = loadjson(jsonfile,weights_file)
  prediction = model.predict(x)
   
  
  #round first
  if method == 'round first':
    scores, preds, ones, zeros = roundfirstscore(proteins,name, prediction,outfile)
  if method == 'average first':
    scores, preds, ones, zeros = avefirstscore(proteins,name, prediction,outfile)
  else:
    RuntimeError('Not a valid method')
  
  conn = sqlite3.connect('/dls/science/users/ycc62267/metrix_db/metrix_db.sqlite')
  cur=conn.cursor()
  n_tp=0
  n_tn=0
  n_fp=0
  n_fn=0 

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
  #  cur.execute('''
  #    UPDATE Phasing SET (ep_score_i, ep_confidence_i, ep_score_o,
  #    ep_confidence_o) = (Null,Null,Null,Null)''')
  
    print(protein_id)
    if protein.endswith('_i'):
      cur.execute('''
        UPDATE Phasing SET (ep_score_i, ep_confidence_i)=(%s,%s) WHERE
        Phasing.pdb_id_id = "%s"''' %(score,pred,protein_id))
      cur.execute('''
        SELECT ep_success_i FROM Phasing WHERE Phasing.pdb_id_id =
        "%s"'''%(protein_id))
      truescore= list(cur.fetchall()[0])[0]
      if truescore ==None:
        continue
      truescore=int(truescore)
      print('score: %s truescore: %s'%(score,truescore))
      #print('updating...')
     # print(score,pred,protein_id)
      if truescore == 1 and score ==1:
        n_tp+=1
      elif truescore ==1 and score ==0:
        n_fn+=1
      elif truescore ==0 and score ==1:
        n_fp+=1
      else:
        n_tn+=1

    else:
      cur.execute('''
        UPDATE Phasing SET (ep_score_o,ep_confidence_o)=(%s,%s) WHERE
        Phasing.pdb_id_id = "%s"''' %(score,pred,protein_id))
      cur.execute('''
        SELECT ep_success_i FROM Phasing WHERE Phasing.pdb_id_id =
        "%s"'''%(protein_id))
      truescore= list(cur.fetchall()[0])[0]
      if truescore ==None:
        continue
      truescore=int(truescore)
      print('score: %s truescore: %s'%(score,truescore))
      #print('updating...')
      #print(score,pred,protein_id)
      if truescore == 1 and score ==1:
        n_tp+=1
      elif truescore ==1 and score ==0:
        n_fn+=1
      elif truescore ==0 and score ==1:
        n_fp+=1
      else:
        n_tn+=1
  conn.commit()
  conn.close()
  print('update EP_success successful')
  print('Number of true positives: %s  Number of true negatives %s'%(n_tp,n_tn))
  print('Number of false positives: %s Number of false negatives %s'%(n_fp,n_fn))
  
  acc = (n_tp+n_tn)/(n_fp+n_fn+n_tp+n_tn)
  print('accuracy = %s'%acc)
  
  text= open(outfile,'a')
  text.write('\nNumber of true positives: %s\nNumber of true negatives: %s\nNumber of false positives: %s\nNumber of false negatives: %s'%(n_tp,n_tn,n_fp,n_fn))
  
if __name__=='__main__':
  main()
