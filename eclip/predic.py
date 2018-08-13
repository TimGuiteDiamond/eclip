################
''' Code to predict new scores for full maps'''
##############
import keras
import sqlite3

from utils.modelmanip import loadjson
from utils.datamanip import importData, avefirstscore, roundfirstscore,trialsplitlist
from utils.visu import plot_test_results
################

def main():

  '''
  predic loads a model from a json file and weights from a weights file and uses
  this to predict the score for a new map - already in image form. the
  predictions are then saved to a txt file and added to the sqlite database. 

  **Warning** this requires the protein to be present in the PDB_id part of the
  database

 
  '''
    ######################################################################
  jsonfile='/dls/science/users/ycc62267/eclip/eclip/paratry/model.json'
  weights_file ='/dls/science/users/ycc62267/eclip/eclip/paratry/model.h5'
  fileloc = '/dls/mx-scratch/ycc62267/imgfdr/blur2_5_maxminbox/'
  protein_split = trialsplitlist('/dls/science/users/ycc62267/eclip/eclip/trialsplit.txt')
  inputshape = [201,201,3]
  #method can be either 'average first' or 'round first'
  method = 'average first'
  #output predictions file
  outdir = '/dls/science/users/ycc62267/eclip/eclip/paratry/'
  date = '070818'

  outfile = os.path.join(outdir,'newpredic'+date+'.txt')
   #######################################################################
  
  x, name, proteins = importData(datafileloc=fileloc,proteinlist=protein_split,
  input_shape=inputshape)
  
  model = loadjson(jsonfile,weights_file)
  prediction = model.predict(x)
   
  
  #round first
  if method == 'round first':
    scores, preds, ones, zeros = roundfirstscore(proteins,name,prediction,outfile,threshold)
  if method == 'average first':
    scores, preds, ones, zeros = avefirstscore(proteins,name,prediction,outfile,threshold)
  else:
    RuntimeError('Not a valid method')
  

  #updating sqlite database
  conn = sqlite3.connect('/dls/science/users/ycc62267/metrix_db/metrix_db.sqlite')
  cur=conn.cursor()
  
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
  
    #adding new protein_id to Phasing
    cur.execute('''
      INSERT OR IGNORE INTO Phasing (pdb_id_id) VALUES (%s) ''' %(protein_id))
  
  
    if protein.endswith('_i'):
      cur.execute('''
        UPDATE Phasing SET (ep_score_i, ep_confidence_i)=(%s,%s) WHERE
        Phasing.pdb_id_id = "%s"''' %(score,pred,protein_id))
    else:
      cur.execute('''
        UPDATE Phasing SET (ep_score_o,ep_confidence_o)=(%s,%s) WHERE
        Phasing.pdb_id_id = "%s"''' %(score,pred,protein_id))
      
  conn.commit()
  conn.close()
  print('update EP_success successful')
    
if __name__=='__main__':
  main()
