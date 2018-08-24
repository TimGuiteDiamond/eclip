################
''' Code to predict new scores for full maps'''
##############
import keras
import sqlite3
import os.path

from utils.modelmanip import load_json
from utils.datamanip import import_data, ave_first_score,round_first_score,trial_split_list
from utils.visu import plot_test_results
################

def main(jsonfile='/dls/science/users/ycc62267/eclip/eclip/paratry1/model.json',
          weights_file ='/dls/science/users/ycc62267/eclip/eclip/paratry1/model.h5',
          sqlite_db = '/dls/science/users/ycc62267/metrix_db/metrix_db.sqlite',
          fileloc = '/dls/mx-scratch/ycc62267/imgfdr/blur2_5_maxminbox/',
          protein_split_list= '/dls/science/users/ycc62267/eclip/eclip/trialsplit.txt',
          inputshape = [201,201,3],
          method = 'average first',
          outdir = '/dls/science/users/ycc62267/eclip/eclip/paratry1/',
          date = '070818a',
          trial_num = 1,
          threshold = 0.5,
          raw = False):


  '''
  predic loads a model from a json file and weights from a weights file and uses
  this to predict the score for a new map - already in image form. the
  predictions are then saved to a txt file and added to the sqlite database. 

  **Warning** this requires the protein to be present in the PDB_id part of the
  database

 
  
  jsonfile='/dls/science/users/ycc62267/eclip/eclip/paratry1/model.json'
  weights_file ='/dls/science/users/ycc62267/eclip/eclip/paratry1/model.h5'
  fileloc = '/dls/mx-scratch/ycc62267/imgfdr/blur2_5_maxminbox/'
  protein_split = trialsplitlist('/dls/science/users/ycc62267/eclip/eclip/trialsplit.txt')
  inputshape = [201,201,3]
  #method can be either 'average first' or 'round first'
  method = 'average first'
  #output predictions file
  outdir = '/dls/science/users/ycc62267/eclip/eclip/paratry1/'
  date = '070818'
  '''

  while os.path.exists(os.path.join(outdir,'datapredic'+ date+'_'+str(trial_num)+'.txt')):
    trial_num+=1
  
  proteinsplit = trial_split_list(protein_split_list)
  outfile = os.path.join(outdir,'datapredic'+date+'_'+str(trial_num)+'.txt')
    
  x, name, proteins = import_data(datafileloc=fileloc,
                                  proteinlist=proteinsplit,
                                  input_shape=inputshape)
  
  model = load_json(jsonfile,weights_file)
  prediction = model.predict(x)
   
  
  #round first
  if method == 'round first':
    scores, preds, ones, zeros = round_first_score(proteins,name,prediction,outfile,threshold)
  if method == 'average first':
    scores, preds, ones, zeros = ave_first_score(proteins,name,prediction,outfile,threshold)
  else:
    RuntimeError('Not a valid method')
  

  #updating sqlite database
  conn = sqlite3.connect(sqlite_db)
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
  
    if raw:
        score_loc_i = 'ep_score_rawi'
        confidence_loc_i = 'ep_confidence_rawi'
        score_loc_o = 'ep_score_rawo'
        confidence_loc_o = 'ep_confidence_rawo'
    else: 
        score_loc_i = 'ep_score_i'
        confidence_loc_i = 'ep_confidence_i'
        score_loc_o = 'ep_score_o'
        confidence_loc_o = 'ep_confidence_o'


    if protein.endswith('_i'):
      cur.execute('''
        UPDATE Phasing SET (%s, %s)=(%s,%s) WHERE Phasing.pdb_id_id = "%s"
        ''' %(score_loc_i,confidence_loc_i,score,pred,protein_id))
    else:
      cur.execute('''
        UPDATE Phasing SET (%s,%s)=(%s,%s) WHERE Phasing.pdb_id_id = "%s"
        ''' %(score_loc_o,confidence_loc_o,score,pred,protein_id))
      
  conn.commit()
  conn.close()
  print('update EP_success successful')
####################################################################    


def run():
  import argparse
  from eclip.utils.datamanip import str2bool

  parser = argparse.ArgumentParser(description = 'command line argument')
  parser.add_argument('--jsn',
                      dest = 'json',
                      type = str,
                      help = 'location of json file for model',
                      default =
                      '/dls/science/users/ycc62267/eclip/eclip/paratry1/model.json')
  parser.add_argument('--wfl',
                      dest = 'weights',
                      type = str,
                      help = 'location of file for weights of model',
                      default =
                      '/dls/science/users/ycc62267/eclip/eclip/paratry1/model.h5')
  parser.add_argument('--db',
                      dest = 'sqlitedb',
                      type = str,
                      help = 'location of database',
                      default =
                      '/dls/mx-scratch/ycc62267/imgfdr/blur2_5_maxminbox/')
  parser.add_argument('--floc',
                      dest = 'floc',
                      type = str,
                      help = 'location to find image directories',
                      default =
                      '/dls/mx-scratch/ycc62267/imgfdr/blur2_5_maxminbox/')

  parser.add_argument('--psl',
                      dest = 'psl',
                      type = str,
                      help = 'location for list of unused proteins.',
                      default =
                      '/dls/science/users/ycc62267/eclip/eclip/trialsplit.txt')
  parser.add_argument('--insh',
                      dest = 'inshape',
                      type = list,
                      help = 'dimensions of input images',
                      default = [201,201,3])
  parser.add_argument('--met',
                      dest = 'met',
                      type = str,
                      help = 'method to round',
                      default = 'average first')
  parser.add_argument('--out',
                      dest = 'out',
                      type = str,
                      help = 'output directory for saved files',
                      default =
                      '/dls/science/users/ycc62267/eclip/eclip/paratry1/')
  parser.add_argument('--date',
                      dest = 'date',
                      type = str,
                      help = 'date to appear on saved files',
                      default = '160818')
  parser.add_argument('--trial',
                      dest = 'trial',
                      type = int,
                      help = 'trial number',
                      default = 1)
  parser.add_argument('--th',
                      dest = 'thresh',
                      type = float,
                      help = 'Value to round up from',
                      default = 0.5)
  parser.add_argument('--raw',
                      dest = 'raw',
                      type = str2bool,
                      help = 'Boolean, True if raw data',
                      default = False)
  
  args = parser.parse_args()

  main(args.json,
        args.weights,
        args.sqlitedb,
        args.floc,
        args.psl,
        args.inshape,
        args.met,
        args.out,
        args.date,
        args.trial,
        args.thresh,
        args.raw)
  
if __name__=='__main__':
  run()

