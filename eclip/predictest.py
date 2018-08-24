################
''' Code to predict new scores for full maps and check this against the true
scores'''
##############
#from keras.models import predict
import sqlite3
import os
import logging

from utils.modelmanip import load_json
from utils.datamanip import import_data, ave_first_score, round_first_score,trial_split_list
from utils.visu import plot_test_results, ConfusionMatrix
################

def main(jsonfile ='/dls/science/users/ycc62267/eclip/eclip/paratry1/model.json',
        weights_file ='/dls/science/users/ycc62267/eclip/eclip/paratry1/model.h5', 
        sqlite_db = '/dls/science/users/ycc62267/metrix_db/metrix_db.sqlite',
        fileloc = '/dls/mx-scratch/ycc62267/ingfdr/blur2_5_maxminbox/', 
        protein_split_list ='/dls/science/users/ycc62267/eclip/eclip/trialsplit.txt',
        inputshape =[201,201,3],
        method = 'average first',
        outdir ='/dls/science/users/ycc62267/eclip/eclip/paratry1/',
        date ='150818',
        trial_num=1,
        threshold = 0.5,
        raw = False):
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
  * **outdir:** output predictions file. default:
  * '/dls/science/users/ycc62267/eclip/eclip/paratry1/'
  * **date:** date. default: date = '100818'
  * **trial_num:** default: trial_num = 5
  * **threshold:** default: threshold= 0.5

  '''


   ########################################################################
   
  proteinsplit = trial_split_list(protein_split_list)
  outfile = os.path.join(outdir,'newpredic'+date+'_'+str(trial_num)+'.txt')
  resoutfile= os.path.join(outdir,'map_results_plot'+date+'_'+str(trial_num)+'.png')
  cnfout=os.path.join(outdir,'map_cnf'+date+'_'+str(trial_num)+'.png')
    #######################################################################
  
  x, name, proteins = import_data(datafileloc=fileloc,proteinlist=proteinsplit,
  input_shape=inputshape)
  
  model = load_json(jsonfile,weights_file)
  prediction = model.predict(x)
   
  
  #round first
  if method == 'round first':
    scores, preds, ones, zeros = round_first_score(proteins,name, prediction,outfile,threshold)
  if method == 'average first':
    scores, preds, ones, zeros = ave_first_score(proteins,name, prediction,outfile,threshold)
  else:
    RuntimeError('Not a valid method')
  

  # Selecting true scores from database and creating a list of various
  # parameters
  conn = sqlite3.connect(sqlite_db)
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
      cur.execute('''
        SELECT ep_success_i FROM Phasing WHERE Phasing.pdb_id_id =
        "%s"'''%(protein_id))
      truescore= list(cur.fetchall()[0])[0]
    else:
      cur.execute('''
        UPDATE Phasing SET (%s, %s)=(%s,%s) WHERE Phasing.pdb_id_id = "%s"
        ''' %(score_loc_o,confidence_loc_o,score,pred,protein_id))
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
 
  #adding to log file 
  logging.info('''\nNumber of true positives:%s\n
              Number of true negatives: %s\n
              Number of false positives: %s\n
              Number of false negatives: %s\n'''
              %(n_tp,n_tn,n_fp,n_fn))
  logging.info('accuracy = %s'%acc)

  # plot test results
  plot_test_results(y_true,preds,resoutfile)
  
  # plot Confusion matrix
  ConfusionMatrix(y_true,scores,cnfout)
  

#########################################################################
def run():
  import argparse
  import time 
  from eclip.utils.datamanip import str2bool
  start_time = time.time()
  date = str(time.strftime("%d%m%y"))

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
                      help = 'location for database',
                      default =
                      '/dls/science/users/ycc62267/metrix_db/metrix_db.sqlite')
  parser.add_argument('--floc',
                      dest = 'fileloc',
                      type = str,
                      help = 'Location to find image files.',
                      default =
                      '/dls/mx-scratch/ycc62267/ingfdr/blur2_5_maxminbox/')
  parser.add_argument('--psl',
                      dest='psl',
                      type = str,
                      help = 'location for list of used proteins',
                      default =
                      '/dls/science/users/ycc62267/eclip/eclip/trialsplit.txt')
  parser.add_argument('--insh',
                      dest='inshape',
                      type = list,
                      help = 'dimensions of input images',
                      default = [201,201,3])
  parser.add_argument('--met',
                      dest = 'met',
                      type = str,
                      help = 'Method to round',
                      default = 'average first')
  parser.add_argument('--out',
                      dest = 'out',
                      type = str,
                      help = 'output directory for saved files.',
                      default =
                      '/dls/science/users/ycc62267/eclip/eclip/paratry1/')
  parser.add_argument('--date',
                      dest = 'date',
                      type = str,
                      help = 'date to appear on saved files',
                      default = date)
  parser.add_argument('--trial',
                      dest = 'trial',
                      type = int,
                      help = 'trial number',
                      default = 1)
  parser.add_argument('--th',
                      dest = 'thresh',
                      type = float,
                      help = 'value to round up from.',
                      default = 0.5)
  parser.add_argument('--raw',
                      dest = 'raw',
                      type = str2bool,
                      help = 'Boolean, True if raw data',
                      default = False)
  
  args = parser.parse_args()
  raw = args.raw
  trialnum = args.trial
  date = args.date
  if raw:
    date = date + 'raw'
  outdir = args.out
  while os.path.exists(os.path.join(outdir,'log'+date+'_'+str(trialnum)+'.txt')):
    trialnum+=1

  logfile = os.path.join(outdir,'log'+date+'_'+str(trialnum)+'.txt')
  logging.basicConfig(filename = logfile, level = logging.DEBUG)
  logging.info('Running predictest.py')

  main(args.json,
        args.weights,
        args.sqlitedb,
        args.fileloc,
        args.psl,
        args.inshape,
        args.met,
        args.out,
        args.date,
        args.trial,
        args.thresh,
        args.raw)
  logging.info('Finished --%s seconds --'%(time.time() - start_time))


if __name__=='__main__':
  run()
