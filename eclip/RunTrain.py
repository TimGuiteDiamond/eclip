##############################################################
'''
Module for creating model in full. The module slices the maps into images,
updates the sqlite database and trains and tests a model. It does not 
have the full flexibility that calling each individually provides, but 
does provide an easier way of running the Eclip package.

Other modules called by RunTrain:

  * ConvMAP
  * EP_success
  * learn
  * predictest

Calling RunTrain results in the creation of:
 
  * Images of slices of the maps
  * populated sqlite Phasing table
  * model.json
  * model.yaml
  * model.hs
  * predict------_-.txt
  * log------_-.txt
  * Plot------__.png
  * CnfM------_-.png
  * Complot------_-.png
  * newpredic------_-.txt
  * map_results------_-.png
  * map_cnf------_-.png

|

Arguments
^^^^^^^^^^^

The command line arguments are as follows.

* **--input:** The location for the imput map directory. Default: /dls/mx-scratch/ycc62267/mapfdrbox
* **--output:** The location for the output directory for the images. Default: /dls/mx-scratch/ycc62267/imgfdr/blur2_5_maxminbox
* **--dtdr:** The location of the directory containing the log files. Default: /dls/mx-scratch/melanie/for_METRIX/results_201710/EP_phasing
* **--db:** The location of the sqlite database file. Default: /dls/science/users/ycc62267/metrix_db/metrix_db.sqlite
* **--raw:** Boolean, if the data is just heavy atom positions or processed. Default: False
* **--epochs:** The number of epochs to run for. Default: 300
* **--nmb:** The number of images to use per axis per protein. Default: 15
* **--trial:** The starting trial number. Default: 1
* **--lgdir:** The output directory for stats. Default: /dls/science/users/ycc62267/eclip/eclip/paratry1
* **--ning:** Boolean, whether to add a name onto filenames to be saved. Default: False
* **--name:** The name to add the the filenames to be saved. Default: ''
* **--para:** Whether or not to train on a cluster, boolean. Default: True

|

Functions in module
^^^^^^^^^^^^^^^^^^^^
|

'''
#############################################################
from eclip.ConvMAP import image_slicing
from eclip.EP_success import main as EPf
from eclip.learn import main as learnf
from eclip.predictest import main as predictestf
import time
import logging
import os

#############################################################
def CreatingNewModel(input_map_directory1,
                      output_directory1,
                      data_directory,
                      sqlite_db,
                      raw,
                      epochs,
                      number,
                      trialnum,
                      lgdir,
                      naming,
                      name,
                      para):
  '''
  CreatingNewModel is the main function of RunTrain

  |

  '''

  start_time = time.time() 
  date = str(time.strftime("%d%m%y"))
  if raw:
    date = date+'raw'
  if naming:
    date = date+name

# creating a log file
  while os.path.exists(os.path.join(lgdir,'log'+date+'_'+str(trialnum)+'.log')):
    trialnum+=1
  logfile = os.path.join(lgdir, 'log'+date+'_'+str(trialnum)+'.log')

  logging.basicConfig(filename = logfile, level = logging.DEBUG)
  logging.info('Running RunTrain.py')
  logging.info('Date: %s'%date)

# calling Convmap
  image_slicing(input_map_directory1,
                output_directory1)

# calling EP_success

  EPf(sqlite_db,
      output_directory1,
      raw,
      data_directory)  

# calling learn
  learnf(epochs=epochs,
          Raw=raw,
          date=date,
          number = number, 
          trialnum = trialnumi,
          parallel = para)

# calling predictest
  predictestf(fileloc = output_directory1, 
                date=date,
                trial_num=trialnum,
                raw=raw, 
                number = number)

  print("--- %s seconds -----" %(time.time()-start_time))
  logging.info('Finished -- %s seconds ---'%(time.time()-start_time))
###################################################################


def run(): 

  '''
  run allows RunTrain to be called from the command line. 

  '''
  import argparse
  from eclip.utils.datamanip import str2bool

  parser = argparse.ArgumentParser(description = 'command line argument')
  parser.add_argument('--input',
                      dest = 'input_m', 
                      type = str,
                      help= 'The input map directory', 
                      default = '/dls/mx-scratch/ycc62267/mapfdrbox')
  parser.add_argument('--output', 
                      dest = 'output_m', 
                      type= str, 
                      help = 'the output directory for the images', 
                      default = '/dls/mx-scratch/ycc62267/imgfdr/blur2_5_maxminbox')
  parser.add_argument('--dtdr',
                      dest = 'datadir',
                      type = str,
                      help = 'The directory to find log files',
                      default =
                      '/dls/mx-scratch/melanie/for_METRIX/results_201710/EP_phasing')
  parser.add_argument('--db', 
                      dest = 'sqlitedb', 
                      type = str, 
                      help='The location of the sqlite database file', 
                      default = '/dls/science/users/ycc62267/metrix_db/metrix_db.sqlite')
  parser.add_argument('--raw', 
                      dest = 'raw', 
                      type = str2bool, 
                      help = 'Boolean, true for raw data', 
                      default = False)
  parser.add_argument('--epochs', 
                      dest = 'epochs', 
                      type = int, 
                      help = 'number of epochs to run for.', 
                      default = 300)
  parser.add_argument('--nmb',
                      dest = 'nmb',
                      type = int,
                      help = 'number of images for learn.py per axis per protein',
                      default = 15)
  parser.add_argument('--trial',
                      dest = 'trial',
                      type = int,
                      help = 'The starting trial number',
                      default = 1)
  parser.add_argument('--lgdir',
                      dest = 'lgdir',
                      type = str,
                      help = 'output directory for stats etc.',
                      default =
                      '/dls/science/users/ycc62267/eclip/eclip/paratry1')
  parser.add_argument('--ning',
                      dest = 'ning',
                      type = str2bool,
                      help = 'Boolean, if true a name is specified',
                      default = False)
  parser.add_argument('--name',
                      dest = 'name',
                      type = str,
                      help = 'name to give if ning is true',
                      default = '')
  parser.add_argument('--para',
                      dest = 'ning',
                      type = str2bool,
                      help = 'Boolean, true if running on a cluster',
                      default = True)


  args=parser.parse_args()
  
  CreatingNewModel(args.input_m,
                  args.output_m,
                  args.datadir,
                  args.sqlitedb,
                  args.raw,
                  args.epochs,
                  args.nmb,
                  args.trial,
                  args.lgdir,
                  args.ning,
                  args.name,
                  args.para)
  
############################################################

if __name__=="__main__":
  run()
