############
'''
RunPred is the module to use an existing model to predict scores on new maps.
The module slices the maps into images, updates an sqlite database, and loads
a pretrained model to predict scores for these images, which are then 
averaged to produce a score for the maps. It does not have the full 
flexibility that calling each indiviuallyprovides, but does provide an easier 
way of running the Eclip package. 

Other modules called by RunPred:

  * ConvMAP
  * EP_add_new
  * predic

Calling RunPred results in the creation of: 

  * Images of slices of the maps
  * log------_-.txt
  * populated columns for predictions in the sqlite database 
  * datapredict------_-.txt

| 

Arguments
^^^^^^^^^^

The command line arguments are as follows.
  
* **--input:** The location for the input map directory. Default: /dls/mx-scratch/ycc62267/mappredfdr
* **--output:** The location for the output direcotry for the images. Default: /dls/mx-scratch/ycc62267/imgfdr/pred
* **--db:** The location of the sqlite database file. Default: /dls/science/users/ycc62267/metrix_db/metrix_db.sqlite
* **--raw:** Boolean, whether using heavy atom positions or processed data. Default: False
* **--op:** The output loction for the prediction file. Default: /dls/science/users/ycc62267/eclip/eclip/paratry1/
* **--ning:** Boolean, true is name should be added to date. Default: False
* **--name:** Name to add. Default: ''
* **--model:** The directory to find the model.json and model.h5 files. Default: /dls/science/users/ycc62267/eclip/eclip/paratry1/

| 

Functions in module
^^^^^^^^^^^^^^^^^^^^
|

'''
##############

from eclip.ConvMAP import image_slicing
from eclip.predic import  main as predicf
from eclip.EP_add_new import main as EP_add_new
import time
import os.path

##########
def Predictfromexisting(input_map_directory1,
                        output_directory1,
                        sqlite_db,
                        raw,
                        outpred,
                        ning,
                        name,
                        model):
  '''
  Predictfromexisting is the main function of RunPred

  |

  '''

  start_time = time.time()
  date = str(time.strftime("%d%m%y"))
  if raw:
    date = date +'raw'
  if ning:
    date = date + name

  jsonfile= os.path.join(model,'model.json')
  weightsfile=os.path.join(model,'model.h5')

#calling ConvMAP
  image_slicing(input_map_directory1,output_directory1)

#updating database
  EP_add_new(sqlitedb=sqlite_db,dirin = output_directory1, raw = raw)

#calling predic
  predicf(jsonfile= jsonfile,
          weights_file = weightsfile,
          sqlite_db=sqlite_db,
          fileloc = output_directory1,
          outdir = outpred,
          date = date, 
          raw = raw)

  print("--- %s seconds ----" %(time.time() - start_time))
#######################################################################
def run():

  '''
  run allows RunPred to be called from the command line.

  '''
  import argparse
  from eclip.utils.datamanip import str2bool

  parser = argparse.ArgumentParser(description = 'command line argument')
  parser.add_argument('--input',
                      dest = 'input_m',
                      type=str,
                      help = 'The input map directory',
                      default = '/dls/mx-scratch/ycc62267/mappredfdr')
  parser.add_argument('--output',
                      dest = 'output_m',
                      type=str,
                      help = 'The output directory for the images',
                      default =
                      '/dls/mx-scratch/ycc62267/imgfdr/pred')
  parser.add_argument('--db',
                      dest = 'sqlitedb',
                      type = str,
                      help = 'The location of the sqlite database file',
                      default = '/dls/science/users/ycc62267/metrix_db/metrix_db.sqlite')
  parser.add_argument('--raw',
                      dest = 'raw',
                      type = str2bool,
                      help = 'Boolean, true for raw data',
                      default = False)
  parser.add_argument('--op',
                      dest = 'op',
                      type = str,
                      help = 'output location for prediction file',
                      default = '/dls/science/users/ycc62267/eclip/eclip/paratry1/')
  parser.add_argument('--ning',
                      dest = 'ning',
                      type = str2bool,
                      help = 'Boolean, True if adding name onto date',
                      default = False)
  parser.add_argument('--name',
                      dest = 'name',
                      type = str,
                      help = 'name to add to date',
                      default = '')
  parser.add_argument('--model',
                      dest = 'model',
                      type = str,
                      help = 'location of directory for model to use, should have model.json and model.h5',
                      default = '/dls/science/users/ycc62267/eclip/eclip/paratry1/')
  
  args=parser.parse_args()

  Predictfromexisting(args.input_m,
                      args.output_m,
                      args.sqlitedb,
                      args.raw,
                      args.op,
                      args.ning,
                      args.name,
                      args.model)


###########
if __name__=="__main__":
  run()


