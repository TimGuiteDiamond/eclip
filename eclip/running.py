##############################################################
'''Creating model in full'''
#############################################################
from ConvMAP import image_slicing
from EP_success import main as EPf
from learn import main as learnf
from predictest import main as predictestf
import time
import logging

#############################################################
def CreatingNewModel(input_map_directory1,
                      output_directory1,
                      sqlite_db,
                      raw,
                      epochs,
                      number,
                      trialnum,
                      lgdir):
  start_time = time.time() 
  date = str(time.strftime("%d%m%y"))
  if raw:
    date = date+'raw'

  date = args.
  while os.path.exists(os.path.join(lgdir,'log'+date+'_'+str(trialnum)+'.log'):
    trialnum+=1
  logfile = os.path.join(lgfir, 'log'+date+'_'+str(trialnum)+'.log')

  logging.basicConfig(filename = logfile, level = logging.DEBUG)
  logging.info('Running running.py')

# calling Convmap
  image_slicing(input_map_directory1,output_directory1)

#calling EP_success

  EPf(sqlite_db,output_directory1,raw)  

#calling learn
  learnf(epochs=epochs,Raw=raw,date=date,number = number, trialnum = trialnum)

#calling predictest
  predictestf(fileloc = output_directory1, date=date, raw=raw)
  print("--- %s seconds -----" %(time.time()-start_time))
  logging.info('Finished')
###################################################################

if __name__=="__main__":
  
  import argparse

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
  parser.add_argument('--db', 
                      dest = 'sqlitedb', 
                      type = str, 
                      help='The location of the sqlite database file', 
                      default = '/dls/science/users/ycc62267/metrix_db/metrix_db.sqlite')
  parser.add_argument('--raw', 
                      dest = 'raw', 
                      type = bool, 
                      help = 'Boolian, true for raw data', 
                      default = False)
  parser.add_argument('--epochs', 
                      dest = 'epochs', 
                      type = int, 
                      help = 'number of epochs to run for.', 
                      default = 100)
  parser.add_argument('--nmb',
                      dest = 'nmb',
                      type = int,
                      help = 'number of images for learn.py per axis per protein',
                      default = 10)
  parser.add_argument('--trial',
                      dest = 'trial',
                      type = int,
                      help = 'The starting trial number',
                      default = 1)
  parser.add_arguemnt('--lgdir',
                      dest = 'lgdir',
                      type = str,
                      help = 'output directory for stats etc.',
                      default =
                      '/dls/sicence/user/ycc62267/eclip/eclip/paratry1')


  args=parser.parse_args()
  
  CreatingNewModel(args.input_m,
                  args.output_m,
                  args.sqlitedb,
                  args.raw,
                  args.epochs,
                  args.nmb,
                  args.trial,
                  args.lgdir)


############################################################
