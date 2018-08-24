############
'''Code to run using existing model'''
##############

from ConvMAP import image_slicing
from predic import  main as predicf
import time

##########
def Predictfromexisting(input_map_directory1,
                        output_directory1,
                        sqlite_db,
                        raw,
                        outpred):

  start_time = time.time()
  date = str(time.strftime("%d%m%y"))
  if raw:
    date = date +'raw'

#calling ConvMAP
  image_slicing(input_map_directory1,output_directory1)

#calling predic
  predicf(sqlite_db=sqlite_db,
          fileloc = output_directory1,
          outdir = outpred,
          date = date, 
          raw = raw)

  print("--- %s seconds ----" %(time.time() - start_time))
#######################################################################
def run():

  import argparse
  from eclip.utils.datamanip import str2bool

  parser = argparse.ArgumentParser(description = 'command line argument')
  parser.add_argument('--input',
                      dest = 'input_m',
                      type=str,
                      help = 'The input map directory',
                      default = '/dls/mx-scratch/ycc62267/mapfdrbox')
  parser.add_argument('--output',
                      dest = 'output_m',
                      type=str,
                      help = 'The output directory for the images',
                      default =
                      '/dls/mx-scratch/ycc62267/imgfdr/blur2_5_maxminbox')
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
                      default = '/dls/science/users/ycc62267/eclip/eclip/paratry/')
  
  args=parser.parse_args()

  Predictfromexisting(args.input_m,
                      args.output_m,
                      args.sqlitedb,
                      args.raw,
                      args.op)


###########
if __name__=="__main__":
  run()


