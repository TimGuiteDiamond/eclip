##############################################################
'''Creating model in full'''
#############################################################
from ConvMAP import image_slicing
from EP_success import main as EPf
from learn import main as learnf
from predictest import main as predictestf

#############################################################
def CreatingNewModel():
  
# calling Convmap
  input_map_directory1 = '/dls/mx-scratch/ycc62267/mapfdrrawbox'
  output_directory1='/dls/mx-scratch/ycc62267/imgfdr/blur2_5_maxminrawbox'

  image_slicing(input_map_directory1,output_directory1)

#calling EP_success

  sqlite_db = '/dls/science/users/ycc62267/metrix_db/metrix_db.sqlite'
  dir_in = output_directory1
  raw=True
  EPf(sqlite_db,dir_in,raw)  

#calling learn
  raw = True
  date='150818raw'
  learnf(Raw=raw,date=date)

#calling predictest
  predictestf(fileloc = output_directory1, date=date)

if __name__=="__main__":
  CreatingNewModel()


############################################################
