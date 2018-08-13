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
  input_map_directory1 = '/dls/mx-scratch/ycc62267/mapfdrbox'
  output_directory1='/dls/mx-scratch/ycc62267/imgfdr/blur2_5_maxminbox'

  image_slicing(input_map_directory1,output_directory1)

#calling EP_success
  EPf()  

#calling learn
  learnf()

#calling predictest
  predictestf()

if __name__=="__main__":
  CreatingNewModel()


############################################################
