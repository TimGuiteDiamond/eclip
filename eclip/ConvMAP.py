'''
ConvMAP is the module that converts .map files to images. This splits the 3D map
into 2D image slices along 3 perpendicular axes.

Calling learn results in the creation of 600 images per protein. 

|

Arguments
^^^^^^^^^^^

The command line arguments for learn are as follows: 

* **--input:** This is a string indicating where to find the .map files. Default: /dls/mx-scratch/ycc62267/mapfdrbox
* **--output:** This is a string indicating where to save the image files. Default: /dls/mx-scratch/ycc62267/imgfdr/blur2_5_maxminbox
* **--axlst:** The axis to section along. Default: ['X','Y','Z']
* **--scskp:** The separation between each section slice. Default: 2 
* **--norm:** Boolean, if to normalise values in a slice to be between mmin and mmax. Default:  True 
* **--mmin:** The minimum value for the map. Default: -2.14  
* **--mmax:** The maximum vlaue for the map. Default: 4.19 
* **--blur:** Boolean, if to blur the data via a gaussian filter. Default: True 
* **--sgma:** If blur is True, this is the standard deviation for the gaussian. Default: 2.5
* **--trial:** The trial number for the logfile name. Default: 1
* **--lgdir:** The directory to save the logfile. Default: /dls/science/users/ycc62267/eclip/eclip/paratry1/
* **--date:** The date to appear on logfile name. Default: current date

|

Functions in module
^^^^^^^^^^^^^^^^^^^^^^
|

'''

import os
from PIL import Image as pimg
import shutil # dont know what this does
import sys #dont know what this does
import scipy.ndimage
import numpy as np
#mrcfile is a CCP-EM library for i/o to MRC format map files
import mrcfile 
import logging

##########################################################################################

def image_slicing(inputmapdirectory, 
                  outputdirectory, 
                  axeslist =['X','Y','Z'],
                  sectionskip=2,
                  normalise = True,
                  mapmin=-2.14,
                  mapmax=4.19,
                  blur=True,
                  sigma=2.5):



  '''
  image_slicing is the main function for ConvMAP. It takes a set of 3D electron
  density maps and slices them into 2 dimensional surfaces and saves these as 
  images. 

  |

  '''



  #checking output_directory exists
  if not os.path.isdir(outputdirectory):
    print("creating output directory "+outputdirectory)
    os.mkdir(outputdirectory)

  
  logging.info('this is a log file for the progress of image_slicing(). \n')

  #opening file
  #m can be used to test when edditing the code - remove at the end
  #m=0
  for file in os.listdir(inputmapdirectory):
  #  m+=1
  #  if m>3:
  #    break
    if file.endswith('.map'):
      f=str(file)
      mrc = mrcfile.open(os.path.join(inputmapdirectory,f))
      filename = os.path.splitext(f)[0]
      logging.info('\n'+filename+'\n')

      if filename.endswith('_i'):
        protein=filename[:-2]
      else:
        protein=filename
      
  
      outputdir1 = os.path.join(outputdirectory,protein)
      #ensuring the output directory exists
      if not os.path.isdir(outputdir1):
        print("creating output directory "+outputdir1)
        os.mkdir(outputdir1)  

      outputdir2 = os.path.join(outputdir1,filename)
      if not os.path.isdir(outputdir2):
        print("creating output directory "+ outputdir2)
        os.mkdir(outputdir2)
     
      #the value of axis_list was specified above, we need to loop over the axes
      #specified in this list and transpose the data columns accordingly- we need the
      #sectioning axis to be in mrc.data[2]
      
      for axis in axeslist:
        if axis in ['X']:
          locmrcdataa=np.transpose(mrc.data,(1,2,0))
        elif axis in ['Y']:
          locmrcdataa=np.transpose(mrc.data,(2,0,1))
        else:
          locmrcdataa=mrc.data
       
        outputdir3 = os.path.join(outputdir2,axis)
        if not os.path.isdir(outputdir3):
          print("creating output directory "+outputdir3)
          os.mkdir(outputdir3)  

        #bluring the data using a gaussian filter
        if blur:
          locmrcdata=scipy.ndimage.filters.gaussian_filter(locmrcdataa,(sigma,sigma,sigma))
          print('bluring map') 
        else:
          locmrcdata=locmrcdataa
       
        #to slice the images
        num=0
        for section in range(0,locmrcdata.shape[2], sectionskip):
          num +=1
          myim = locmrcdata[:,:,section].copy(order='C')
         
          #to normalise the contrast of the image
          if normalise:
             
            myim = 255.0*np.clip((myim-mapmin)/(mapmax-mapmin),0,1)  

          #this is to same the files in order and is otherwise unimportant
          if num < 10:
            imgoutfilename = filename +'_00'+str(num)+'_'+str(section)+axis+'.png'
          elif num < 100:
            imgoutfilename = filename +'_0'+str(num)+'_'+str(section)+axis+'.png'
          else:
            imgoutfilename =filename+'_'+str(num)+'_'+str(section)+axis+'.png'

         #saving file
          imgout = pimg.fromarray(myim)
          imgnew =  imgout.convert('RGB')
          imgnew.save(outputdir3+'/'+imgoutfilename)
          
          if not os.path.exists(os.path.join(outputdir3,imgoutfilename)):
            logging.info(imageoutfilename+' did not work')
            continue

                 
    else:
      continue
    print("Finished. The images are saved in ",outputdir1 )
##########################################################################################

def run():
  '''
  run allows ConvMAP to be called from the command line. 

  '''

  import argparse
  import time 
  from eclip.utils.datamanip import str2bool
  
  start_time = time.time()
  date = str(time.strftime("%d%m%y"))
  parser = argparse.ArgumentParser(description = 'command line argument')
  parser.add_argument('--input', 
                      dest = 'inputmapdir', 
                      type = str,
                      help='the input map directory',
                      default='/dls/mx-scratch/ycc62267/mapfdrbox')
  parser.add_argument('--output',
                      dest='outputdir', 
                      type=str,
                      help='the output directory for the images',
                      default = '/dls/mx-scratch/ycc62267/imgfdr/blur2_5_maxminbox')
  parser.add_argument('--axlst',
                      dest = 'axlst',
                      type = list,
                      help = 'the axes to slice on',
                      default = ['X','Y','Z'])
  parser.add_argument('--scskp',
                      dest = 'scskp',
                      type = int,
                      help = 'separation between each slice',
                      default = 2)
  parser.add_argument('--norm',
                      dest = 'normalise',
                      type = str2bool,
                      help = 'boolean , whether to normalise values in slice between mapmin and mapmax',
                      default = True)
  parser.add_argument('--mmin',
                      dest = 'mapmin',
                      type = float,
                      help = 'minimum value for the map',
                      default = -2.14)
  parser.add_argument('--mmax',
                      dest = 'mapmax',
                      type = float,
                      help = 'maximum value for the map',
                      default = 4.19)
  parser.add_argument('--blur',
                      dest = 'blur',
                      type = str2bool,
                      help = 'boolean, whether to blur data via gausian filter',
                      default = True)
  parser.add_argument('--sgma',
                      dest = 'sigma',
                      type = float,
                      help = 'standard deviation for gausian filter',
                      default = 2.5)

  parser.add_argument('--trial',
                      dest = 'trial',
                      type = int,
                      help = 'the trial number for the logfile',
                      default = 1)
  parser.add_argument('--lgdir',
                      dest = 'lgdir',
                      type = str,
                      help='the directory to save the logfile',
                      default =
                      '/dls/science/users/ycc62267/eclip/eclip/paratry1/')
  parser.add_argument('--date',
                      dest = 'date',
                      type = str,
                      help = 'date to appear on logfile name',
                      default = date)

  args=parser.parse_args()

  trialnum = args.trial
  date = args.date
  outdir = args.lgdir
  while os.path.exists(os.path.join(outdir,'log'+date+'_'+str(trialnum)+'.txt')):
    trialnum+=1
  logfile = os.path.join(outdir, 'log'+date+'_'+str(trialnum)+'.txt')
  logging.basicConfig(filename = logfile, level = logging.DEBUG)

  logging.info('Running ConvMAP.py')

  image_slicing(args.inputmapdir,
                args.outputdir,
                args.axlst,
                args.scskp,
                args.normalise,
                args.mapmin,
                args.mapmax,
                args.blur,
                args.sigma)

  logging.info('Finished -- %s seconds --'%(time.time() - start_time))


###########################################################################################

if __name__=="__main__":
  run()


