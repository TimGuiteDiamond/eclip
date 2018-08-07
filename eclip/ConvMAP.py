#this is a code to convert .map to numpy array, following the same structure as
#from cyro EM and edited
import os
from PIL import Image as pimg
import shutil # dont know what this does
import sys #dont know what this does
import scipy.ndimage
import numpy as np
#mrcfile is a CCP-EM library for i/o to MRC format map files
import mrcfile 




def image_slicing(input_map_directory, output_directory, axes_list =
['X','Y','Z'],window_size=10,offset =
10,section_skip=2,normalise = True
,map_min=-2.14,map_max=4.19,blur=True,sigma=2.5):



  '''

  ConvMAP takes a set of 3D electron density maps and slices them into 2d
  surfaces and saves these as images. 
|

**Arguments for image_slicing:**

* **input_map_directory:** this is a string indicating where to find the .map files\ 
* **output_directory:** this is a string indicating where to save the image files\ 
* **axes_list:** axis to section along (default: axes_list = ['X','Y','Z'])\ 
* **window_size:** the linear dimension of the scanning window for each slice (default:  window_size=20)\ 
* **offset:** the offset distance between each window position. (default: offset = 10)\ 
* **section_skip:** the separation between each section slice. (default: section_skip =2)\ 
* **normalise:** boolean, decide whether to normalise values in a slice to be between map_min and map_max (default: normalise = True)\ 
* **map_min:** minimum value for the map (default: map_min = -2.14)\  
* **map_max:** maximum vlaue for the map (default: map_max =  4.19)\ 
* **blur:** boolean, decide whether to blur the data via a gaussian filter. (default: blur = True)\ 
* **sigma:** if blur is True, this is the standard deviation for the gaussian.(default: sigma = 2.5)\ 
  
  
  '''



  #checking output_directory exists
  if not os.path.isdir(output_directory):
    print("creating output directory "+output_directory)
    os.mkdir(output_directory)

  #creating logfile  
  logfile = os.path.join(output_directory, 'loffile_image_slicing.txt')
  text = open(logfile,'a')
  text.write('this is a log file for the progress of image_slicing(). \n')
  text.close()
  #opening file
  m=0
  for file in os.listdir(input_map_directory):
    m+=1
    if m>3:
      break

    if file.endswith('.map'):
      f=str(file)
      mrc = mrcfile.open(os.path.join(input_map_directory,f))
      #mrc.print_header()
      print('Opening file')
      file_name = os.path.splitext(f)[0]
      print(file_name)
      text=open(logfile,'a')
      text.write('\n'+file_name+'\n')



      if file_name.endswith('_i'):
        protein=file_name[:-2]
      else:
        protein=file_name
      print(protein)
  
      output_dir1 = os.path.join(output_directory,protein)
     #ensuring the output directory exists
      if not os.path.isdir(output_dir1):
        print("creating output directory "+output_dir1)
        #text.write('creating output directory '+output_dir1+'\n')
        os.mkdir(output_dir1)  

      output_dir2 = os.path.join(output_dir1,file_name)
      if not os.path.isdir(output_dir2):
        print("creating output directory "+ output_dir2)
        #text.write('creating output directory '+output_dir2+'\n')
        os.mkdir(output_dir2)
     
#    '''the value of axis_list was specified above, we need to loop over the axes
 #     specified in this list and transpose the data columns accordingly- we need the
 #     sectioning axis to be in mrc.data[2]'''
      text.close()

      for axis in axes_list:
        if axis in ['X']:
          #text.write('\nFor the '+axis+' axis. \n\n')
          print("transposing")
          #text.write('transposing...\n')
          loc_mrcdata_a=np.transpose(mrc.data,(1,2,0))
        elif axis in ['Y']:
          loc_mrcdata_a=np.transpose(mrc.data,(2,0,1))
        else:
          loc_mrcdata_a=mrc.data
       
        output_dir3 = os.path.join(output_dir2,axis)
        if not os.path.isdir(output_dir3):
          print("creating output directory "+output_dir3)
          #text.write('creating output directory '+output_dir3+'\n')
          os.mkdir(output_dir3)  

#'''a variable to log image file names in order- completely optional, but can
#be helpful.'''
       #bluring the data using a gaussian filter
        if blur:
          loc_mrcdata=scipy.ndimage.filters.gaussian_filter(loc_mrcdata_a,(sigma,sigma,sigma))
          print('bluring map')
          #text.write('bluring map\n')
        else:
          loc_mrcdata=loc_mrcdata_a
       
        num=0
        for section in range(0,loc_mrcdata.shape[2], section_skip):
          print("processing section",section)
          #text.write('\nprocessing section '+str(section)+'\n')
          num +=1
          myim = loc_mrcdata[:,:,section].copy(order='C')
         
          #to normalise the contrast of the image
          if normalise:
            print('normalising')
            #text.write('normalising\n')
            myim = 255.0*np.clip((myim-map_min)/(map_max-map_min),0,1)  

          #this is to same the files in order and otherwise unimportant
          if num < 10:
            imgout_filename = file_name +'_00'+str(num)+'_'+str(section)+axis+'.png'
          elif num < 100:
            imgout_filename = file_name +'_0'+str(num)+'_'+str(section)+axis+'.png'
          else:
            imgout_filename =file_name+'_'+str(num)+'_'+str(section)+axis+'.png'
         #saving file
          img_out = pimg.fromarray(myim)
          img_new =  img_out.convert('RGB')
          img_new.save(output_dir3+'/'+imgout_filename)
          print('saved image: '+imgout_filename+' in directory '+output_dir3+'\n')
          text=open(logfile,'a')
          if not os.path.exists(os.path.join(output_dir3,imgout_filename)):
            text.write(imageout_filename+' did not work')
            continue

          text.write('saved image: '+imgout_filename+'\n')
          text.close()
       
    else:
      continue
    print("Finished. The images are saved in ",output_dir1 )
    text=open(logfile,'a')
    text.write('Finished. The images are saved in %s' %output_dir1)
    text.close()
##########################################################################################


###########################################################################################
#to allow it to be called in the command line 

if __name__=="__main__":
  import argparse

  parser = argparse.ArgumentParser(description = 'command line argument')
  parser.add_argument('--input', dest = 'input_map_dir', type = str,help='the input map directory',default='/dls/mx-scratch/ycc62267/mapfdrbox')
  parser.add_argument('--output',dest='output_dir', type=str,help='the output directory for the images',default = '/dls/mx-scratch/ycc62267/imgfdr/blur2_5_maxminbox')

  args=parser.parse_args()

  image_slicing(args.input_map_dir,args.output_dir)


###########################################################################################
#input and output directories
#input_map_directory1 = '/dls/science/users/ycc62267/mapfdr/gauss'
#output_directory1 = '/dls/science/users/ycc62267/imgfdr/gauss/blur25'
#output_directory3 = '/dls/science/users/ycc62267/imgfdr/gauss/blur10'
#output_directory2 = '/dls/science/users/ycc62267/imgfdr/gauss/noblur'
#
#
#image_slicing(input_map_directory1,output_directory1)
#image_slicing(input_map_directory1,output_directory2,blur=False)
#image_slicing(input_map_directory1, output_directory3,sigma=10)

  








'''this section makes a scanning window of images, but is not necessary to run the
      code at the moment.'''

        #assert window_size < loc_mrcdata.shape[1]
        #assert window_size < loc_mrcdata.shape[0]


       # for col in range(0,loc_mrcdata.shape[1]-window_size,offset):
       #   for row in range(0,loc_mrcdata.shape[0]-window_size,offset):
    
            #myslice =loc_mrcdata[row:row+window_size,col:col+window_size,section].copy(order='C')
    
            #annotate here the cyro em code seems to use a reference map and
            #compares the two to find the measurement for how good the map is- here
            #we need some means of labeling how good/bad the data is.
            #annotate = 'b_'
            #n_ref_dens= 0
            #for x in range(row,row+window_size):
            #  for y in range(col,col+window_size):
            #    if loc_mrcrefdata[x,y,section]>ref_thresh:
            #      n_ref_dens +=1
            #if a fraction of grid points in reference map have significant density
            #then label as good
            #if n_ref_dens>window_size*window_size*good_fraction:
            #  annotate = 'g_'
            #          
            #if normalise:
            #  myslice = 255.0*np.clip((myslice-map_min)/(map_max - map_min),0,1)
    
            #pillow - forming images 
            #imgout_filename =file_name+str(num)+str(section)+'_'+str(col)+'_'+str(row)+axis+'.png'
            #img_out = pimg.fromarray(myslice)
            #img_new =  img_out.convert('RGB')
            #img_new.save(output_directory+'/'+imgout_filename)
            #print("just saved an image: "+imgout_filename)

