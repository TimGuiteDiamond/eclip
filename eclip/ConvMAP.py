#this is a code to convert .map to numpy array, following the same structure as
#for cyro EM
import os
from PIL import Image as pimg
import shutil # dont know what this does
import sys #dont know what this does

import numpy as np
#mrcfile is a CCP-EM library for i/o to MRC format map files
import mrcfile 

#necesary files
input_map_file = '3S6E_i.map'
input_map_directory = '/dls/science/users/ycc62267/mapfdr/tests'
output_directory = '/dls/science/users/ycc62267/imgfdr'
#reference_map_file = '?????'
#
###################################################################
#we need a set of conditions
#axis to section along
axes_list = ['Z']
#linear dimension of slice
window_size=20
#offset from one slice to the next within a section
offset = 10
#offset from one section to the next 
section_skip =10
#normalise values in a slice to be between map_min and map_max
normalise = True
#map_min =np.amin(mrc.data)
#map_max=np.amax(mrc.data)
map_min = -0.02
map_max = 0.05
#threshold for counting ref map density
ref_thresh = 0.1
#if a fraction of grip points in reference map have significant density, then
#label as good
good_fraction = 0.05
##########################################################################
#opening file
print("opening file..")#
mrc = mrcfile.open('../../mapfdr/tests/3S6E_i.map')
mrc.print_header()

#print("opening reference map file ...")
#mrcref = mrcfile.open(reference_map_file)
#mrcref.print_header()

##ensuring the output directory exists
if not os.path.isdir(output_directory):
  print("creating output directory...")
  os.mkdir(output_directory)

#'''the value of axis_list was specified above, we need to loop over the axes
#specified in this list and transpose the data columns accordingly- we need the
#sectioning axis to be in mrc.data[2]'''
for axis in axes_list:
  if axis in ['X']: 
    print("transposing")
    loc_mrcdata=np.transpose(mrc.data,(1,2,0))
 #   loc_mrcrefdata=np.transpose(mrcref.data,(1,2,0))
  elif axis in ['Y']:
    loc_mrcdata=np.transpose(mrc.data,(2,0,1))
 #   loc_mrcrefdata=np.transpose(mrcref.data,(2,0,1))
  else:
    loc_mrcdata=mrc.data
  #  loc_mrcrefdata=mrcref.data

  assert window_size < loc_mrcdata.shape[1]
  assert window_size < loc_mrcdata.shape[0]

  for section in range(0,loc_mrcdata.shape[2]-window_size, offset):
    print("processing section",section)

    for col in range(0,loc_mrcdata.shape[1]-window_size,offset):
      for row in range(0,loc_mrcdata.shape[0]-window_size,offset):

        myslice =loc_mrcdata[row:row+window_size,col:col+window_size,section].copy(order='C')

        #annotate here the cyro em code seems to use a reference map and
        #compares the two to find the measurement for how good the map is- here
        #we need some means of labeling how good/bad the data is.
        #annotate = 'b_'
        #n_ref_dens= 0
        #for x in range(row,row+window_size):
        #  for y in range(col,col+window_size):
        #    if loc_mrcrefdata[x,y,section]>ref_thresh:
        #      n_ref_dens +=1
        ##if a fraction of grid points in reference map have significant density
        ##then label as good
        #if n_ref_dens>window_size*window_size*good_fraction:
        #  annotate = 'g_'
        
        if normalise:
          myslice = 255.0*np.clip((myslice-map_min)/(map_max - map_min),0,1)

        #pillow - forming images 
        imgout_filename ='3S6E_i'+str(section)+'_'+str(col)+'_'+str(row)+axis+'.png'
        img_out = pimg.fromarray(myslice)
        img_new =  img_out.convert('RGB')
        img_new.save(output_directory+'/'+imgout_filename)
        print("just saved an image: "+imgout_filename)

    myim = loc_mrcdata[:,:,section].copy(order='C')
    if normalise:
      myim = 255.0*np.clip((myim-map_min)/(map_max-map_min),0,1)

    imgout_filename ='3S6E_i'+str(section)+axis+'.png'
    img_out = pimg.fromarray(myim)
    img_new =  img_out.convert('RGB')
    img_new.save(output_directory+'/'+imgout_filename)
    print("just saved a section image")

print("done!")













