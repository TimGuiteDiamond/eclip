''' Module for functions for forming Classification collumn for EP images '''

#############################################################################
import os.path
from os import listdir
############################################################################

#get log_filename
def getlogfilename(name):
  out= '/dls/mx-scratch/melanie/for_METRIX/results_201710/EP_phasing'
  log_filename=os.path.join(out,name,'simple_xia2_to_shelxcde.log')
  return log_filename
  
#find best space group
def BestSym(log_filename):
  if log_filename is None:
    raise RuntimeError('Need to specify hklin filename')
  elif not os.path.exists(log_filename):
    raise RuntimeError('%s does not exist' %log_filename)
  
  tx=open(log_filename).read()
  index=tx.find('Best space group: ')
  shift = len('Best space group: ')
  start = int( index+shift)
  stop =int(tx.find(' ',start,start+10))
  if stop == -1:
    print('BestSym did not work for %s'%log_filename)
    best=-1
  else:
    best=tx[start:stop]
  return best

#get lst_filename
def getlstfilename(name,name_i,best_space_group):
  out = '/dls/mx-scratch/melanie/for_METRIX/results_201710/EP_phasing'
  lst_filename= os.path.join(out,name,best_space_group,(name_i+'.lst'))
  return lst_filename 

#find percentage
def percentfind(lst_filename):
  if lst_filename is None:
    raise RuntimeError('Need to specify hklin filename')
  elif not os.path.exists(lst_filename):
    raise RuntimeError('%s does not exist' %lst_filename)

  tx=open(lst_filename).read()
  index=tx.find('with CC ', -1000,-700)
  shift = len('with CC ')
  start=int(index+shift)
  stop=int(tx.find('%',start,start+6))
  if stop == -1:
    give_up = int(tx.find('CC is less than zero - giving up',start))
    if give_up == -1:
      print('percentfind did not work for %s'%lst_filename)
      Percent = -1
    else: Percent = 0
  else:
    Percent = float(tx[start:stop])
  return Percent


def makelistoriginal(dir_in):
  name_list=[]
  for item in listdir(dir_in):
    if os.path.isdir(os.path.join(dir_in,item,item)):
      name_list.append(item)
    else: continue
  return name_list

def makelistinverse(dir_in):
  name_list = []
  for item in listdir(dir_in):
    item_i=item+'_i'
    if os.path.isdir(os.path.join(dir_in,item,item_i)):
      name_list.append(item_i)
    else: continue
  return name_list
  

