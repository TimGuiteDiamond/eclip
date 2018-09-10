''' 
Module for functions for forming Classification collumn for EP images

|

'''

#############################################################################
import os.path
from os import listdir
############################################################################

#get log_filename
def get_log_filename(out, name, filename):
  '''
  gets log file name, combines out, name and filename

  **Arguments:**

  * **out:** The directory
  * **name:** subdirectory - name of protein
  * **filename:** subsubdirectory - name of protein with phasing label (e.g. _i)

  |

  '''
  log_filename=os.path.join(out,name,filename)
  return log_filename
  
#find best space group
def best_sym(log_filename):
  '''
  gets the best symmetry from logfile

  **Arguments:**

  * **log_filename:** location of logfile containing best symmetry

  |

  '''
  if log_filename is None:
    raise RuntimeError('Need to specify hklin filename')
  elif not os.path.exists(log_filename):
    print('%s does not exist' %log_filename)
    best = -1
    #raise RuntimeError('%s does not exist' %log_filename)
  else:
    tx=open(log_filename).read()
    index=tx.find('Best space group: ')
    shift = len('Best space group: ')
    start = int( index+shift)
    stop =int(tx.find(' ',start,start+8))
    if stop == -1:
      print('BestSym did not work for %s'%log_filename)
      best=-1
    else:
      best=tx[start:stop]
  return best

#get lst_filename
def get_lst_filename(out,name,name_i,best_space_group):
  '''
  gets lst filename by combining arguments

  **Arguments:**

  * **out:** directory
  * **name:** subdirectory - protein name
  * **name_i: ** subsubdirectory - protein name with phase label
  * **best_space_group:** the name of the best space group

  |

  '''
  lst_filename= os.path.join(out,name,best_space_group,(name_i+'.lst'))
  return lst_filename 

#find percentage
def percent_find(lst_filename):
  '''
  finds the CC percentage for scoring

  **Arguments**

  * **lst_filename:** the location of the lst with the percentage information

  |

  '''
  if lst_filename is None:
    raise RuntimeError('Need to specify hklin filename')
  elif not os.path.exists(lst_filename):
    raise RuntimeError('%s does not exist' %lst_filename)

  tx=open(lst_filename).read()
  index=tx.find('with CC ', -1000,-700)
  shift = len('with CC ')
  start=int(index+shift)
  stop=int(tx.find('%',start,start+10))
  if stop == -1:
    give_up = int(tx.find('CC is less than zero - giving up',start))
    if give_up == -1:
      print('percentfind did not work for %s'%lst_filename)
      Percent = -1
    else: Percent = 0
  else:
    Percent = float(tx[start:stop])
  return Percent


def make_list_original(dir_in):
  '''
  makes a list of the proteins directories with original phases

  **Arguments:**

  * **dir_in:** The input directory

  |

  '''
  name_list=[]
  for item in listdir(dir_in):
    if os.path.isdir(os.path.join(dir_in,item,item)):
      name_list.append(item)
    else: continue
  return name_list

def make_list_inverse(dir_in):
  '''
  makes a list of the protein directories with inverse phases

  **Arguments**

  * **dir_in:** the input directory

  |

  '''
  name_list = []
  for item in listdir(dir_in):
    item_i=item+'_i'
    if os.path.isdir(os.path.join(dir_in,item,item_i)):
      name_list.append(item_i)
    else: continue
  return name_list
  

