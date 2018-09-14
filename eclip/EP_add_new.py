'''
EP_add_new is the module for adding or updating columns for specific proteins
when predicting new scores. 

In the Phasing table, the 'score' collumns contain the scores, 1 is good, 0 is
bad. If the ep_img_o, ep_img_i, ep_raw_i or ep_raw_o were populated, they would
have been cleared and repopulated. 

Other classes and functions called in the module:

 From eclip.utils.get_data:
  * make_list_inverse
  * make_list_original
  
Calling EP_add_new results in the population of collumns pdb_id_id, scores, and
image locatoins of the Phasing table in the sqlite database. 

|

Arguments
^^^^^^^^^^^

The command line arguments are as follows. 

* **--db:** The location for the sqlite database. Default: /dls/science/users/ycc62267/metrix_db/metrix_db/sqlite
* **--dirin:** The directory location for the input images. Default: /dls/mx-scratch/ycc62267/imgfdr/blur2_5_maxminbox'
* **--raw:** Boolean, whether the data is just the heavy atom positions. Default: False


|

Functions in module 
^^^^^^^^^^^^^^^^^^^^^^
|

'''

#############################################################################

import sqlite3
import os.path

from eclip.utils.get_data import make_list_inverse, make_list_original



###########################################################################################  
def main(sqlitedb='/dls/science/users/ycc62267/metrix_db/metrix_db.sqlite', 
        dirin='/dls/mx-scratch/ycc62267/imgfdr/blur2_5_maxminbox', 
        raw=False):
  
  '''
  main is the overall function for EP_add_new. 

  |

  '''
  
  # columns to clear
  column_list=['ep_img_o','ep_img_i','ep_raw_o','ep_raw_i']

  #connecting to SQLite database
  conn=sqlite3.connect(sqlitedb)
  cur=conn.cursor()
  
   
  #I have used two different methods to fill the columns of the database, both
  #work and i haven't decided which is better.
  
  
  #for original:
  origlst= make_list_original(dirin)
  for n in origlst:
    #find label
    name=str(n)
    #get pdb_id
    cur.execute('''
      SELECT id FROM PDB_id WHERE PDB_id.pdb_id="%s" ''' % (name))
    pdbpk = cur.fetchone()[0]
    #clear column
    for i in column_list:
      cur.execute('''UPDATE Phasing SET %s =Null WHERE pdb_id_id =
      %s'''%(i,pdbpk))
    cur.execute('''
      INSERT OR IGNORE INTO Phasing (pdb_id_id) VALUES (%s) ''' %(pdbpk))
    if raw:
      directory = os.path.join(dirin,name,name)
      cur.execute('''
        UPDATE Phasing SET ep_raw_o = "%s" WHERE 
        Phasing.pdb_id_id = "%s"'''%(directory,pdbpk))
    else:
      directory = os.path.join(dirin,name,name)
      cur.execute('''
        UPDATE Phasing SET ep_img_o = "%s" WHERE 
        Phasing.pdb_id_id = "%s"'''%(directory,pdbpk))
  
  invlst = make_list_inverse(dirin)
  for n in invlst:
    name_i=str(n)
    name=name_i[:-2]
    #get pdb_id
    cur.execute('''
      SELECT id FROM PDB_id WHERE PDB_id.pdb_id="%s" ''' % (name))
    pdbpk = cur.fetchone()[0]
    #clear column
    for i in column_list:
      cur.execute('''UPDATE Phasing SET %s =Null WHERE pdb_id_id =
      %s'''%(i,pdbpk))
    cur.execute('''
      INSERT OR IGNORE INTO Phasing (pdb_id_id) VALUES (%s) ''' % (pdbpk))
    if raw:
      directory = os.path.join(dirin,name,name_i)
      cur.execute('''
        UPDATE Phasing SET ep_raw_i = "%s" WHERE Phasing.pdb_id_id =
        "%s"'''%(directory,pdbpk))
    else:
      directory = os.path.join(dirin,name,name_i)
      cur.execute('''
        UPDATE Phasing SET ep_img_i = "%s" WHERE Phasing.pdb_id_id = "%s"'''%(directory,pdbpk))

  
  conn.commit()
  conn.close()
  print('EP_add_new successful')

############################################################################


def run():  

  '''
  run allows EP_add_new to be called from the command line. 

  '''
  import argparse
  from eclip.utils.datamanip import str2bool

  parser = argparse.ArgumentParser(description = 'command line argument')
  parser.add_argument('--db',
                      dest = 'sqlitedb', 
                      type = str, 
                      help = 'the location of the sqlite database',
                      default = '/dls/science/users/ycc62267/metrix_db/metrix_db.sqlite')
  parser.add_argument('--dirin',
                      dest='dirin',
                      type=str,
                      help='the directory input image location',
                      default = '/dls/mx-scratch/ycc62267/imgfdr/blur2_5_maxminbox')
  parser.add_argument('--raw',
                      dest='raw',
                      type=str,
                      help='parameter specifying raw or not',
                      default=False)
  
  args = parser.parse_args()
  
  main(args.sqlitedb,
        args.dirin,
        args.raw)
  
if __name__=="__main__":
  run()
