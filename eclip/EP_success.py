''' Forming Classification collumn for EP images '''

#############################################################################

import sqlite3
import os.path
from eclip.utils.get_data import getlogfilename, BestSym, getlstfilename
from eclip.utils.get_data import percentfind, makelistinverse, makelistoriginal

###########################################################################################  
def main(sqlite_db='/dls/science/users/ycc62267/metrix_db/metrix_db.sqlite', 
        dir_in='/dls/mx-scratch/ycc62267/imgfdr/blur2_5_maxminbox', 
        raw=False):
  
  
  #connecting to SQLite database
  conn=sqlite3.connect(sqlite_db)
  cur=conn.cursor()
  
   
  #I have used two different methods to fill the columns of the database, both
  #work and i haven't decided which is better.
  
  
  #for original:
  originals_list= makelistoriginal(dir_in)
  for n in originals_list:
    #find label
    name=str(n)
    log_filename=getlogfilename(name)
    best_space_group = BestSym(log_filename)
    if best_space_group == -1:
      continue
    lst_filename = getlstfilename(name,name,best_space_group)
    percent=percentfind(lst_filename)
    if percent == -1:
      continue
    #get pdb_id
    cur.execute('''
      SELECT id FROM PDB_id WHERE PDB_id.pdb_id="%s" ''' % (name))
    pdb_pk = cur.fetchone()[0]
    cur.execute('''
      INSERT OR IGNORE INTO Phasing (pdb_id_id) VALUES (%s) ''' %(pdb_pk))
    if percent > 25:
      cur.execute('''
        UPDATE Phasing SET (ep_success_o, ep_percent_o)=(1,%s) WHERE
        Phasing.pdb_id_id = "%s"''' %(percent, pdb_pk))
    else:
      cur.execute('''
        UPDATE Phasing SET (ep_success_o,ep_percent_o)=(0,%s) WHERE
        Phasing.pdb_id_id = "%s"''' %(percent,pdb_pk))

    if raw:
      directory = os.path.join(dir_in,name,name)
      cur.execute('''
        UPDATE Phasing SET ep_raw_o = "%s" WHERE Phasing.pdb_id_id = "%s"'''%(directory,pdb_pk))
    else:
      directory = os.path.join(dir_in,name,name)
      cur.execute('''
        UPDATE Phasing SET ep_img_o = "%s" WHERE Phasing.pdb_id_id = "%s"'''%(directory,pdb_pk))
  
  inverse_list = makelistinverse(dir_in)
  for n in inverse_list:
    name_i=str(n)
    name=name_i[:-2]
    log_filename=getlogfilename(name)
    best_space_group = BestSym(log_filename)
    if best_space_group == -1:
      continue
    lst_filename=getlstfilename(name,name_i,best_space_group)
    percent=percentfind(lst_filename)
    if percent == -1:
      continue
    #get pdb_id
    cur.execute('''
      SELECT id FROM PDB_id WHERE PDB_id.pdb_id="%s" ''' % (name))
    pdb_pk = cur.fetchone()[0]
    cur.execute('''
      INSERT OR IGNORE INTO Phasing (pdb_id_id) VALUES (%s) ''' % (pdb_pk))
    cur.execute('''
      UPDATE Phasing SET ep_percent_i = %s WHERE Phasing.pdb_id_id = "%s"'''
      %(percent,pdb_pk))
    cur.execute('''
      UPDATE Phasing SET ep_success_i=1 WHERE  ep_percent_i > 25''' )
    cur.execute('''
      UPDATE Phasing SET ep_success_i=0 WHERE ep_percent_i < 25''' )
    if raw:
      directory = os.path.join(dir_in,name,name_i)
      cur.execute('''
        UPDATE Phasing SET ep_raw_i = "%s" WHERE Phasing.pdb_id_id =
        "%s"'''%(directory,pdb_pk))
    else:
      directory = os.path.join(dir_in,name,name_i)
      cur.execute('''
        UPDATE Phasing SET ep_img_i = "%s" WHERE Phasing.pdb_id_id = "%s"'''%(directory,pdb_pk))

  
  #to catch other exceptions- there may be a better way to do this- there should
  #not be many exceptions!
  cur.execute('''
    SELECT pdb_id_id FROM Phasing WHERE (ep_percent_o - ep_percent_i)>10 AND
    (ep_percent_o NOT NULL AND ep_percent_i NOT NULL)''')
  list_o=cur.fetchall()
  
  cur.execute('''
    SELECT pdb_id_id FROM Phasing WHERE (ep_percent_i - ep_percent_o)>10 AND
    (ep_percent_o NOT NULL AND ep_percent_i NOT NULL)''')
  list_i=cur.fetchall()
  
  number=len(list_o)+len(list_i)
  print('Number of exceptions: %s' %number)
  
  
  cur.execute('''
    UPDATE Phasing SET ep_success_o =1 WHERE (ep_percent_o - ep_percent_i)>10 AND
    (ep_percent_o NOT NULL AND ep_percent_i NOT NULL)''')
  cur.execute('''
    UPDATE Phasing SET ep_success_i =1 WHERE (ep_percent_i-ep_percent_o)>10 AND
    (ep_percent_o NOT NULL AND ep_percent_i NOT NULL)''')
  
  
  
  conn.commit()
  conn.close()
  print('EP_success successful')

############################################################################
if __name__=="__main__":
  

  import argparse

  parser = argparse.ArgumentParser(description = 'command line argument')
  parser.add_argument('--sqlitedb',
                      dest = 'sqlitedb', 
                      type = str, 
                      help = 'the location of the sqlite database',
                      default = '/dls/science/users/ycc62267/metrix_db/metrix_db.sqlite')
  parser.add_argument('--dir_in',
                      dest='dir_in',
                      type=str,
                      help='the directory input image location',
                      default = '/dls/mx-scratch/ycc62267/imgfdr/blur2_5_maxminbox')
  parser.add_argument('--raw',
                      dest='raw',
                      type=str,
                      help='parameter specifying raw or not',
                      default=False)

  args = parser.parse_args()
  
  main(args.sqlitedb,args.dir_in,args.raw)
