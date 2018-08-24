''' Forming Classification collumn for EP images '''

#############################################################################

import sqlite3
import os.path
from eclip.utils.get_data import get_log_filename, best_sym, get_lst_filename
from eclip.utils.get_data import percent_find, make_list_inverse, make_list_original

###########################################################################################  
def main(sqlitedb='/dls/science/users/ycc62267/metrix_db/metrix_db.sqlite', 
        dirin='/dls/mx-scratch/ycc62267/imgfdr/blur2_5_maxminbox', 
        raw=False,
        out = '/dls/mx-scratch/melanie/for_METRIX/results_201710/EP_phasing',
        filename = 'simple_xia2_to_shelxcde.log'):
  
  
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
    logfnm=get_log_filename(out,name,filename)
    bsgrp = best_sym(logfnm)
    if bsgrp == -1:
      continue
    lstfnm = get_lst_filename(out,name,name,bsgrp)
    percent=percent_find(lstfnm)
    if percent == -1:
      continue
    #get pdb_id
    cur.execute('''
      SELECT id FROM PDB_id WHERE PDB_id.pdb_id="%s" ''' % (name))
    pdbpk = cur.fetchone()[0]
    cur.execute('''
      INSERT OR IGNORE INTO Phasing (pdb_id_id) VALUES (%s) ''' %(pdbpk))
    if percent > 25:
      cur.execute('''
        UPDATE Phasing SET (ep_success_o, ep_percent_o)=(1,%s) WHERE
        Phasing.pdb_id_id = "%s"''' %(percent, pdbpk))
    else:
      cur.execute('''
        UPDATE Phasing SET (ep_success_o,ep_percent_o)=(0,%s) WHERE
        Phasing.pdb_id_id = "%s"''' %(percent,pdbpk))

    if raw:
      directory = os.path.join(dirin,name,name)
      cur.execute('''
        UPDATE Phasing SET ep_raw_o = "%s" WHERE Phasing.pdb_id_id = "%s"'''%(directory,pdbpk))
    else:
      directory = os.path.join(dirin,name,name)
      cur.execute('''
        UPDATE Phasing SET ep_img_o = "%s" WHERE Phasing.pdb_id_id = "%s"'''%(directory,pdbpk))
  
  invlst = make_list_inverse(dirin)
  for n in invlst:
    name_i=str(n)
    name=name_i[:-2]
    logfnm=get_log_filename(out,name,filename)
    bsgrp = best_sym(logfnm)
    if bsgrp == -1:
      continue
    lstfnm=get_lst_filename(out,name,name_i,bsgrp)
    percent=percent_find(lstfnm)
    if percent == -1:
      continue
    #get pdb_id
    cur.execute('''
      SELECT id FROM PDB_id WHERE PDB_id.pdb_id="%s" ''' % (name))
    pdbpk = cur.fetchone()[0]
    cur.execute('''
      INSERT OR IGNORE INTO Phasing (pdb_id_id) VALUES (%s) ''' % (pdbpk))
    cur.execute('''
      UPDATE Phasing SET ep_percent_i = %s WHERE Phasing.pdb_id_id = "%s"'''
      %(percent,pdbpk))
    cur.execute('''
      UPDATE Phasing SET ep_success_i=1 WHERE  ep_percent_i > 25''' )
    cur.execute('''
      UPDATE Phasing SET ep_success_i=0 WHERE ep_percent_i < 25''' )
    if raw:
      directory = os.path.join(dirin,name,name_i)
      cur.execute('''
        UPDATE Phasing SET ep_raw_i = "%s" WHERE Phasing.pdb_id_id =
        "%s"'''%(directory,pdbpk))
    else:
      directory = os.path.join(dirin,name,name_i)
      cur.execute('''
        UPDATE Phasing SET ep_img_i = "%s" WHERE Phasing.pdb_id_id = "%s"'''%(directory,pdbpk))

  
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


def run():  

  import argparse

  parser = argparse.ArgumentParser(description = 'command line argument')
  parser.add_argument('--sqlitedb',
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
  parser.ad_argument('--out',
                      dest = 'out',
                      type = str,
                      help = 'directory to find log files etc',
                      default =
                      '/dls/science/melanie/for_METRIX/results_201710/EP_phasing')
  parser.add_argument('--fnm',
                      dest = 'fnm',
                      type = str,
                      help = 'name of log file',
                      default = 'simple_xia2_to shelxcde.log')

  args = parser.parse_args()
  
  main(args.sqlitedb,
        args.dirin,
        args.raw,
        args.out,
        args.fnm)
  
if __name__=="__main__":
  run()
