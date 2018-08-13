import sqlite3

conn=sqlite3.connect('/dls/science/users/ycc62267/metrix_db/metrix_db.sqlite')
cur=conn.cursor()

cur.execute('''
  SELECT pdb_id_id FROM Phasing WHERE (ep_percent_0 - ep_percent_i)>20 AND 
  (ep_percent_0 NOT NULL AND ep_percent_i NOT NULL)''')
list_o=cur.fetchall()

cur.execute('''
  SELECT pdb_id_id FROM Phasing WHERE (ep_percent_i - ep_percent_0)>20 AND 
  (ep_percent_0 NOT NULL AND ep_percent_i NOT NULL)''')
list_i=cur.fetchall()

number20= len(list_o)+len(list_i)
print('number20: %s'%number20)

cur.execute('''
  SELECT pdb_id_id FROM Phasing WHERE (ep_percent_0 - ep_percent_i)>15 AND 
  (ep_percent_0 NOT NULL AND ep_percent_i NOT NULL)''')
list_o=cur.fetchall()

cur.execute('''
  SELECT pdb_id_id FROM Phasing WHERE (ep_percent_i - ep_percent_0)>15 AND 
  (ep_percent_0 NOT NULL AND ep_percent_i NOT NULL)''')
list_i=cur.fetchall()

number15= len(list_o)+len(list_i)
print('number15: %s'%number15)



cur.execute('''
  SELECT pdb_id_id FROM Phasing WHERE (ep_percent_0 - ep_percent_i)>10 AND 
  (ep_percent_0 NOT NULL AND ep_percent_i NOT NULL)''')
list_o=cur.fetchall()

cur.execute('''
  SELECT pdb_id_id FROM Phasing WHERE (ep_percent_i - ep_percent_0)>10 AND 
  (ep_percent_0 NOT NULL AND ep_percent_i NOT NULL)''')
list_i=cur.fetchall()

number10= len(list_o)+len(list_i)
print('number10: %s'%number10)


cur.execute('''
  SELECT pdb_id_id FROM Phasing WHERE (ep_percent_0 - ep_percent_i)>5 AND 
  (ep_percent_0 NOT NULL AND ep_percent_i NOT NULL)''')
list_o=cur.fetchall()

cur.execute('''
  SELECT pdb_id_id FROM Phasing WHERE (ep_percent_i - ep_percent_0)>5 AND 
  (ep_percent_0 NOT NULL AND ep_percent_i NOT NULL)''')
list_i=cur.fetchall()

number5= len(list_o)+len(list_i)
print('number5: %s'%number5)

cur.execute('''
  SELECT pdb_id_id FROM Phasing WHERE (ep_percent_0 - ep_percent_i)>2 AND 
  (ep_percent_0 NOT NULL AND ep_percent_i NOT NULL)''')
list_o=cur.fetchall()

cur.execute('''
  SELECT pdb_id_id FROM Phasing WHERE (ep_percent_i - ep_percent_0)>2 AND 
  (ep_percent_0 NOT NULL AND ep_percent_i NOT NULL)''')
list_i=cur.fetchall()

number2= len(list_o)+len(list_i)
print('number10: %s'%number2)

conn.close()

