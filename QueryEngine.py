import sys
import os
sys.path.insert(0, os.path.abspath('/raid/phundh/demo/spider'))
sys.path.insert(0, os.path.abspath('/raid/phundh/nltk_data'))

import os
import sqlite3
import argparse
from datasets import load_dataset
from spider.script_add_double_flashes import *
from spider.evaluation import Evaluator
from spider.process_sql import get_schema, Schema, get_sql
import sqlite3
import os
import nltk
nltk.download('punkt')

os.makedirs("db_demo", exist_ok=True)

sample_example = {
    "schema": """CREATE TABLE IF NOT EXISTS table_42529 (
        "Mùa giải" REAL,
        "Division" TEXT,
        "Thắng" REAL,
        "Thua" REAL,
        "Hòa" REAL,
        "Vị trí cuối cùng" TEXT,
        "Ghi chú" TEXT
    )
    """,
    "value":['INSERT INTO "table_42529" VALUES("2004","SPL","6","3","1","5","tồn tại")',
            'INSERT INTO "table_42529" VALUES("2004","SPL","5","4","1","6","tồn tại")',
            'INSERT INTO "table_42529" VALUES("2005","SPL","6","2","2","7","tồn tại")',
            'INSERT INTO "table_42529" VALUES("2002","SPL","6","2","2","7","tồn tại")'],
    "question": "",
    "query": """SELECT AVG ("Hòa") FROM table_42529 WHERE "Thắng" = '6' AND "Mùa giải" > '2004'""",
    "predict": """SELECT AVG ("Hòa") FROM table_42529 WHERE "Thắng" = '6' AND "Mùa giải" > '2004'""",
    "db_id": "example",
}

def generate_db_file(db_path, schema: str, db_name: str = "example.sqlite",list_value=None):
    # Step 1: Create a directory named 'db' if it doesn't exist
    os.makedirs(db_path, exist_ok=True)

    # Step 2: Define the path for the SQLite database file
    db_path = os.path.join(db_path, db_name)

    # Step 3: Connect to the SQLite database (or create it if it doesn't exist)
    conn = sqlite3.connect(db_path)

    # Step 4: Create a cursor object
    cursor = conn.cursor()

    # cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    # tables = cursor.fetchall()
    # print(tables)
    
    for schema_ele in schema:
    # Step 5: Execute the SQL command to create the table
        cursor.execute(schema_ele)
    
    if list_value != None:
        for val in list_value:
            try:
                cursor.execute(val)
            except:
                continue
    
    # Step 6: Commit the changes
    conn.commit()

    # Step 7: Query the table names
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    # Step 8: Print the table names
    print(tables)

    # Step 9: Close the connection
    conn.close()

def execute_query(db,db_name, p_str):
    """
    return 1 if the values between prediction and gold are matching
    in the corresponding index. Currently not support multiple col_unit(pairs).
    """

    db_path = os.path.join(db, db_name)
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    try:
        res = cursor.execute(p_str)
        row = res.fetchall()
    except:
        conn.close()
        return False
    
    list_dict = []
    conn.close()
    
    for res in row:
        data_dict = {col: res[col] for col in res.keys()}
        list_dict.append(data_dict)
    return list_dict
    