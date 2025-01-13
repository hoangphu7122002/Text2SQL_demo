from datetime import datetime
import pandas as pd

def generate_db_name():
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    # Generate names
    return f"__db_{timestamp}"

def generate_table_name():
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    # Generate names
    return f"__table_{timestamp}"

def isfloat(str_num):
    try:
        float(str_num.replace(",", "."))
        return True
    except:
        return False

def isinteger(str_num):
    if isfloat(str_num):
        try:
            int(str_num)
            return True
        except:
            return False
    return False

def map_dtypes(dtypes_list):
    # Define the mapping
    dtype_mapping = {
        'float64': 'float',
        'int64': 'integer',
        'object': 'string',
        'bool': 'boolean',
        'datetime64[ns]': 'datetime',
        # Add more mappings as needed
    }
    
    # Map the data types using the defined mapping
    mapped_dtypes = [dtype_mapping.get(str(dtype), 'unknown') for dtype in dtypes_list]
    
    return mapped_dtypes

def generate_create_table_sql(database_info):
    sql_commands = []

    for table_index, table_name in enumerate(database_info['table_names']):

        # Generate CREATE TABLE statement
        create_table_sql = f'CREATE TABLE "{table_name}" ('
        column_sql = []
        for column_index, column_name, column_type in zip(database_info['column_indices'], database_info['column_names'], database_info['column_types']):
            if column_index == table_index:
                column_sql += [f'"{column_name}" {column_type}']

        create_table_sql += ", ".join(column_sql) + ")"

        sql_commands.append(create_table_sql)
    database_info['schema'] = "; ".join(sql_commands) + ";"
    return database_info