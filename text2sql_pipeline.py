from InferEngine import *
from QueryEngine import *
import json
import re
from copy import copy

#==========HYPER PARAMETER==========
schema_json = ""
schema_name = "transaction"
inferEngine = InferEngine()

def print_schema(schema_name):
    global schema_json
    with open(f'db/{schema_name}.json','r') as f:
        schema_json = json.load(f)
    return '\n'.join(schema_json['schema'])

print_schema(schema_name)
# mode = None
# express = None
# query_show = None
# list_choice = os.listdir('db_demo/db')
# list_choice.remove('.ipynb_checkpoints')
#==========HYPER PARAMETER==========

def get_df(shema_json):
    cnx = sqlite3.connect(f'db_demo/db/{shema_json["db_id"]}/{shema_json["db_id"]}.sqlite')
    cursor = cnx.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    
    df = pd.read_sql_query(f"SELECT * FROM '{tables[0][0]}'", cnx)
    return df

def schema_to_db(sample_example):
    generate_db_file(
        db_path=f"db_demo/db/{sample_example['db_id']}",
        db_name=f"{sample_example['db_id']}.sqlite",
        schema=sample_example["schema"],
        list_value=sample_example["value"],
    )

    return get_df(sample_example)

def add_double_flashes(sql_str):
    global sql_keywords, sql_symbols

    toks = tokenize(sql_str)
    words = []
    aliases = list(scan_alias(toks).keys())
    for tok in toks:
        if tok in sql_keywords + sql_symbols + aliases or isfloat(tok):
            words.append(tok)
        elif (
            tok.count(".") == 1 and " " not in tok and tok[0] != '"' and tok[-1] != '"'
        ):  # case < "Computer Info. Systems"
            alias, name = tok.split(".")
            if alias in aliases:
                words.append(alias + "." + '"' + name + '"')
            else:  # case "vai trò.mô tả về vai trò"
                words.append('"' + alias + '"' + "." + '"' + name + '"')
        else:  # case < "2", = "Hello World"
            words.append('"' + tok + '"')

    sql_str = " ".join(words)
    sql_str = sql_str.replace('" "', " ")
    sql_str = sql_str.replace('""', '"')  # = "Hello World"

    return sql_str

def tokenize_sql_query(query):
    sql_keywords = [
        "SELECT", "FROM", "WHERE", "GROUP BY", "ORDER BY", "JOIN", "INNER JOIN",
        "LEFT JOIN", "RIGHT JOIN", "OUTER JOIN", "ON", "AND", "OR", "NOT", "IN",
        "LIKE", "BETWEEN", "IS NULL", "INSERT INTO", "VALUES", "UPDATE", "SET",
        "DELETE FROM", "CREATE TABLE", "ALTER TABLE", "DROP TABLE", "DISTINCT",
        "HAVING", "AS", "ASC", "DESC", "COUNT", "SUM", "AVG", "MAX", "MIN", "TOP",
        "ALL", "ANY", "UNION", "EXCEPT", "INTERSECT", "CASE", "WHEN", "THEN",
        "ELSE", "END", "BEGIN", "ROLLBACK", "COMMIT", "SAVEPOINT", "TRANSACTION",
        "PRIMARY KEY", "FOREIGN KEY", "REFERENCES", "INDEX", "CONSTRAINT"
    ]
    
    for sql_keyword in sql_keywords:
        elements = sql_keyword.split(' ')
        query = query.replace('_'.join(elements), ' '.join(elements))
    # Define the regular expression pattern to capture tokens
    pattern = r"""[\w'|\w"|*]+|[()=><.,;]|'|""" + re.escape(' ')

    # Use regular expression to find all occurrences of tokens
    # Split for all tokens
    tokens = re.findall(pattern, query)
    # Maybe some token with combine word and keyword (e.g học_việnWHERE) -> split it
    new_tokens = []
    for idx, token in enumerate(tokens):
        spaced_token = copy(token)
        for sql_keyword in sql_keywords:
            if sql_keyword in spaced_token and all(sql_keyword not in other_sql for other_sql in sql_keywords if other_sql != sql_keyword):
                spaced_token = spaced_token.replace(sql_keyword, f" {sql_keyword} ")
        while "  " in spaced_token:
            spaced_token = spaced_token.replace("  ", " ")
        spaced_token = re.findall(pattern, spaced_token)
        new_tokens += spaced_token
    new_tokens = "".join(new_tokens)
    new_tokens = new_tokens.strip()
    while "  " in new_tokens:
        new_tokens = new_tokens.replace("  ", " ")
    new_query = "".join(new_tokens)
    tokens = re.findall(pattern, new_query)

    # combine token open " and token close "
    _open = {
        "index": -1,
        "active": False
    }
    new_tokens = []
    for idx, token in enumerate(tokens):
        if _open['active']:
            if token[-1] in ["'", '"']:
                _open['active'] = False
                new_tokens += ["".join(tokens[_open['index']: idx + 1])]
            continue
        if token[0] in ["'", '"'] and token[-1] not in ["'", '"']: # just catch '"học' to match 'sinh"'
            _open['active'] = True
            _open['index'] = idx
        else:
            new_tokens.append(token)
    return "".join(new_tokens)

def query_output(schema_json,sql):
    result = execute_query(
            db=f"db_demo/db/{schema_json['db_id']}",
            db_name=f"{schema_json['db_id']}.sqlite",
            p_str=sql
        )

    return result

def format_express(result):
    pair_list = []
    for ele in result:
        for key,val in ele.items():
            # pair_list.append(f"{key.replace('SUM','tổng số').replace('(',' ').replace(')',' ')} : {val}")
            pair_list.append(f"{key} : {val}")
    
    return pair_list

def query(message):
    global schema_name
    question = message.split('##')[0]
    if '##' not in message:
        mode = 'sql'
    else: mode = message.split('##')[1]

    text = {'schema' : schema_json['schema'],'question' : question}
    try:
        res_infer = inferEngine.inference(text,mode)
        sql = res_infer['sql'].replace('\n','').strip()
        print("++++++++++1++++++++++")
        print(sql)
        print("++++++++++1++++++++++")
        if mode == 'CoT':
            CoT_express = res_infer['CoT'].strip()
        sql = tokenize_sql_query(sql)
        sql = sql.replace('FROM',' FROM ').replace('WHERE',' WHERE ').replace('GROUP BY', ' GROUP BY').replace('\"FROM','\" FROM').replace('_',' ')
        sql = add_double_flashes(sql)
        list_replace = ['20241020','20241021','20241022','20241023','20241024','20241025','20241026','20241027','20241028','20241029','20241030']
        list_temp = ['2024/10/20','2024/10/21','2024/10/22','2024/10/23','2024/10/24','2024/10/25','2024/10/26','2024/10/27','2024/10/28','2024/10/29','2024/10/30']
        for key,rep in zip(list_replace,list_temp):
            if key in sql:
                sql = sql.replace(key,rep)
        print(sql)
        result = query_output(schema_json,sql)
    except:
        sql = inferEngine.inference(text,'CoT')['sql'].replace('\n','').strip()
        print("++++++++++3++++++++++")
        print(sql)
        print("++++++++++3++++++++++")
        sql = tokenize_sql_query(sql)
        sql = sql.replace('FROM',' FROM ').replace('WHERE',' WHERE ').replace('GROUP BY', ' GROUP BY').replace('\"FROM','\" FROM').replace('_',' ')
        sql = add_double_flashes(sql)
        list_replace = ['20241020','20241021','20241022','20241023','20241024','20241025','20241026','20241027','20241028','20241029','20241030']
        list_temp = ['2024/10/20','2024/10/21','2024/10/22','2024/10/23','2024/10/24','2024/10/25','2024/10/26','2024/10/27','2024/10/28','2024/10/29','2024/10/30']
        for key,rep in zip(list_replace,list_temp):
            if key in sql:
                sql = sql.replace(key,rep)
        print(sql)
        result = query_output(schema_json,sql)

    ##Done Translate sql to query
    pair_list = format_express(result)
    result_save = pair_list.__str__()
    text = {'schema' : f'{pair_list.__str__()}','question' : question}
    infer_res = inferEngine.inference(text,'express')['sql'].strip()
    result = infer_res

    if mode == 'CoT':
        return str(result),sql,result_save,CoT_express
    return str(result),sql,result_save,''

if __name__ == "__main__":
    print(query('Các thành phố nào đăng cai thế vận hội'))