import bw2io as bi
import bw2data as bd
import os
import pandas as pd
from pathlib import Path
path=r'C:\Users\Administrator\Documents\Alex\external_db\inventory_premise'

def get_filenames(path: str):
    """
    get paths from folder
    """

    files_names=[]
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path,file)):
            files_names.append(str((os.path.join(path,file))))
    return files_names

files=get_filenames(path)

for file in files:
    bd.projects.set_current('TFM_Lex')
    bi.bw2setup()
    # spolds
    try:
        file_ex= Path(file)
        excel = bi.ExcelImporter(file)
        excel.match_database('ecoinvent')
        excel.apply_strategies()
        excel.write_database()

    except:


            print(f"Error with {file_ex}")
            continue
    pass







#
#excel.match_database("db_experiments_2")
#excel.apply_strategies()
#excel.write_database()