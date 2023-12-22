import warnings
import bw2io as bi
import bw2data as bd
import bw2data.backends
import pandas as pd
from ProspectBackground.const.const import bw_project,bw_db
import json
from collections import defaultdict
import os
bd.projects.set_current(bw_project)            # Select your project
ei = bd.Database(bw_db)        # Select your db
from typing import List,Dict,Union,Optional


# Try to import hydrogen techs
#file="/home/lex/Downloads/lci-hydrogen-electrolysis.xlsx"
#excel=bi.ExcelImporter(file)
#excel.match_database("db_experiments_2")
#excel.apply_strategies()
#excel.write_database()
#pass

print(list(bd.databases))