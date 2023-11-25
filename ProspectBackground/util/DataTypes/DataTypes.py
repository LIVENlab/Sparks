"""
Define some standard classes to organize data
"""
from typing import Dict,Union,Optional
from collections import defaultdict
import bw2data.errors
import pandas as pd
pd.options.mode.chained_assignment = None
import bw2data as bd
import warnings
from ProspectBackground.const.const import bw_project,bw_db
from typing import Dict,Union,Optional
bd.projects.set_current(bw_project)            # Select your project
database = bd.Database(bw_db)        # Select your db

class ActivityData:
    """
    Store some basic information for the scenario data
    {alias : [unit, amount]}
    """
    def __init__(self):
        self.data=defaultdict(list)

    def add_activity(self,alias:str, unit: str, value: float):
        self.data[alias].extend([unit,value])

class Scenarios_Dict:
    """
    Store the scenario data
    { scenario name: {'activities':ActivityDict}}
    """
    def __init__(self):
        self.scenario_data: Dict[str,Union[str,ActivityData]]=defaultdict(dict)
    def add_scenario(self, scenario : str, ActivityData):
        self.scenario_data[scenario]={'activities':ActivityData}


class ActitiesDict:
    """
    Store information for the Activities dictionary
    {alias :
        "id":
        {"code": bw code}}
    """
    def __init__(self):
        self.activities: Dict[str,Union[str,Union[str,str]]] = defaultdict(dict)
    def add_activity(self, alias,code):
        self.activities[alias]={'id':{ "code":code}}




