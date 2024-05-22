import json
import random
import bw2data as bd
from collections import defaultdict
import pandas as pd
from bw2data.errors import UnknownObject
from typing import Optional,Dict,Union,List
from Sparks.const.const import bw_project,bw_db
import warnings
from dataclasses import dataclass, field
from Sparks.util.preprocess.cleaner import BaseFileActivity

bd.projects.set_current(bw_project)            # Select your project
database = bd.Database(bw_db)        # Select your db



@dataclass
class Activity_scenario:
    """ Class for each activity in a specific scenario"""
    alias: str
    amount: int
    unit: str #TODO: adapt


@dataclass
class Scenario:
    """ Basic Scenario"""
    name: str # scenario name
    activities: List['Activity_scenario'] = field(default_factory=list)

    def __post_init__(self):
        self.activities_dict = {x.alias: [
            x.unit,x.amount
        ] for x in self.activities}

    def to_dict(self):
        return {'name': self.name, 'nodes':self.activities_dict}

@dataclass
class Last_Branch:
    """ Last Branch before leaf. Leaf is a BaseFileActivity"""
    name: str
    level: str
    parent: str
    adapter='bw'
    origin: List['BaseFileActivity'] = field(default_factory=list)
    leafs: List = field(init=False)

    def __post_init__(self):
        self.leafs = [{'name': x.alias, 'adapter': 'bw', 'config': {'code': x.code}} for x in self.origin]
        pass


@dataclass
class Branch:
    name: str
    level: str
    parent :Optional[str] = None
    origin: List[Union['Branch', 'Last_Branch']]=field(default_factory=list)
    leafs: List = field(init=False)
    def __post_init__(self):
        self.leafs=[
            {
                'name': x.name, 'aggregator': 'sum', 'children': x.leafs
            }
            for x in self.origin]

@dataclass
class Method:
    method: tuple

    def to_dict(self):
        return {self.method[2].split('(')[1].split(')')[0]: [
            self.method[0], self.method[1], self.method[2]
        ]}



class SoftLinkCalEnb():
    """
    This class allows to create an ENBIOS-like input
    """

    def __init__(self,calliope,
                 mother_data: list,
                 motherfile,
                 smaller_vers=None):


        self.calliope=calliope
        self.motherfile=motherfile
        self.dict_gen=None
        self.scens=None
        self.aliases=[]
        self.final_acts={}
        self.hierarchy_tree=None
        self.smaller_vers=smaller_vers
        self.mother_data=mother_data



    def _generate_scenarios(self):

        cal_dat=self.calliope
        cal_dat['scenarios']=cal_dat['scenarios'].astype(str)
        try:
            scenarios = cal_dat['scenarios'].unique().tolist()
        except KeyError as e:
            cols=cal_dat.columns
            raise KeyError(f'Input data error. Columns are {cols}.', f' and expecting {e}.')
        if self.smaller_vers is not None:  # get a small version of the data ( only 3 scenarios )
            try:
                scenarios = scenarios[:self.smaller_vers]
            except:
                raise ValueError('Scenarios out of bonds')

        return self._get_scenarios()



    def _get_scenarios(self): # Implementing: work fine

        cal_dat = self.calliope
        cal_dat['scenarios'] = cal_dat['scenarios'].astype(str)
        scenarios = [str(x) for x in cal_dat['scenarios'].unique()]  # Convert to string, just in case the scenario is a number

        scenarios=[
            Scenario(name=str(scenario),
                     activities=[
                         Activity_scenario(
                             alias=row['aliases'],
                             amount = row['flow_out_sum'],
                             unit=row['new_units']
                         )
                         for _,row in group.iterrows()
                     ]).to_dict()
            for scenario,group in cal_dat.groupby('scenarios')
        ]
        return scenarios


    def _get_methods(self):
        processors = pd.read_excel(self.motherfile, sheet_name='Methods')
        methods=[Method(meth).to_dict() for meth in processors['Formula'].apply(eval)]

        return  {key: value for key, value in [list(item.items())[0] for item in methods]}



    def run(self, path= None):
        """public function """

        self.hierarchy=Hierarchy(base_path=self.motherfile, motherdata=self.mother_data).generate_hierarchy()
        enbios2_methods= self._get_methods()

        self.enbios2_data = {
            "bw_project": bw_project,
            "hierarchy": self.hierarchy,
            "methods": enbios2_methods,
            "scenarios": self._generate_scenarios()
        }
        pass
        if path is not None:
            with open(path, 'w') as gen_diction:
                json.dump(self.enbios2_data, gen_diction, indent=4)
            gen_diction.close()

        print('Input data for ENBIOS created')



class Hierarchy:
    def __init__(self, base_path: str, motherdata):
        self.parents = pd.read_excel(base_path, sheet_name='Dendrogram_top')
        self.motherdata=motherdata
        self.data=self._transform_motherdata()


    def _transform_motherdata(self):
        """ Transform mother data into a config dictionary
        This should be equal to the last level of the hierarchy"""
        return [
            {'name': x.alias, 'adapter': 'bw', 'config': {'code': x.code}} for x in self.motherdata
        ]


    def generate_hierarchy(self):
        last_level=None
        last_level_branches=[]
        last_level_hierachy=[] # official list

        for level in reversed(self.parents['Level'].unique().tolist()):
            data=self.parents.loc[self.parents['Level']==level]
            if last_level is None:
                last_level_branches = [
                    Last_Branch(
                        name=row['Processor'],
                        level=level,
                        parent=row['ParentProcessor'],
                        origin=[x for x in self.motherdata if x.parent == row['Processor']],
                    )
                    for _, row in data.iterrows()
                ]
                last_level_hierachy=[x.leafs for x in last_level_branches]
                last_level=level
                continue

            if last_level is not None and level!=self.parents['Level'].unique().tolist()[0]:
                last_level_branches = [
                    Branch(
                        name = row['Processor'],
                        level=level,
                        parent= row['ParentProcessor'],
                        origin=[x for x in last_level_branches if x.parent == row['Processor']]
                    )
                    for _, row in data.iterrows()]

                last_level_hierachy=[x.leafs for x in last_level_branches]

            else:
                last_level_branches = [
                    Branch(
                        name=row['Processor'],
                        level=level,
                        parent=row['ParentProcessor'],
                        origin=[x for x in last_level_branches if x.parent == row['Processor']]
                    )
                    for _, row in data.iterrows()]
                return {'name': last_level_branches[0].name, 'aggregator': 'sum', 'children': last_level_branches[0].leafs}





