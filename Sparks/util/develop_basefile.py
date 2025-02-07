import pandas as pd
import bw2data as bd
import bw2io as bi
from dataclasses import dataclass, field, InitVar
from typing import Union, Optional, List,Tuple
import warnings
from bw2data.errors import UnknownObject
from bw2data.backends import Activity, ActivityDataset



class Support():
    def __init__(self,
                 file: [str],
                 project: [str],
                 calliope: [str]
                 ):
        """
       
        """
        self.file = file
        self.project = project
        self.calliope= calliope
        bd.projects.set_current(self.project)


        self._read_excel(self.file)
        self._get_technologies()
        self._get_locations()
        self._append_contry_techs()

      

    def _read_excel(self, file):  # Added self as the first parameter
    
        self.om = pd.read_excel(file, sheet_name='o&m')  # Changed o&m to om
        self.infrastructure = pd.read_excel(file, sheet_name='infrastructure')
        


    def _get_technologies(self):

        """
        get pairs of technology - filename  in a list of tuples
        """
        # Append technology name and source file in tuples 
        list_om = []
        for index, row in self.om.iterrows():
            act=BaseAct(name=row['technology_name_calliope'],
            ecoinvent_name = row['life_cycle_inventory_name_o&m'],
            file_name=row['calliope_om_file'],
            factor= row['prod_scaling_factor'],
            ecoinvent_location = row['prod_location'],
            database = row['prod_database'] ,
            calliope_units = row['calliope_prod_unit'])
           
            list_om.append(act)
        pass
        self.list_om =list_om


        list_inf = []
        for index, row in self.infrastructure.iterrows():
            pair = (row['technology_name_calliope'],
             row['calliope_capacity_file'])
            list_inf.append(pair)
        self.list_inf = list_inf

    
    
    def _get_locations(self):
        """
        assume that one csv will define all the countries
        """
        data=pd.read_csv(self.calliope['energy_cap.csv'])
        self.locs=data['locs'].unique().tolist()
        



    @staticmethod
    def _create_empty_dataframe():
        # Define the columns
        columns = [
            'Processor', 
            'Region', 
            '@SimulationCarrier', 
            'ParentProcessor', 
            '@SimulationToEcoinventFactor', 
            'Ecoinvent_key_code', 
            'File_source'
        ]
        
        # Create an empty DataFrame with the specified columns
        return pd.DataFrame(columns=columns)
    

    def _open_file(self, name):
        return setattr(self, name, pd.read_csv(self.calliope[name]))  
        
        

    def _append_contry_techs(self):
        
        df= self._create_empty_dataframe
        for element in self.list_inf:
            element=element[1] + '.csv'
            if  not hasattr(self, element):
                self._open_file(element) 
            at_val = getattr(self, element)
            pass

        


@dataclass
class BaseAct:
    """ Base Activity Definition"""
    name: str
    ecoinvent_name: str
    file_name: str
    factor: str
    ecoinvent_location: str
    database: str
    calliope_units: str

    init_post: InitVar[bool]=True # Allow to create an instance without calling alias modifications


    def __post_init__(self,init_post):
        if not init_post:
            return
       
        self.activity = self._load_activity(name=self.ecoinvent_name,
                                            database=self.database,
                                            location= self.ecoinvent_location)



    def _load_activity(self, name, database, location) -> Optional['Activity']:

        try:
            activity=list(ActivityDataset.select().where(
                (ActivityDataset.name == name) & 
                (ActivityDataset.database == database)))
            """
            pass
            if len(activity)>1:
                warnings.warn(f"More than one activity found with name {name}",Warning)
                pass
            if len(activity)<1:
                raise UnknownObject(f'No activity with code {name}')
            """
            if len(activity) != 0:
                act=[Activity(act) for act in activity]
                
                act=[a for a in act if a['location'] == location][0]
                if len(act) > 1:
                    raise('warning, weird stuff going on')
                return act[0]
            

        except (bw2data.errors.UnknownObject, KeyError):
            message = (f"\n{key} not found in the database. Please check your database / basefile."
                       f"\nThis activity won't be included.")
            warnings.warn(message, Warning)
            return None


