"""
@LexPascal
"""

import pandas as pd
import bw2data as bd
import bw2io as bi
from dataclasses import dataclass, field, InitVar
from typing import Union, Optional, List, Tuple
import warnings
from bw2data.errors import UnknownObject
from bw2data.backends import Activity, ActivityDataset



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
    locations: list
        
    activities: List[Tuple[str, str, str]] = field(default_factory=list)
    
    generic_location: [bool] = False
    init_post: InitVar[bool]=True 


    def __post_init__(self, init_post):
        pass
        if self.ecoinvent_location != 'country': # Unique search 
            
            self.generic_location = True 
            self.activities.append(self._load_activity(
                                    name=self.ecoinvent_name,
                                    database=self.database,
                                    location= self.ecoinvent_location))

        else:
            self._load_multi_activity(name= self.ecoinvent_name,
                                    database=self.database,
                                    locations= self.locations)

        if not init_post:
            return
        


    def _load_activity(self, name:str, database:str, location:str): #TODO: add typing       

        activity=list(ActivityDataset.select().where(
                (ActivityDataset.name == name) & 
                (ActivityDataset.database == database)))
        pass

        if len(activity) != 0:

            act=[Activity(act) for act in list(activity)]                
            act=[a for a in act if a['location'] == location]

            if len(act) < 1:  # Raise KeyError if no activity found for the specified location
                
                message = (f"No activity in database with name {name} and location {location}")
                warnings.warn(message, Warning)

                if self.generic_location:
                    return (name, location, 'DB Undefined')
                
                else:
                    return(name,location, 'unfound location')
            
            act= [a for a in act if a['location'] == location][0]
        
            pass
            return  (name, location, act)

        else:
            message = (f"No activity in database with name {name}")
            warnings.warn(message, Warning)
            return (name, location, 'DB Undefined')


       
    
    def _load_multi_activity(self, name:str, database: str, locations: list):
        """
        if activity specific for countries, iter list 
        """

        for element in self.locations:
            loc = element[1]
            act=self._load_activity(name=self.ecoinvent_name,
                                database=self.database,
                                location= loc)
            self.activities.append(act)






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

        pass
        self._get_locations()
        pass
        self._append_contry_techs()
        pass
        self._get_technologies()
        
        

      

    def _read_excel(self, file):  # Added self as the first parameter
    
        self.om = pd.read_excel(file, sheet_name='o&m')  # Changed o&m to om
        self.infrastructure = pd.read_excel(file, sheet_name='infrastructure')
        self._iter_sheet(self.om)
        

    
    def _iter_sheet(self, df) -> List[BaseAct]:

        # create excel

        activities = []

        for _,row in df.iterrows():
            # 1: look for the file
            # 2: get unique combinations
            combinations = self._extract_combinations(row['technology_name_calliope'], row['calliope_file'])
            
            activities.append(BaseAct(name=row['technology_name_calliope'],
            ecoinvent_name = row['life_cycle_inventory_name_o&m'],
            file_name=row['calliope_file'],
            factor= row['prod_scaling_factor'],
            ecoinvent_location = row['prod_location'],
            database = row['prod_database'] ,
            calliope_units = row['calliope_prod_unit'],
            locations = combinations))
            
           


            # 3: search for db code
            # 4: store them in the final excel



    def _extract_combinations(self, name, file) -> List[Tuple[str, str]]:
        """
        open file and extract combinations of name - location
        """
        
        file = file +'.csv'

        if  not hasattr(self, file):
            self._open_file(file) 
        
        doc = getattr(self, file)
        combinations = list(doc.loc[doc['techs'] == name, ['techs', 'locs']].itertuples(index=False, name=None))  
        return  list(set(combinations))






    def _get_technologies(self):

        """
        Transform a list of basic activities
        """
       
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
        

    def _append_contry_techs(self):
        """
        Open the different files, get couples of tech-location, and get a unique list
        """
        elements = []
        for element in self.calliope:
            print('parsing element', element)
            df = pd.read_csv(self.calliope[element])
            df['name'] = element  # Assign the name of the file to the 'name' column
            
            # Append tuples of (techs, carriers, locs, file_name) to elements
            elements.extend(zip(df['techs'], df['carriers'], df['locs'], df['name']))  # Use df['name'] to get the file name

        
        pass
        pass
        df= self._create_empty_dataframe
        for element in self.list_inf:
            element=element[1] + '.csv'
            if  not hasattr(self, element):
                self._open_file(element) 
            at_val = getattr(self, element)
            pass


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
        
        