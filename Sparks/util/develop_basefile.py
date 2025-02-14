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
    ecoinvent_unit: str
    database: str
    calliope_units: str
    locations: list

        
    activities: List[Tuple[str, str, str]] = field(default_factory=list)
    basefile_activities : List[Tuple] = field(default_factory = list)


    generic_location: [bool] = False
    init_post: InitVar[bool]=True 


    def __post_init__(self, init_post):
        
        if self.ecoinvent_location != 'country': # Unique search 
            
            self.generic_location = True 
            self.activities.append(self._load_activity(
                                    name=self.ecoinvent_name,
                                    database=self.database,
                                    location= self.ecoinvent_location,
                                    unit= self.ecoinvent_unit))
        
        else:
            self._load_multi_activity(name= self.ecoinvent_name,
                                    database=self.database,
                                    locations= self.locations,
                                    unit = self.ecoinvent_unit)


        self._combine_activities_and_locs()
        
        
        if not init_post:
            return
        

    def _load_activity(self, name:str, database:str, location:str, unit: str) -> Tuple[str, str, str]:      
        """Load a single activity from the database."""

        activity=list(ActivityDataset.select().where(
                (ActivityDataset.name == name) & 
                (ActivityDataset.database == database)))

        if len(activity) != 0:

            act=[Activity(act) for act in list(activity)]    

            act= [a for a in act if a['unit'] == unit]            
            
            act=[a for a in act if a['location'] == location]

            if len(act) < 1:  
                
                message = (f"No activity in database with name {name} and location {location}")
                warnings.warn(message, Warning)

                if self.generic_location:
                    return (name, location, 'DB Undefined')
                
                else:
                    return(name, location, 'unknown location')
            
            act= [a for a in act if a['location'] == location][0]
            
            return  (name, location, act['code'])

        else:
            message = (f"No activity in database with name {name}")
            warnings.warn(message, Warning)
            
            return (name, location, 'DB Undefined')


       
    
    def _load_multi_activity(self, name:str, database: str, locations: list, unit : str) -> None:
    
        """Load multiple activities based on provided locations."""
    
        for element in self.locations:
            loc = element[1]
            act=self._load_activity(name=self.ecoinvent_name,
                                database=self.database,
                                location= loc,
                                unit = unit)
            
            self.activities.append(act)


    def _combine_activities_and_locs(self):

        if self.generic_location:
            # logic 1: all the locations will contain the same bd.code
            for element in self.locations:
                element = element + self.activities[0]
                self.basefile_activities.append(element)
                
        else:
            # logic_2: search location in two list and extend.
            # Use dictionaries to improve efficiency
            grouped_dict = {}
            
            for item in self.locations:
                key = item[1]  
                if key not in grouped_dict:
                    grouped_dict[key] = []
                grouped_dict[key].append(item)
            
            for item in self.activities:
                key = item[1]  

                if key in grouped_dict:          
                    for match in grouped_dict[key]:                      
                        summed_tuple = (match + item)  
                        self.basefile_activities.append(summed_tuple)
                        

        
class Support():
    def __init__(self,
                 file: [str],
                 project: [str],
                 calliope: [str]
                 ):
      
        self.file = file
        self.project = project
        self.calliope= calliope
        bd.projects.set_current(self.project)

        self.df = self._create_empty_dataframe()    


    def run(self) -> None:   
        """Run the processing on the provided file."""
        self.om = pd.read_excel(self.file, sheet_name='o&m')  
        self.infrastructure = pd.read_excel(self.file, sheet_name='infrastructure')
        
        self._iter_sheet(self.infrastructure)
        self._iter_sheet(self.om)
        
            
    def _iter_sheet(self, df) -> None:
            
        activities = []

        for _,row in df.iterrows():
            
            # 1: look for the file
            # 2: get unique combinations
            combinations = self._extract_combinations(row['technology_name_calliope'], row['calliope_file'])
            
            activities.append(BaseAct(name=row['technology_name_calliope'],
            ecoinvent_name = row['life_cycle_inventory_name'],
            file_name=row['calliope_file'],
            factor= row['prod_scaling_factor'],
            ecoinvent_location = row['prod_location'],
            ecoinvent_unit = row['prod_unit'],
            database = row['prod_database'] ,
            calliope_units = row['calliope_prod_unit'],
            locations = combinations))
        
        # 4: store them in the final excel
        self._store_excel(activities)


    def _store_excel(self,activities: List[BaseAct]) -> None:
        """
        Store activities in a DatFrame formated Excel file.
        """
        new_rows=[]
        
        for activity in activities:
            for loc in activity.basefile_activities:
                new_row ={
            'Processor' : activity.name, 
            'Region': loc[1], 
            '@SimulationCarrier' : activity.calliope_units, 
            'ParentProcessor' : 'Unknown', 
            '@SimulationToEcoinventFactor' : activity.factor, 
            'Ecoinvent_key_code' : loc[-1], 
            'File_source' : activity.file_name + '.csv',
            'activity_name_passed' : loc[2],
            'location_passed': loc[-2]
                }
            
                new_rows.append(new_row)
        
        new_df = pd.DataFrame(new_rows)
        self.df = pd.concat([self.df, new_df], ignore_index = True)
    
           


    def _extract_combinations(self, name, file) -> List[Tuple[str, str]]:
        """
        Extract unique combinations of name and location from the file.
        """
        
        file = file + '.csv'
        if not hasattr(self, file):
            self._open_file(file) 
        
        doc = getattr(self, file)
        combinations = list(doc.loc[doc['techs'] == name, ['techs', 'locs']].itertuples(index=False, name=None))  
        
        return list(set(combinations))


    @staticmethod
    def _create_empty_dataframe():
        # Define columns
        columns = [
            'Processor', 
            'Region', 
            '@SimulationCarrier', 
            'ParentProcessor', 
            '@SimulationToEcoinventFactor', 
            'Ecoinvent_key_code', 
            'File_source',
            'activity_name_passed',
            'location_passed'
        ]
        
        # Create an empty DataFrame with the specified columns
        return pd.DataFrame(columns=columns)
    

    def _open_file(self, name):
        return setattr(self, name, pd.read_csv(self.calliope[name]))  
        
        