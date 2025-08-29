from dataclasses import dataclass, field, InitVar
import bw2data
from typing import Union, Optional, List,Tuple, Dict
import warnings
import bw2data as bd
from bw2data.errors import UnknownObject
from bw2data.backends import Activity, ActivityDataset
from Sparks.const.const import bw_project
bd.projects.set_current(bw_project)            
from dataclasses import dataclass, field
from typing import List
from collections import defaultdict
import warnings


@dataclass
class BaseFileActivity:
    """ Base class for motherfile data"""

    name: str
    region: str
    carrier: str
    parent: str
    code:str
    alias_filename_loc: str
    full_alias:str
    factor: Union[int, float]
    database: Optional[str] = None
    unit: Optional[str] = None
    init_post: InitVar[bool]=True # Allow to create an instance without calling alias modifications

    activity_cache = defaultdict(lambda: None)  # init cache to speedup db search!


    def __post_init__(self,init_post):

        if not init_post:
            return

        self.alias_carrier = f"{self.name}_{self.carrier}"
        self.alias_carrier_region=f"{self.name}__{self.carrier}___{self.region}"
        #self.alias_carrier_parent_loc =f"{self.alias_carrier}_{self.alias_carrier_parent_loc}"
        self.activity = self._load_activity(key=self.code)

        try:
            if isinstance(self.activity, Activity):
                self.unit = self.activity['unit']
        except:
            self.unit = None


    def _load_activity(self, key:str) -> Optional['Activity']:
        """
        key: code
        """

        if key in BaseFileActivity.activity_cache:
            return BaseFileActivity.activity_cache[key]

        try:
            activity=list(ActivityDataset.select().where(ActivityDataset.code == key))

            if len(activity)>1:
                warnings.warn(f"More than one activity found with code {key}",Warning)
            if len(activity)<1:

                raise UnknownObject(f'No activity with code {key}')

            result =  Activity(activity[0])

        except (bw2data.errors.UnknownObject, KeyError):
            message = (f"\n{key} not found in the database. Please check your database / basefile."
                       f"\nThis activity won't be included.")
            warnings.warn(message, Warning)
            result = None

        # save activity into the cache
        BaseFileActivity.activity_cache[key] = result
        return result


@dataclass
class Activity_scenario:
    """ Class for each activity in a specific scenario"""
    alias: str
    amount: int
    unit: str


@dataclass
class Scenario:
    """ Basic Scenario"""
    name: str
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

        self._filter_unique_origin_by_full_alias()  # Filter unique values


        self.leafs = [
            {
                'name': x.full_alias,
                'adapter': 'bw',
                'config': (
                    {'code': x.code, 'database': x.database}
                    if x.database is not None else
                    {'code': x.code}
                )
            }
            for x in self.origin]

        if not self.leafs:
            warnings.warn(f"leafs not found for Last Tree Branch {self.name}, at {self.level}. This error can induce critical erros when using this data in enbios. Please, check the dendrogram structure")


    def _filter_unique_origin_by_full_alias(self):
        """Filters out duplicate full_alias entries from self.origin and logs duplicates."""
        alias_map = defaultdict(list)
        for activity in self.origin:
            alias_map[activity.full_alias].append(activity)

        unique = []
        duplicates_reported = False

        for alias, group in alias_map.items():
            if len(group) == 1:
                unique.append(group[0])
            else:
                duplicates_reported = True
                warning_msg = (
                    f"⚠️  Found {len(group)} entries with duplicated full_alias: '{alias}' "
                    f"in Last_Branch '{self.name}' at level '{self.level}'. Only the first occurrence will be kept.\n"
                )
                for i, item in enumerate(group, 1):
                    warning_msg += f"    [{i}] {item}\n"
                warnings.warn(warning_msg.strip())


                unique.append(group[0])

        self.origin = unique




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
        if not self.leafs:
            warnings.warn(f"leafs not found for Last Tree Branch {self.name}, at {self.level}. This  can induce critical errors when using this data in enbios. Please, check the dendrogram structure")



@dataclass
class Method:
    """
     For methods we need to pass a dictionary,
     where the keys are arbitrary names that we give to the method and the tuple of strings, 
     which are the names/identifiers of methods in brightway
    """
    method: tuple
    _used_keys = set() #Fixes issue #12
    def __post_init__(self):
        if self.method not in bd.methods:
            raise ValueError(f"Method {self.method} not found in brightway. Please, introduce the full method")
    
        


    def to_dict(self,*args):
        """
        @Enbios2: "For methods we need to pass a dictionary, 
        where the keys are arbitrary names that we give to the method and the tuple of strings,
          which are the names/identifiers of methods in brightway"
        This function returns a dictionary with a key being the second arbitrary element
        """
        base_key = self.method[2]
        
        if base_key in self._used_keys:
            counter = 1
            while f"{base_key}_{counter}" in self._used_keys:
                counter += 1
            key = f"{base_key}_{counter}"
        else:
            key = base_key 
        
        self._used_keys.add(key) #Issue #12
        return {key: list(args)}

    