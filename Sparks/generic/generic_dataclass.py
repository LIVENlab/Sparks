from dataclasses import dataclass, field, InitVar
import bw2data
from typing import Union, Optional, List,Tuple
import warnings
import bw2data as bd
from bw2data.errors import UnknownObject
from bw2data.backends import Activity, ActivityDataset
from Sparks.const.const import bw_project,bw_db
bd.projects.set_current(bw_project)            # Select your project
database = bd.Database(bw_db)


@dataclass
class BaseFileActivity:
    """ Base class for motherfile data"""
    name: str
    region: str
    carrier: str
    parent: str
    code:str
    factor: Union[int, float]
    alias_carrier: Optional[str] = None
    alias_carrier_region: Optional[str] = None
    unit: Optional[str] = None
    init_post: InitVar[bool]=True # Allow to create an instance without calling alias modifications


    def __post_init__(self,init_post):
        if not init_post:
            return

        self.alias_carrier = f"{self.name}_{self.carrier}"
        self.alias_carrier_region=f"{self.name}__{self.carrier}___{self.region}"
        self.activity = self._load_activity(key=self.code)

        try:
            if isinstance(self.activity, Activity):
                self.unit = self.activity['unit']
        except:
            self.unit = None

    def _load_activity(self, key) -> Optional['Activity']:
        pass
        try:
            activity=list(ActivityDataset.select().where(ActivityDataset.code == key))

            if len(activity)>1:
                warnings.warn(f"More than one activity found with code {key}",Warning)
            if len(activity)<1:

                raise UnknownObject(f'No activity with code {key}')
            return Activity(activity[0])

        except (bw2data.errors.UnknownObject, KeyError):
            message = (f"\n{key} not found in the database. Please check your database / basefile."
                       f"\nThis activity won't be included.")
            warnings.warn(message, Warning)
            return None


@dataclass
class Activity_scenario:
    """ Class for each activity in a specific scenario"""

    alias: str
    amount: int
    unit: str #TODO: adapt
    pass

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
        self.leafs = [{'name': x.alias_carrier_region, 'adapter': 'bw', 'config': {'code': x.code}} for x in self.origin]



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


