import copy
import json
import pandas as pd
from pathlib import Path
import pandera as pa
import logging
import ast
from Sparks.generic.basefile_schema import calliope_cleaning_schema, methods_schema
from Sparks.generic.generic_dataclass import *


logger = logging.getLogger("sparks")
bd.projects.set_current(bw_project)            


class SoftLinkCalEnb():
    """
    Transform table-like input into a hierarchy (ENBIOS) like format
    """

    def __init__(self,
                 calliope: pd.DataFrame,
                 mother_data: list[BaseFileActivity],
                 sublocations: list[str],
                 motherfile: Path,
                 smaller_vers: bool =None):

        self.calliope=calliope.copy()
        self.motherfile=Path(motherfile)
        self.sublocations=sublocations
        self.smaller_vers=smaller_vers
        self.mother_data=mother_data

        logger.info("===Soflink class initiated===")
        self._validate_inputs()

    def _validate_inputs(self)-> None:
        """
        Validate input data using pandera schemas
        :return:
        """
        logger.info("Validating input data")
        #try:
         #   calliope_cleaning_schema.validate(self.calliope, lazy=True)
        #except pa.errors.SchemaErrors as e:
         #   logger.error(f"Input data validation error: {e.failure_cases}")
          #  raise
        #logger.info("Input data validated successfully")
        
        # check unique entries
        act_names = [getattr(a, "full_name", None) for a in self.mother_data]
        if len(act_names) != len(set(act_names)):
            logger.error("Duplicate full_name detected among mother_data activities. Consider deduplicating.")
            raise ValueError("Duplicate full_name detected among mother_data activities. Consider deduplicating.")




    def _generate_scenarios(self):
        cal_dat=self.calliope 
        cal_dat['scenarios']=cal_dat['scenarios'].astype(str)
        try:
            scenarios = cal_dat['scenarios'].unique().tolist()
        except KeyError as e:
            cols=cal_dat.columns
            raise KeyError(f'Input data error. Columns are {cols}.', f' and expecting {e}.')

        return self._get_scenarios()

    def _get_scenarios(self):
        cal_dat = self.calliope
        cal_dat['scenarios'] = cal_dat['scenarios'].astype(str)

        if self.smaller_vers:  # get a small version of the data (only the first)
            try:
                scenario = cal_dat['scenarios'].unique().tolist()[0]
                cal_dat['scenarios'] = cal_dat['scenarios'].astype(str)
                cal_dat = cal_dat[cal_dat['scenarios'] == str(scenario)]
                logger.info(f"Using only scenario {scenario}")
            except:
                raise ValueError('Scenarios out of bonds')

        scenarios_check = [str(x) for x in
                           cal_dat['scenarios'].unique()]  # Convert to string, just in case the scenario is a number

        scenarios = [
            Scenario(name=str(scenario),
                     activities=[
                         Activity_scenario(
                             alias=row['full_name'],
                             amount=row['energy_value'],
                             unit=row['new_units']
                         )
                         for _, row in group.iterrows()
                     ]).to_dict()
            for scenario, group in cal_dat.groupby('scenarios')
        ]
        assert (len(scenarios) == len(scenarios_check))
        return scenarios


    def _get_methods(self):
        """ Get methods from the motherfile"""
        methods_data = pd.read_excel(self.motherfile, sheet_name='Methods')
        logger.info("Validating methods sheet")

        try:
           methods_schema.validate(methods_data, lazy=True)
        except pa.errors.SchemaErrors as e:
            logger.error(f"Input data validation error: {e.failure_cases}")
            raise

        methods = []

        for meth in methods_data['Formula']:
            try:
                parsed = ast.literal_eval(meth)
                methods.append(
                    Method(parsed).to_dict(*parsed)
                )
            except (ValueError, SyntaxError) as e:
                logger.error(f"Failed to parse method entry '{meth}': {e}")
                continue

        return  {key: value for d in methods for key, value in d.items()}


    def run(self, path= None):
        """public function """
        logger.info("Starting ENBIOS hierarchy generation")
        self.hierarchy=Hierarchy(base_path=self.motherfile,
                                 motherdata=self.mother_data,
                                 sublocations=self.sublocations ).generate_hierarchy()

        enbios2_methods= self._get_methods()
        logger.info(f"{len(enbios2_methods)} Methods extracted from motherfile")

        self.enbios2_data = {
            "adapters": [
                {
                    "adapter_name": "brightway-adapter",
                    "config": {"bw_project": bw_project},
                    "methods": enbios2_methods
                }
            ],
            "hierarchy": self.hierarchy,
            "scenarios": self._generate_scenarios()
        }

        if path is not None:
            with open(path, 'w') as gen_diction:
                json.dump(self.enbios2_data, gen_diction, indent=4)
            gen_diction.close()
        print('Input data for ENBIOS created')


class Hierarchy:
    def __init__(self, base_path: str, motherdata, sublocations:list):
        self.parents = pd.read_excel(base_path, sheet_name='Dendrogram_top')
        self.motherdata=motherdata
        self.subloc = sublocations
        logger.debug("Hierarchy class initiated")

        self.motherdata = self.manage_subregions()
        self.data=self._transform_motherdata()


        


    def _create_copies(self,
                       existing_act: BaseFileActivity,
                       new_names: List[str])->List[BaseFileActivity]:
            """ Pass a the name of an existing BasefileAct,
             a list of new names, and return a list of copies"""

            copies=[]
            for new_name in new_names:
                new_act=BaseFileActivity(
                    name=new_name,
                    region=existing_act.region,
                    carrier=existing_act.carrier,
                    parent=existing_act.parent,
                    code=existing_act.code,
                    factor=existing_act.factor,
                    full_alias = existing_act.full_alias,
                    alias_filename_loc=existing_act.alias_filename_loc,
                    init_post=False
                )
                new_act = copy.deepcopy(existing_act)
                new_act.name = new_name
                new_act.alias_carrier_region = new_name
                copies.append(new_act)

            return copies


    def manage_subregions(self):
        logger.debug("Managing subregions in hierarchy")
        seen = set()
        final_list = []
        for act in self.motherdata:
            new_names = [x for x in self.subloc if act.full_alias in str(x)]
            if new_names:
                copies = self._create_copies(act, new_names)
                for copy in copies:
                    if copy.alias_carrier_region not in seen:
                        final_list.append(copy)
                        seen.add(copy.alias_carrier_region)
            else:
                final_list.append(act)

        return final_list


    def _transform_motherdata(self):
        """ Transform mother data into a config dictionary
        This should be equal to the last level of the hierarchy"""

        unique_dict = {}
        for x in self.motherdata:
            if x.alias_carrier_region not in unique_dict:
                unique_dict[x.alias_carrier_region] = {'name': x.alias_carrier_region, 'adapter': 'bw',
                                                       'config': {'code': x.code}}
        unique_items = list(unique_dict.values())
        return unique_items


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