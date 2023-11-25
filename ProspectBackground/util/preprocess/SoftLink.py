import pprint
import json
import random
from logging import getLogger
import bw2data as bd
from collections import defaultdict
import pandas as pd
from bw2data.errors import UnknownObject
from typing import Optional,Dict,Union,List
from ProspectBackground.const.const import bw_project,bw_db
import warnings
from ProspectBackground.errors.errors import HierarchyError
from ProspectBackground.util.DataTypes.DataTypes import ActivityData,Scenarios_Dict,ActitiesDict,Hierarchy




bd.projects.set_current(bw_project)            # Select your project
database = bd.Database(bw_db)        # Select your db

class SoftLinkCalEnb():
    """
    This class allows to create an ENBIOS-like input

    Class that allows to move from Calliope-mother file data to input data for enbios

    """
    __scenarios=[]
    __delete_keys=[]
    def __init__(self,calliope,motherfile):
        self.calliope=calliope
        self.motherfile=motherfile
        self.dict_gen=None
        self.scens=None
        self.filtered_mother=[pd.DataFrame]
        self.aliases=[]
        self.final_acts={}
        self.hierarchy_tree=None


    # Define some satandard classes for the data





    def generate_scenarios(self, smaller_vers=None):
        """SoftLink
        Iterate through the data from calliope (data.csv, output results...)
            -The function includes an intermediate step to create the hierarchy


        :param calliope_data:
        :param smaller_vers: BOOL, if true, a small version of the data for testing gets produced
        :return:scen_dict, acts
                *scen dict --> dictionary of the different scenarios
                *acts --> list including all the unique activities
        """
        if isinstance(self.calliope, pd.DataFrame):
            cal_dat=self.calliope

        elif isinstance(self.calliope,str):

            cal_dat = pd.read_csv(self.calliope, delimiter=',')
        else:
            raise Exception(f'Input data error in {self.generate_scenarios.__name__}')

        #cal_dat['aliases'] = cal_dat['techs'] + '__' + cal_dat['carriers'] + '___' + cal_dat['locs']  # Use ___ to split the loc for the recognision of the activities
        cal_dat['scenarios']=cal_dat['scenarios'].astype(str)

        try:
            scenarios = cal_dat['scenarios'].unique().tolist()
            self.__scenarios = scenarios
        except KeyError as e:
            cols=cal_dat.columns
            raise KeyError(f'Input data error. Columns are {cols}.', f' and expecting {e}.')




        scenarios=[str(x) for x in scenarios] # Convert to string, just in case the scenario is a number
        scenarios_dictionary=Scenarios_Dict()

        for scenario in scenarios:
            df = cal_dat[cal_dat['scenarios'] == scenario]
            info = SoftLinkCalEnb.get_scenario(df)
            scenarios_dictionary.add_scenario(scenario,info)


        # GENERATE KEYS FOR THE SCENARIOS

        scen_dict=scenarios_dictionary.scenario_data

        acts = list(scen_dict[random.choice(list(scen_dict.keys()))]['activities'].keys())
        activities=set(acts)
        # Create intermediate information for the hierarchy


        dict_gen=SoftLinkCalEnb.get_regionalized_processors(*activities)

        self.dict_gen=dict(dict_gen) # avoid missing key errors
        self.acts=activities
        self.scens=scenarios_dictionary.scenario_data
        pass

    @staticmethod
    def get_regionalized_processors(*args)-> Dict[str,list [str]]:
        """
        This function returns the following information:
        Each processor has multiple aliases, depending on the regions
        For example: 'wind_onshore__electricity': ['wind_onshore__electricity___ESP', 'wind_onshore__electricity___DEU']
        Return a dictionary with this information
        """
        general=defaultdict(list)
        pass
        for act in args:
            act_key=act.split('___')[0]
            general[act_key].append(act)
        return general

    @staticmethod

    def get_scenario(df) -> Dict[str , List[Union[str,int]]]:
        """
        Iters through 1 scenario of the calliope_flow_out_sum.csv (scenarios data), storing basic data in a dictionary
        Get {
        activities : {
            alias : [
            unit,
            amount]}}
        :param df:
        :return:
        """
        activity_data=ActivityData()
        for index,row in df.iterrows():
            alias = row['aliases']
            flow_out_sum = (row['flow_out_sum'])
            unit = row['units']
            activity_data.add_activity(alias,unit,flow_out_sum)
        return dict(activity_data.data)




    def generate_activities(self,*args) -> dict:
        """
        This function reads the Excel "mother file" and creates a dictionary.
        It reads the BWs' codes and extracts information from each activity and stores them in a dictionary

        :param args:
        :return:
        """
        pass
        #TODO: THIS ACTIVITY IS CAUSING PROBLEMS
        processors = pd.read_excel(self.motherfile, sheet_name='Processors')

        delete={}
        activities_cool = {}
        # Create an instance of the activities dictionary

        activities_dictionary=ActitiesDict()

        for index, row in processors.iterrows():
            code = str(row['BW_DB_FILENAME'])
            try:
                act = database.get_node(code)

            except UnknownObject:
                mess=f'''\n{row['Processor']},has an unknown object in the db \n
                It will not be included in the input data'''
                warnings.warn(mess,Warning)

                continue
            name = act['name']
            alias=row['aliases']


            activities_cool[alias] = {
                'name': name,
                'code': code,
            }

        pass
        activities = {}
        for element in args:
            new_element = element.split('___')[0]  # This should match the name
            for key in activities_cool.keys():
                if new_element == key:
                    new_code = activities_cool[key]['code']
                    activities[element] = {
                        "id": {
                            'code': new_code
                        }
                    }

        pass
        self.final_acts = activities

        print('Activities stored as a dict')



    def generate_aternative_activities(self):
        calliope=self.calliope
        # Get all the aliases
        aliases_unique=calliope['aliases'].unique().tolist()
        processors_mother=pd.read_excel(self.motherfile, sheet_name='Processors')
        activities=ActitiesDict()
        for _,row in processors_mother.iterrows():
            alias=row['aliases']
            if str(alias) not in aliases_unique:
                # if alias not in the calliope filter; delete the row
                processors_mother.drop(index=row.name, inplace=True)
                continue
            else:
                activities.add_activity(alias,row['BW_DB_FILENAME'])

        pass
        self.filtered_mother=processors_mother
        self.final_acts = activities.activities
        return activities.activities

    def get_non_existing_hierarchy(self):
        """
        Check if some dendrogrm category it's not being used.
        Return an error requesting to delete non-used categories
        """
        levels = pd.read_excel(self.motherfile, sheet_name='Dendrogram_top')
        processors = self.filtered_mother

        processors=processors['ParentProcessor'].tolist()

        levels=levels['ParentProcessor'].tolist()
        # Join the two list

        levels_keys=levels+processors
        list_non_hierarchy=[]
        for level in levels_keys:

            if str(level) == 'nan':
                continue
            elif levels_keys.count(level) <2:
                list_non_hierarchy.append(level)

        if len(list_non_hierarchy)>0:
            raise HierarchyError(f'A hierarchy element is defined but not used. Please, check it. \n'
                                 f'{list_non_hierarchy}. Check deleted activities in warnings (if any)')







        pass


    def alternative_hierarchy(self,*args):
        """
                This function creates the hierarchy tree.
                It uses two complementary functions (generate_dict and tree_last_level).

                It reads the information contained in the mother file starting by the bottom (n-lowest) level
                :param data:
                :param args:
                :return:
                """


        print('Creating tree following the structure defined in the basefile')
        self.get_non_existing_hierarchy()
        df = pd.read_excel(self.motherfile, sheet_name='Dendrogram_top')
        #df2 = pd.read_excel(self.motherfile, sheet_name='Processors')
        df2=self.filtered_mother
        #a = Hierarchy(processors=df2,parents=df)
        #a.process_input()
        #a.process_top_levels()

        # Do some changes to match the regions and aliases

        df2['Processor'] = df2['Processor'] + '__' + df2['@SimulationCarrier']  # Mark, '__' for carrier split
        df2=df2.drop('Processor',axis=1)
        df2=df2.rename(columns={'aliases':'Processor'})
        # Start by the last level of parents
        levels = df['Level'].unique().tolist()
        last_level_parent = int(levels[-1].split('-')[-1])

        last_level_processors = 'n-' + str(last_level_parent + 1)
        df2['Level'] = last_level_processors
        df = pd.concat([df, df2[['Processor', 'ParentProcessor', 'Level']]], ignore_index=True, axis=0)
        pass
        levels = df['Level'].unique().tolist()

        list_total = []
        for level in reversed(levels):
            df_level = df[df['Level'] == level]
            if level == levels[0]:
                break
            elif level == levels[-1]:
                last = self.tree_last_level(df_level, *args)
                global last_list
                last_list = last
            else:
                df_level = df[df['Level'] == level]
                list_2 = self.generate_dict(df_level, last_list)
                last_list = list_2
                list_total.append(list_2)

        dict_tree = list_total[-1]
        self.hierarchy_tree = dict_tree[-1]
        pass

    @staticmethod
    def tree_last_level(df, *args):
        """
        This function supports the creation of the tree.
        It's specific for the lowest level of the dendrogram
        Return the act
        :param df:
        :param names: comes from generate scenarios. List of unique aliases
        :return:
        """
        # TODO: let's avoid hierarchy refinement
        new_rows = []
        for index, row in df.iterrows():
            processor = row['Processor']
            for element in args:
                cop = element.split('___')[0]
                if cop == processor:
                    new_row = row.copy()
                    new_row['Processor'] = element
                    new_rows.append(new_row)

        df = pd.concat([df] + new_rows, ignore_index=True)
        last_level_list = []
        # Return a list of dictionaries
        parents = list(df['ParentProcessor'].unique())
        for parent in parents:
            last_level = {}
            df3 = df[df['ParentProcessor'] == parent]
            childs = df3['Processor'].unique().tolist()
            last_level[parent] = childs
            last_level_list.append(last_level)
        pass

        return last_level_list

    @staticmethod
    def generate_dict(df, list_pre):
        """
        Pass a list of the lower level and the dataframe of the present
        Returns a list of the dictionary corresponding that branch

        :param df:
        :param list:
        :return:
        """
        parents = df['ParentProcessor'].unique().tolist()
        list_branches = []
        for parent in parents:
            branch = {}
            df_parent = df[df['ParentProcessor'] == parent]
            branch[parent] = {}

            for _, row in df_parent.iterrows():
                child_df = row['Processor']
                for element in list_pre:
                    if child_df in element:
                        branch[parent][child_df] = element[child_df]
            list_branches.append(branch)

        return list_branches




    def hierarchy_refinement(self,hierarchy_dict):
        """
        Include different regions in the tree

        Read the hierarchy dictionary and do some modifications:
            * Replace the names of the activities by the ones defined in the mapping dictionary
            Ex: hydro_run_of_river__electricity : [hydro_run_of_river__electricity___PRT_1, hydro_run_of_river__electricity___PRT_2]

        :param hierarchy_dict:
        :return: same dictionary modified
        """
        # 1 look for the lists. List contain the last levels of the dendrogram, where the names need to be modified


        map_names=self.dict_gen


        for key,value in hierarchy_dict.items():

            if isinstance(value, list):
                if len(value) <1:
                    print(f'value {value}', key)
                # Copy the list

                values_copy = value[:]

                value.clear()

                for element in values_copy:
                    for key, val in map_names.items():
                        if element == key:
                            list_names = map_names[key]
                            # 3. Include the new names
                            for name in list_names:
                                value.append(name)


            elif isinstance(value, dict):

                self.hierarchy_refinement(value)

            for key,value in hierarchy_dict.items():
                if isinstance(value,list) and len(value)<1:
                    self.__delete_keys.append(key)
                    #del hierarchy_dict[key]
                    print('I GOT U',key,value)
        pass
        # Check if you have an empty list

        self.hierarchy_tree = hierarchy_dict

    def clean_key(self, key_delete):
        def delete_recursive(d):
            if key_delete in d:
                del d[key_delete]
            for value in d.values():
                if isinstance(value, dict):
                    delete_recursive(value)

        delete_recursive(self.hierarchy_tree)
    def get_methods(self):
        methods = {}
        processors = pd.read_excel(
            self.motherfile,
            sheet_name='Methods')
        processors['Formula'] = processors['Formula'].apply(eval)

        for method in processors['Formula']:
            methods.update({str(method[-1]): method})

        return methods



    def run(self, path= None):

        self.generate_scenarios()
        self.generate_aternative_activities()
        self.alternative_hierarchy(*self.final_acts)
        #self.hierarchy_refinement(hierarchy_dict=self.hierarchy_tree)
        for element in self.__delete_keys:
            self.clean_key(element)

        enbios2_methods= self.get_methods()

        self.enbios2_data = {
            "bw_project": bw_project,
            "activities": self.final_acts,
            "hierarchy": self.hierarchy_tree,
            "methods": enbios2_methods,
            "scenarios": self.scens
        }
        pass
        if path is not None:
            with open(path, 'w') as gen_diction:
                json.dump(self.enbios2_data, gen_diction, indent=4)
            gen_diction.close()


        print('Input data for ENBIOS created')

    @property
    def _scenarios_list_(self):
        return self.__scenarios



