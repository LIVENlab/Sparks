"""
@LexPascal
"""
import json
from typing import Union,Optional
import pandas as pd
import bw2data as bd
from enbios2.base.experiment import Experiment
import os
from ProspectBackground.util.preprocess.cleaner import Cleaner
from ProspectBackground.util.preprocess.SoftLink import SoftLinkCalEnb
import bw2io as bi
from ProspectBackground.const import const
from dataclasses import dataclass
from ProspectBackground.util.preprocess.template_market_4_electricity import Market_for_electricity
from ProspectBackground.util.updater.background_updater import Updater
import time
import pprint

@dataclass
class Prospect():

    __scenarios=[]
    __Softlink=None
    __exec_time={}

    def __init__(self, caliope : Union[str, pd.DataFrame], mother_file: [str], project : [str], database : [str]):
        """
        @param caliope: path to the caliope data (flow_out_sum.csv)
        @type caliope: either str path or pd.Dataframe
        @param mother_file: path to the mother file
        @type mother_file: str
        @param project: project name in bw
        @type project: str
        @param database: db name in bw
        @type database: str
        """
        self.default_market = None #
        self.project=project
        self.calliope=caliope
        self.mother=mother_file
        self.techs=[]

        self.scenarios=[]

        self.preprocessed_starter=None
        self.template_electricity_market = None

        self.database=database
        self.exp=None

        #Check project and db
        self.BW_project_and_DB()

    @staticmethod
    def timer(func):
        def wrapper(*args, **kwargs):
            start = time.time()
            func(*args,**kwargs)
            end = time.time()
            total=end-start
            Prospect.__exec_time[func.__name__]=total
        return wrapper


    def BW_project_and_DB(self):
        """
        Check the BW project and database.
        It also allows for the creation of a new one
        """
        projects=list(bd.projects)

        if self.project not in str(projects):
            ans=input(f'Project {self.project} not in projects. Want to create a new project? (y/n)')
            if ans =='y':
                database=input('Enter the DB name that you want to create:')
                spolds=input('Enter path to the spold files ("str"):')
                self.create_BW_project(self.project, database,spolds)
                const.bw_project=self.project
                const.bw_db=database
            else:
                raise Warning('Please, create a project before continue')

        bd.projects.set_current(self.project)
        if self.database not in list(bd.databases):
            print(list(bd.databases))
            raise Warning(f"database {self.database} not in bw databases")
        print('Project and Database existing...')

    @staticmethod
    def create_BW_project(project,database,spolds):
        """
        Create a new project and database
        """
        bd.projects.set_current(project)
        bi.bw2setup()
        ei=bi.SingleOutputEcospold2Importer(spolds,database,use_mp=False)
        ei.apply_strategies()
        ei.write_database()
        pass

    @timer
    def preprocess(self,subregions : [bool,Optional] = False):
        """
        cal_file: str path to flow_out_sum data

        moth_file: path file to excel data (check examples)

        returns:
            -pd.DataFrame: modified Calliope file with the following changes:
                -Unit adaptation
                -Scaling according to the conversion factor
                -Filtered activities contained in the mother file
        ___________________________
        """

        # Create an instance of the Cleaner class
        cleaner=Cleaner(self.calliope,self.mother,subregions)
        self.preprocessed_starter=cleaner.preprocess_data()
        self.preprocessed_units=cleaner.adapt_units()
        self.locations=cleaner.locations
        self.exluded_techs_and_regions=cleaner.techs_region_not_included



    @timer
    def data_for_ENBIOS(self,path_save=None,smaller_vers=None):
        """
        Transform the data into enbios like dictionary
        """
        # Create an instance of the SoftLInkCalEnb
        self.__Softlink=SoftLinkCalEnb(self.preprocessed_units,self.mother,smaller_vers)
        self.__Softlink.run(path_save)
        self.enbios2_data = self.__Softlink.enbios2_data
        self.save_json_data(self.enbios2_data, path_save)


    @timer
    def template_electricity(self, final_key, Units):
        """
        This function creates the template activity for the market for electricity using the data of enbios.
        It gets all the activities with _electicity_ in the alias
        """
        # Create an instance of the class


        market_class=Market_for_electricity(self.enbios2_data,final_key,regions=self.locations, units=Units)
        temp=market_class.build_templates()
        self.default_market=temp
        # DO SOME TEST HERE
        updater = Updater(self.enbios2_data, self.default_market)
        updater.update_results('1')
        self.default_market=updater.template # Update the dictionary with the new info from Updater
        pass
        #for reg in self.locations:


        #self.electricity_activities = market_class.get_elec_acts()  # run get list to

        #self.default_market = market_class.template_market_4_electricity(Location,Activity_name,Activity_code,Reference_product,Units)
        #self.template_code=Activity_code


    def classic_run(self):

        general_path=self.path_saved
        self.exp = Experiment(general_path)
        self.exp.run()
        result=self.exp.result_to_dict()



    def updater_run(self):


        # check if template created

        if self.template_code is None:
            raise TypeError(
                f'An error occurred. The template for electricity is {self.template_code}. Please, consider running {self.template_electricity.__name__} before')

        general = self.enbios2_data
        general_path = self.path_saved
        scenarios = list(general['scenarios'].keys())
        try:
            exp = Experiment(general_path)
        except Exception as e:
            # Generally the exception is the unspecificEcoinvent error from ENBIOS
            from enbios2.base.unit_registry import ecoinvent_units_file_path
            text_to_write = 'unspecificEcoinventUnit = []'
            # Abre el archivo en modo escritura ('w')
            with open(ecoinvent_units_file_path, 'w') as file:
                file.write(text_to_write)
            print(f'error {e} covered and solved')
            exp=Experiment(general_path)
            pass



        updater=Updater(general,self.default_market)

        for scenario in scenarios:
            print(f'parsing scenario {scenario}')

            template=updater.inventoryModify(scenario)
            self.template_electricity_market=template # update the table
            updater.exchange_updater(self.template_code)
            exp.run_scenario(scenario)


        pass


    def save_json_data(self,data, path):
        if path is not None:
            try:
                with open(path, 'w') as file:
                    json.dump(data, indent=4)
                self.path_saved=path
            except FileNotFoundError:
                raise FileNotFoundError(f'Path {path} does not exist. Please check it')
        else:
            current=os.path.dirname(os.path.abspath(__file__))
            folder_path=os.path.join(os.path.dirname(current),'Default')

            print(current)

            os.makedirs(folder_path,exist_ok=True)
            file_path=os.path.join(folder_path,'data_enbios.json')
            print(file_path)

            with open(file_path, 'w') as file:
                json.dump(data, file,indent=4)
            print(f'Data for enbios saved in {file_path}')
            self.path_saved=file_path

    @classmethod
    def get_execution_times(cls):
        return cls.__exec_time





if __name__=='__main__':
    tr=UpdaterExperiment(r'C:\Users\altz7\PycharmProjects\enbios__git\projects\seed\MixUpdater\data\flow_out_sum.csv',r'C:\Users\altz7\PycharmProjects\enbios__git\projects\seed\MixUpdater\data\base_file_simplified.xlsx','Seeds_exp4','db_experiments')
    tr.preprocess()
    tr.data_for_ENBIOS()
    tr.template_electricity('Electricity_generation', Location='PT', Reference_product='electricity production, 2050 in Portugal test',Units='kWh')
    tr.run()


