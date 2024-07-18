"""
@LexPascal
"""
import json
import os
import time
from typing import Union, Optional
import pandas as pd
import bw2data as bd
import bw2io as bi
import warnings
from Sparks.util.preprocess.cleaner import Cleaner
from Sparks.util.preprocess.SoftLink import SoftLinkCalEnb
from Sparks.const import const


class SoftLink():

    def __init__(self,
                 caliope : Union[str, pd.DataFrame],
                 mother_file: [str],
                 project : [str],
                 database: str
                 ):
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
        self.project=project
        self.calliope=caliope
        self.mother=mother_file
        self.database = database
        self.SoftLink=None


        #Check project and db
        self._bw_project_and_DB()
        self._cleaner=Cleaner(caliope, mother_file) # Instance cleaner


    @staticmethod
    def timer(func):
        def wrapper(*args, **kwargs):
            start = time.time()
            func(*args,**kwargs)
            end = time.time()
            print(f'function {func.__name__} executed in {end - start} seconds')
        return wrapper


    def _bw_project_and_DB(self):
        """
        Check the BW project and database.
        It also allows for the creation of a new one
        """
        projects=list(bd.projects)
        if self.project not in str(projects):
            ans=input(f'Project {self.project} not in projects.'
                      f' Want to create a new project? (y/n)')
            if ans =='y':

                spolds=input('Enter path to the spold files ("str") in order to fill the database:')
                self._create_BW_project(self.project, self.database,spolds)
                const.bw_project=self.project
                const.bw_db=self.database
            else:
                raise Warning('Please, create a project before continue')

        bd.projects.set_current(self.project)
        if self.database not in list(bd.databases):
            print(list(bd.databases))
            warnings.warn(f"database {self.database} not in bw databases")
            ans=input("Do you want to import an ecoinvent database? y/n")
            if ans == 'y':
                spolds=str(input('Enter path to the spold files in order to fill the database:'))
                pass
                self._create_BW_project(self.project, self.database, spolds)

        self._save_const(self.project, self.database)
        print('Project and Database existing...')


    @staticmethod
    def _create_BW_project(project,database,spolds):
        """Create a new project and database"""
        pass
        bd.projects.set_current(project)
        bi.bw2setup()
        ei=bi.SingleOutputEcospold2Importer(spolds,database,use_mp=False)
        ei.apply_strategies()
        ei.write_database()


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

        """
        # Create an instance of the Cleaner class
        self._cleaner.subregions = subregions
        pass
        self._cleaner.preprocess_data()
        self.preprocessed_units= self._cleaner.adapt_units()
        self.exluded_techs_and_regions = self._cleaner.techs_region_not_included


    def data_for_ENBIOS(self, path_save=None,smaller_vers=False):
        """
        Transform the data into enbios like dictionary
        """
        # Create an instance of the SoftLInkCalEnb


        self.SoftLink=SoftLinkCalEnb(calliope=self.preprocessed_units,
                                     motherfile=self.mother,
                                     mother_data=self._cleaner.base_activities,
                                     sublocations=self._cleaner._techs_sublocations,
                                     smaller_vers=smaller_vers)
        self.SoftLink.run(path_save)
        self.enbios2_data = self.SoftLink.enbios2_data
        self._save_json_data(self.enbios2_data, path_save)


    def _save_json_data(self,data, path: str):
        if path is not None:
            try:
                with open(path, 'w') as file:
                    json.dump(data,file, indent=4)
                self.path_saved=path
            except FileNotFoundError:
                raise FileNotFoundError(f'Path {path} does not exist. Please check it')
        else:
            current=os.path.dirname(os.path.abspath(__file__))
            folder_path=os.path.join(os.path.dirname(current),'Default')

            os.makedirs(folder_path,exist_ok=True)
            file_path=os.path.join(folder_path,'data_enbios.json')
            print(file_path)
            with open(file_path, 'w') as file:
                json.dump(data, file,indent=4)
            print(f'Data for enbios saved in {file_path}')
            self.path_saved=file_path

    @staticmethod
    def _save_const(project: str, db: str):
        """ get the project and db name and store in a const file"""
        try:
            with open(r'Sparks/const/const.py', 'w') as f:
                f.write(f"bw_project = '{project}'\n")
                f.write(f"bw_db = '{db}'\n")
        except:
            base_path=os.path.abspath(os.path.join('..', '..', 'Sparks', 'const'))
            os.makedirs(base_path, exist_ok=True)

            file_path=file_path = os.path.join(base_path, 'const.py')
            with open(file_path, 'w') as f:
                f.write(f"bw_project = '{project}'\n")
                f.write(f"bw_db = '{db}'\n")












