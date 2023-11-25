"""
Created on Mon Sep  4 12:05:22 2023

@author: Alexander de Tomás (ICTA-UAB)
        -LexPascal
"""
import pathlib

import bw2data as bd
import pandas as pd
import time
from tqdm import tqdm
from ProspectBackground.const.const import bw_project,bw_db


bd.projects.set_current(bw_project)            # Select your project
ei = bd.Database(bw_db)        # Select your db





def InventoryFromExcel(data)->str:

    """
    This function creates bw activities reading an excel file.
    Please check the expected structure of the Excel file on ____

    :param data: can either use a predefined dataframe or a path to a csv
    :return: a list containing the code of the activity created for further uses
    """
    starter_time=time.time()
    #check
    if isinstance(data, str):
        try:
            df=pd.read_csv(data, delimiter=',')
        except:
            raise FileNotFoundError(f"The path '{data}' does not exist.")

    elif isinstance(data,pathlib.Path):
        try:
            df = pd.read_csv(data, delimiter=',')
        except:
            raise FileNotFoundError(f"The path '{data}' does not exist.")

    elif isinstance(data ,pd.DataFrame):
        df=data

    else:
        print(f"Input error {type(data)} is not admitted")
        return None

    df.fillna('NA',inplace=True)

    activ=[]
    for index,row in df.iterrows():
        origin=str(row['Act_to'])

        if origin == "marker":            # If Act origin=NA == This is the new activity to create

            activity_name = str(row['Activity name'])
            activity_code=str(row['Activity_code'])
            activ.append(activity_code)

            pass
            try:
                new_activity = ei.new_activity(name=activity_name, code=activity_code, unit=str(row['Unit']))
                new_activity.save()
            # create a df containing the rows
            except bd.errors.DuplicateNode:
                new_activity = ei.get_node(activity_code)
                new_activity.delete()
                new_activity = ei.new_activity(name=activity_name, code=activity_code, unit=str(row['Unit']))
                new_activity.save()

            act_df = df.loc[df['Act_to'] == activity_name]
            pass
            # Subset all the activities that have origin in the new activity created
            for _, row2 in act_df.iterrows():
                if row2['Technosphere'] == 'Yes':
                    try:
                        act = bd.Database(row2['Database']).get_node(row2['Activity_code'])
                    except bd.errors.UnknownObject as e:

                        code = row2['Activity_code']
                        name=row2['Database']
                        print(f"Error: Code '{code}', in   db {name} is not in the db")

                        raise e

                    if row2['Reference_product'] != 'NA':
                        exchange = new_activity.new_exchange(input=act, type='technosphere',unit=row2['Unit'], amount=row2['Amount'],location=row2['Location'])
                        exchange.save()
                    else:
                        pass
                else:
                    act = bd.Database('biosphere3').get_node(row2['Activity_code'])
                    if row2['Reference_product'] != 'NA':
                        exchange = new_activity.new_exchange(input=act, type='biosphere', amount=row2['Amount'])
                        exchange.save()
                    else:
                        pass
            new_activity.save()





    return(activ)








