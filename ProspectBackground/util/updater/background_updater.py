import bw2data as bd
import pandas as pd
import typing
from ProspectBackground.const.const import bw_project,bw_db
from typing import Dict,List
from decimal import Decimal, getcontext
bd.projects.set_current(bw_project)            # Select your project
ei = bd.Database(bw_db)



class Updater():
    """
    This class updates the "amounts" of each technology for the future market for electricity
    """
    def __init__(self,enbios_data,templates):
        self.template=templates
        self.enbios_data=enbios_data
        pass

    def inventoryModify(self,data: pd.DataFrame ,scenario: str,region : str) -> pd.DataFrame:

        """
        This function updates the values of the template inventory.
        ** Update
        :param scenario: str --> scenario to modify
        :return: pandas Dataframe --> df with the market for electricity modified
        """
        df=data

        dict=self.enbios_data
        subdict=self.get_region_activities(dict,scenario,region)
        #subdict = dict['scenarios'][scenario]['activities']
        pass
        for key in subdict.keys():
            name = key
            amount = subdict[key][1]
            for index, row in df.iterrows():
                if row['Activity name'] == name:
                    df.loc[df['Activity name'] == name,'Amount'] = amount

        df_gruoped = df.groupby('Activity_code')['Amount'].transform('sum')
        df['Amount'] = df_gruoped
        df = df.drop_duplicates(subset='Amount')
        getcontext().prec = 50
        # Normalize all the flows to 1
        sum_of_column = sum(map(Decimal, df['Amount'][1:]))
        df['Amount'] = [1] + [Decimal(x) / sum_of_column for x in df['Amount'][1:]]


        return df

    @staticmethod
    def get_region_activities(dict,scenario,region) -> Dict[str,List[str]]:
        """
              This function returns the enbios data filtered by the scenario and region
        """
        subdict = dict['scenarios'][scenario]['activities']
        # FIlter for the region
        sub={key: val for key,val in subdict.items() if key.endswith(str(region))}
        return sub


    def exchange_updater(self,df,code):

        """"
        Opens the bw activity and update the results with the modified dataframe
        """
        market = ei.get_node(code)
        # Eval exchanges from market

        for index, row in df.iterrows():
            code_iter = row['Activity_code']
            amount = row['Amount']
            act_iter = ei.get_node(code_iter)
            name = act_iter['name']

            for ex in market.exchanges():
                if name in str(ex.input):
                    old_am = ex['amount']
                    ex['amount'] = float(amount)
                    ex.save()
                    #print(f"Amount modified in exchange {name}, moved from {old_am} to {ex['amount']}")
                else:
                    pass

    def update_results(self,scenario):

        template_dic=self.template
        for key,value in self.template.items():
            pass
            # Access the dataframe with the data
            data=template_dic[key][0]
            #Update the dictionary
            updated_data=self.inventoryModify(data,scenario,key)
            # save it
            template_dic[key][0]=updated_data
            # Modify the new echanges in the db
            self.exchange_updater(updated_data,code=template_dic[key][1])
        return self.template







