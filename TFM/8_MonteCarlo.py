
import bw2data as bd
import pandas as pd
from bw2calc import LCA
from bw2calc.monte_carlo import MonteCarloLCA, MultiMonteCarlo
import json
from enbios2.base.experiment import Experiment

a=Experiment()



bd.projects.set_current("TFM_Lex")  # Select your project
ei = bd.Database("ecoinvent")

act = ei.get_node("fd6356e68720aec505c69cdadb3046fc")
pass
#act2 = ei.get_node('413bc4617794c6e8b07038dbeca64adb')


from enbios2.base.stacked_MultiLCA import StackedMultiLCA
from enbios2.models.experiment_models import BWCalculationSetup


""""
from prospect, export the "preprocessed_units df first"
"""

class SetUp:

    def __init__(self, motherfile):
        self.list=self.generate_list()
        self.motherfile=self.open_motherfile(motherfile)
        self.final=self.get_activities()

        pass
    def generate_list(self)->list[dict[str,int]]:
        data=[]
        df=pd.read_csv('data_for_uncertainty.csv')
        df=df[df['scenarios']==0]

        for index,row in df.iterrows():
            dic={row['aliases'] : row['flow_out_sum']}
            data.append(dic)
        return data




    def get_activities(self):
        df_mother=self.motherfile
        dat=self.list
        pass
        final_list=[]
        for element in dat:
            key=list(element.keys())[0]
            value=list(element.values())[0]
            filter=df_mother[df_mother['aliases']==key]
            code=filter.iloc[0]['BW_DB_FILENAME']
            act=ei.get_node(code)
            dict={act : value}
            print(type(act))
            final_list.append(dict)
            pass
        return final_list
        pass

    def open_motherfile(self,motherfile):
        """
        open and add aliases
        """
        df=pd.read_excel(motherfile,sheet_name='Processors')
        df['aliases']=df['Processor']+'__' +df['@SimulationCarrier']+'___'+df['Region']

        return df



b=SetUp(r'/home/lex/Downloads/basefile_off.xlsx')
acti=b.final
meths=[('CML v4.8 2016', 'climate change', 'global warming potential (GWP100)')]
setup=BWCalculationSetup('stuff',acti,meths)
pass


Monte=StackedMultiLCA(setup,use_distributions=True)


#demands = [{act: 1,
 #           act2: 10}]
