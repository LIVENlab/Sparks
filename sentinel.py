import pandas as pd
from Sparks.util.update_experiment import SoftLink

enbios_mod = SoftLink(r'/home/lex/Documents/data_sentinel/flow_out_sum.csv',
                       r'/home/lex/Downloads/basefile_sentinel.xlsx',
                       'TFM_Lex', 'ecoinvent')

enbios_mod.preprocess(subregions=True)
enbios_mod.data_for_ENBIOS(smaller_vers=1)


#df=pd.read_csv(r'C:\Users\altz7\PycharmProjects\enbios__git\projects\data_sentinel\flow_out_sum.csv',delimiter=',')