import pandas as pd
from ProspectBackground.util.update_experiment import Prospect

enbios_mod = Prospect(r'C:\Users\Administrator\Documents\Alex\flow_out.csv',
                      r'C:\Users\Administrator\Documents\Alex\BASEFILE\basefile_off.xlsx',
                       'TFM_Lex', 'ecoinvent')
pass
enbios_mod.preprocess(subregions=False)
enbios_mod.data_for_ENBIOS()
#enbios_mod.classic_run()


#df=pd.read_csv(r'C:\Users\altz7\PycharmProjects\enbios__git\projects\data_sentinel\flow_out_sum.csv',delimiter=',')