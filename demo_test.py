from Sparks.util.base import SoftLink


enbios_mod = SoftLink(r'C:\Users\Administrator\Downloads\flow_out_sum (1).csv',
                      r'C:\Users\Administrator\Documents\Alex\BASEFILE\basefile_off_heat.xlsx',
                       'github',
                      'ecoinvent')

enbios_mod.preprocess(subregions=False)
enbios_mod.data_for_ENBIOS(smaller_vers=True, path_save=r'data/test.json')