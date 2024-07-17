from Sparks.util.base import SoftLink


enbios_mod = SoftLink(r'C:\Users\1496051\PycharmProjects\Sparks_local\testing\data_test\ElectricgenerationTIMES-Sinergia.xlsx',
                      r'C:\Users\1496051\PycharmProjects\Sparks_local\testing\data_test\TIMES_ecoinvent.xlsx',
                       'github_3',
                      'ecoinvent')



enbios_mod.preprocess(subregions=False)
enbios_mod.data_for_ENBIOS(smaller_vers=False,
                           path_save=r'test.json')




