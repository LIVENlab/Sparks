
import json


from enbios2.base.experiment import Experiment

try:

    exp = Experiment(r'/home/lex/PycharmProjects/Sparks_2/ProspectBackground/Default/data_enbios.json')

    exp.run()

except:

    # Generally the exception is the unspecificEcoinvent error from ENBIOS

    from enbios2.base.unit_registry import ecoinvent_units_file_path

    text_to_write = 'unspecificEcoinventUnit = []'

    with open(ecoinvent_units_file_path, 'w') as file:

        file.write(text_to_write)

    exp = Experiment(r'/home/lex/PycharmProjects/Sparks_2/ProspectBackground/Default/data_enbios.json')

    exp.run()

print("done")

bb = exp.result_to_dict()

path = 'results_check_MonteCarlo_3_.json'

with open(path, 'w') as file:
    json.dump(bb, file, indent=4)

