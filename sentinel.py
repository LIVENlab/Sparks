from enbios2.base.experiment import Experiment

import json
try:
    exp = Experiment(r'C:\Users\Administrator\PycharmProjects\SEEDS\Data_enbios_paper\final_run_v1.json')
    exp.run()
except:
    # Generally the exception is the unspecificEcoinvent error from ENBIOS
    from enbios2.base.unit_registry import ecoinvent_units_file_path

    text_to_write = 'unspecificEcoinventUnit = []'

    with open(ecoinvent_units_file_path, 'w') as file:
        file.write(text_to_write)


    exp = Experiment(r'C:\Users\Administrator\PycharmProjects\SEEDS\Data_enbios_paper\final_run_v1.json')

    exp.run()

res=exp.result_to_dict()

with open('results.json', 'w') as file:
    json.dump(res, file, indent=4)