
import json
from typing import Union,Optional
import pandas as pd
import bw2data as bd

from enbios2.base.experiment import Experiment
import os
from ProspectBackground.util.preprocess.cleaner import Cleaner
from ProspectBackground.util.preprocess.SoftLink import SoftLinkCalEnb
import bw2io as bi
from ProspectBackground.const import const
from dataclasses import dataclass
from ProspectBackground.util.preprocess.template_market_4_electricity import Market_for_electricity
from ProspectBackground.util.updater.background_updater import Updater
import time

try:
    exp = Experiment(r'C:\Users\Administrator\PycharmProjects\TFM__Lex\ProspectBackground\Default\data_enbios.json')
    exp.run()
except:
    # Generally the exception is the unspecificEcoinvent error from ENBIOS
    from enbios2.base.unit_registry import ecoinvent_units_file_path

    text_to_write = 'unspecificEcoinventUnit = []'
    # Abre el archivo en modo escritura ('w')
    with open(ecoinvent_units_file_path, 'w') as file:
        file.write(text_to_write)

    exp = Experiment(r'C:\Users\Administrator\PycharmProjects\TFM__Lex\ProspectBackground\Default\data_enbios.json')
    exp.run()


bb=exp.result_to_dict()
path='results_TFM.json'
import json
with open(path, 'w') as file:
    json.dump(bb, file, indent=4)