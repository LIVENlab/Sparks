import json
import os
from pathlib import Path
from enbios2.base.experiment import Experiment

os.environ["CONFIG_FILE"] = "exp.json"
os.environ["RUN_SCENARIOS"] = '["Scenario 0"]'

data = {
    "bw_project": "ecoinvent_391",
    "activities": [
        {
            "id": {
                "name": "electricity production, wind, >3MW turbine, onshore",
                "code": "0d48975a3766c13e68cedeb6c24f6f74",
                "alias": "electricity production, wind, >3MW turbine, onshore",
            }
        },
        {
            "id": {
                "name": "electricity production, solar tower power plant, 20 MW",
                "code": "f2700b2ffcb6b32143a6f95d9cca1721",
                "alias": "electricity production, solar tower power plant, 20 MW",
            }
        },
        {
            "id": {
                "name": "electricity production, solar thermal parabolic trough, 50 MW",
                "code": "19040cdacdbf038e2f6ad59814f7a9ed",
                "alias": "electricity production, solar thermal parabolic trough, 50 MW",
            }
        },
    ],
    "methods": [
        {
            "id": [
                "EF v3.1 no LT",
                "ecotoxicity: freshwater, organics no LT",
                "comparative toxic unit for ecosystems (CTUe) no LT",
            ]
        }
    ],
    "hierarchy": {
        "wind": ["electricity production, wind, >3MW turbine, onshore"],
        "solar": [
            "electricity production, solar tower power plant, 20 MW",
            "electricity production, solar thermal parabolic trough, 50 MW",
        ],
    },
    "scenarios": [
        {
            "activities": {
                "electricity production, wind, >3MW turbine, onshore": [
                    "kilowatt_hour",
                    8,
                ],
                "electricity production, solar tower power plant, 20 MW": [
                    "kilowatt_hour",
                    6,
                ],
                "electricity production, solar thermal parabolic trough, 50 MW": [
                    "kilowatt_hour",
                    3,
                ],
            }
        },
        {
            "activities": {
                "electricity production, wind, >3MW turbine, onshore": [
                    "kilowatt_hour",
                    10,
                ],
                "electricity production, solar tower power plant, 20 MW": [
                    "kilowatt_hour",
                    4,
                ],
                "electricity production, solar thermal parabolic trough, 50 MW": [
                    "kilowatt_hour",
                    3,
                ],
            }
        },
        {
            "activities": {
                "electricity production, wind, >3MW turbine, onshore": [
                    "kilowatt_hour",
                    10,
                ],
                "electricity production, solar tower power plant, 20 MW": [
                    "kilowatt_hour",
                    8,
                ],
                "electricity production, solar thermal parabolic trough, 50 MW": [
                    "kilowatt_hour",
                    2,
                ],
            }
        },
        {
            "activities": {
                "electricity production, wind, >3MW turbine, onshore": [
                    "kilowatt_hour",
                    9,
                ],
                "electricity production, solar tower power plant, 20 MW": [
                    "kilowatt_hour",
                    3,
                ],
                "electricity production, solar thermal parabolic trough, 50 MW": [
                    "kilowatt_hour",
                    2,
                ],
            }
        },
        {
            "activities": {
                "electricity production, wind, >3MW turbine, onshore": [
                    "kilowatt_hour",
                    4,
                ],
                "electricity production, solar tower power plant, 20 MW": [
                    "kilowatt_hour",
                    5,
                ],
                "electricity production, solar thermal parabolic trough, 50 MW": [
                    "kilowatt_hour",
                    6,
                ],
            }
        },
        {
            "activities": {
                "electricity production, wind, >3MW turbine, onshore": [
                    "kilowatt_hour",
                    4,
                ],
                "electricity production, solar tower power plant, 20 MW": [
                    "kilowatt_hour",
                    3,
                ],
                "electricity production, solar thermal parabolic trough, 50 MW": [
                    "kilowatt_hour",
                    8,
                ],
            }
        },
        {
            "activities": {
                "electricity production, wind, >3MW turbine, onshore": [
                    "kilowatt_hour",
                    9,
                ],
                "electricity production, solar tower power plant, 20 MW": [
                    "kilowatt_hour",
                    7,
                ],
                "electricity production, solar thermal parabolic trough, 50 MW": [
                    "kilowatt_hour",
                    7,
                ],
            }
        },
        {
            "activities": {
                "electricity production, wind, >3MW turbine, onshore": [
                    "kilowatt_hour",
                    4,
                ],
                "electricity production, solar tower power plant, 20 MW": [
                    "kilowatt_hour",
                    10,
                ],
                "electricity production, solar thermal parabolic trough, 50 MW": [
                    "kilowatt_hour",
                    3,
                ],
            }
        },
    ],
}

Path("exp.json").write_text(json.dumps(data))
exp = Experiment()
# exp = Experiment("exp.json")
print(exp)
res = exp.run()

print(res)
pass
