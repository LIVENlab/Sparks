import pandas as pd
import pandera.pandas as pa
from pandera import Check

schema = pa.DataFrameSchema(
    {
    "Processor": pa.Column(str, required=True, nullable=False),
    "Region": pa.Column(str, required=True, nullable=False),
    "@SimulationCarrier": pa.Column(str, required=True, nullable=False),
    "ParentProcessor": pa.Column(str, required=True, nullable=False),
    "@SimulationToEcoinventFactor": pa.Column(object,Check(lambda s: s.apply(lambda x: isinstance(x, (float,int)))),
                                              required=True,
                                              nullable=False),
    "Ecoinvent_key_code": pa.Column(object,Check(lambda s: s.apply(lambda x: isinstance(x, (float,int)))),
                                    required=True, nullable=False),
    "File_source": pa.Column(str, required=True, nullable=False),
    "geo_loc": pa.Column(str, required=True, nullable=False)
    },
    strict = False  # Allow extra columns
    )


