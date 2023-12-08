import json
from pathlib import Path

import numpy
import torch
from bw2data.backends import Activity, ActivityDataset, ExchangeDataset
from sentence_transformers import SentenceTransformer
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

import bw2data
from tqdm import tqdm

from enbios2.experiment.nlp.models import NLPModels

bw2data.projects.set_current("ecoinvent")
db = bw2data.Database("cutoff_3.9.1_default")
# act: Activity = db.random()

feature_path = Path("features")

# here we create node features from the name and the product.
# we store them as jsonl arrays in the features folder

model = NLPModels.get_sentence_model()


name_feature_files = (feature_path / "names.jsonl").open("w")
# product_feature_files = (feature_path / "product.jsonl").open("w")

default_len = len(model.encode("cool"))
for act in tqdm(ActivityDataset.select().order_by(ActivityDataset.id)):
    name_feature_files.write(json.dumps(model.encode(act.name).tolist()) + "\n")
    # if act.product:
    #     product_feature_files.write(json.dumps(model.encode(act.product).tolist()) + "\n")
    # else:
    #     product_feature_files.writelines(json.dumps(numpy.zeros(default_len).tolist()) + "\n")
name_feature_files.close()
# product_feature_files.close()

# Now lets create edges and their features
# ids are indices... note that our ids start with 0. while those of bw (sqllite database start with 1)
# we store them in a n x k array, where k is min 2, first 2. are in/out nodes
code2idMap = [act.code for act in tqdm(ActivityDataset.select().order_by(ActivityDataset.id))]
edges_file = (feature_path / "edges.jsonl").open("w")
for e in tqdm(ExchangeDataset.select().order_by(ExchangeDataset.id)):
    edges_file.write(json.dumps([
        code2idMap.index(e.output_code),
        code2idMap.index(e.input_code),
        e.data.get("amount",0)
    ]) + "\n")
edges_file.close()

# X = torch.tensor(node_features, dtype=torch.float)
# edges = torch.tensor(edge_list, dtype=torch.long)
# edge_values = torch.tensor(edge_values, dtype=torch.float)
# scores = torch.tensor(node_scores, dtype=torch.float)
#
# # Create a PyG Data object
# data = Data(x=X, edge_index=edges.t().contiguous(), edge_attr=edge_values, y=scores)


import numpy as np
import torch

# Assume we have some node data in the variables node_features
# node_features is a numpy array of shape (num_nodes, num_features)
