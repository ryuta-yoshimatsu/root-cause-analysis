# Databricks notebook source
# MAGIC %sh 
# MAGIC sudo apt-get -qq update
# MAGIC sudo apt-get -y -qq install graphviz libgraphviz-dev

# COMMAND ----------

# MAGIC %pip install -r ./requirements.txt --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./99_utils

# COMMAND ----------

catalog = 'causal_solacc'
db = 'rca'

# Make sure that the catalog exists
_ = spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog}")

# Make sure that the schema exists
_ = spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{db}")

# COMMAND ----------

import mlflow
import pickle
import numpy as np
from scipy.stats import bernoulli, norm, halfnorm, poisson, uniform
import matplotlib.pyplot as plt
import pandas as pd
import dowhy
import networkx as nx

# Get the current user name
current_user_name = spark.sql("SELECT current_user()").collect()[0][0]

# Set the experiment name
experiment_name = f"/Users/{current_user_name}/rca_manufacturing"
mlflow.set_experiment(experiment_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Generation

# COMMAND ----------

X = generate_data(catalog, db, 1000)
display(X)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Causal Graph Generation

# COMMAND ----------

from IPython.display import Image, display
Image('./images/manufacturing-process-A.png')

# COMMAND ----------

true_graph = nx.DiGraph()
true_graph.add_nodes_from = X.columns

dependencies = {
        "position_alignment": ['worker', 'machine'],
        "force_torque": ['raw_material', 'machine', 'material'],
        "temperature": ['chamber_temperature', 'chamber_humidity', 'chamber_pressure'],
        "dimensions": ['position_alignment', 'force_torque'],
        "torque_checks": ['force_torque', 'temperature'],
        "visual_inspection": ['temperature'],
        "quality": ['dimensions', 'torque_checks', 'visual_inspection'],
    }

for child in dependencies:
    for parent in dependencies[child]:
        true_graph.add_edge(parent, child)

dowhy.gcm.util.plot(true_graph, figure_size=(20, 20))

# COMMAND ----------

with mlflow.start_run(run_name="causal_graph") as run:
    
    # save graph object to file
    pickle.dump(true_graph, open('/databricks/driver/causal_graph.pickle', 'wb'))

    # log the pickle file to mlflow
    mlflow.log_artifact("/databricks/driver/causal_graph.pickle", artifact_path="graph")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Appendix: Causal Discovery (WIP)

# COMMAND ----------

from causallearn.search.ConstraintBased.PC import pc

# default parameters
data = X.copy().drop('id', axis=1)
cg = pc(np.vstack(data.to_numpy()), node_names=data.columns)

# visualization using pydot
cg.draw_pydot_graph()

# COMMAND ----------

# MAGIC %md
# MAGIC © 2025 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License. All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC
# MAGIC | library                                | description             | license    | source                                              |
# MAGIC |----------------------------------------|-------------------------|------------|-----------------------------------------------------|
# MAGIC | Graphviz | An open source graph visualization software | Common Public License Version 1.0 | https://graphviz.org/download/
# MAGIC | pygraphviz | A Python interface to the Graphviz graph layout and visualization package | BSD | https://pypi.org/project/pygraphviz/
# MAGIC | networkx | A Python package for the creation, manipulation, and study of the structure, dynamics, and functions of complex networks. | BSD | https://pypi.org/project/networkx/
# MAGIC | dowhy | A Python library for causal inference that supports explicit modeling and testing of causal assumptions | MIT | https://pypi.org/project/dowhy/
# MAGIC | causal-learn | A python package for causal discovery that implements both classical and state-of-the-art causal discovery algorithms, which is a Python translation and extension of Tetrad. | MIT | https://pypi.org/project/causal-learn/

# COMMAND ----------


