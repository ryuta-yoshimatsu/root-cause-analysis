# Databricks notebook source
# MAGIC %sh 
# MAGIC sudo apt-get -qq update
# MAGIC sudo apt-get -y -qq install graphviz libgraphviz-dev

# COMMAND ----------

# MAGIC %pip install -r ../requirements.txt --quiet
# MAGIC dbutils.library.restartPython()

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
import matplotlib.pyplot as plt
import pandas as pd
import dowhy
import networkx as nx
#from scipy.special import expit, logit

np.random.seed(1)

# Get the current user name
current_user_name = spark.sql("SELECT current_user()").collect()[0][0]

# Set the experiment name
experiment_name = f"/Users/{current_user_name}/rca_manufacturing"
mlflow.set_experiment(experiment_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Generation --> Stays in 01_causal_graph

# COMMAND ----------

n = 10000

raw_material = np.random.choice([0, 1], size=n, p=[0.75, 0.25])                             # Raw material
material = np.random.choice([0, 1], size=n, p=[0.5, 0.5])                                   # Additional material
worker = np.random.binomial(n=1, p=0.5, size=n)                                             # Manual worker
machine = np.random.choice([1, 2, 3, 4, 5], size=n, p=[0.5, 0.125, 0.125, 0.125, 0.125])    # Machine setting

chamber_temperature = np.random.normal(loc=20, scale=5, size=n)
chamber_humidity = np.random.normal(loc=0.5, scale=0.05, size=n)
chamber_pressure = np.random.normal(loc=1013.25, scale=15, size=n)

X = pd.DataFrame(
    {
        'raw_material': raw_material,
        'material': material,
        'worker': worker,
        'machine': machine,
        'chamber_temperature': chamber_temperature,
        'chamber_humidity': chamber_humidity,
        'chamber_pressure': chamber_pressure,
    }
)

dependencies = {
    "force_torque": (
        ['raw_material', 'material', 'machine'],
        [100, 100, -100]
    ),
    "position_alignment": (
        ['worker', 'machine'],
        [0.1, 0.1]
    ),
    "process_duration": (
        ['machine', 'chamber_temperature', 'chamber_humidity', 'chamber_pressure'],
        [-1.0, -0.1, 0.01, 0.001]
    ),
    "vibration": (
        ['force_torque', 'position_alignment'],
        [0.00001, 0.01]
    ),
    "sound": (
        ['force_torque', 'position_alignment'],
        [0.01, 5]
    ),
    "temperature": (
        ['force_torque', 'position_alignment'],
        [0.1, -100]
    ),
    "quality": (
        ['process_duration', 'sound', 'vibration', 'temperature'],
        [-0.1, -0.1, -0.1, 0.01]
    ),
}


def linear(df, dependency, const, loc, scale):
    noise = np.random.normal(loc=loc, scale=scale, size=n)
    return df[dependency[0]] @ dependency[1] + const + noise

# Newtons (N)
X['force_torque'] = linear(X,
    dependency = dependencies['force_torque'],
    const = 1000,
    loc = 0,
    scale = 50,
    )

# Millimeters (mm)
X['position_alignment'] = linear(X,
    dependency = dependencies['position_alignment'],
    const = 1,
    loc = 0,
    scale = 0.1,
    )

# Seconds (s)
X['process_duration'] = linear(X,
    dependency = dependencies['process_duration'],
    const = 10,
    loc = 0,
    scale = 1,
    )

# Amplitude: millimeters (mm)
X['vibration'] = linear(X,
    dependency = dependencies['vibration'],
    const = 0.1,
    loc = 0,
    scale = 0.01,
    )

# Decibels (dB)
X['sound'] = linear(X,
    dependency = dependencies['sound'],
    const = 60,
    loc = 0,
    scale = 5,
    )

# Celsius (°C)
X['temperature'] = linear(X,
    dependency = dependencies['temperature'],
    const = 1000,
    loc = 0,
    scale = 100,
    )

# Customized quality score (0-100). Anything below 60 is considered defective.
X['quality'] = linear(X,
    dependency = dependencies['quality'],
    const = 80,
    loc = 0,
    scale = 5,
    )

spark.createDataFrame(X).write.format("delta").mode("overwrite").saveAsTable(f"{catalog}.{db}.data_manufacturing")

display(X)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Causal Graph Generation --> Stays in 01_causal_graph

# COMMAND ----------

from IPython.display import Image, display
display(Image('../images/manufacturing-processes.png', width=750, height=450), display_id='centered_image')

# COMMAND ----------

Image('../images/manufacturing-process-A.png')

# COMMAND ----------

true_graph = nx.DiGraph()
true_graph.add_nodes_from = X.columns

for child in dependencies:
    for parent in dependencies[child][0]:
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
# MAGIC ## Causal Modeling --> Goes to 02_causal_modeling

# COMMAND ----------

# MAGIC %md
# MAGIC ### Read dataset

# COMMAND ----------

table_name = f"{catalog}.{db}.data_manufacturing"
sdf = spark.read.format("delta").table(table_name)
pdf = sdf.toPandas()
pdf.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load graph from mlflow

# COMMAND ----------

# Find all the runs from the prior notebook for causal discovery
client = mlflow.MlflowClient()
experiment = mlflow.get_experiment_by_name(experiment_name)
discovery_runs = client.search_runs(
    experiment_ids=[experiment.experiment_id], 
    filter_string="attributes.run_name='causal_graph'",
    order_by=["start_time DESC"],
    max_results=1,
    )

# Make sure there is at least one run available
assert len(discovery_runs) == 1, "please run this notebook from the beginning at least once"

# The only result should be the latest based on our search_runs call
latest_discovery_run = discovery_runs[0]
latest_discovery_run.info.artifact_uri

# Load the graph artifact from the run
local_path = mlflow.artifacts.download_artifacts(latest_discovery_run.info.artifact_uri + "/graph/causal_graph.pickle")

with open(local_path, "rb") as f:
    causal_graph = pickle.load(f)

# COMMAND ----------

dowhy.gcm.util.plot(causal_graph, figure_size=(20, 20))

# COMMAND ----------

from dowhy import gcm

# Create the structural causal model object
scm = gcm.StructuralCausalModel(causal_graph)

# Automatically assign generative models to each node based on the given data
auto_assignment_summary = gcm.auto.assign_causal_mechanisms(
  scm, 
  pdf, 
  override_models=True, 
  quality=gcm.auto.AssignmentQuality.GOOD
  )

# COMMAND ----------

print(auto_assignment_summary)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Fit data to graph

# COMMAND ----------

gcm.fit(scm, pdf)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Evaluate the fitted graph

# COMMAND ----------

print(gcm.evaluate_causal_model(
  scm,
  pdf, 
  compare_mechanism_baselines=True, 
  evaluate_invertibility_assumptions=True))

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Appendix: Causal Discovery (WIP) --> Stays in 01_causal_graph

# COMMAND ----------

from causallearn.search.ConstraintBased.PC import pc

# default parameters
cg = pc(np.vstack(X.to_numpy()), node_names=X.columns)

# visualization using pydot
cg.draw_pydot_graph()

# COMMAND ----------


