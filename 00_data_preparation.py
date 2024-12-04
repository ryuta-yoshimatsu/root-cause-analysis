# Databricks notebook source
# MAGIC %pip install causalAssembly
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %pip install rpy2

# COMMAND ----------

# MAGIC %r install.packages('drf')

# COMMAND ----------

from causalAssembly.drf_fitting import fit_drf

# COMMAND ----------

import pandas as pd

from causalAssembly.models_dag import ProductionLineGraph
#from causalAssembly.drf_fitting import fit_drf

seed = 2024
n_select = 500

assembly_line_data = ProductionLineGraph.get_data()

# take subsample for demonstration purposes
assembly_line_data = assembly_line_data.sample(
    n_select, random_state=seed, replace=False
)

# load in ground truth
assembly_line = ProductionLineGraph.get_ground_truth()

# fit drf and sample for entire line
#assembly_line.drf = fit_drf(assembly_line, data=assembly_line_data)
#assembly_line_sample = assembly_line.sample_from_drf(size=n_select)

# fit drf and sample for station3
#assembly_line.Station3.drf = fit_drf(assembly_line.Station3, data=assembly_line_data)
#station3_sample = assembly_line.Station3.sample_from_drf(size=n_select)


# COMMAND ----------

assembly_line_data

# COMMAND ----------

assembly_line.show()

# COMMAND ----------


