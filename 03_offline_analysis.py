# Databricks notebook source
# MAGIC %md
# MAGIC This solution accelerator notebook is available at [Databricks Industry Solutions](https://github.com/databricks-industry-solutions/).

# COMMAND ----------

# MAGIC %md
# MAGIC # Answer causal questions
# MAGIC In this notebook, we will perform causal analysis such as causal attributions and root cause analysis using the fitted graph from the previous notebook.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cluster configuration
# MAGIC We recommend using a cluster with the following or similar specifications to run this solution accelerator:
# MAGIC - Unity Catalog enabled cluster
# MAGIC - Databricks Runtime 15.4 LTS ML or above
# MAGIC - Single-node cluster: e.g. `m5d.2xlarge` on AWS or `Standard_D8ds_v5` on Azure Databricks

# COMMAND ----------

# DBTITLE 1,Install graphviz from nicer visualization
# MAGIC %sh 
# MAGIC sudo apt-get -qq update
# MAGIC sudo apt-get -y -qq install graphviz libgraphviz-dev

# COMMAND ----------

# MAGIC %md
# MAGIC We install the required packages from the `requirements.txt`.

# COMMAND ----------

# DBTITLE 1,Install requirements
# MAGIC %pip install -r ./requirements.txt --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC In the next cell, we run the `99_utils` notebook, which defines a few utility functions that we will use along the way.

# COMMAND ----------

# DBTITLE 1,Run utils notebook
# MAGIC %run ./99_utils

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define variables and set MLflow experiment

# COMMAND ----------

import mlflow
from mlflow import MlflowClient
import pandas as pd
import numpy as np
from dowhy import gcm

# COMMAND ----------
user_name = spark.sql("SELECT current_user()").collect()[0][0]
first_name = user_name.split(".")[0]

# Set up Unity Catalog
catalog = f'causal_solacc_{first_name}'     # Change this to your catalog name
schema = f'rca_{first_name}'                # Change this to your schema name
model = f"manufacturing_{first_name}"   # Change this to your model name

setup_unity_catalog(catalog, schema)
# COMMAND ----------

# MAGIC %md
# MAGIC ## Load the causal model
# MAGIC
# MAGIC Let's load the fitted causal model from the previous notebook. We will use the "champion" alias to ensure the correct version is loaded.

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")
mlflow_client = MlflowClient()

registered_model_name = f"{catalog}.{schema}.{model}"
model = f"models:/{registered_model_name}@champion"

# Load model as a PyFuncModel
loaded_model = mlflow.pyfunc.load_model(model)
loaded_scm = loaded_model.unwrap_python_model().load_scm()
loaded_causal_graph = loaded_model.unwrap_python_model().load_causal_graph()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Conduct root cause analysis

# COMMAND ----------

# MAGIC %md
# MAGIC ### What are the key factors influencing the variance in quality?
# MAGIC
# MAGIC Suppose we want to understand the factors driving changes in quality. To begin, let's perform a simple descriptive analysis of this target variable. We'll plot quality against product ID, with quality on the Y-axis and product ID on the X-axis. Before creating the plot, we need to load the data.

# COMMAND ----------

train = spark.read.table(f"{catalog}.{schema}.data_manufacturing")
train = train.toPandas()
train.head()

# COMMAND ----------

train['quality'].plot(ylabel='quality', figsize=(15,5), rot=45)

# COMMAND ----------

# MAGIC %md
# MAGIC It seems that defective products are evenly distributed across the IDs. Next, let's examine the rate of defective products:

# COMMAND ----------

train.describe()

# COMMAND ----------

# MAGIC %md
# MAGIC The defect rate is 0.075. Examining the causal graph, we observe that dimensional verification, torque resistance checks, and visual inspection directly impact quality. But which of these factors contributes the most to the variance? To determine this, we use the [direct arrow strength algorithm](https://www.pywhy.org/dowhy/v0.9.1/user_guide/gcm_based_inference/answering_causal_questions/quantify_arrow_strength.html), which quantifies the causal influence of each specific arrow in the graph:

# COMMAND ----------

import numpy as np
from dowhy.utils import plot

# Note: The percentage conversion only makes sense for purely positive attributions.
def convert_to_percentage(value_dictionary):
    total_absolute_sum = np.sum([abs(v) for v in value_dictionary.values()])
    return {k: abs(v) / total_absolute_sum * 100 for k, v in value_dictionary.items()}


arrow_strengths = gcm.arrow_strength(loaded_scm, target_node='quality')

plot(loaded_causal_graph, 
     causal_strengths=convert_to_percentage(arrow_strengths), 
     figure_size=[15, 10])

# COMMAND ----------

# MAGIC %md
# MAGIC In this causal graph, we observe how much each node contributes to the variance in quality, with contributions expressed as percentages. Since quality is defined as: `max(dimensions, torque_checks, visual_inspection)` (see the function `generate_data` in the notebook `99_utils` for more detail), there should be no additional factors influencing its variance. As shown, `torque_checks` has a greater impact than `dimensions` or `visual_inspection`.
# MAGIC
# MAGIC While understanding the direct influences is important for identifying which immediate parents contribute most to the variance in quality, the deeper question remains: what ultimately drives `quality = 1`? For example, `torque_checks` depends on `force_torque` and `temperature`, which are themselves influenced by other upstream factors.
# MAGIC
# MAGIC To identify the key causal factors driving the variance in quality, we can use the [intrinsic causal contribution method](https://www.pywhy.org/dowhy/v0.11/user_guide/causal_tasks/quantify_causal_influence/icc.html). This approach attributes variance in quality to upstream nodes in the causal graph, accounting only for the unique contribution by each node, excluding contributions inherited from its parents. For more details, refer to the [research paper](https://arxiv.org/abs/2007.00714).
# MAGIC
# MAGIC Now, let's apply this method to the data:

# COMMAND ----------

iccs = gcm.intrinsic_causal_influence(loaded_scm, target_node='quality', num_samples_randomization=500)

# COMMAND ----------

from dowhy.utils import bar_plot
bar_plot(convert_to_percentage(iccs), ylabel='Variance attribution in %')

# COMMAND ----------

# MAGIC %md
# MAGIC The scores shown in this bar chart represent percentages that quantify how much variance each node contributes to quality—excluding any variance inherited from its parent nodes in the causal graph. As shown clearly, `chamber_temperature` and `chamber_humidity` have the most significant influence on quality variance. This highlights the process's sensitivity to some environmental conditions, which may have been previously overlooked. Additionally, factors such as `worker` and `machine` also exhibit a non-negligible influence.

# COMMAND ----------

# MAGIC %md
# MAGIC ### What are the key factors explaining the quality of a particular product?
# MAGIC
# MAGIC Up to this point, we've focused on causal analysis at an aggregated level. However, DoWhy also enables us to perform the same analysis at the sample level. Let’s generate some new samples using the fitted graph:

# COMMAND ----------

np.random.seed(1)

new_batch = gcm.draw_samples(loaded_scm, num_samples=100)
display(new_batch)

# COMMAND ----------

# MAGIC %md
# MAGIC We generated an additional 100 samples, including 7 products with `quality = 1`. These products originated from 2 instances of `dimension = 1`, 2 instances of `torque_checks = 1`, and 3 instances of `visual_inspection = 1`.
# MAGIC
# MAGIC To identify the factors contributing to the quality drop, we can leverage DoWhy's [anomaly attribution feature](https://www.pywhy.org/dowhy/v0.11/user_guide/causal_tasks/root_causing_and_explaining/anomaly_attribution.html). This feature requires specifying the target node of interest (`quality`) and the anomaly sample to be analyzed. The results are displayed in a bar chart, showing the attribution scores of each node for the specified anomaly sample:

# COMMAND ----------

defects_dimensions = new_batch[new_batch['dimensions'] == 1]
display(defects_dimensions)

# COMMAND ----------

attributions = gcm.attribute_anomalies(
  loaded_scm, 
  target_node='quality', 
  anomaly_samples=pd.DataFrame([defects_dimensions.iloc[0]])
  )
bar_plot({k: v[0] for k, v in attributions.items()}, ylabel='Anomaly attribution score')

# COMMAND ----------

# MAGIC %md
# MAGIC The bar chart above displays the anomaly attribution scores for the nodes related to a product that failed the quality check due to dimensional verification. Positive values indicate nodes that increased the likelihood of the sample being classified as an anomaly, while negative values indicate the opposite. More details about the interpretation of the score can be found in the corresponding [reserach paper](https://proceedings.mlr.press/v162/budhathoki22a.html).
# MAGIC
# MAGIC Notably, `worker` and `machine` emerge as the dominant factors influencing `quality`. This aligns with our data generation function, where `worker = 1` and `machine = 1` were configured to be less precise in positioning and aligning materials and equipment. This effect is also reflected in the positive contribution of `position_alignment`. However, DoWhy attributes the cause more strongly to `worker` and `machine` than to `position_alignment` because it recognizes `position_alignment` as a consequence of the combined impact of `worker = 1` and `machine = 1`.
# MAGIC
# MAGIC ***Note: Given the stochastic nature of sampling, the displayed anomaly distribution might differ slightly from what is discussed here. We encourage users to experiment with the `attribute_anomalies` feature using a variety of samples.***

# COMMAND ----------

defects_torque_checks = new_batch[new_batch['torque_checks'] == 1]
display(defects_torque_checks)

# COMMAND ----------

attributions = gcm.attribute_anomalies(
  loaded_scm, 
  target_node='quality', 
  anomaly_samples=pd.DataFrame([defects_torque_checks.iloc[0]])
  )
bar_plot({k: v[0] for k, v in attributions.items()}, ylabel='Anomaly attribution score')

# COMMAND ----------

# MAGIC %md
# MAGIC This time, the bar chart above shows the anomaly attribution scores for the nodes related to a product that failed the torque resistance checks. `chamber_humidity` emerges as the primary factor influencing `quality`. This is not surprising considering our data generation logic, where higher `chamber_humidity` is associated with lower material-machine interface `temperature` due to condensation and evaporation. Examining the values of `chamber_humidity` (0.79) and `temperature` (899) for this sample in the table above, we see that they deviate significantly from their expected standard values (0.5 and 1250—see the function `generate_data` in the notebook `99_utils` for more detail), which is consistent with the attribution results.
# MAGIC
# MAGIC ***Note: Given the stochastic nature of sampling, the displayed anomaly distribution might differ slightly from what is discussed here. We encourage users to experiment with the `attribute_anomalies` feature using a variety of samples.***

# COMMAND ----------

defects_visual_inspection = new_batch[new_batch['visual_inspection'] == 1]
display(defects_visual_inspection)

# COMMAND ----------

attributions = gcm.attribute_anomalies(
  loaded_scm, 
  target_node='quality', 
  anomaly_samples=pd.DataFrame([defects_visual_inspection.iloc[0]])
  )
bar_plot({k: v[0] for k, v in attributions.items()}, ylabel='Anomaly attribution score')

# COMMAND ----------

# MAGIC %md
# MAGIC Our final bar chart above displays the anomaly attribution scores for the nodes related to a product that failed the visual inspection. `chamber_humidity` and `chamber_temperature` stand out as the primary factors influencing `quality`. Interestingly, unlike the previous example, this anomaly is attributed to `chamber_humidity` being significantly lower than the standard value. Combined with the elevated `chamber_temperature`, this led to a higher material-machine interface `temperature`, creating conditions for undesirable outcomes such as excessive welding spatters, hence failing the visual inspection checks. As with the earlier examples, causal machine learning enables us to transparently identify combinations of attributes that contribute to desired or undesired outcomes, offering greater insight compared to traditional correlation-based machine learning techniques.
# MAGIC
# MAGIC ***Note: Given the stochastic nature of sampling, the displayed anomaly distribution might differ slightly from what is discussed here. We encourage users to experiment with the `attribute_anomalies` feature using a variety of samples.***

# COMMAND ----------

# MAGIC %md
# MAGIC Although this method provides a point estimate of the attributions based on the specific models and parameters learned, we can also leverage DoWhy’s [confidence interval feature](https://www.pywhy.org/dowhy/v0.9.1/user_guide/gcm_based_inference/estimating_confidence_intervals.html). This feature accounts for uncertainties in the fitted model parameters and algorithmic approximations:

# COMMAND ----------

# this disables ML autolog as we are just trying to optimise the process and don't need to log everything
import mlflow
mlflow.autolog(disable=True)

# COMMAND ----------

gcm.config.disable_progress_bars()  # We turn off the progress bars here to reduce the number of outputs.
median_attributions, confidence_intervals, = gcm.confidence_intervals(
    gcm.fit_and_compute(
        gcm.attribute_anomalies,
        loaded_scm,
        bootstrap_training_data=train,
        target_node='quality',
        anomaly_samples=pd.DataFrame([defects_dimensions.iloc[0]])
        ),
    num_bootstrap_resamples=10
    )

# COMMAND ----------

bar_plot(median_attributions, confidence_intervals, 'Anomaly attribution score')

# COMMAND ----------

# MAGIC %md
# MAGIC The results are similar to before, but the confidence interval for some nodes, like `chamber_humidity`, includes zero, indicating that its contribution is insignificant.

# COMMAND ----------

# MAGIC %md
# MAGIC ### What caused the quality drop in the new batch of products?

# COMMAND ----------

# MAGIC %md
# MAGIC In the previous section, we focused on anomaly attribution for a single observation. Now, let’s explore a shift in quality across batches of processed products. To illustrate this, we’ll simulate a scenario where `worker = 0`, who previously processed 75% of the products, has taken leave, and `worker = 1` has taken over her role. In this new batch, `worker = 1` processes 75% of the products as a manual operator, while `worker = 0` processes only 25%. We can use the same `generate_data` function from the `99_utils` notebook to create this data by specifying the `p_worker` argument. For more details, refer to the `99_utils` notebook.

# COMMAND ----------

test = generate_data(catalog, schema, 100, p_worker=0.25, train=False)
test.describe()

# COMMAND ----------

# MAGIC %md
# MAGIC Indeed, the defect rate has risen from 0.075 to 0.2 compared to the dataset used to train the models. Let’s now see if DoWhy can pinpoint the root cause of this. We will apply the [distribution change method](https://proceedings.mlr.press/v130/budhathoki21a.html) to identify the part in the system that has changed:

# COMMAND ----------

median_attributions, confidence_intervals = gcm.confidence_intervals(
    lambda: gcm.distribution_change(loaded_scm,
                                    train,
                                    test,
                                    target_node='quality',
                                    # Here, we are intersted in explaining the differences in the mean.
                                    difference_estimation_func=lambda x, y: np.mean(y) - np.mean(x)) 
)

# COMMAND ----------

bar_plot(median_attributions, confidence_intervals, 'Change attribution in defect rate')

# COMMAND ----------

# MAGIC %md
# MAGIC In our case, the distribution change method explains the change in the defect rate, i.e., a positive value to an increase of the mean and a negative value indicates that a node contributes to a decrease. Using the bar chart, we get a very clear picture that the change in `worker` has actually a significant positive contribution to the expected defect rate due to the increase of `position_alignment`.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Wrap up
# MAGIC
# MAGIC In this notebook, we performed causal analysis, including causal attributions and root cause analysis, across different levels of data granularity. We explored how to use DoWhy's functions, such as `attribute_anomalies` and `confidence_intervals`, and interpreted their results. In the next notebook, we will deploy the fitted graph to Databricks Model Serving and explore how to enable real-time causal analysis.

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
# MAGIC | lime | Local Interpretable Model-Agnostic Explanations for machine learning classifiers | BSD | https://pypi.org/project/lime/
# MAGIC | shap | A unified approach to explain the output of any machine learning model | MIT | https://pypi.org/project/shap/
