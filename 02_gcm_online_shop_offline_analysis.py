# Databricks notebook source
# MAGIC %pip install dowhy networkx --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

catalog = "ryuta"
db = "causal"
model = "root_cause_analysis_scm"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Answer causal questions
# MAGIC ### Load the model

# COMMAND ----------

import mlflow
from mlflow import MlflowClient
import pandas as pd

mlflow.set_registry_uri("databricks-uc")
mlflow_client = MlflowClient()

registered_model_name = f"{catalog}.{db}.{model}"
model = f"models:/{registered_model_name}@Champion"

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(model)
loaded_scm = loaded_model.unwrap_python_model().load_scm()
loaded_causal_graph = loaded_model.unwrap_python_model().load_causal_graph()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Generate new samples
# MAGIC
# MAGIC Since we learned about the data generation process, we can also generate new samples:

# COMMAND ----------

from dowhy import gcm
gcm.draw_samples(loaded_scm, num_samples=10)

# COMMAND ----------

# MAGIC %md
# MAGIC We have drawn 10 samples from the joint distribution following the learned causal relationships.

# COMMAND ----------

# MAGIC %md
# MAGIC ### What are the key factors influencing the variance in profit?

# COMMAND ----------

# MAGIC %md
# MAGIC At this point, we want to understand which factors drive changes in the Profit. Let us first have a closer look at the Profit over time. For this, we plot the Profit over time for 2021, where the produced plot shows the Profit in dollars on the Y-axis and the time on the X-axis.

# COMMAND ----------

data_2021 = spark.read.table(f"{catalog}.{db}.data_2021")
data_2021 = data_2021.toPandas().set_index("Date")
data_2021.head()

# COMMAND ----------

data_2021['Profit'].plot(ylabel='Profit in $', figsize=(15,5), rot=45)

# COMMAND ----------

# MAGIC %md
# MAGIC We see some significant spikes in the Profit across the year. We can further quantify this by looking at the standard deviation:

# COMMAND ----------

data_2021['Profit'].std()

# COMMAND ----------

# MAGIC %md
# MAGIC The estimated standard deviation of ~259247 dollars is quite significant. Looking at the causal graph, we see that Revenue and Operational Cost have a direct impact on the Profit, but which of them contribute the most to the variance? To find this out, we can make use of the direct arrow strength algorithm that quantifies the causal influence of a specific arrow in the graph:

# COMMAND ----------

import numpy as np
from dowhy.utils import plot

# Note: The percentage conversion only makes sense for purely positive attributions.
def convert_to_percentage(value_dictionary):
    total_absolute_sum = np.sum([abs(v) for v in value_dictionary.values()])
    return {k: abs(v) / total_absolute_sum * 100 for k, v in value_dictionary.items()}


arrow_strengths = gcm.arrow_strength(loaded_scm, target_node='Profit')

plot(loaded_causal_graph, 
     causal_strengths=convert_to_percentage(arrow_strengths), 
     figure_size=[15, 10])

# COMMAND ----------

arrow_strengths

# COMMAND ----------

# MAGIC %md
# MAGIC In this causal graph, we see how much each node contributes to the variance in Profit. For simplicity, the contributions are converted to percentages. Since Profit itself is only the difference between Revenue and Operational Cost, we do not expect further factors influencing the variance. As we see, Revenue has more impact than Operational Cost. This makes sense seeing that Revenue typically varies more than Operational Cost due to the stronger dependency on the number of sold units. Note that the direct arrow strength method also supports the use of other kinds of measures, for instance, KL divergence. 
# MAGIC
# MAGIC While the direct influences are helpful in understanding which direct parents influence the most on the variance in Profit, this mostly confirms our prior belief. The question of which factor is ultimately responsible for this high variance is, however, still unclear. For instance, Revenue itself is based on Sold Units and the Unit Price. Although we could recursively apply the direct arrow strength to all nodes, we would not get a correctly weighted insight into the influence of upstream nodes on the variance.
# MAGIC
# MAGIC What are the important causal factors contributing to the variance in Profit? To find this out, we can use the intrinsic causal contribution method that attributes the variance in Profit to the upstream nodes in the causal graph by only considering information that is newly added by a node and not just inherited from its parents. For instance, a node that is simply a rescaled version of its parent would not have any intrinsic contribution. See the corresponding [research paper](https://arxiv.org/abs/2007.00714) for more details.
# MAGIC
# MAGIC Let's apply the method to the data:

# COMMAND ----------

iccs = gcm.intrinsic_causal_influence(loaded_scm, target_node='Profit', num_samples_randomization=500)

# COMMAND ----------

from dowhy.utils import bar_plot

bar_plot(convert_to_percentage(iccs), ylabel='Variance attribution in %')

# COMMAND ----------

# MAGIC %md
# MAGIC The scores shown in this bar chart are percentages indicating how much variance each node is contributing to Profit — without inheriting the variance from its parents in the causal graph. As we see quite clearly, the Shopping Event has by far the biggest influence on the variance in our Profit. This makes sense, seeing that the sales are heavily impacted during promotion periods like Black Friday or Prime Day and, thus, impact the overall profit. Surprisingly, we also see that factors such as the number of sold units or number of page views have a rather small influence, i.e., the large variance in profit can be almost completely explained by the shopping events. Let’s check this visually by marking the days where we had a shopping event. To do so, we use the pandas plot function again, but additionally mark all points in the plot with a vertical red bar where a shopping event occured:

# COMMAND ----------

import matplotlib.pyplot as plt

data_2021['Profit'].plot(ylabel='Profit in $', figsize=(15,5), rot=45)
plt.vlines(np.arange(0, data_2021.shape[0])[data_2021['Shopping_Event']], data_2021['Profit'].min(), data_2021['Profit'].max(), linewidth=10, alpha=0.3, color='r')

# COMMAND ----------

# MAGIC %md
# MAGIC We clearly see that the shopping events coincide with the high peaks in profit. While we could have investigated this manually by looking at all kinds of different relationships or using domain knowledge, the tasks gets much more difficult as the complexity of the system increases. With a few lines of code, we obtained these insights from DoWhy.

# COMMAND ----------

# MAGIC %md
# MAGIC ### What are the key factors explaining the Profit drop on a particular day?

# COMMAND ----------

# MAGIC %md
# MAGIC After a successful year in terms of profit, newer technologies come to the market and, thus, we want to keep the profit up and get rid of excess inventory by selling more devices. In order to increase the demand, we therefore lower the retail price by 10% at the beginning of 2022. Based on a prior analysis, we know that a decrease of 10% in the price would roughly increase the demand by 13.75%, a slight surplus. Following the price elasticity of demand model, we expect an increase of around 37.5% in number of Sold Units. Let us take a look if this is true by loading the data for the first day in 2022 and taking the fraction between the numbers of Sold Units from both years for that day:

# COMMAND ----------

data_first_day_2022 = spark.read.table(f"{catalog}.{db}.data_first_day_2022")
data_first_day_2022 = data_first_day_2022.toPandas().set_index("Date")
(data_first_day_2022['Sold_Units'][0] / data_2021['Sold_Units'][0] - 1) * 100

# COMMAND ----------

# MAGIC %md
# MAGIC Surprisingly, we only increased the number of sold units by ~19%. This will certainly impact the profit given that the revenue is much smaller than expected. Let us compare it with the previous year at the same time:

# COMMAND ----------

(1 - data_first_day_2022['Profit'][0] / data_2021['Profit'][0]) * 100

# COMMAND ----------

# MAGIC %md
# MAGIC Indeed, the profit dropped by ~8.5%. Why is this the case seeing that we would expect a much higher demand due to the decreased price? Let us investigate what is going on here.
# MAGIC
# MAGIC In order to figure out what contributed to the Profit drop, we can make use of DoWhy’s anomaly attribution feature. Here, we only need to specify the target node we are interested in (the Profit) and the anomaly sample we want to analyze (the first day of 2022). These results are then plotted in a bar chart indicating the attribution scores of each node for the given anomaly sample:

# COMMAND ----------

data_first_day_2022

# COMMAND ----------

attributions = gcm.attribute_anomalies(loaded_scm, target_node='Profit', anomaly_samples=data_first_day_2022)

bar_plot({k: v[0] for k, v in attributions.items()}, ylabel='Anomaly attribution score')

# COMMAND ----------

# MAGIC %md
# MAGIC A positive attribution score means that the corresponding node contributed to the observed anomaly, which is in our case the drop in Profit. A negative score of a node indicates that the observed value for the node is actually reducing the likelihood of the anomaly (e.g., a higher demand due to the decreased price should increase the profit). More details about the interpretation of the score can be found in the corresponding [reserach paper](https://proceedings.mlr.press/v162/budhathoki22a.html). Interestingly, the Page Views stand out as a factor explaining the Profit drop that day as indicated in the bar chart shown here.
# MAGIC
# MAGIC While this method gives us a point estimate of the attributions for the particular models and parameters we learned, we can also use DoWhy’s confidence interval feature, which incorporates uncertainties about the fitted model parameters and algorithmic approximations:

# COMMAND ----------

# this disables ML autolog as we are just trying to optimise the process and don't need to log everything
import mlflow
mlflow.autolog(disable=True)

# COMMAND ----------

gcm.config.disable_progress_bars()  # We turn off the progress bars here to reduce the number of outputs.

median_attributions, confidence_intervals, = gcm.confidence_intervals(
    gcm.fit_and_compute(gcm.attribute_anomalies,
                        loaded_scm,
                        bootstrap_training_data=data_2021,
                        target_node='Profit',
                        anomaly_samples=data_first_day_2022),
    num_bootstrap_resamples=10)

# COMMAND ----------

bar_plot(median_attributions, confidence_intervals, 'Anomaly attribution score')

# COMMAND ----------

# MAGIC %md
# MAGIC Note, in this bar chart we see the median attributions over multiple runs on smaller data sets, where each run re-fits the models and re-evaluates the attributions. We get a similar picture as before, but the confidence interval of the attribution to Sold Units also contains zero, meaning its contribution is insignificant. But some important questions still remain: Was this only a coincidence and, if not, which part in our system has changed? To find this out, we need to collect some more data.

# COMMAND ----------

# MAGIC %md
# MAGIC > Note that the results differ depending on the selected data, since they are sample specific. On other days, other factors could be relevant. Furthermore, note that the analysis (including the confidence intervals) always relies on the modeling assumptions made. In other words, if the models change or have a poor fit, one would also expect different results.

# COMMAND ----------

# MAGIC %md
# MAGIC ### What caused the profit drop in Q1 2022?

# COMMAND ----------

# MAGIC %md
# MAGIC While the previous analysis is based on a single observation, let us see if this was just coincidence or if this is a persistent issue. When preparing the quarterly business report, we have some more data available from the first three months. We first check if the profit dropped on average in the first quarter of 2022 as compared to 2021. Similar as before, we can do this by taking the fraction between the average Profit of 2022 and 2021 for the first quarter:

# COMMAND ----------

data_first_quarter_2021 = data_2021[data_2021.index <= '2021-03-31']
data_first_quarter_2022 = spark.read.table(f"{catalog}.{db}.data_first_quarter_2022")
data_first_quarter_2022 = data_first_quarter_2022.toPandas().set_index("Date")

(1 - data_first_quarter_2022['Profit'].mean() / data_first_quarter_2021['Profit'].mean()) * 100

# COMMAND ----------

# MAGIC %md
# MAGIC Indeed, the profit drop is persistent in the first quarter of 2022. Now, what is the root cause of this? Let us apply the [distribution change method](https://proceedings.mlr.press/v130/budhathoki21a.html) to identify the part in the system that has changed:

# COMMAND ----------

median_attributions, confidence_intervals = gcm.confidence_intervals(
    lambda: gcm.distribution_change(loaded_scm,
                                    data_first_quarter_2021,
                                    data_first_quarter_2022,
                                    target_node='Profit',
                                    # Here, we are intersted in explaining the differences in the mean.
                                    difference_estimation_func=lambda x, y: np.mean(y) - np.mean(x)) 
)

# COMMAND ----------

bar_plot(median_attributions, confidence_intervals, 'Profit change attribution in $')

# COMMAND ----------

# MAGIC %md
# MAGIC In our case, the distribution change method explains the change in the mean of Profit, i.e., a negative value indicates that a node contributes to a decrease and a positive value to an increase of the mean. Using the bar chart, we get now a very clear picture that the change in Unit Price has actually a slightly positive contribution to the expected Profit due to the increase of Sold Units, but it seems that the issue is coming from the Page Views which has a negative value. While we already understood this as a main driver of the drop at the beginning of 2022, we have now isolated and confirmed that something changed for the Page Views as well. Let’s compare the average Page Views with the previous year.

# COMMAND ----------

(1 - data_first_quarter_2022['Page_Views'].mean() / data_first_quarter_2021['Page_Views'].mean()) * 100

# COMMAND ----------

# MAGIC %md
# MAGIC Indeed, the number of Page Views dropped by ~14%. Since we eliminated all other potential factors, we can now dive deeper into the Page Views and see what is going on there. This is a hypothetical scenario, but we could imagine it could be due to a change in the search algorithm which ranks this product lower in the results and therefore drives fewer customers to the product page. Knowing this, we could now start mitigating the issue.

# COMMAND ----------

# MAGIC %md
# MAGIC © 2024 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License. All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC
# MAGIC | library                                | description             | license    | source                                              |
# MAGIC |----------------------------------------|-------------------------|------------|-----------------------------------------------------|
# MAGIC | dowhy | A Python library for causal inference that supports explicit modeling and testing of causal assumptions | MIT | https://pypi.org/project/dowhy/
# MAGIC | networkx | A Python package for the creation, manipulation, and study of the structure, dynamics, and functions of complex networks. | BSD License | https://pypi.org/project/networkx/

# COMMAND ----------


