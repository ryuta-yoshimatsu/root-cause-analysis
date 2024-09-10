# Databricks notebook source
# MAGIC %pip install dowhy networkx --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC # Finding the Root Cause of Elevated Latencies in a Microservice Architecture
# MAGIC
# MAGIC In this case study, we identify the root causes of "unexpected" observed latencies in cloud services that empower an
# MAGIC online shop. We focus on the process of placing an order, which involves different services to make sure that
# MAGIC the placed order is valid, the customer is authenticated, the shipping costs are calculated correctly, and the shipping
# MAGIC process is initiated accordingly. The dependencies of the services is shown in the graph below.

# COMMAND ----------

# MAGIC %md
# MAGIC <img src="images/microservice-architecture-dependencies.png" width="800"/>

# COMMAND ----------

# MAGIC %md
# MAGIC This kind of dependency graph could be obtained from services like [Amazon X-Ray](https://aws.amazon.com/xray/) or
# MAGIC defined manually based on the trace structure of requests.
# MAGIC
# MAGIC We assume that the dependency graph above is correct and that we are able to measure the latency (in seconds) of each node for an order request. In case of `Website`, the latency would represent the time until a confirmation of the order is shown. For simplicity, let us assume that the services are synchronized, i.e., a service has to wait for downstream services in order to proceed. Further, we assume that two nodes are not impacted by unobserved factors (hidden confounders) at the same time (i.e., causal sufficiency). Seeing that, for instance, network traffic affects multiple services, this assumption might be typically violated in a real-world scenario. However, weak confounders can be neglected, while stronger ones (like network traffic) could falsely render multiple nodes as root causes. Generally, we can only identify causes that are part of the data.
# MAGIC
# MAGIC Under these assumptions, the observed latency of a node is defined by the latency of the node itself (intrinsic latency), and the sum over all latencies of direct child nodes. This could also include calling a child node multiple times.
# MAGIC
# MAGIC Let us load data with observed latencies of each node.

# COMMAND ----------

import pandas as pd

normal_data = pd.read_csv("data/rca_microservice_architecture_latencies.csv")
normal_data.head()

# COMMAND ----------

# MAGIC %md
# MAGIC Let us also take a look at the pair-wise scatter plots and histograms of the variables.

# COMMAND ----------

axes = pd.plotting.scatter_matrix(normal_data, figsize=(10, 10), c='#ff0d57', alpha=0.2, hist_kwds={'color':['#1E88E5']});
for ax in axes.flatten():
    ax.xaxis.label.set_rotation(90)
    ax.yaxis.label.set_rotation(0)
    ax.yaxis.label.set_ha('right')

# COMMAND ----------

# MAGIC %md
# MAGIC In the matrix above, the plots on the diagonal line are histograms of variables, whereas those outside of the diagonal are scatter plots of pair of variables. The histograms of services without a dependency, namely `Customer DB`, `Product DB`, `Order DB` and `Shipping Cost Service`, have shapes similar to one half of a Gaussian distribution. The scatter plots of various pairs of variables (e.g., `API` and `www`, `www` and `Website`, `Order Service` and `Order DB`) show linear relations. We shall use this information shortly to assign generative causal models to nodes in the causal graph.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setting up the causal model
# MAGIC
# MAGIC If we look at the `Website` node, it becomes apparent that the latency we experience there depends on the latencies of
# MAGIC all downstream nodes. In particular, if one of the downstream nodes takes a long time, `Website` will also take a
# MAGIC long time to show an update. Seeing this, the causal graph of the latencies can be built by inverting the arrows of the
# MAGIC service graph.

# COMMAND ----------

import networkx as nx
from dowhy import gcm
from dowhy.utils import plot, bar_plot

causal_graph = nx.DiGraph([('www', 'Website'),
                           ('Auth Service', 'www'),
                           ('API', 'www'),
                           ('Customer DB', 'Auth Service'),
                           ('Customer DB', 'API'),
                           ('Product Service', 'API'),
                           ('Auth Service', 'API'),
                           ('Order Service', 'API'),
                           ('Shipping Cost Service', 'Product Service'),
                           ('Caching Service', 'Product Service'),
                           ('Product DB', 'Caching Service'),
                           ('Customer DB', 'Product Service'),
                           ('Order DB', 'Order Service')])

# COMMAND ----------

plot(causal_graph, figure_size=[13, 13])

# COMMAND ----------

# MAGIC %md
# MAGIC > Here, we are interested in the causal relationships between latencies of services rather than the order of calling the services.

# COMMAND ----------

# MAGIC %md
# MAGIC We will use the information from the pair-wise scatter plots and histograms to manually assign causal models. In particular, we assign half-Normal distributions to the root nodes (i.e., `Customer DB`, `Product DB`, `Order DB` and `Shipping Cost Service`). For non-root nodes, we assign linear additive noise models (which scatter plots of many parent-child pairs indicate) with empirical distribution of noise terms.

# COMMAND ----------

from scipy.stats import halfnorm

causal_model = gcm.StructuralCausalModel(causal_graph)

for node in causal_graph.nodes:
    if len(list(causal_graph.predecessors(node))) > 0:
        causal_model.set_causal_mechanism(node, gcm.AdditiveNoiseModel(gcm.ml.create_linear_regressor()))
    else:
        causal_model.set_causal_mechanism(node, gcm.ScipyDistribution(halfnorm))

# COMMAND ----------

# MAGIC %md
# MAGIC > Alternatively, we can also automate this **if** we don't have prior knowledge or are not familiar with the statistical implications:
# MAGIC ```
# MAGIC gcm.auto.assign_causal_mechanisms(causal_model, normal_data)
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC Before we contiue with the first scenario, let's first evaluate our causal model:

# COMMAND ----------

gcm.fit(causal_model, normal_data)
print(gcm.evaluate_causal_model(causal_model, normal_data))

# COMMAND ----------

# MAGIC %md
# MAGIC This confirms the goodness of our causal model. However, we also see that for two nodes ('Product Service' and 'Caching Service'), the additive noise model assumption is violated. This also aligns with the data generation process, where these two nodes follow non-additive noise models. As we see in the following, most of the algorithms are still fairly robust against such violations or poor performance of the causal mechanism.

# COMMAND ----------

# MAGIC %md
# MAGIC > For more detailed insights, set compare_mechanism_baselines to True. However, this will take significantly longer.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Scenario 1: Observing a single outlier
# MAGIC
# MAGIC Suppose we get an alert from our system where a customer experienced an unusually high latency when
# MAGIC an order is placed. Our task is now to investigate this issue and to find the root cause of this behaviour.
# MAGIC
# MAGIC We first load the latency to the corresponding alert.

# COMMAND ----------

outlier_data = pd.read_csv("data/rca_microservice_architecture_anomaly.csv")
outlier_data

# COMMAND ----------

# MAGIC %md
# MAGIC We are interested in the increased latency of `Website` which the customer directly experienced.

# COMMAND ----------

outlier_data.iloc[0]['Website']-normal_data['Website'].mean()

# COMMAND ----------

# MAGIC %md
# MAGIC For this customer, `Website` was roughly 2 seconds slower than for other customers on average. Why?

# COMMAND ----------

# MAGIC %md
# MAGIC ### Attributing an outlier latency at a target service to other services
# MAGIC
# MAGIC To answer why `Website` was slower for this customer, we attribute the outlier latency at `Website` to upstream services in the causal graph. We refer the reader to [Janzing et al., 2019](https://arxiv.org/abs/1912.02724) for scientific details behind this API. We will calculate a 95% bootstrapped confidence interval of our attributions. In particular, we learn the causal models from a random subset of normal data and attribute the target outlier score using those models, repeating the process 10 times. This way, the confidence intervals we report account for (a) the uncertainty of our causal models as well as (b) the uncertainty in the attributions due to the variance in the samples drawn from those causal models.

# COMMAND ----------

# this disables ML autolog as we are just trying to optimise the process and don't need to log everything
import mlflow
mlflow.autolog(disable=True)

# COMMAND ----------

gcm.config.disable_progress_bars() # to disable print statements when computing Shapley values

median_attribs, uncertainty_attribs = gcm.confidence_intervals(
    gcm.fit_and_compute(gcm.attribute_anomalies,
                        causal_model,
                        normal_data,
                        target_node='Website',
                        anomaly_samples=outlier_data),
    num_bootstrap_resamples=10)

# COMMAND ----------

# MAGIC %md
# MAGIC > By default, a quantile-based anomaly score is used that estimates the negative log-probability of a sample being
# MAGIC normal. This is, the higher the probabilty of an outlier, the larger the score. The library offers different kinds of outlier scoring functions, such as the <a href="https://en.wikipedia.org/wiki/Standard_score">z-score</a>, where the mean is the expected value based on the causal model.

# COMMAND ----------

# MAGIC %md
# MAGIC Let us visualize the attributions along with their uncertainty in a bar plot.

# COMMAND ----------

bar_plot(median_attribs, uncertainty_attribs, 'Attribution Score')

# COMMAND ----------

# MAGIC %md
# MAGIC The attributions indicate that `Caching Service` is the main driver of high latency in `Website` which is expected as we perturb the causal mechanism of `Caching Service` to generate an outlier latency in `Website` (see Appendix below). Attributions to `Customer DB` and `Product Service` can be explained by misspecification of causal models. First, some of the parent-child relationships in the causal graph are non-linear (by looking at the scatter matrix). Second, the parent child-relationship between `Caching Service` and `Product DB` seems to indicate two mechanisms. This could be due to an unobserved binary variable (e.g., Cache hit/miss) that has a multiplicative effect on `Caching Service`. An additive noise cannot capture the multiplicative effect of this unobserved variable.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Scenario 2: Observing permanent degradation of latencies
# MAGIC
# MAGIC In the previous scenario, we attributed a *single* outlier latency in `Website` to services that are nodes in the causal graph, which is useful for anecdotal deep dives. Next, we consider a scenario where we observe a permanent degradation of latencies and we want to understand its drivers. In particular, we attribute the change in the average latency of `Website` to upstream nodes.
# MAGIC
# MAGIC Suppose we get additional 1000 requests with higher latencies as follows.

# COMMAND ----------

outlier_data = pd.read_csv("data/rca_microservice_architecture_anomaly_1000.csv")
outlier_data.head()

# COMMAND ----------

# MAGIC %md
# MAGIC We are interested in the increased latency of `Website` on average for 1000 requests which the customers directly experienced.

# COMMAND ----------

outlier_data['Website'].mean() - normal_data['Website'].mean()

# COMMAND ----------

# MAGIC %md
# MAGIC The _Website_ is slower on average (by almost 2 seconds) than usual. Why?

# COMMAND ----------

# MAGIC %md
# MAGIC ### Attributing permanent degradation of latencies at a target service to other services
# MAGIC
# MAGIC To answer why `Website` is slower for those 1000 requests compared to before, we attribute the change in the average latency of `Website` to services upstream in the causal graph. We refer the reader to [Budhathoki et al., 2021](https://assets.amazon.science/b6/c0/604565d24d049a1b83355921cc6c/why-did-the-distribution-change.pdf) for scientific details behind this API. As in the previous scenario, we will calculate a 95% bootstrapped confidence interval of our attributions and visualize them in a bar plot.

# COMMAND ----------

import numpy as np

median_attribs, uncertainty_attribs = gcm.confidence_intervals(
    lambda : gcm.distribution_change(causal_model,
                                     normal_data.sample(frac=0.6),
                                     outlier_data.sample(frac=0.6),
                                     'Website',
                                     difference_estimation_func=lambda x, y: np.mean(y) - np.mean(x)),
    num_bootstrap_resamples = 10)

bar_plot(median_attribs, uncertainty_attribs, 'Attribution Score')

# COMMAND ----------

# MAGIC %md
# MAGIC We observe that `Caching Service` is the root cause that slowed down `Website`. In particular, the method we used tells us that the change in the causal mechanism (i.e., the input-output behaviour) of `Caching Service` (e.g., Caching algorithm) slowed down `Website`. This is also expected as the outlier latencies were generated by changing the causal mechanism of `Caching Service` (see Appendix below).

# COMMAND ----------

# MAGIC %md
# MAGIC ## Scenario 3: Simulating the intervention of shifting resources
# MAGIC
# MAGIC Next, let us imagine a scenario where permanent degradation has happened as in scenario 2 and we've successfully identified `Caching Service` as the root cause. Furthermore, we figured out that a recent deployment of the `Caching Service` contained a bug that is causing the overloaded hosts. A proper fix must be deployed, or the previous deployment must be rolled back. But, in the meantime, could we mitigate the situation by shifting over some resources from `Shipping Service` to `Caching Service`? And would that help? Before doing it in reality, let us simulate it first and see whether it improves the situation.
# MAGIC
# MAGIC <img src="shifting-resources.png" width="600"/>
# MAGIC
# MAGIC Let’s perform an intervention where we say we can reduce the average time of `Caching Service` by 1s. But at the same time we buy this speed-up by an average slow-down of 2s in `Shipping Cost Service`.

# COMMAND ----------

median_mean_latencies, uncertainty_mean_latencies = gcm.confidence_intervals(
    lambda : gcm.fit_and_compute(gcm.interventional_samples,
                                 causal_model,
                                 outlier_data,
                                 interventions = {
                                    "Caching Service": lambda x: x-1,
                                    "Shipping Cost Service": lambda x: x+2
                                 },
                                 observed_data=outlier_data)().mean().to_dict(),
    num_bootstrap_resamples=10)

# COMMAND ----------

# MAGIC %md
# MAGIC Has the situation improved? Let's visualize the results.

# COMMAND ----------

avg_website_latency_before = outlier_data.mean().to_dict()['Website']
bar_plot(dict(before=avg_website_latency_before, after=median_mean_latencies['Website']),
                  dict(before=np.array([avg_website_latency_before, avg_website_latency_before]), after=uncertainty_mean_latencies['Website']),
                  ylabel='Avg. Website Latency',
                  figure_size=(3, 2),
                  bar_width=0.4,
                  xticks=['Before', 'After'],
                  xticks_rotation=45)

# COMMAND ----------

# MAGIC %md
# MAGIC Indeed, we do get an improvement by about 1s. We’re not back at normal operation, but we’ve mitigated part of the problem. From here, maybe we can wait until a proper fix is deployed.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Appendix: Data generation process
# MAGIC
# MAGIC The scenarios above work on synthetic data. The normal data was generated using the following functions:

# COMMAND ----------

from scipy.stats import truncexpon, halfnorm


def create_observed_latency_data(unobserved_intrinsic_latencies):
    observed_latencies = {}
    observed_latencies['Product DB'] = unobserved_intrinsic_latencies['Product DB']
    observed_latencies['Customer DB'] = unobserved_intrinsic_latencies['Customer DB']
    observed_latencies['Order DB'] = unobserved_intrinsic_latencies['Order DB']
    observed_latencies['Shipping Cost Service'] = unobserved_intrinsic_latencies['Shipping Cost Service']
    observed_latencies['Caching Service'] = np.random.choice([0, 1], size=(len(observed_latencies['Product DB']),),
                                                             p=[.5, .5]) * \
                                            observed_latencies['Product DB'] \
                                            + unobserved_intrinsic_latencies['Caching Service']
    observed_latencies['Product Service'] = np.maximum(np.maximum(observed_latencies['Shipping Cost Service'],
                                                                  observed_latencies['Caching Service']),
                                                       observed_latencies['Customer DB']) \
                                            + unobserved_intrinsic_latencies['Product Service']
    observed_latencies['Auth Service'] = observed_latencies['Customer DB'] \
                                         + unobserved_intrinsic_latencies['Auth Service']
    observed_latencies['Order Service'] = observed_latencies['Order DB'] \
                                          + unobserved_intrinsic_latencies['Order Service']
    observed_latencies['API'] = observed_latencies['Product Service'] \
                                + observed_latencies['Customer DB'] \
                                + observed_latencies['Auth Service'] \
                                + observed_latencies['Order Service'] \
                                + unobserved_intrinsic_latencies['API']
    observed_latencies['www'] = observed_latencies['API'] \
                                + observed_latencies['Auth Service'] \
                                + unobserved_intrinsic_latencies['www']
    observed_latencies['Website'] = observed_latencies['www'] \
                                    + unobserved_intrinsic_latencies['Website']

    return pd.DataFrame(observed_latencies)


def unobserved_intrinsic_latencies_normal(num_samples):
    return {
        'Website': truncexpon.rvs(size=num_samples, b=3, scale=0.2),
        'www': truncexpon.rvs(size=num_samples, b=2, scale=0.2),
        'API': halfnorm.rvs(size=num_samples, loc=0.5, scale=0.2),
        'Auth Service': halfnorm.rvs(size=num_samples, loc=0.1, scale=0.2),
        'Product Service': halfnorm.rvs(size=num_samples, loc=0.1, scale=0.2),
        'Order Service': halfnorm.rvs(size=num_samples, loc=0.5, scale=0.2),
        'Shipping Cost Service': halfnorm.rvs(size=num_samples, loc=0.1, scale=0.2),
        'Caching Service': halfnorm.rvs(size=num_samples, loc=0.1, scale=0.1),
        'Order DB': truncexpon.rvs(size=num_samples, b=5, scale=0.2),
        'Customer DB': truncexpon.rvs(size=num_samples, b=6, scale=0.2),
        'Product DB': truncexpon.rvs(size=num_samples, b=10, scale=0.2)
    }


normal_data = create_observed_latency_data(unobserved_intrinsic_latencies_normal(10000))

# COMMAND ----------

# MAGIC %md
# MAGIC This simulates the latency relationships under the assumption of having synchronized services and that there are no
# MAGIC hidden aspects that impact two nodes at the same time. Furthermore, we assume that the Caching Service has to call through to the Product DB only in 50% of the cases (i.e., we have a 50% cache miss rate). Also, we assume that the Product Service can make calls in parallel to its downstream services Shipping Cost Service, Caching Service, and Customer DB and join the threads when all three service have returned.

# COMMAND ----------

# MAGIC %md
# MAGIC > We use <a href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.truncexpon.html">truncated exponential</a> and
# MAGIC <a href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.halfnorm.html">half-normal</a> distributions,
# MAGIC since their shapes are similar to distributions observed in real services.

# COMMAND ----------

# MAGIC %md
# MAGIC The anomalous data is generated in the following way:

# COMMAND ----------

def unobserved_intrinsic_latencies_anomalous(num_samples):
    return {
        'Website': truncexpon.rvs(size=num_samples, b=3, scale=0.2),
        'www': truncexpon.rvs(size=num_samples, b=2, scale=0.2),
        'API': halfnorm.rvs(size=num_samples, loc=0.5, scale=0.2),
        'Auth Service': halfnorm.rvs(size=num_samples, loc=0.1, scale=0.2),
        'Product Service': halfnorm.rvs(size=num_samples, loc=0.1, scale=0.2),
        'Order Service': halfnorm.rvs(size=num_samples, loc=0.5, scale=0.2),
        'Shipping Cost Service': halfnorm.rvs(size=num_samples, loc=0.1, scale=0.2),
        'Caching Service': 2 + halfnorm.rvs(size=num_samples, loc=0.1, scale=0.1),
        'Order DB': truncexpon.rvs(size=num_samples, b=5, scale=0.2),
        'Customer DB': truncexpon.rvs(size=num_samples, b=6, scale=0.2),
        'Product DB': truncexpon.rvs(size=num_samples, b=10, scale=0.2)
    }

outlier_data = create_observed_latency_data(unobserved_intrinsic_latencies_anomalous(1000))

# COMMAND ----------

# MAGIC %md
# MAGIC Here, we significantly increased the average time of the *Caching Service* by two seconds, which coincides with our
# MAGIC results from the RCA. Note that a high latency in *Caching Service* would lead to a constantly higher latency in upstream
# MAGIC services. In particular, customers experience a higher latency than usual.
