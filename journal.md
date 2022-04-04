# Journal

- week of 1/17
Looked into Wilds benchmark (https://wilds.stanford.edu/) and scoped out a representation learning project.
TODO: set up datasets and start thinking about baseline methods to evaluate.

Learned about cmip6 dataset and thinking about alternative project where I fit an implicit function (energy based model) to the data. Could be an interesting way to combine the predictions from many models? Have to see how the ensembling in cmip6 is done already. My understanding is that all of this is simulation data, not raw data, but could still be a cool exploration.

- week of 1/24

Met with some folks from LEAP to figure out if there is some alignment between their needs and this class project. Here are meeting notes:

(1) Benmarks
Carl and Ryan identified two kinds of benchmarks, both of which seem really interesting.
One for ML practitioners/researchers to evaluate their models (even if the resulting models are not immediately useful in a "production" setting).
The second for benchmarking models upon which decisions (e.g., policies) are actively being made. It is up to climate experts to come together and agree to define this benchmark.
Other thoughts
The first benchmark is much easier to define, but Ryan brought up the point that it is not clear what train/val/test set looks like here as we do not know the future.
One idea here is to have the benchmark task test some notions of interpolation and extrapolation, both spatially and temporally. For example, consider the task of predicting near surface air-temperature (tas) as a function of latitude, longitude, time, and other variables. What would it mean for a model to generalize on this task? If I train on data from 1850-1920, 1940-2010, how accurate are my predictions from 1920-1940 (interpolation regime) and from 2010-2020 (extrapolation regime)? One could imagine similar kinds of splits to evaluate generalization over lat/long as well.
With this benchmark there is definitely an open question of what the training data is. If the data is generated from CMIP6 models, which ensemble should be used to generate it? Is there a possibility to use raw data points?
Ryan had a fantastic idea for a classification task. Given some data simulated from the CMIP6 ensembles {1, ..., n}, predict which ensemble generated the data (n-way classification). At test time, feed in empirical data and use the classifier to predict which of the ensembles is most likely to have generated the observed data. This is useful as it gives us a way to select which climate model to look at more closely for a region we are interested in.
If CMIP6 data is of coarse resolution can we expect the classifier to generalize to the empirical data, which may not not be at the same resolution?
A potentially strong baseline here is nearest-neighbors between the empirical observation and the corresponding ensemble predictions.
(2) Super Resolution
Current climate models have a limitation that they must discretize the earth. Improving the resolution of climate models is an active area of research.
One idea here is to pretrain a network on CMIP6 data (again for say surface temperature prediction task). Does pre-training on coarse resolution data and fine-tuning on localized empirical data allow us to get the best of both worlds?

- week of Jan 31

Made a visualization of the different CMIP6 ensemble members and realized there is a lot of disagreement about the predictions...
Started bringing in some network code, but need to figure out which of the ensembles to fit. Moving in a direction of super resolution. Thinking that I am going to use a NeRF type formulation to fit a neural network to CMIP6 data. Thinking about this as pre-training, fine-tuning could then be about bringing in data from a specific region and time period.
also looking at this paper: https://iopscience.iop.org/article/10.1088/1748-9326/ac1ed9
to try to figure out why the models disagree so much.

- week of Feb 7
Chatted with Dylan from class and got some strategies for how to reconcile the CMIP6 model discrepancies. Seems like mean surface temperature is a way to go. For early data fitting I might also just consider just using one set of GT for simplicity rather than trying to find the best predictions from different climate models.

- week of Feb 14
Reviewed the NeRF paper to see how many of the principles can map over for climate modeling. The volumetric rendering piece does not seem super useful, but the fact that we can fit an mlp to pretty high frequency data is encouraging. This week I hope to start on implementation and over fitting to a simple training set. I will also consider using a transformer and not just mlp.

- week of Feb 21
As I dig into the data more, I also realize that it is more complex. There are many different simulations. I will stick to the historical split, which seems to be fit on historically collected temperatures.

- week of Feb 28
Not much progress made this week as shooting for a paper deadline.

- week of Mar 7
Not much progress made this week as shooting for a paper deadline.

## 2022-03-07 check in: alp

Looking good. Would definitely encourage you to try to get some early results. Fine if you stick with the network architectures that were used by other researchers. Would also err on the side of simplicity.

- week of Mar 14
Spring break

- week of Mar 22
started writing data generation script, dataloader, and training loop.
deciding on the train/test splits following original idea of interpolation and extrapolation regimes both in spatial and temporal dimension.

- week of Mar 29
started writing data generation script, dataloader, and training loop.
deciding on the train/test splits following original idea of interpolation and extrapolation regimes both in spatial and temporal dimension.

