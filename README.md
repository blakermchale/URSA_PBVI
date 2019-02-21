This folder constains codes for point-based value iteration implemented for
ursa application at a checkpoint described in https://arxiv.org/abs/1902.05644

Three algorithms are implemented: (1) normal point-based value iteration in belief space (rho-POMDP), reference:

Araya, Mauricio, et al. "A POMDP extension with belief-dependent rewards." Advances in neural information processing systems. 2010.

(2) Robust version of (1), reference:

Osogami, Takayuki. "Robust partially observable Markov decision process." International Conference on Machine Learning. 2015.

(3) Chance-Constrained version of (1).

Dependency julia packages:
JuMP, Clp, Distributions
