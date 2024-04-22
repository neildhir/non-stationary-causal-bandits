# Non-stationary (actually piecewise stationary) SCM-MAB a.k.a. _chronological causal bandit_

This paper studies an instance of the multi-armed bandit (MAB) problem, specifically where several causal MABs operate chronologically in the same dynamical system. Practically the reward distribution of each bandit is governed by the same non-trivial dependence structure, which is a dynamic causal model. Dynamic because we allow for each causal MAB to depend on the preceding MAB and in doing so are able to transfer information between agents. Our contribution, the Chronological Causal Bandit (CCB), is useful in discrete decision-making settings where the causal effects are changing across time and can be informed by earlier interventions in the same system. In this paper, we present some early findings of the CCB as demonstrated on a toy problem.

Visual representation of model and method.

![CCB](./figures/Screenshot%202024-04-22%20at%2010.24.25.png)

## Installation

```python
pip install .
```

## Paper

This work is published in [this paper](https://arxiv.org/pdf/2112.01819.pdf).

## Implementation

This implementation builds heavily upon the original [SCM-MAB code](https://github.com/sanghack81/SCMMAB-NIPS2018).

## Cite

If you use this code in your research, please consider citing:

```bib
@inproceedings{CCB,
 author = {Dhir, Neil},
 booktitle = {NeurIPS 2021 workshop Causal Inference Challenges in Sequential Decision Making: Bridging Theory and Practice}
 title = {Chronological Causal Bandits}
 volume = {35},
 year = {2021}
}
```
