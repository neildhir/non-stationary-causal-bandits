# -*- coding: utf-8 -*-
# =============================================
# Title:  Discrete forecasting models
# File:   forecast.py
# Date:   03 September 2021
# =============================================

from numpy import ndarray, vectorize
from scipy.stats import bernoulli


def make_conditional_bernoulli(trans_mat: ndarray):
    assert isinstance(trans_mat, ndarray)
    assert all([x == 1 for x in trans_mat.sum(axis=1)])
    transmat = {val: row for val, row in zip(range(len(trans_mat), trans_mat))}
    return vectorize(lambda val: bernoulli.rvs(p=transmat[val], size=1))

