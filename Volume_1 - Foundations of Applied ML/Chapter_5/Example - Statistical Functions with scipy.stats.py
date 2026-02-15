# -*- coding: utf-8 -*-
"""
Created on Mon Dec  1 14:52:08 2025

@author: Admin
"""

import numpy as np
from scipy import stats

# Normal distribution PDF
x = np.linspace(-3, 3, 100)
pdf = stats.norm.pdf(x, loc=0, scale=1)

print("First 5 PDF values:", pdf[:5])

# Hypothesis test (t-test)
group1 = [2.9, 3.0, 2.5, 2.6, 3.2]
group2 = [3.8, 2.7, 4.0, 2.4, 2.8]

t_stat, p_val = stats.ttest_ind(group1, group2)

print("T-statistic:", t_stat)
print("P-value:", p_val)
