# -*- coding: utf-8 -*-
"""
Created on Fri Nov  7 01:53:51 2025

@author: Admin
"""

# Conceptual diagram: End-to-End scikit-learn Pipeline (black & white)
# Saves: end_to_end_pipeline_structure.pdf

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrow

fig, ax = plt.subplots(figsize=(12, 8))
ax.axis("off")

def box(x, y, w, h, text):
    rect = Rectangle((x, y), w, h, fill=False, linewidth=1.5, edgecolor="black")
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, text, va="center", ha="center", fontsize=10)

def arrow_right(x1, y1, w1, h1, x2, y2, w2, h2):
    dx = (x2 - (x1 + w1))
    dy = (y2 + h2/2 - (y1 + h1/2))
    ax.add_patch(FancyArrow(x1 + w1, y1 + h1/2, dx, dy,
                            width=0.02, length_includes_head=True,
                            head_width=0.12, head_length=0.25, color="black"))

# Layout
w = 3
h = 0.9
gap_x = 0.7
start_x = 0.5
y_main = 4.0

# Main flow
box(start_x, y_main, w, h, "Data\nIngestion\n(CSV /\n DB /\n API)")
x2 = start_x + w + gap_x
box(x2, y_main, w, h, "Train/Test\n Split\n(Stratify,\n Seed)")
arrow_right(start_x, y_main, w, h, x2, y_main, w, h)

x3 = x2 + w + gap_x
box(x3, y_main, w, h, "Preprocessor\n(Column\nTransformer)")
arrow_right(x2, y_main, w, h, x3, y_main, w, h)

# Subcomponents for Preprocessor
sub_w = 3
sub_h = 0.7
sub_y_top = y_main - 1.2
box(x3 - 0.1, sub_y_top, sub_w, sub_h, "Numeric →\nStandardScaler")
sub_y_bottom = sub_y_top - 0.9
box(x3 - 0.1, sub_y_bottom, sub_w, sub_h, "Categorical →\n OneHotEncoder")
ax.plot([x3 + w/2, x3 + w/2], [y_main, sub_y_top + sub_h], linewidth=1.0, color="black")
ax.plot([x3 + w/2, x3 + w/2], [sub_y_top - 0.2, sub_y_bottom + sub_h], linewidth=1.0, color="black")

x4 = x3 + w + gap_x
box(x4, y_main, w, h, "Estimator\n(LogReg /\n RF /\n SVC)")
arrow_right(x3, y_main, w, h, x4, y_main, w, h)

x5 = x4 + w + gap_x
box(x5, y_main, w, h, "Evaluation\n(Accuracy /\n Recall /\n F1 /\n AUC)")
arrow_right(x4, y_main, w, h, x5, y_main, w, h)

x6 = x5 + w + gap_x
box(x6, y_main, w, h, "Persist\n Model\n(joblib.dump)")
arrow_right(x5, y_main, w, h, x6, y_main, w, h)

# Inference branch
y_inf = 1.5
box(x6, y_inf, w, h, "Load\n Pipeline\n(joblib.\nload)")
x_inf2 = x6 + w + gap_x
box(x_inf2, y_inf, w, h, "Transform &\n Predict\n(pipeline.\npredict)")
arrow_right(x6, y_inf, w, h, x_inf2, y_inf, w, h)

# Down arrow from Persist to Load
ax.add_patch(FancyArrow(x6 + w/2, y_main, 0, y_inf + h - y_main,
                        width=0.02, length_includes_head=True,
                        head_width=0.12, head_length=0.25, color="black"))

# Title
ax.text(0.5, 5.35, "End-to-End scikit-learn Pipeline (Conceptual Structure)",
        ha="left", va="center", fontsize=12, color="black")

plt.savefig("end_to_end_pipeline_structure.pdf", format="pdf", bbox_inches="tight", dpi=300)
plt.show()
