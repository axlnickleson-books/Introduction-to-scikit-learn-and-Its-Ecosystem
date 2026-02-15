# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 00:29:37 2025

@author: Admin
"""

import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

# Load digits dataset
digits = load_digits()

# Select first 10 images for demonstration
images = digits.images[:10]
labels = digits.target[:10]

# Create figure
plt.figure(figsize=(10, 3))

for i, (img, lbl) in enumerate(zip(images, labels)):
    plt.subplot(2, 5, i + 1)
    plt.imshow(img, cmap="gray", interpolation="nearest")
    plt.title(f"{lbl}", fontsize=10)
    plt.axis("off")

plt.tight_layout()

# Save figure
plt.savefig("Figure_6.2_digits_examples.png", dpi=300, bbox_inches="tight")
# plt.close()
plt.show()

print("Saved figure as digits_examples.png")
