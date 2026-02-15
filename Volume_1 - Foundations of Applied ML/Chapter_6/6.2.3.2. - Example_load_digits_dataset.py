# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 01:54:31 2025

@author: Admin
"""
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

# Load the dataset
digits = load_digits()

# Features and labels
X = digits.data
y = digits.target

print("Shape of X:", X.shape)
print("Unique classes:", digits.target_names)

# -------------------------------
# Plot multiple digit examples
# -------------------------------

plt.figure(figsize=(10, 3))

# Display first 10 digit images
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(digits.images[i], cmap="gray", interpolation="nearest")
    plt.title(f"{digits.target[i]}", fontsize=10)
    plt.axis("off")

plt.tight_layout()

# Save the figure for LaTeX
# plt.savefig("digits_examples.png", dpi=300, bbox_inches="tight")
# plt.close()
plt.show()


print("Saved: digits_examples.png")
