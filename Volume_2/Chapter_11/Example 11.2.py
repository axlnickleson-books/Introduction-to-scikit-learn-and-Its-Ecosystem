
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE
from collections import Counter
	
X, y = make_classification(n_classes=2, weights=[0.9, 0.1], n_samples=1000, random_state=42)
print("Before:", Counter(y))
	
X_res, y_res = SMOTE(random_state=42).fit_resample(X, y)
print("After :", Counter(y_res))
# This visualization is optional USed for creation of Figure 11.2 
import matplotlib.pyplot as plt 
plt.rcParams["font.family"] = "Times New Roman"

SMALL_SIZE = 24
MEDIUM_SIZE = 28
BIGGER_SIZE = 30

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
fig, axes = plt.subplots(1, 2, figsize=(30, 16))


axes[0].scatter(X[y == 0][:, 0], X[y == 0][:, 1],
label="Majority Class", alpha=0.6)
axes[0].scatter(X[y == 1][:, 0], X[y == 1][:, 1],
label="Minority Class", alpha=0.8, color='red')
axes[0].set_title("Before SMOTE")
axes[0].legend()
axes[0].set_xlabel("Feature 1")
axes[0].set_ylabel("Feature 2")
axes[0].grid(True)


axes[1].scatter(X_res[y_res == 0][:, 0], X_res[y_res == 0][:, 1],
label="Majority Class", alpha=0.6)
axes[1].scatter(X_res[y_res == 1][:, 0], X_res[y_res == 1][:, 1],
label="Minority Class (Synthetic)", alpha=0.8, color='red')
axes[1].set_title("After SMOTE")
axes[1].legend()
axes[1].set_xlabel("Feature 1")
axes[1].set_ylabel("Feature 2")
axes[1].grid(True)
plt.suptitle("Effect of SMOTE Oversampling on Class Distribution", fontsize=32)
plt.tight_layout()
# plt.show()
plt.savefig("smote_distribution_visualization.pdf",
            dpi = 300 )