from sklearn.datasets import fetch_openml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 0) Load
# df = pd.read_csv("Titanic-Dataset.csv")
titanic = fetch_openml(name="titanic", version=1, as_frame=True)

# Extract the DataFrame and target
df = titanic.frame
print(df.head())
print(df.info())
# print(df.describe())
# # df.describe().to_csv("Titanic_describe_statistics.csv", index=False)

# # 1) Target (keep the original df intact for EDA)
# y = df["Survived"].copy()
# print(y.value_counts(dropna=False))
# print((y.value_counts(normalize=True) * 100).round(1).astype(str) + "%")
# print("y dtype:", y.dtype)
# # 2) EDA on a copy
# df_eda = df.copy()

# print("Missing values per column (EDA):")
# print(df_eda.isnull().sum())

# # ---- EDA FIGURES ----
# plt.figure(figsize=(8,6))
# df_eda["Age"].hist(bins=20, edgecolor="black", zorder=3)
# plt.xlabel("Age"); plt.ylabel("Frequency"); plt.title("Distribution of Passenger Age")
# plt.grid(True, zorder=0)
# plt.savefig("Chapter_5_Figure_5.1.png", dpi=300, bbox_inches="tight")
# plt.show()

# plt.figure(figsize=(8,6))
# sns.boxplot(x="Survived", y="Age", data=df_eda, zorder=3)
# plt.title("Passenger Age vs Survival")
# plt.grid(True, zorder=0)
# plt.xticks([0,1], ['0\n(did not survive)', "1\n(survived)"])
# plt.savefig("Chapter_5_Figure_5.2.png", dpi=300, bbox_inches="tight")
# plt.show()

# plt.figure(figsize=(8,6))
# # get_dummies later will create Sex_male; here use original column safely:
# sns.barplot(x="Sex", y="Survived", data=df_eda, estimator=np.mean, errorbar=None, zorder=3)
# plt.title("Survival Rate by Sex")
# plt.grid(True, zorder=0)
# plt.savefig("Chapter_5_Figure_5.3.png", dpi=300, bbox_inches="tight")
# plt.show()

# plt.figure(figsize=(8,6))
# sns.barplot(x="Pclass", y="Survived", data=df_eda, order=[1,2,3], estimator=np.mean, errorbar=None, zorder=3)
# plt.title("Survival Rate by Passenger Class")
# plt.xticks([0,1,2], ["1\n(Upper class)", "2\n(Middle class)", "3\n(Lower class)"])
# plt.grid(True, zorder=0)
# plt.savefig("Chapter_5_Figure_5.4.png", dpi=300, bbox_inches="tight")
# plt.show()

# plt.figure(figsize=(8,6))
# sns.heatmap(df_eda.corr(numeric_only=True), cmap="coolwarm", center=0, annot=True, fmt=".2f")
# plt.title("Correlation Heatmap")
# plt.savefig("Chapter_5_Figure_5.5.png", dpi=300, bbox_inches="tight")
# plt.show()

# # 3) Preprocessing for modeling (new working copy)
# df_model = df.copy()

# # Drop high-missing / low-signal columns first (keep Cabin temp if you want an indicator)
# # Create a 'CabinKnown' indicator from the original, then drop the text Cabin
# df_model["CabinKnown"] = df_model["Cabin"].notna().astype(int)
# df_model.drop(columns=["Cabin", "Ticket"], inplace=True, errors="ignore")

# # 4) Basic imputations
# df_model["Age"] = df_model["Age"].fillna(df_model["Age"].mean())
# df_model["Embarked"] = df_model["Embarked"].fillna(df_model["Embarked"].mode()[0])
# df_model["Fare"] = df_model["Fare"].fillna(df_model["Fare"].median())

# # 5) Feature engineering
# df_model["FamilySize"] = df_model["SibSp"] + df_model["Parch"] + 1
# df_model["FarePerPerson"] = df_model["Fare"] / df_model["FamilySize"]

# # Extract Title from Name, group rares, then encode
# df_model["Title"] = df_model["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
# title_map = {
#     "Mlle":"Miss","Ms":"Miss","Mme":"Mrs","Lady":"Royalty","Countess":"Royalty","Sir":"Royalty",
#     "Don":"Royalty","Dona":"Royalty","Jonkheer":"Royalty","Capt":"Officer","Col":"Officer",
#     "Major":"Officer","Dr":"Officer","Rev":"Officer"
# }
# df_model["Title"] = df_model["Title"].replace(title_map)
# rare_titles = df_model["Title"].value_counts()[df_model["Title"].value_counts()<10].index
# df_model["Title"] = df_model["Title"].replace(dict.fromkeys(rare_titles, "Rare"))

# # 6) One-hot encode categoricals; drop_first to avoid multicollinearity
# categoricals = ["Sex", "Embarked", "Title"]
# df_model = pd.get_dummies(df_model, columns=categoricals, drop_first=True, dtype=int)

# # 7) Build X at the end (exclude target & identifiers you don’t want)
# cols_to_drop = ["Survived", "PassengerId", "Name"]  # drop Name now that Title is encoded
# X = df_model.drop(columns=cols_to_drop, errors="ignore")
# y = df_model["Survived"].copy()

# print("Feature matrix shape:", X.shape)
# print("Target vector shape:", y.shape)
# print("Remaining missing values (should be 0):")
# print(X.isnull().sum().sum())

# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# 	
# # Split DataFrame into train/test
# X_train, X_test, y_train, y_test = train_test_split(
# X, y, test_size=0.2, random_state=42
# )
# 	
# # Train classifier
# clf = RandomForestClassifier()
# clf.fit(X_train, y_train)

# print("Accuracy:", clf.score(X_test, y_test))
