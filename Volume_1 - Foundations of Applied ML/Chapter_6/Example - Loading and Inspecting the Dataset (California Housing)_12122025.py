import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 
from sklern.metrics import r2_score, mean_squared_error 

plt.rcParams["font.family"] = "Times New Roman"
SMALL_SIZE = 20
MEDIUM_SIZE = 22
BIGGER_SIZE = 24

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
housing = fetch_california_housing(as_frame=True)
df = housing.frame

print(df.head())
	
print("Number of samples:", df.shape[0])
print("Number of features:", len(housing.feature_names))
print("Target variable:", housing.target.name)

plt.figure(figsize=(12,8))
plt.scatter(
    df["MedInc"], 
    df["MedHouseVal"], 
    alpha=0.3, 
    color="black",           # black points
    edgecolor="none"
)
plt.xlabel("Median Income (10k USD)")
plt.ylabel("Median House Value (100k USD)")
plt.title("Relationship Between Income and House Value")
plt.grid(True, linestyle="--", linewidth=0.6, color="gray")
plt.tight_layout()
plt.savefig("california_income_vs_value.png",
            dpi = 300, 
            bbox_inches = "tight")
plt.show()

plt.figure(figsize=(12,8))
plt.scatter(
    df["Longitude"], 
    df["Latitude"], 
    c=df["MedHouseVal"], 
    cmap="gray",             # grayscale colormap
    s=20, 
    alpha=0.7
)
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("California Housing Prices by Location")
cbar = plt.colorbar(label="Median House Value (100k USD)")
cbar.ax.tick_params(colors="black")
plt.grid(True, linestyle="--", linewidth=0.6, color="gray")
plt.tight_layout()
plt.savefig("california_geo_map.png",
            dpi = 300, 
            bbox_inches = "tight")
plt.show()

X = df[housing.feature_names]      # all 8 numerical features
y = df["MedHouseVal"]              # target: median house value

X_train, X_test, y_train, y_test = train_test_split(
	X,
	y,
	test_size=0.2,
	random_state=42
	)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
	
print("R^2 Score:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))




