import os              # Built-in module
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from contourpy.util import data

#Path
BASE_DIR = r"C:\Users\Hp\OneDrive\Desktop\2nd Year\Sem-4\ML\Project"
DATA_PATH = os.path.join(BASE_DIR, "Data")
PLOT_PATH = os.path.join(BASE_DIR, "Output", "plot")
REPORT_PATH = os.path.join(BASE_DIR, "Output", "report")
os.makedirs(PLOT_PATH, exist_ok=True)
os.makedirs(REPORT_PATH, exist_ok=True)

#loading data
df = pd.read_csv(os.path.join(DATA_PATH, "placement_predict_50k Dataset (2).csv"))
print(df.shape)

#Count Plot
plt.figure(figsize = (10,10))
sns.countplot(data = df, x="PlacementStatus")
plt.title("Placement Status Distribution")
plt.xlabel("Placement Status")
plt.ylabel("Number of Students")
plt.savefig(os.path.join(PLOT_PATH, "PlacementStatusCount.png"))
plt.show()

#Histogram
plt.figure(figsize = (10,10))
plt.hist(df["CGPA"], bins = 10, edgecolor = "black")
plt.title("CGPA Distribution")
plt.xlabel("CGPA")
plt.ylabel("Frequency")
plt.savefig(os.path.join(PLOT_PATH, "Histogram.png"))
plt.show()

#Pie chart
df["Gender"].value_counts().plot(
    kind="pie",
    autopct="%1.1f%%",
    startangle=90,
)
plt.title("Gender Status Distribution")
plt.ylabel(" ")
plt.savefig(os.path.join(PLOT_PATH, "PieChart.png"))
plt.show()

#Scatter Plot
plt.figure(figsize = (10,10))
sns.scatterplot(x = "CGPA", y = "AttendancePercent", data = df, color = "blue")
plt.title("CGPA vs Attendance Percentage")
plt.savefig(os.path.join(PLOT_PATH, "ScatterPlot.png"))
plt.show()

#Box Plot
plt.figure(figsize = (10,10))
sns.boxplot(x = "PlacementStatus", y = "CGPA", data = df)
plt.title("CGPA vs Placement Status")
plt.savefig(os.path.join(PLOT_PATH, "BoxPlot.png"))
plt.show()

#Count Plot
plt.figure(figsize = (10,10))
sns.countplot(x = "Gender", hue = "PlacementStatus", data = df)
plt.title("Gender vs Placement Status")
plt.savefig(os.path.join(PLOT_PATH, "GendervsPlacementCount.png"))
plt.show()

#Corelation Heatmap
corr = df.select_dtypes(include=['number']).corr()
sns.heatmap(corr, annot=True, cmap="YlGnBu", )
plt.title("Correlation Matrix")
plt.savefig(os.path.join(PLOT_PATH, "CorrelationMatrix.png"))
plt.show()

# Pair Plot
# plt.figure(figsize=(10,10))
# sns.pairplot(df.sample(1).select_dtypes(include=['number']))
# plt.suptitle("Pairwise Relationships Among Numerical Variables", y=1.02)
# plt.savefig(os.path.join(PLOT_PATH, "PairPlot.png"))
# plt.show()

# Dot Plot (Strip Plot)
plt.figure(figsize=(10,10))
sns.stripplot(x="PlacementStatus", y="CGPA", data=df, jitter=True, size=5, color="purple")
plt.title("Dot Plot of CGPA by Placement Status")
plt.xlabel("Placement Status")
plt.ylabel("CGPA")
plt.savefig(os.path.join(PLOT_PATH, "DotPlot.png"))
plt.show()
