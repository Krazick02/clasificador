import pandas as pd
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import DecisionBoundaryDisplay

from matplotlib.colors import ListedColormap


dataset = pd.read_csv("datosPesoCi3.csv",)

#Caracteristicas
X = dataset[['peso','cirf']].values
#Clases
y = dataset['clasificacion'].values
print(dataset.shape)
print(dataset.head(20))
print(dataset.describe())

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.70, random_state=73)

gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)

print("El numero de puntos mal etiquetados de un totalde %d points  es: %d"
      % (X_test.shape[0], (y_test != y_pred).sum()))

#Clasificamos un nuevo dato
DatoNuevo=[[171	,22.5]]
prediccion = gnb.predict(DatoNuevo)
print("El nuevo dato pertenece a", prediccion)

### Create color maps
cmap_light = ListedColormap(["#FFAAAA", "#AAFFAA"])
cmap_bold = ["#FF0000",  "#00FF00"]
_, ax = plt.subplots()
DecisionBoundaryDisplay.from_estimator(
    gnb,
    X,
    cmap=cmap_light,
    ax=ax,
    response_method="predict",
    plot_method="pcolormesh",
    xlabel="Peso",
    ylabel="Circunferencia",
    shading="auto",
)

# Dibujar el dato nuevo
sns.scatterplot(
    x=X[:, 0],
    y=X[:, 1],
    hue= dataset['clasificacion'] ,
    palette=cmap_bold,
    alpha=1.0,
    edgecolor="black",)
sns.scatterplot(
    x=DatoNuevo[0][0],
    y=DatoNuevo[0][1],
    hue= dataset['clasificacion'] ,
    palette=cmap_bold,
    alpha=1.0,
    edgecolor="yellow",)
    
    
plt.title("Clasificador de Bayes" )

plt.show()
