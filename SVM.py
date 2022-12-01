import pandas as pd
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split

#from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import seaborn as sns
from sklearn.inspection import DecisionBoundaryDisplay
#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from matplotlib.colors import ListedColormap
from sklearn.svm import SVC


dataset = pd.read_csv("datosPesoCi3.csv",)

#Caracteristicas
X = dataset[['peso','cirf']].values
#Clases
y = dataset['clasificacion'].values
print(dataset.shape)
print(dataset.head(20))
print(dataset.describe())

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.20, random_state=73)
svm = SVC(gamma='scale')
svm.fit(X_train,y_train)
predictions = svm.predict(X_test)

#Clasificamos un nuevo dato
DatoNuevo=[[171	,22.5]]
prediccion = svm.predict(DatoNuevo)
#predictions = lda.predict(X_test)
print("El nuevo dato pertenece a", prediccion)
print("El numero de puntos mal etiquetados de un total de %d puntos es: %d" % (X_test.shape[0], (y_test != prediccion).sum()))

### Create color maps
cmap_light = ListedColormap(["#FFAAAA", "#AAFFAA"])
cmap_bold = ["#FF0000",  "#00FF00"]


_, ax = plt.subplots()
DecisionBoundaryDisplay.from_estimator(
    svm,
    X,
    cmap=cmap_light,
    ax=ax,
    response_method="predict",
    plot_method="pcolormesh",
    xlabel="Peso",
    ylabel="Circunferencia",
    shading="auto",
)

# Plot also the training points
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
    
    
plt.title("Maquina de vector soporte" )

plt.show()

