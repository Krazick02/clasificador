import pandas as pd
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as bayesmap
import matplotlib.pyplot as ldamap
import matplotlib.pyplot as svmsmap
from sklearn.metrics import classification_report
import seaborn as sns
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from matplotlib.colors import ListedColormap
from sklearn.neural_network import MLPClassifier 

dataset = pd.read_csv("datosPesoCi3.csv",)

#Caracteristicas
X = dataset[['peso','cirf']].values

#Clases
y = dataset['clasificacion'].values
print(dataset.shape)
print(dataset.head(20))
print(dataset.describe())

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.20, random_state=73)
clf = MLPClassifier(hidden_layer_sizes=(11,20,1),solver='lbfgs',alpha=1e-5,random_state=1,max_iter=100)
clf.fit(X_train,y_train)

#Clasificamos un nuevo dato
DatoNuevo=[[100	,20.5]]
prediccion = clf.predict(DatoNuevo)
print("\n\nMaquina de vector soporte")
print("El nuevo dato pertenece a", prediccion)
print("El numero de puntos mal etiquetados de un total de %d puntos es: %d" % (X_test.shape[0], (y_test != prediccion).sum()))



X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.70, random_state=73)

gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)

prediccion = gnb.predict(DatoNuevo)
print("\n\nClasificador bayes")
print("El nuevo dato pertenece a", prediccion)
print("El numero de puntos mal etiquetados de un total de %d puntos  es: %d" % (X_test.shape[0], (y_test != y_pred).sum()))



X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.70, random_state=73)
lda=LinearDiscriminantAnalysis()
y_pred=lda.fit(X_train,y_train).predict(X_test)

#Clasificamos un nuevo dato
prediccion = lda.predict(DatoNuevo)
print("\n\nAnalisis de Discriminante Lineal")
print("El nuevo dato pertenece a", prediccion)
print("El numero de puntos mal etiquetados de un total de %d puntos es: %d" % (X_test.shape[0], (y_test != y_pred).sum()))

### Create color maps
cmap_light = ListedColormap(["#FFAAAA", "#AAFFAA"])
cmap_bold = ["#FF0000",  "#00FF00"]


_, ax = svmsmap.subplots()
DecisionBoundaryDisplay.from_estimator(
    clf,
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
    
    
svmsmap.title("\n\nMaquina de vector soporte" )



_, ax = ldamap.subplots()
DecisionBoundaryDisplay.from_estimator(
    lda,
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
     
ldamap.title("Analisis de Discriminante Lineal" )

cmap_light = ListedColormap(["#FFAAAA", "#AAFFAA"])
cmap_bold = ["#FF0000",  "#00FF00"]
_, ax = bayesmap.subplots()
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
    
    
bayesmap.title("Clasificador de Bayes" )


svmsmap.show()
ldamap.show()
svmsmap.show()