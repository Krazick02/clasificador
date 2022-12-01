from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
from sklearn .inspection import DecisionBoundaryDisplay
import pandas as pd

n_neighbors = 3

#Leer el archivo con los datos
dataset = pd.read_csv("datosPesoCi2.csv ")

#Caracteristicas
x = dataset[['ingresos', 'gastos']].values

#Clases
y = dataset['decision'].values

x_train, y_test_split(x,y, str)

# el mapa de color
cmap_light = ListedColormap(["orange", "cyan"])
cmap_bold = ["red", "blue"]


# uniforme todos los vecinos tienen el mismo peso
#distance los vecinos que estan mas cerca tienen mayor peso

#
for weights in["uniform", "distance"]:
    knn = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
    knn.fit(x, y)
    #Clasifique un nuevo dato
    DatoNuevo=[[0.049170482, 0.689384745]]
    clasifica = knn.predict(DatoNuevo)
    print("El nuevo dato es de la clasificación: ", clasifica)
    _ax = plt.subplots()
    DecisionBoundaryDisplay.from_estimator(
        knn,
        x,
        cmap=cmap_light,
        response_method="predict",
        plot_method="pcolormesh",
        xlabel="ingresos",
        ylabel="gastos",
        shading="auto",
        )
    #
    sns.scatterplot(
        x=x[:,0],
        y=x[:,0],
        hue=dataset["decision"],
        palette=cmap_bold,
        alpha=1.0,
        edgecolor="black",)
    sns.scatterplot(
        x=DatoNuevo[0][0],
        y=DatoNuevo[0][1],
        hue=dataset["decision"],
        palette=cmap_bold,
        alpha=1.0,
        edgecolor="red",)
    plt.title("Clasificador perron")

plt.show()