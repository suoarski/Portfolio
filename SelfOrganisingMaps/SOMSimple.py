
import numpy as np
import pandas as pd
import pyvista as pv
import plotly.express as px
from minisom import MiniSom
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def normalizeList(L, minValue=0.0, maxValue=1.0):
    L = L - min(L)
    L = L / (max(L) - min(L))
    return L * (maxValue - minValue) + minValue

#Read data from file
dataFileName='Mobile Device Data for Assignment 2.xlsx'
dataFrame = pd.read_excel(dataFileName)
dataArray = dataFrame.to_numpy().T
attributeArray = dataArray[4:].astype(float)
attributeNames = dataFrame.columns.values[4:]
releaseYear = dataArray[2].astype(float)
modelNames = dataFrame['Model'].astype(str)

#Extract hardware based attributes and set zero values to next lowest values
ram = np.copy(attributeArray[0])
storage = np.copy(attributeArray[1])
cpu = np.copy(attributeArray[2])
ram[ram==0] = np.min(ram[ram!=0])
storage[storage==0] = np.min(storage[storage!=0])
cpu[cpu==0] = np.min(cpu[cpu!=0])

#Create XYZ coordinates from hardware based attributes
ram = normalizeList(np.log(ram))
storage = normalizeList(np.log(storage))
cpu = normalizeList(np.log(cpu))
XYZ = np.stack((ram, storage, cpu), axis=1)

#Create the scatter cloud
dataMarker = pv.Sphere(radius=0.01)
boundaryBox = pv.Box(bounds=(0, 1, 0, 1, 0, 1))
meshXYZ = pv.PolyData(XYZ)
meshXYZ['Release Year'] = releaseYear
scatterCloud = meshXYZ.glyph(geom=dataMarker, scale=False)

#Identify the plane that is spanned by the first two eigenvectors
mapSize = 80
pca = PCA(n_components=2).fit(XYZ)
eigVectors = pca.components_
planeCentre = np.mean(XYZ, axis=0)
planeNormal = np.cross(eigVectors[0], eigVectors[1])
planeNormal = planeNormal / np.linalg.norm(planeNormal)
plane = pv.Plane(center=planeCentre, direction=planeNormal, i_resolution=mapSize-1, j_resolution=mapSize-1)
planeVerts = plane.points

#Initiate a self organizing map with plane obtained by PCA as initialweights
som = MiniSom(x=mapSize, y=mapSize, input_len=3, sigma=2.0, learning_rate=0.4)
weights = som.get_weights().copy()
for i in range(mapSize):
    for j in range(mapSize):
        weights[i, j] = planeVerts[i*mapSize + j]
som._weights = weights

#Train the self organizing map and update coordinates of plane mesh used to represent weights
som.train_random(XYZ, 40000)
weights = som.get_weights().copy()
plane = pv.Plane(i_resolution=mapSize-1, j_resolution=mapSize-1)
planeVerts = plane.points
weigh = []
for i in range(mapSize):
    for j in range(mapSize):
        planeVerts[i*mapSize + j] = weights[i, j]
        weigh.append(weights[i, j])

#Create plot of distance map
plt.pcolor(som.distance_map().T, cmap='bone_r')
plt.title('SOM Distance Map')
plt.colorbar()
#plt.show()

#Set up 3D plotter and add objects to scene
plotter = pv.Plotter()
plane['scalars'] = som.distance_map().flatten()
plotter.add_mesh(scatterCloud, scalars='Release Year')
plotter.add_mesh(plane, scalars='scalars', opacity=0.77, show_edges=True, show_scalar_bar=False)
plotter.add_mesh(boundaryBox.extract_all_edges(), color='black')
plotter.add_mesh(pv.Line([ 0, 0, 0], [1, 0, 0]), color='red')
plotter.add_mesh(pv.Line([ 0, 0, 0], [0, 1, 0]), color='green')
plotter.add_mesh(pv.Line([ 0, 0, 0], [0, 0, 1]), color='blue')
plotter.add_axes(interactive=True)
plotter.show()
