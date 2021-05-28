
import numpy as np
import pandas as pd
import pyvista as pv
import plotly.express as px
from minisom import MiniSom
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

#Brings a list to desired value range
def normalizeList(L, minValue=0.0, maxValue=1.0):
    L = L - min(L)
    L = L / (max(L) - min(L))
    return L * (maxValue - minValue) + minValue

#Specify properties to be used
mapSize = 80
animationFrames = 600
iterationsPerFrame = mapSize**2

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
dataMarker = pv.Sphere(radius=0.002)
boundaryBox = pv.Box(bounds=(0, 1, 0, 1, 0, 1))
meshXYZ = pv.PolyData(XYZ)
meshXYZ['Release Year'] = releaseYear
scatterCloud = meshXYZ.glyph(geom=dataMarker, scale=False)

#Identify the plane that is spanned by the first two eigenvectors
pca = PCA(n_components=2).fit(XYZ)
eigVectors = pca.components_
planeCentre = np.mean(XYZ, axis=0)
planeNormal = np.cross(eigVectors[0], eigVectors[1])
planeNormal = planeNormal / np.linalg.norm(planeNormal)
plane = pv.Plane(center=planeCentre, direction=planeNormal, i_resolution=mapSize-1, j_resolution=mapSize-1)
planeVerts = plane.points

#Initiate a self organizing map with plane obtained by PCA as initialweights
som = MiniSom(x=mapSize, y=mapSize, input_len=3, sigma=3.0, learning_rate=0.04)
weights = som.get_weights().copy()
for i in range(mapSize):
    for j in range(mapSize):
        weights[i, j] = planeVerts[i*mapSize + j]
som._weights = weights
dist = som.distance_map().flatten()
plane['scalars'] = dist

#Set up plotter for 3D animation
plotter = pv.Plotter()
plotter.add_mesh(scatterCloud, scalars='Release Year')
planeAct = plotter.add_mesh(plane, scalars='scalars', opacity=0.77, show_edges=True, show_scalar_bar=False)
plotter.add_mesh(boundaryBox.extract_all_edges(), color='black')
plotter.add_mesh(pv.Line([ 0, 0, 0], [1, 0, 0]), color='red')
plotter.add_mesh(pv.Line([ 0, 0, 0], [0, 1, 0]), color='green')
plotter.add_mesh(pv.Line([ 0, 0, 0], [0, 0, 1]), color='blue')
plotter.add_axes(interactive=True)
plotter.show(auto_close=False, window_size=[800, 608])
if True:
    plotter.open_movie('SOM.mp4')
    plotter.write_frame()


#Train SOM for a bit and write frame to movie
for f in range(animationFrames):
    som.train_random(XYZ, iterationsPerFrame)
    weights = som.get_weights().copy()
    
    #Update coordinates of plane that is used to represent the SOM
    plane = pv.Plane(i_resolution=mapSize-1, j_resolution=mapSize-1)
    planeVerts = plane.points
    for i in range(mapSize):
        for j in range(mapSize):
            planeVerts[i*mapSize + j] = weights[i, j]
    
    #Update plane to mesh and write frame to animation
    plotter.remove_actor(planeAct)
    planeAct = plotter.add_mesh(plane, opacity=0.77, show_edges=True, show_scalar_bar=False)
    plotter.write_frame()
    print(f)






