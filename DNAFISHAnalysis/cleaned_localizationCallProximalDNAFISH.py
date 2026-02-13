import numpy as np
from matplotlib import pyplot as plt
from openpyxl import Workbook
import openpyxl
from importlib import reload
import localizationPSFCore
reload(localizationPSFCore)
from localizationCall.localizationPSFCore import preProcess
from localizationCall.localizationPSFCore import findProximalPairs
from localizationCall.localizationPSFCore import findProximalPairsFinal
from localizationCall.localizationPSFCore import localizationPSFDistanceAutomatic1
from skimage.feature import peak_local_max
import math
import os





analysisFolder = '../Images/Meta Results'
lab = "mike"
saveOption = False





folderPath = "../Images/"

uncorrectedDistanceDataTotal = []
coordinatesPrimaryTotal = [[], [], []]
coordinatesSecondaryTotal = [[], [], []]
initial_guess = [5, 7000, 1.5]
workPath = analysisFolder + '/Compiled Distribution.xlsx'
workbook = Workbook()
sheet = workbook.active
sheet.append(['Image', 'parameters3D_primary (x, y, z)', 'parameters3D_secondary (x, y, z)', 'Uncorrected Distance (micron)',
              'CorrectedDistance (micron)'])

image_index = 0
for image in os.listdir(folderPath):

    c1 = 0.3  
    c2 = 0.35  

    if str(image)[-3:] == 'png':
        continue
    name = folderPath + "/" + str(image)[:-4] + '_cp_masks.png'
    imagePath = folderPath + "/" + str(image)
    print(str(image))
    image_channels, voxel_size, image_shape = preProcess(imagePath)
    paired_peaks, nucleiMatrix = findProximalPairsFinal(image_channels, 8, c1, c2, name)

    primaryList = [x[0] for x in paired_peaks]
    secondaryList = [x[1] for x in paired_peaks]


    uncorrected_distances, primary_coordinates, secondary_coordinates = localizationPSFDistanceAutomatic1(image_channels, voxel_size,
                                                        image_shape, primaryList, secondaryList, analysisFolder, saveOption, sheet,
                                                                                                          image_index, initial_guess, lab)
    print(uncorrected_distances)
    print(primary_coordinates)
    print(secondary_coordinates)

    pair_index = 0
    for value in primary_coordinates[0]:
        coordinatesPrimaryTotal[0].append(round(primary_coordinates[0][pair_index], 3))
        coordinatesPrimaryTotal[1].append(round(primary_coordinates[1][pair_index], 3))
        coordinatesPrimaryTotal[2].append(round(primary_coordinates[2][pair_index], 3))
        coordinatesSecondaryTotal[0].append(round(secondary_coordinates[0][pair_index], 3))
        coordinatesSecondaryTotal[1].append(round(secondary_coordinates[1][pair_index], 3))
        coordinatesSecondaryTotal[2].append(round(secondary_coordinates[2][pair_index], 3))
        uncorrectedDistanceDataTotal.append(uncorrected_distances[pair_index])

        pair_index += 1

    print(str(image) + " number of entries: ")
    print(pair_index)

    image_index += 1


uncorrectedDistanceBootstrap = []
bootstrap_index = 0
bootstrap_iterations = len(uncorrectedDistanceDataTotal)
while bootstrap_index < bootstrap_iterations:
    temporary = np.random.choice(uncorrectedDistanceDataTotal, size=int(len(uncorrectedDistanceDataTotal) / 2), replace=False)
    uncorrectedDistanceBootstrap.append(np.mean(temporary))
    bootstrap_index += 1

plt.figure(0)
plt.hist(uncorrectedDistanceDataTotal, bins=np.linspace(0, 1.2, 26),
                                  edgecolor='black')
plt.title("Uncorrected Distance Bootstrap")
plt.xlabel("Intensity")
plt.ylabel("Frequency")
plt.show()

print("uncorrected median bootstrap")
print(np.median(uncorrectedDistanceBootstrap))
print(np.std(uncorrectedDistanceBootstrap))

correctionMatrix = [[], [], []]
axis_index = 0
delta = [[], [], []]
for axes in coordinatesPrimaryTotal:
    coordinateHolder = 0
    for coordinate in axes:
        delta[axis_index].append(coordinatesSecondaryTotal[axis_index][coordinateHolder] -
                                 coordinatesPrimaryTotal[axis_index][coordinateHolder])
        coordinateHolder += 1

    fit, covariance = np.polyfit(coordinatesSecondaryTotal[axis_index], delta[axis_index], 1, cov=True)
    fitFunction = np.poly1d(fit)
    correctionMatrix[axis_index].append(fit[0])
    correctionMatrix[axis_index].append(fit[1])

    plt.scatter(coordinatesSecondaryTotal[axis_index], delta[axis_index])
    plt.plot(coordinatesSecondaryTotal[axis_index], fitFunction(coordinatesSecondaryTotal[axis_index]), color='red')
    plt.title("Axes-" + str(axis_index) + "-Distribution")
    plt.xlabel("Pixels")
    plt.ylabel("Uncorrected Axial Distance (pixels)")
    plt.show()

    axis_index += 1

print(correctionMatrix)
final_axis_index = 0
deltaCorrected = [[], [], []]
for axes in coordinatesPrimaryTotal:
    finalCoordinateHolder = 0
    for coordinate in axes:
        temp = coordinatesSecondaryTotal[final_axis_index][finalCoordinateHolder]
        deltaCorrected[final_axis_index].append(delta[final_axis_index][finalCoordinateHolder] -
                                                   correctionMatrix[final_axis_index][0]*temp - correctionMatrix[final_axis_index][1])
        finalCoordinateHolder += 1

    fit, covariance = np.polyfit(coordinatesSecondaryTotal[final_axis_index], deltaCorrected[final_axis_index], 1, cov=True)
    fitFunction = np.poly1d(fit)

    plt.scatter(coordinatesSecondaryTotal[final_axis_index], deltaCorrected[final_axis_index])
    plt.plot(coordinatesSecondaryTotal[final_axis_index], fitFunction(coordinatesSecondaryTotal[final_axis_index]), color='red')
    plt.title("Axes-" + str(final_axis_index) + "-Distribution")
    plt.xlabel("Pixels")
    plt.ylabel("Corrected Axial Distance (pixels)")
    plt.show()

    final_axis_index += 1

correctedDistances = []
distance_index = 0
for value in coordinatesPrimaryTotal[0]:
    correctedDistances.append((((deltaCorrected[0][distance_index])*voxel_size[0])**2 +
                                ((deltaCorrected[1][distance_index])*voxel_size[1])**2 +
                                ((deltaCorrected[2][distance_index])*voxel_size[2])**2) ** 0.5)
    distance_index += 1

correctedDistancestemp = correctedDistances

for value in correctedDistancestemp:
   if value > 1.5:
       correctedDistancestemp.remove(value)

bootstrap_index = 0
meanCorrectedDistances = []
bootstrap_iterations = len(correctedDistancestemp)
while bootstrap_index < bootstrap_iterations:
    bootstrap = np.random.choice(correctedDistancestemp, size=int(len(correctedDistancestemp)/2), replace=False)
    meanCorrectedDistances.append(np.mean(bootstrap))
    bootstrap_index += 1


plt.hist(correctedDistances, bins=np.linspace(0, 2.0, 36),
                                  edgecolor='black')
plt.title("Total Corrected Distance Histogram")
plt.xlabel("Distance (um)")
plt.ylabel("Frequency")
plt.show()

fig, ax = plt.subplots()
n, bins, patches = ax.hist(correctedDistancestemp, 1000, density=False, histtype='step',
                           cumulative=True, label='Empirical')
ax.set_title('Cumulative Distribution')
ax.set_xlabel('Distance (um)')
ax.set_ylabel('N')

plt.xlim([0, 0.84])
plt.ylim([0, len(correctedDistancestemp)])
plt.show()

print("Corrected Median Distance: " + str(np.median(meanCorrectedDistances)))
print("Corrected Standard Error: " + str(np.std(meanCorrectedDistances)))
print("Dataset Size: " + str(len(correctedDistances)))
