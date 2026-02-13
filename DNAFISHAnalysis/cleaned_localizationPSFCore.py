import sys
import os, glob
from aicsimageio import AICSImage
import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Rectangle
import scipy.optimize as opt
import statistics
from skimage.feature import peak_local_max
import math
from openpyxl import Workbook
import openpyxl
from PIL import Image

def distance2D(point1, point2):

    distanceValue = ((point1[0] - point2[0])**2 +
                     (point1[1] - point2[1])**2) ** 0.5

    return distanceValue

def distance3D(point1, point2):

    distanceValue = ((point1[0] - point2[0])**2 +
                     (point1[1] - point2[1])**2 +
                     (point1[2] - point2[2])**2) ** 0.5

    return distanceValue

def gaussian1D(i, amplitude, i0, sigma):
    i0 = float(i0)
    g = amplitude*np.exp((-(i - i0)**2)/2*sigma**2)
    return g.ravel()

def gaussian2D(ij, amplitude, i0, j0, sigmaI, sigmaJ):
    i, j = ij
    i0 = float(i0)
    j0 = float(j0)
    g = amplitude*np.exp(-((i - i0)**2)/(2*sigmaI**2) - ((j - j0)**2)/(2*sigmaJ**2))
    return g.ravel()

def preProcess(imagePath):
    image = AICSImage(imagePath)
    voxel_size = [image.physical_pixel_sizes.X, image.physical_pixel_sizes.Y, image.physical_pixel_sizes.Z]
    z_stack_dict = {}
    image_shape = image.image_shape
    print(image_shape)

    zHolder = 0
    cHolder = 0
    while zHolder < image.image_shape[2]:
        temp = []
        while cHolder < image.image_shape[1]:
            data = np.asarray(image.get_image_data("YX", Z=zHolder, C=cHolder))
            temp.append(data)
            cHolder += 1

        z_stack_dict['Z-Stack: ' + str(zHolder)] = temp

        zHolder += 1
        cHolder = 0

    CLAHE = cv2.createCLAHE(clipLimit=5)
    image_channels = [[], []]
    for stack in z_stack_dict:
        channel_index = 0
        for channel in z_stack_dict[stack]:
            if channel_index == 2:
                continue

            image_channels[channel_index].append(z_stack_dict[stack][channel_index])
            channel_index += 1

    return image_channels, voxel_size, image_shape

def preProcess3(imagePath):
    image = AICSImage(imagePath)
    voxel_size = [image.physical_pixel_sizes.X, image.physical_pixel_sizes.Y, image.physical_pixel_sizes.Z]
    z_stack_dict = {}
    image_shape = image.image_shape
    print(image_shape)

    zHolder = 0
    cHolder = 0
    while zHolder < image.image_shape[2]:
        temp = []
        while cHolder < image.image_shape[1]:
            data = np.asarray(image.get_image_data("YX", Z=zHolder, C=cHolder))
            temp.append(data)
            cHolder += 1

        z_stack_dict['Z-Stack: ' + str(zHolder)] = temp

        zHolder += 1
        cHolder = 0

    CLAHE = cv2.createCLAHE(clipLimit=5)
    image_channels = [[], [], []]
    for stack in z_stack_dict:
        channel_index = 0
        for channel in z_stack_dict[stack]:

            image_channels[channel_index].append(z_stack_dict[stack][channel_index])
            channel_index += 1

    return image_channels, voxel_size, image_shape

def preProcessR(imagePath):
    image = AICSImage(imagePath)
    voxel_size = [image.physical_pixel_sizes.X, image.physical_pixel_sizes.Y, image.physical_pixel_sizes.Z]
    z_stack_dict = {}
    image_shape = image.image_shape
    print(image_shape)

    zHolder = 0
    cHolder = 0
    while zHolder < image.image_shape[2]:
        temp = []
        while cHolder < image.image_shape[1]:
            data = np.asarray(image.get_image_data("YX", Z=zHolder, C=cHolder))
            temp.append(data)
            cHolder += 1

        z_stack_dict['Z-Stack: ' + str(zHolder)] = temp

        zHolder += 1
        cHolder = 0

    image_channels = [[], [], []]
    for stack in z_stack_dict:
        channel_index = 0
        for channel in z_stack_dict[stack]:

            image_channels[channel_index].append(z_stack_dict[stack][channel_index])
            channel_index += 1

    return image_channels, voxel_size, image_shape

def findProximalPairs(image_channels, window_radius, c1, c2):
    channel_index = 0
    peaks = []
    for channel in image_channels:

        if channel_index == 0:
            threshold = c1 
        if channel_index == 1:
            threshold = c2 

        coordinates = peak_local_max(np.max(image_channels[channel_index], axis=0),  
                                     min_distance=4,
                                     threshold_rel=threshold,
                                     exclude_border=int(window_radius + 1))

        peaks.append(coordinates)
        plt.imshow(np.max(image_channels[channel_index], axis=0))
        plt.colorbar()
        if (channel_index == 0):
            plt.clim(0, 6000)
        if channel_index == 1:
            plt.clim(0, 5000)
        plt.scatter(coordinates[:, 1], coordinates[:, 0], facecolors='none', edgecolors='r')
        plt.xlabel("X-Dimension")
        plt.ylabel("Y-Dimension")
        plt.title("MIP, channel " + str(channel_index))


        plt.show()

        channel_index += 1

    paired_peaks = []
    channel0_peaks = peaks[0].tolist()
    channel1_peaks = peaks[1].tolist()
    for candidate0 in channel0_peaks:
        redundant = []
        for candidate1 in channel1_peaks:
            upperBoundX = candidate0[1] + (window_radius - 1)
            lowerBoundX = candidate0[1] - (window_radius - 1)
            upperBoundY = candidate0[0] + (window_radius - 1)
            lowerBoundY = candidate0[0] - (window_radius - 1)
            if upperBoundX >= candidate1[1] >= lowerBoundX:
                if upperBoundY >= candidate1[0] >= lowerBoundY:
                    redundant.append([candidate1,
                                      distance2D(candidate0, candidate1)])

        if len(redundant) == 1:
            paired_peaks.append([candidate0, redundant[0][0]])
            channel1_peaks.remove(redundant[0][0])
        if len(redundant) > 1:
            distances = [d[1] for d in redundant]
            index = distances.index(min(distances))
            paired_peaks.append([candidate0, redundant[index][0]])

    print(len(paired_peaks))
    plt.imshow(np.max(image_channels[1], axis=0))
    plt.colorbar()
    plt.clim(0, 8000)

    plt.xlabel("X-Dimension")
    plt.ylabel("Y-Dimension")
    plt.title("MIP, channel 1")
    for pair in paired_peaks:
        plt.plot(pair[1][1], pair[1][0], marker='+', color="white")
        plt.plot(pair[0][1], pair[0][0], marker='+', color="red")
    plt.show()

    return paired_peaks

def findSpots(image_channels, window_radius, c1):
    channel_index = 0
    peaks = []
    for channel in image_channels:
        if channel_index == 0: 
            threshold = c1 
        if channel_index != 0:
            continue
        coordinates = peak_local_max(np.max(image_channels[channel_index], axis=0),  
                                     min_distance=4,
                                     threshold_rel=threshold,
                                     exclude_border=int(window_radius + 1))

        peaks = coordinates
        plt.imshow(np.max(image_channels[channel_index], axis=0))
        plt.colorbar()
        if (channel_index == 0):
            plt.clim(0, 6000)
        plt.scatter(coordinates[:, 1], coordinates[:, 0], facecolors='none', edgecolors='r')
        plt.xlabel("X-Dimension")
        plt.ylabel("Y-Dimension")
        plt.title("MIP, channel " + str(channel_index))


        plt.show()

        channel_index += 1

    return peaks

def findProximalPairsFinal(image_channels, window_radius, c1, c2, name):
    channel_index = 0
    peaks = []

    segmentation = Image.open(name)
    nucleiMatrix = np.array(segmentation)

    for channel in image_channels:

        if channel_index == 0:
            threshold = c1 
        if channel_index == 1:
            threshold = c2 

        coordinates = peak_local_max(np.max(image_channels[channel_index], axis=0),  
                                     min_distance=4,
                                     threshold_rel=threshold,
                                     exclude_border=int(window_radius + 1))

        peaks.append(coordinates)
        plt.imshow(np.max(image_channels[channel_index], axis=0))
        plt.colorbar()
        if (channel_index == 0):
            plt.clim(0, 6000)
        if channel_index == 1:
            plt.clim(0, 5000)
        plt.scatter(coordinates[:, 1], coordinates[:, 0], facecolors='none', edgecolors='r')
        plt.xlabel("X-Dimension")
        plt.ylabel("Y-Dimension")
        plt.title("MIP, channel " + str(channel_index))

        plt.show()

        channel_index += 1

    paired_peaks = []
    channel0_peaks = peaks[0].tolist()
    channel1_peaks = peaks[1].tolist()
    for candidate0 in channel0_peaks:
        redundant = []
        if nucleiMatrix[candidate0[0]][candidate0[1]] == 0:
            continue
        for candidate1 in channel1_peaks:
            nucleiCheck = True
            upperBoundX = candidate0[1] + (window_radius - 1)
            lowerBoundX = candidate0[1] - (window_radius - 1)
            upperBoundY = candidate0[0] + (window_radius - 1)
            lowerBoundY = candidate0[0] - (window_radius - 1)

            temp = nucleiMatrix[(candidate1[0] - 2):(candidate1[0] + 3), (candidate1[1] - 2):(candidate1[1] + 3)]
            if nucleiMatrix[candidate0[0]][candidate0[1]] in temp.flatten():
                nucleiCheck = True
            else:
                nucleiCheck = False
            if not nucleiCheck:
                continue
            if upperBoundX >= candidate1[1] >= lowerBoundX:
                if upperBoundY >= candidate1[0] >= lowerBoundY:
                    redundant.append([candidate1,
                                      distance2D(candidate0, candidate1)])

        if len(redundant) == 1:
            paired_peaks.append([candidate0, redundant[0][0]])
            channel1_peaks.remove(redundant[0][0])
        if len(redundant) > 1:
            distances = [d[1] for d in redundant]
            index = distances.index(min(distances))
            paired_peaks.append([candidate0, redundant[index][0]])

    print(len(paired_peaks))
    plt.imshow(np.max(image_channels[1], axis=0))
    plt.colorbar()
    plt.clim(0, 8000)

    plt.xlabel("X-Dimension")
    plt.ylabel("Y-Dimension")
    plt.title("MIP, channel 1")
    for pair in paired_peaks:
        plt.plot(pair[1][1], pair[1][0], marker='+', color="white")
        plt.plot(pair[0][1], pair[0][0], marker='+', color="red")
    plt.show()

    return paired_peaks, nucleiMatrix

def findProximalPairsFinal3(image_channels, window_radius, c1, c2, c3, name):
    channel_index = 0
    peaks = []

    segmentation = Image.open(name)
    nucleiMatrix = np.array(segmentation)

    for channel in image_channels:

        if channel_index == 0:
            threshold = c1 
        if channel_index == 1:
            threshold = c2 
        if channel_index == 2:
            threshold = c3 
        if channel_index == 0 or channel_index == 1:
            coordinates = peak_local_max(np.max(image_channels[channel_index], axis=0),  
                                     min_distance=6,
                                     threshold_rel=threshold,
                                     exclude_border=int(window_radius + 1))
        if channel_index == 2:
            channel488max = np.max(np.array(image_channels[2], dtype=np.uint16), axis=0)
            normalized = channel488max / 65535
            corrected = np.power(normalized, 1.25)
            corrected_image = np.clip(corrected * 65535, 0, 65535).astype(np.uint16)
            coordinates = peak_local_max(corrected_image,  
                                     min_distance=6,
                                     threshold_rel=threshold,
                                     exclude_border=int(window_radius + 1))

        peaks.append(coordinates)
        plt.imshow(np.max(image_channels[channel_index], axis=0))
        plt.colorbar()
        if (channel_index == 0):
            plt.clim(0, 6000)
        if channel_index == 1:
            plt.clim(0, 5000)
        if channel_index == 2:
            plt.clim(0, 6000)
        plt.scatter(coordinates[:, 1], coordinates[:, 0], facecolors='none', edgecolors='r')
        plt.xlabel("X-Dimension")
        plt.ylabel("Y-Dimension")
        plt.title("MIP, channel " + str(channel_index))




        plt.show()

        channel_index += 1

    paired_peaks = []
    channel0_peaks = peaks[0].tolist()
    channel1_peaks = peaks[1].tolist()
    channel2_peaks = peaks[2].tolist()
    for candidate0 in channel0_peaks:
        redundant = []
        redundant2 = []
        if nucleiMatrix[candidate0[0]][candidate0[1]] == 0:
            continue
        for candidate1 in channel1_peaks:
            nucleiCheck = True
            upperBoundX = candidate0[1] + (window_radius - 1)
            lowerBoundX = candidate0[1] - (window_radius - 1)
            upperBoundY = candidate0[0] + (window_radius - 1)
            lowerBoundY = candidate0[0] - (window_radius - 1)
            if nucleiMatrix[candidate0[0]][candidate0[1]] == nucleiMatrix[candidate1[0]][candidate1[1]]:
                nucleiCheck = True
            else:
                nucleiCheck = False
            if not nucleiCheck:
                continue
            if upperBoundX >= candidate1[1] >= lowerBoundX:
                if upperBoundY >= candidate1[0] >= lowerBoundY:
                    redundant.append([candidate1,
                                      distance2D(candidate0, candidate1)])
        for candidate2 in channel2_peaks:
            nucleiCheck = True
            upperBoundX = candidate0[1] + (window_radius - 1)
            lowerBoundX = candidate0[1] - (window_radius - 1)
            upperBoundY = candidate0[0] + (window_radius - 1)
            lowerBoundY = candidate0[0] - (window_radius - 1)
            if nucleiMatrix[candidate0[0]][candidate0[1]] == nucleiMatrix[candidate2[0]][candidate2[1]]:
                nucleiCheck = True
            else:
                nucleiCheck = False
            if not nucleiCheck:
                continue
            if upperBoundX >= candidate2[1] >= lowerBoundX:
                if upperBoundY >= candidate2[0] >= lowerBoundY:
                    redundant2.append([candidate2,
                                      distance2D(candidate0, candidate2)])

        if len(redundant) == 1:
            if len(redundant2) == 1:
                paired_peaks.append([candidate0, redundant[0][0], redundant2[0][0]])
                channel1_peaks.remove(redundant[0][0])
                channel2_peaks.remove(redundant2[0][0])
            if len(redundant2) > 1:
                distances = [d[1] for d in redundant2]
                index = distances.index(min(distances))
                paired_peaks.append([candidate0, redundant[0][0], redundant2[index][0]])
                channel1_peaks.remove(redundant[0][0])
                channel2_peaks.remove(redundant2[index][0])
        if len(redundant) > 1:
            if len(redundant2) == 1:
                distances = [d[1] for d in redundant]
                index = distances.index(min(distances))
                paired_peaks.append([candidate0, redundant[index][0], redundant2[0][0]])
                channel1_peaks.remove(redundant[index][0])
                channel2_peaks.remove(redundant2[0][0])
            if len(redundant2) > 1:
                distances = [d[1] for d in redundant]
                index = distances.index(min(distances))
                distances2 = [d[1] for d in redundant2]
                index2 = distances2.index(min(distances2))
                paired_peaks.append([candidate0, redundant[index][0], redundant2[0][0]])
                channel1_peaks.remove(redundant[index][0])
                channel2_peaks.remove(redundant2[index2][0])

    print(len(paired_peaks))
    plt.imshow(np.max(image_channels[1], axis=0))
    plt.colorbar()
    plt.clim(0, 8000)

    plt.xlabel("X-Dimension")
    plt.ylabel("Y-Dimension")
    plt.title("MIP, channel 1")
    for pair in paired_peaks:
        plt.plot(pair[1][1], pair[1][0], marker='+', color="white")
        plt.plot(pair[0][1], pair[0][0], marker='+', color="red")
        plt.plot(pair[2][1], pair[2][0], marker='+', color='blue')
    plt.show()

    return paired_peaks, nucleiMatrix

def findProximalPairsFinalR1(image_channels, window_radius, c1, c2, cR, name):
    channel_index = 0
    peaks = []

    segmentation = Image.open(name)
    nucleiMatrix = np.array(segmentation)
    activeRNAHolder = []

    for channel in image_channels:

        if channel_index == 0:
            threshold = c1 
        if channel_index == 1:
            threshold = c2 
        if channel_index == 2:
            threshold = cR 

        coordinates = peak_local_max(np.max(image_channels[channel_index], axis=0),  
                                     min_distance=4,
                                     threshold_rel=threshold,
                                     exclude_border=int(window_radius + 1))

        peaks.append(coordinates)
        plt.imshow(np.max(image_channels[channel_index], axis=0))
        plt.colorbar()
        if (channel_index == 0):
            plt.clim(0, 6000)
        if channel_index == 1:
            plt.clim(0, 5000)
        plt.scatter(coordinates[:, 1], coordinates[:, 0], facecolors='none', edgecolors='r')
        plt.xlabel("X-Dimension")
        plt.ylabel("Y-Dimension")
        plt.title("MIP, channel " + str(channel_index))
        plt.show()

        channel_index += 1

    paired_peaks = []
    channel0_peaks = peaks[0].tolist()
    channel1_peaks = peaks[1].tolist()
    peakChannelR = peaks[2].tolist()
    candidateHolder = -1
    for candidate0 in channel0_peaks:
        candidateHolder += 1
        redundant = []
        if nucleiMatrix[candidate0[0]][candidate0[1]] == 0:
            continue
        for candidate1 in channel1_peaks:
            nucleiCheck = True
            upperBoundX = candidate0[1] + (window_radius - 1)
            lowerBoundX = candidate0[1] - (window_radius - 1)
            upperBoundY = candidate0[0] + (window_radius - 1)
            lowerBoundY = candidate0[0] - (window_radius - 1)
            if nucleiMatrix[candidate0[0]][candidate0[1]] == nucleiMatrix[candidate1[0]][candidate1[1]]:
                nucleiCheck = True
            else:
                nucleiCheck = False
            if not nucleiCheck:
                continue
            if upperBoundX >= candidate1[1] >= lowerBoundX:
                if upperBoundY >= candidate1[0] >= lowerBoundY:
                    redundant.append([candidate1,
                                      distance2D(candidate0, candidate1)])

        dumbLoopCheck = False
        if len(redundant) == 1:
            paired_peaks.append([candidate0, redundant[0][0]])
            channel1_peaks.remove(redundant[0][0])
            for candidateR in peakChannelR:
                upperBoundX = candidate0[1] + (window_radius - 1)
                lowerBoundX = candidate0[1] - (window_radius - 1)
                upperBoundY = candidate0[0] + (window_radius - 1)
                lowerBoundY = candidate0[0] - (window_radius - 1)
                if nucleiMatrix[candidate0[0]][candidate0[1]] == nucleiMatrix[candidateR[0]][candidateR[1]]:
                    activeRNAHolder.append(True)
                    dumbLoopCheck = True
                    break
            if not dumbLoopCheck:
                activeRNAHolder.append(False)
        if len(redundant) > 1:
            distances = [d[1] for d in redundant]
            index = distances.index(min(distances))
            paired_peaks.append([candidate0, redundant[index][0]])
            for candidateR in peakChannelR:
                upperBoundX = candidate0[1] + (window_radius - 1)
                lowerBoundX = candidate0[1] - (window_radius - 1)
                upperBoundY = candidate0[0] + (window_radius - 1)
                lowerBoundY = candidate0[0] - (window_radius - 1)
                if nucleiMatrix[candidate0[0]][candidate0[1]] == nucleiMatrix[candidateR[0]][candidateR[1]]:
                    activeRNAHolder.append(True)
                    dumbLoopCheck = True
                    break
            if not dumbLoopCheck:
                activeRNAHolder.append(False)

    print(len(paired_peaks))
    plt.imshow(np.max(image_channels[1], axis=0))
    plt.colorbar()
    plt.clim(0, 8000)

    plt.xlabel("X-Dimension")
    plt.ylabel("Y-Dimension")
    plt.title("MIP, channel 1")
    for pair in paired_peaks:
        plt.plot(pair[1][1], pair[1][0], marker='+', color="white")
        plt.plot(pair[0][1], pair[0][0], marker='+', color="red")
    plt.show()

    return paired_peaks, nucleiMatrix, activeRNAHolder

def localizationPSFSMFISH1C(image_channels, voxel_size, image_shape, listPoints,
                            channel, analysisFolder, backgroundValue, saveOption, initial_guess):
    window_radius = 5
    actualAmplitude = []
    lateralSigma = []

    positionHolder = []
    for point in listPoints:

        check = True
        print(point)
        xCenter = int(point[0])
        yCenter = int(point[1])
        initial2DGuess = (initial_guess[1], window_radius, window_radius, initial_guess[2], initial_guess[2])

        fit_parameters_per_slice = []
        slice_indices = []
        slice_intensities = []
        x = np.linspace(0, window_radius * 2 - 1, (window_radius * 2))
        y = np.linspace(0, window_radius * 2 - 1, (window_radius * 2))
        x, y = np.meshgrid(x, y)

        z_index = 0
        for stack in image_channels[0]:
            fit_failed = False
            local_region = stack[(yCenter - window_radius):(yCenter + window_radius),
                             (xCenter - window_radius):(xCenter + window_radius)]
            try:
                parameters2D, covariance2D = opt.curve_fit(gaussian2D, (x, y),
                                                           np.concatenate(local_region).ravel(),
                                                           p0=initial2DGuess, maxfev=20000)
                if not (0 <= parameters2D[1] <= window_radius * 2) or not \
                        (0 <= parameters2D[2] <= window_radius * 2) or (parameters2D[0] < 0):
                    fit_failed = True
            except (RuntimeError, ValueError) as e:
                fit_failed = True
            if not fit_failed:
                fit_parameters_per_slice.append(parameters2D)
                slice_indices.append(z_index)
                slice_intensities.append(local_region[min(window_radius*2 - 1, round(parameters2D[2]))]
                                [min(window_radius*2 - 1, round(parameters2D[1]))])
                peakFitted = gaussian2D((x, y), *parameters2D)
                fig, ax = plt.subplots(1, 1)
                ax.imshow(local_region, cmap=cm.binary, origin='lower',
                        extent=(x.min(), x.max(), y.min(), y.max()))
                ax.contour(x, y, peakFitted.reshape((window_radius * 2), (window_radius * 2)), 8, colors='r')
                plt.gca().invert_yaxis()
                plt.title(str(point[0]) + "-" + str(point[1]) + " Fit")
                plt.xlabel("X")
                plt.ylabel("Y")



                plt.close()
            else:
                fit_parameters_per_slice.append([local_region[window_radius][window_radius], "idiot", "dumbass", 0, 0])
                slice_indices.append(z_index)
                slice_intensities.append(local_region[window_radius][window_radius])
            z_index += 1

        tempHolder = ([stack[0] for stack in fit_parameters_per_slice])
        indexMax = max(range(len(tempHolder)), key=tempHolder.__getitem__)
        z = np.linspace(indexMax - 4, indexMax + 3, 8)
        zTemp = np.linspace(indexMax - 4, indexMax + 3, 100)
        initial1DGuess = (initial_guess[1], indexMax, initial_guess[2])

        if indexMax < (4) or (indexMax + 4) > image_shape[2]:
            indexMax = 4
        if (indexMax + 4) > image_shape[2]:
            indexMax = image_shape[2] - 4

        try:
            parameters1D, covariance1D = opt.curve_fit(gaussian1D, z,
                                                           tempHolder[indexMax - 4:indexMax + 4],
                                                           p0=initial1DGuess, maxfev=20000)
            print(parameters1D)
        except (RuntimeError, ValueError) as e:
            check = False

        if parameters1D[1] < 0:
            check = False

        if check:
            length = 0
            dumbIdiot = [stack[3] for stack in fit_parameters_per_slice] + [stack[4] for stack in fit_parameters_per_slice]
            for value in dumbIdiot:
                if value != 0:
                    length += 1

            try:
                actualAmplitude.append(slice_intensities[math.ceil(parameters1D[1])]*(math.ceil(parameters1D[1]) - parameters1D[1]) +
                                slice_intensities[math.floor(parameters1D[1])]*(1-(math.ceil(parameters1D[1]) - parameters1D[1])))
                positionHolder.append(point)
            except (IndexError) as error:
                print("z error")
    return actualAmplitude, lateralSigma, positionHolder

def localizationPSFSMFISH2C(image_channels, voxel_size, image_shape, listPoints,
                            channel, analysisFolder, backgroundValue, saveOption, initial_guess):
    window_radius = 5
    actualAmplitude = []
    lateralSigma = []

    for point in listPoints:
        channel_index = 0
        print(point)
        fit_valid = True

        actualAmplitudetemp = []
        lateralSigmatemp = []

        for channel in image_channels:
            xCenter = int(point[0])
            yCenter = int(point[1])
            initial2DGuess = (initial_guess[1], window_radius, window_radius, initial_guess[2], initial_guess[2])
            if channel_index == 2:
                continue

            fit_parameters_per_slice = []
            slice_indices = []
            slice_intensities = []
            x = np.linspace(0, window_radius * 2 - 1, (window_radius * 2))
            y = np.linspace(0, window_radius * 2 - 1, (window_radius * 2))
            x, y = np.meshgrid(x, y)

            z_index = 0
            for stack in image_channels[channel_index]:
                fit_failed = False
                local_region = stack[(yCenter - window_radius):(yCenter + window_radius),
                             (xCenter - window_radius):(xCenter + window_radius)]
                try:
                    parameters2D, covariance2D = opt.curve_fit(gaussian2D, (x, y),
                                                           np.concatenate(local_region).ravel(),
                                                           p0=initial2DGuess, maxfev=20000)
                    if not (0 <= parameters2D[1] <= window_radius * 2) or not \
                        (0 <= parameters2D[2] <= window_radius * 2) or (parameters2D[0] < 0):
                        fit_failed = True
                except (RuntimeError, ValueError) as e:
                    fit_failed = True
                if not fit_failed:
                    fit_parameters_per_slice.append(parameters2D)
                    slice_indices.append(z_index)
                    slice_intensities.append(local_region[min(window_radius*2 - 1, round(parameters2D[2]))]
                                [min(window_radius*2 - 1, round(parameters2D[1]))])
                    peakFitted = gaussian2D((x, y), *parameters2D)
                    fig, ax = plt.subplots(1, 1)
                    ax.imshow(local_region, cmap=cm.binary, origin='lower',
                        extent=(x.min(), x.max(), y.min(), y.max()))
                    ax.contour(x, y, peakFitted.reshape((window_radius * 2), (window_radius * 2)), 8, colors='r')
                    plt.gca().invert_yaxis()
                    plt.title(str(point[0]) + "-" + str(point[1]) + " Fit")
                    plt.xlabel("X")
                    plt.ylabel("Y")



                    plt.close()
                else:
                    fit_parameters_per_slice.append([local_region[window_radius][window_radius], "idiot", "dumbass", 0, 0])
                    slice_indices.append(z_index)
                    slice_intensities.append(local_region[window_radius][window_radius])
                z_index += 1

            tempHolder = ([stack[0] for stack in fit_parameters_per_slice])
            indexMax = max(range(len(tempHolder)), key=tempHolder.__getitem__)
            z = np.linspace(indexMax - 4, indexMax + 3, 8)
            zTemp = np.linspace(indexMax - 4, indexMax + 3, 100)
            initial1DGuess = (initial_guess[1], indexMax, initial_guess[2])

            if indexMax < (4) or (indexMax + 4) > image_shape[2]:
                indexMax = 4
            if (indexMax + 4) > image_shape[2]:
                indexMax = image_shape[2] - 4

            try:
                parameters1D, covariance1D = opt.curve_fit(gaussian1D, z,
                                                           tempHolder[indexMax - 4:indexMax + 4],
                                                           p0=initial1DGuess, maxfev=20000)
                print(parameters1D)
            except (RuntimeError, ValueError) as e:
                fit_valid = False
                break

            if parameters1D[1] < 0:
                fit_valid = False
                break

            plt.plot(slice_indices, [stack[0] for stack in fit_parameters_per_slice], label='2D Amplitude')
            plt.plot(zTemp, gaussian1D(zTemp, *parameters1D), color='red', label='1D Gaussian Fit')
            plt.plot(slice_indices, slice_intensities, color='green', label='Actual Intensity')
            plt.scatter(parameters1D[1], parameters1D[0], color='red')
            plt.text(len(slice_indices)/2 - 1, 7000,
                    "Z-" + str(round(parameters1D[1], 3)) + " A-" + str(round(parameters1D[0], 3)),
                    fontsize='15')
            plt.title(str([point[0], point[1]]) + "-Amplitude-Z")
            plt.legend(loc="upper left")
            plt.ylim(0, 30000)
            plt.xlim(0, image_shape[2]-1)


            plt.close()

            length = 0
            dumbIdiot = [stack[3] for stack in fit_parameters_per_slice] + [stack[4] for stack in fit_parameters_per_slice]
            for value in dumbIdiot:
                if value != 0:
                    length += 1
            lateralSigmatemp.append(sum(dumbIdiot)/length)
            try:
                actualAmplitudetemp.append(slice_intensities[math.ceil(parameters1D[1])]*(math.ceil(parameters1D[1]) - parameters1D[1]) +
                                slice_intensities[math.floor(parameters1D[1])]*(1-(math.ceil(parameters1D[1]) - parameters1D[1])))
            except (IndexError) as error:
                print("z-error")
                fit_valid = False

            channel_index += 1

        if fit_valid:
            actualAmplitude.append(actualAmplitudetemp)
            lateralSigma.append(lateralSigmatemp)
        else:
            print("DOUBLECHECK FAILURE DOUBLECHECK FAILURE DOUBLECHECK FAILURE DOUBLECHECK FAILURE")

    return actualAmplitude, lateralSigma

def localizationPSFDistanceManual1(image_channels, voxel_size, image_shape, listPoints, analysisFolder, saveOption, sheet,
                                  image_index, initial_guess):
    window_radius = initial_guess[0]
    primary_coordinates = [[], [], []]
    secondary_coordinates = [[], [], []]
    actualAmplitude = [[], []]
    lateralSigma = [[], []]

    print("calling manual")

    uncorrected_distances = []
    for point in listPoints:
        print(point)
        channel_index = 0
        primary_coordinates_temp = []
        secondary_coordinates_temp = []
        fit_valid = True
        for channel in image_channels:

            xCenter = int(point[0])
            yCenter = int(point[1])

            if channel_index == 0:
                initial2DGuess = (initial_guess[1], window_radius, window_radius, initial_guess[2], initial_guess[2])
            if channel_index == 1:
                initial2DGuess = (initial_guess[1], window_radius, window_radius, initial_guess[2], initial_guess[2])
            if channel_index == 2:
                continue

            fit_parameters_per_slice = []
            slice_indices = []
            slice_intensities = []
            x = np.linspace(0, window_radius * 2 - 1, (window_radius * 2))
            y = np.linspace(0, window_radius * 2 - 1, (window_radius * 2))
            x, y = np.meshgrid(x, y)

            z_index = 0
            for stack in channel:
                fit_failed = False
                local_region = stack[(yCenter - window_radius):(yCenter + window_radius),
                                      (xCenter - window_radius):(xCenter + window_radius)]
                try:
                    parameters2D, covariance2D = opt.curve_fit(gaussian2D, (x, y),
                                                                  np.concatenate(local_region).ravel(),
                                                                  p0=initial2DGuess, maxfev=20000)

                    if not (0 <= parameters2D[1] <= window_radius * 2) or not \
                            (0 <= parameters2D[2] <= window_radius * 2) or (parameters2D[0] < 0):
                        fit_failed = True
                except (RuntimeError, ValueError) as e:
                    fit_failed = True
                if not fit_failed:
                    fit_parameters_per_slice.append(parameters2D)
                    slice_indices.append(z_index)
                    slice_intensities.append(local_region[min(window_radius*2 - 1, round(parameters2D[2]))]
                                    [min(window_radius*2 - 1, round(parameters2D[1]))])
                    peakFitted = gaussian2D((x, y), *parameters2D)
                    fig, ax = plt.subplots(1, 1)
                    ax.imshow(local_region, cmap=cm.binary, origin='lower',
                          extent=(x.min(), x.max(), y.min(), y.max()))
                    ax.contour(x, y, peakFitted.reshape((window_radius * 2), (window_radius * 2)), 8, colors='r')
                    plt.gca().invert_yaxis()
                    plt.title(str(point[channel_index]) + " - " + str(z_index))
                    plt.xlabel("X")
                    plt.ylabel("Y")
                    if saveOption:
                        plt.savefig(analysisFolder + '/' + str(image_index) + "-" + str([point[0], point[1]]) + "-" + str(
                        z_index) + "-" + str(channel_index))
                    plt.close()
                else:
                    fit_parameters_per_slice.append([local_region[window_radius][window_radius], 'dumbass', 'idiot', 0, 0])
                    slice_indices.append(z_index)
                    slice_intensities.append(local_region[window_radius][window_radius])

                z_index += 1

            tempHolder = ([stack[0] for stack in fit_parameters_per_slice])
            indexMax = max(range(len(tempHolder)), key=tempHolder.__getitem__)
            z = np.linspace(indexMax - 4, indexMax + 3, 8)
            zTemp = np.linspace(indexMax - 4, indexMax + 3, 100)
            initial1DGuess = (initial_guess[1], indexMax, initial_guess[2])

            if indexMax < (4) or (indexMax + 4) > image_shape[2]:
                indexMax = 4
            if (indexMax + 4) > image_shape[2]:
                indexMax = image_shape[2] - 4

            try:
                parameters1D, covariance1D = opt.curve_fit(gaussian1D, z,
                                                           tempHolder[indexMax - 4:indexMax + 4],
                                                           p0=initial1DGuess, maxfev=20000)
                print(parameters1D)
            except (RuntimeError, ValueError) as e:
                fit_valid = False
                break

            if parameters1D[1] < 0:
                fit_valid = False
                break

            plt.plot(slice_indices, [stack[0] for stack in fit_parameters_per_slice], label='2D Amplitude')
            plt.plot(zTemp, gaussian1D(zTemp, *parameters1D), color='red', label='1D Gaussian Fit')
            plt.plot(slice_indices, slice_intensities, color='green', label='Actual Intensity')
            plt.scatter(parameters1D[1], parameters1D[0], color='red')
            plt.text(len(slice_indices)/2 - 1, 7000,
                     "Z-" + str(round(parameters1D[1], 3)) + " A-" + str(round(parameters1D[0], 3)),
                     fontsize='15')
            plt.title(str([point[0], point[1]]) + "-Amplitude-Z-" + str(channel_index))
            plt.legend(loc="upper left")
            plt.ylim(0, initial_guess[1]*4)
            plt.xlim(0, image_shape[2]-1)
            if saveOption:
                plt.savefig(analysisFolder + '/' + str(image_index) + "-" + str([point[0], point[1]]) + "-Amplitude-Z-" + str(channel_index))
            plt.close()

            try:
                xCoordinate = fit_parameters_per_slice[math.floor(parameters1D[1])][1]*(math.ceil(parameters1D[1]) - parameters1D[1]) + \
                          fit_parameters_per_slice[math.ceil(parameters1D[1])][1]*(1 - math.ceil(parameters1D[1]) + parameters1D[1])
                yCoordinate = fit_parameters_per_slice[math.floor(parameters1D[1])][2]*(math.ceil(parameters1D[1]) - parameters1D[1]) + \
                          fit_parameters_per_slice[math.ceil(parameters1D[1])][2]*(1 - math.ceil(parameters1D[1]) + parameters1D[1])
            except (IndexError, TypeError) as error:
                print('xy coordinate error')
                fit_valid = False
                continue

            if channel_index == 0:
                primary_coordinates_temp.append(round(xCoordinate, 3) + xCenter - window_radius)
                primary_coordinates_temp.append(round(yCoordinate, 3) + yCenter - window_radius)
                primary_coordinates_temp.append(round(parameters1D[1], 3))
                primary_coordinates_temp.append(str(xCenter) + "," + str(yCenter))
                primary_coordinates_temp.append(str(round(xCoordinate + xCenter - window_radius, 3)) + "," +
                                   str(round(yCoordinate + yCenter - window_radius, 3)) + "," +
                                   str(round(parameters1D[1], 3)))
            if channel_index == 1:
                secondary_coordinates_temp.append(round(xCoordinate, 3) + xCenter - window_radius)
                secondary_coordinates_temp.append(round(yCoordinate, 3) + yCenter - window_radius)
                secondary_coordinates_temp.append(round(parameters1D[1], 3))
                secondary_coordinates_temp.append(str(xCenter) + "," + str(yCenter))
                secondary_coordinates_temp.append(str(round(xCoordinate + xCenter - window_radius, 3)) + "," +
                                   str(round(yCoordinate + yCenter - window_radius, 3)) + "," +
                                   str(round(parameters1D[1], 3)))
            channel_index += 1

        if fit_valid:
            primary_coordinates[0].append(primary_coordinates_temp[0])
            primary_coordinates[1].append(primary_coordinates_temp[1])
            primary_coordinates[2].append(primary_coordinates_temp[2])
            secondary_coordinates[0].append(secondary_coordinates_temp[0])
            secondary_coordinates[1].append(secondary_coordinates_temp[1])
            secondary_coordinates[2].append(secondary_coordinates_temp[2])
            distanceTemp = (((primary_coordinates_temp[0] - secondary_coordinates_temp[0]) * voxel_size[0]) ** 2 +
                                  ((primary_coordinates_temp[1] - secondary_coordinates_temp[1]) * voxel_size[1]) ** 2 +
                                  ((primary_coordinates_temp[2] - secondary_coordinates_temp[2]) * voxel_size[2]) ** 2) ** 0.5
            uncorrected_distances.append(distanceTemp)
            if saveOption:
                sheet.append([primary_coordinates_temp[3], primary_coordinates_temp[4], distanceTemp])
                sheet.append([secondary_coordinates_temp[3], secondary_coordinates_temp[4], distanceTemp])
        else:
            print("DOUBLECHECK FAILURE DOUBLECHECK FAILURE DOUBLECHECK FAILURE DOUBLECHECK FAILURE")
            if saveOption:
                for filename in os.listdir(analysisFolder):
                    if filename.startswith(str([point[0], point[1]])):
                        os.remove(os.path.join(analysisFolder, filename))
                for filename in os.listdir(analysisFolder):
                    if filename.startswith(str([point[0], point[1]])):
                        os.remove(os.path.join(analysisFolder, filename))

    return uncorrected_distances, primary_coordinates, secondary_coordinates, actualAmplitude, lateralSigma

def localizationPSFDistanceAutomatic1(image_channels, voxel_size, image_shape, primaryList, secondaryList, analysisFolder, saveOption, sheet,
                                  image_index, initial_guess, lab):
    window_radius = initial_guess[0]
    primary_coordinates = [[], [], []]
    secondary_coordinates = [[], [], []]

    uncorrected_distances = []
    pointHolder = 0
    print("calling auto")
    for point in primaryList:
        channel_index = 0
        primary_coordinates_temp = []
        secondary_coordinates_temp = []
        fit_valid = True
        for channel in image_channels:
            if channel_index == 0:
                xCenter = int(primaryList[pointHolder][1 if lab == "mike" else 0]) 
                yCenter = int(primaryList[pointHolder][0 if lab == "mike" else 1]) 
            if channel_index == 1:
                xCenter = int(secondaryList[pointHolder][1 if lab == "mike" else 0])
                yCenter = int(secondaryList[pointHolder][0 if lab == "mike" else 1])

            if channel_index == 0:
                initial2DGuess = (initial_guess[1], window_radius, window_radius, initial_guess[2], initial_guess[2])
            if channel_index == 1:
                initial2DGuess = (initial_guess[1], window_radius, window_radius, initial_guess[2], initial_guess[2])

            fit_parameters_per_slice = []
            slice_indices = []
            slice_intensities = []
            x = np.linspace(0, window_radius * 2 - 1, (window_radius * 2))
            y = np.linspace(0, window_radius * 2 - 1, (window_radius * 2))
            x, y = np.meshgrid(x, y)

            z_index = 0
            for stack in channel:
                fit_failed = False
                local_region = stack[(yCenter - window_radius):(yCenter + window_radius),
                                      (xCenter - window_radius):(xCenter + window_radius)]
                try:
                    parameters2D, covariance2D = opt.curve_fit(gaussian2D, (x, y),
                                                                  np.concatenate(local_region).ravel(),
                                                                  p0=initial2DGuess, maxfev=20000)
                    if not (0 <= parameters2D[1] <= window_radius * 2) or not \
                            (0 <= parameters2D[2] <= window_radius * 2) or (parameters2D[0] < 0):
                        fit_failed = True
                except (RuntimeError, ValueError) as e:
                    fit_failed = True
                if not fit_failed:
                    fit_parameters_per_slice.append(parameters2D)
                    slice_indices.append(z_index)
                    slice_intensities.append(local_region[min(window_radius*2 - 1, round(parameters2D[2]))]
                                    [min(window_radius*2 - 1, round(parameters2D[1]))])
                    peakFitted = gaussian2D((x, y), *parameters2D)
                    fig, ax = plt.subplots(1, 1)
                    ax.imshow(local_region, cmap=cm.binary, origin='lower',
                          extent=(x.min(), x.max(), y.min(), y.max()))
                    ax.contour(x, y, peakFitted.reshape((window_radius * 2), (window_radius * 2)), 8, colors='r')
                    plt.gca().invert_yaxis()
                    plt.title(str(point[channel_index]) + " - " + str(z_index))
                    plt.xlabel("X")
                    plt.ylabel("Y")
                    if saveOption:
                        plt.savefig(analysisFolder + '/' + str(image_index) + "-" + str([point[0], point[1]]) + "-" + str(
                        z_index) + "-" + str(channel_index))
                    plt.close()
                else:
                    fit_parameters_per_slice.append([local_region[window_radius][window_radius], "idiot", "dumbass", 0, 0])
                    slice_indices.append(z_index)
                    slice_intensities.append(local_region[window_radius][window_radius])

                z_index += 1

            tempHolder = ([stack[0] for stack in fit_parameters_per_slice])
            indexMax = max(range(len(tempHolder)), key=tempHolder.__getitem__)
            z = np.linspace(indexMax - 4, indexMax + 3, 8)
            zTemp = np.linspace(indexMax - 4, indexMax + 3, 100)
            initial1DGuess = (initial_guess[1], indexMax, initial_guess[2])

            if indexMax < (4) or (indexMax + 4) > image_shape[2]:
                indexMax = 4
            if (indexMax + 4) > image_shape[2]:
                indexMax = image_shape[2] - 4

            try:
                parameters1D, covariance1D = opt.curve_fit(gaussian1D, z,
                                                           tempHolder[indexMax - 4:indexMax + 4],
                                                           p0=initial1DGuess, maxfev=20000)

            except (RuntimeError, ValueError) as e:

                fit_valid = False
                continue

            if parameters1D[1] < 0:
                fit_valid = False

            plt.plot(slice_indices, [stack[0] for stack in fit_parameters_per_slice], label='2D Amplitude')
            plt.plot(zTemp, gaussian1D(zTemp, *parameters1D), color='red', label='1D Gaussian Fit')
            plt.plot(slice_indices, slice_intensities, color='green', label='Actual Intensity')
            plt.scatter(parameters1D[1], parameters1D[0], color='red')
            plt.text(len(slice_indices)/2 - 1, 7000,
                     "Z-" + str(round(parameters1D[1], 3)) + " A-" + str(round(parameters1D[0], 3)),
                     fontsize='15')
            plt.title(str([point[0], point[1]]) + "-Amplitude-Z-" + str(channel_index))
            plt.legend(loc="upper left")
            plt.ylim(0, initial_guess[1]*4)
            plt.xlim(0, image_shape[2]-1)
            if saveOption:
                plt.savefig(analysisFolder + '/' + str(image_index) + "-" + str([point[0], point[1]]) + "-Amplitude-Z-" + str(channel_index))
            plt.close()

            try:
                xCoordinate = fit_parameters_per_slice[math.floor(parameters1D[1])][1]*(math.ceil(parameters1D[1]) - parameters1D[1]) + \
                          fit_parameters_per_slice[math.ceil(parameters1D[1])][1]*(1 - math.ceil(parameters1D[1]) + parameters1D[1])
                yCoordinate = fit_parameters_per_slice[math.floor(parameters1D[1])][2]*(math.ceil(parameters1D[1]) - parameters1D[1]) + \
                          fit_parameters_per_slice[math.ceil(parameters1D[1])][2]*(1 - math.ceil(parameters1D[1]) + parameters1D[1])
            except (IndexError, TypeError) as error:

                fit_valid = False
                continue




            if channel_index == 0:
                primary_coordinates_temp.append(round(xCoordinate, 3) + xCenter - window_radius)
                primary_coordinates_temp.append(round(yCoordinate, 3) + yCenter - window_radius)
                primary_coordinates_temp.append(round(parameters1D[1], 3))
                primary_coordinates_temp.append(str(xCenter) + "," + str(yCenter))
                primary_coordinates_temp.append(str(round(xCoordinate + xCenter - window_radius, 3)) + "," +
                                   str(round(yCoordinate + yCenter - window_radius, 3)) + "," +
                                   str(round(parameters1D[1], 3)))
            if channel_index == 1:
                secondary_coordinates_temp.append(round(xCoordinate, 3) + xCenter - window_radius)
                secondary_coordinates_temp.append(round(yCoordinate, 3) + yCenter - window_radius)
                secondary_coordinates_temp.append(round(parameters1D[1], 3))
                secondary_coordinates_temp.append(str(xCenter) + "," + str(yCenter))
                secondary_coordinates_temp.append(str(round(xCoordinate + xCenter - window_radius, 3)) + "," +
                                   str(round(yCoordinate + yCenter - window_radius, 3)) + "," +
                                   str(round(parameters1D[1], 3)))
            channel_index += 1

        if fit_valid:
            primary_coordinates[0].append(primary_coordinates_temp[0])
            primary_coordinates[1].append(primary_coordinates_temp[1])
            primary_coordinates[2].append(primary_coordinates_temp[2])
            secondary_coordinates[0].append(secondary_coordinates_temp[0])
            secondary_coordinates[1].append(secondary_coordinates_temp[1])
            secondary_coordinates[2].append(secondary_coordinates_temp[2])
            distanceTemp = (((primary_coordinates_temp[0] - secondary_coordinates_temp[0]) * voxel_size[0]) ** 2 +
                                  ((primary_coordinates_temp[1] - secondary_coordinates_temp[1]) * voxel_size[1]) ** 2 +
                                  ((primary_coordinates_temp[2] - secondary_coordinates_temp[2]) * voxel_size[2]) ** 2) ** 0.5
            uncorrected_distances.append(distanceTemp)
            if saveOption:
                sheet.append([primary_coordinates_temp[3], primary_coordinates_temp[4], distanceTemp])
                sheet.append([secondary_coordinates_temp[3], secondary_coordinates_temp[4], distanceTemp])
        else:

            if saveOption:
                for filename in os.listdir(analysisFolder):
                    if filename.startswith(str([point[0], point[1]])):
                        os.remove(os.path.join(analysisFolder, filename))
                for filename in os.listdir(analysisFolder):
                    if filename.startswith(str([point[0], point[1]])):
                        os.remove(os.path.join(analysisFolder, filename))
        pointHolder += 1

    return uncorrected_distances, primary_coordinates, secondary_coordinates

def localizationPSFDistanceAutomatic(image_channels, voxel_size, image_shape, primaryList, secondaryList, analysisFolder, saveOption, sheet,
                                  image_index, initial_guess, lab):
    window_radius = initial_guess[0]
    primary_coordinates = [[], [], []]
    secondary_coordinates = [[], [], []]

    uncorrected_distances = []
    pointHolder = 0
    for point in primaryList:
        channel_index = 0
        primary_coordinates_temp = []
        secondary_coordinates_temp = []
        fit_valid = True
        for channel in image_channels:
            if channel_index == 0:
                xCenter = int(primaryList[pointHolder][1 if lab == "mike" else 0]) 
                yCenter = int(primaryList[pointHolder][0 if lab == "mike" else 1]) 
            if channel_index == 1:
                xCenter = int(secondaryList[pointHolder][1 if lab == "mike" else 0])
                yCenter = int(secondaryList[pointHolder][0 if lab == "mike" else 1])

            if channel_index == 0:
                initial2DGuess = (initial_guess[1], window_radius, window_radius, initial_guess[2], initial_guess[2])
            if channel_index == 1:
                initial2DGuess = (initial_guess[1], window_radius, window_radius, initial_guess[2], initial_guess[2])

            fit_parameters_per_slice = []
            slice_indices = []
            slice_intensities = []
            x = np.linspace(0, window_radius * 2 - 1, (window_radius * 2))
            y = np.linspace(0, window_radius * 2 - 1, (window_radius * 2))
            x, y = np.meshgrid(x, y)

            z_index = 0
            for stack in channel:
                fit_failed = False
                local_region = stack[(yCenter - window_radius):(yCenter + window_radius),
                                      (xCenter - window_radius):(xCenter + window_radius)]
                try:
                    parameters2D, covariance2D = opt.curve_fit(gaussian2D, (x, y),
                                                                  np.concatenate(local_region).ravel(),
                                                                  p0=initial2DGuess, maxfev=20000)
                    if not (0 <= parameters2D[1] <= window_radius * 2) or not \
                            (0 <= parameters2D[2] <= window_radius * 2) or (parameters2D[0] < 0):
                        fit_failed = True
                except (RuntimeError, ValueError) as e:
                    fit_failed = True
                if not fit_failed:
                    fit_parameters_per_slice.append(parameters2D)
                    slice_indices.append(z_index)
                    slice_intensities.append(local_region[min(window_radius*2 - 1, round(parameters2D[2]))]
                                    [min(window_radius*2 - 1, round(parameters2D[1]))])
                    peakFitted = gaussian2D((x, y), *parameters2D)
                    fig, ax = plt.subplots(1, 1)
                    ax.imshow(local_region, cmap=cm.binary, origin='lower',
                          extent=(x.min(), x.max(), y.min(), y.max()))
                    ax.contour(x, y, peakFitted.reshape((window_radius * 2), (window_radius * 2)), 8, colors='r')
                    plt.gca().invert_yaxis()
                    plt.title(str(point[channel_index]) + " - " + str(z_index))
                    plt.xlabel("X")
                    plt.ylabel("Y")
                    if saveOption:
                        plt.savefig(analysisFolder + '/' + str(image_index) + "-" + str([point[0], point[1]]) + "-" + str(
                        z_index) + "-" + str(channel_index))
                    plt.close()
                else:
                    fit_parameters_per_slice.append([local_region[window_radius][window_radius], window_radius, window_radius, 0, 0])
                    slice_indices.append(z_index)
                    slice_intensities.append(local_region[window_radius][window_radius])

                z_index += 1

            tempHolder = ([stack[0] for stack in fit_parameters_per_slice])
            indexMax = max(range(len(tempHolder)), key=tempHolder.__getitem__)
            z = np.linspace(indexMax - 4, indexMax + 3, 8)
            zTemp = np.linspace(indexMax - 4, indexMax + 3, 100)
            initial1DGuess = (initial_guess[1], indexMax, initial_guess[2])

            if indexMax < (4) or (indexMax + 4) > image_shape[2]:
                indexMax = 4
            if (indexMax + 4) > image_shape[2]:
                indexMax = image_shape[2] - 4

            try:
                parameters1D, covariance1D = opt.curve_fit(gaussian1D, z,
                                                           tempHolder[indexMax - 4:indexMax + 4],
                                                           p0=initial1DGuess, maxfev=20000)
                print(parameters1D)
            except (RuntimeError, ValueError) as e:
                fit_valid = False
                break

            if parameters1D[1] < 0:
                fit_valid = False
                break

            plt.plot(slice_indices, [stack[0] for stack in fit_parameters_per_slice], label='2D Amplitude')
            plt.plot(zTemp, gaussian1D(zTemp, *parameters1D), color='red', label='1D Gaussian Fit')
            plt.plot(slice_indices, slice_intensities, color='green', label='Actual Intensity')
            plt.scatter(parameters1D[1], parameters1D[0], color='red')
            plt.text(len(slice_indices)/2 - 1, 7000,
                     "Z-" + str(round(parameters1D[1], 3)) + " A-" + str(round(parameters1D[0], 3)),
                     fontsize='15')
            plt.title(str([point[0], point[1]]) + "-Amplitude-Z-" + str(channel_index))
            plt.legend(loc="upper left")
            plt.ylim(0, initial_guess[1]*4)
            plt.xlim(0, image_shape[2]-1)
            if saveOption:
                plt.savefig(analysisFolder + '/' + str(image_index) + "-" + str([point[0], point[1]]) + "-Amplitude-Z-" + str(channel_index))
            plt.close()

            try:
                xCoordinate = fit_parameters_per_slice[math.floor(parameters1D[1])][1]*(math.ceil(parameters1D[1]) - parameters1D[1]) + \
                          fit_parameters_per_slice[math.ceil(parameters1D[1])][1]*(1 - math.ceil(parameters1D[1]) + parameters1D[1])
                yCoordinate = fit_parameters_per_slice[math.floor(parameters1D[1])][2]*(math.ceil(parameters1D[1]) - parameters1D[1]) + \
                          fit_parameters_per_slice[math.ceil(parameters1D[1])][2]*(1 - math.ceil(parameters1D[1]) + parameters1D[1])
            except (IndexError, TypeError) as error:
                print("x/y coordinate index error")
                fit_valid = False
                break




            if channel_index == 0:
                primary_coordinates_temp.append(round(xCoordinate, 3) + xCenter - window_radius)
                primary_coordinates_temp.append(round(yCoordinate, 3) + yCenter - window_radius)
                primary_coordinates_temp.append(round(parameters1D[1], 3))
                primary_coordinates_temp.append(str(xCenter) + "," + str(yCenter))
                primary_coordinates_temp.append(str(round(xCoordinate + xCenter - window_radius, 3)) + "," +
                                   str(round(yCoordinate + yCenter - window_radius, 3)) + "," +
                                   str(round(parameters1D[1], 3)))
            if channel_index == 1:
                secondary_coordinates_temp.append(round(xCoordinate, 3) + xCenter - window_radius)
                secondary_coordinates_temp.append(round(yCoordinate, 3) + yCenter - window_radius)
                secondary_coordinates_temp.append(round(parameters1D[1], 3))
                secondary_coordinates_temp.append(str(xCenter) + "," + str(yCenter))
                secondary_coordinates_temp.append(str(round(xCoordinate + xCenter - window_radius, 3)) + "," +
                                   str(round(yCoordinate + yCenter - window_radius, 3)) + "," +
                                   str(round(parameters1D[1], 3)))
            channel_index += 1

        if fit_valid:
            primary_coordinates[0].append(primary_coordinates_temp[0])
            primary_coordinates[1].append(primary_coordinates_temp[1])
            primary_coordinates[2].append(primary_coordinates_temp[2])
            secondary_coordinates[0].append(secondary_coordinates_temp[0])
            secondary_coordinates[1].append(secondary_coordinates_temp[1])
            secondary_coordinates[2].append(secondary_coordinates_temp[2])
            distanceTemp = (((primary_coordinates_temp[0] - secondary_coordinates_temp[0]) * voxel_size[0]) ** 2 +
                                  ((primary_coordinates_temp[1] - secondary_coordinates_temp[1]) * voxel_size[1]) ** 2 +
                                  ((primary_coordinates_temp[2] - secondary_coordinates_temp[2]) * voxel_size[2]) ** 2) ** 0.5
            uncorrected_distances.append(distanceTemp)
            if saveOption:
                sheet.append([primary_coordinates_temp[3], primary_coordinates_temp[4], distanceTemp])
                sheet.append([secondary_coordinates_temp[3], secondary_coordinates_temp[4], distanceTemp])
        else:
            print("DOUBLECHECK FAILURE DOUBLECHECK FAILURE DOUBLECHECK FAILURE DOUBLECHECK FAILURE")
            if saveOption:
                for filename in os.listdir(analysisFolder):
                    if filename.startswith(str([point[0], point[1]])):
                        os.remove(os.path.join(analysisFolder, filename))
                for filename in os.listdir(analysisFolder):
                    if filename.startswith(str([point[0], point[1]])):
                        os.remove(os.path.join(analysisFolder, filename))
        pointHolder += 1

    return uncorrected_distances, primary_coordinates, secondary_coordinates

def localizationPSFDistanceAutomaticR(image_channels, voxel_size, image_shape, primaryList, secondaryList, analysisFolder, saveOption, sheet,
                                  image_index, initial_guess, lab, activeRNAHolder):
    window_radius = initial_guess[0]
    primary_coordinates = [[], [], []]
    secondary_coordinates = [[], [], []]

    uncorrected_distances = []
    pointHolder = 0
    for point in primaryList:
        channel_index = 0
        primary_coordinates_temp = []
        secondary_coordinates_temp = []
        fit_valid = True
        for channel in image_channels:
            if channel_index == 2:
                continue
            if channel_index == 0:
                xCenter = int(primaryList[pointHolder][1 if lab == "mike" else 0]) 
                yCenter = int(primaryList[pointHolder][0 if lab == "mike" else 1]) 
            if channel_index == 1:
                xCenter = int(secondaryList[pointHolder][1 if lab == "mike" else 0])
                yCenter = int(secondaryList[pointHolder][0 if lab == "mike" else 1])

            if channel_index == 0:
                initial2DGuess = (initial_guess[1], window_radius, window_radius, initial_guess[2], initial_guess[2])
            if channel_index == 1:
                initial2DGuess = (initial_guess[1], window_radius, window_radius, initial_guess[2], initial_guess[2])

            fit_parameters_per_slice = []
            slice_indices = []
            slice_intensities = []
            x = np.linspace(0, window_radius * 2 - 1, (window_radius * 2))
            y = np.linspace(0, window_radius * 2 - 1, (window_radius * 2))
            x, y = np.meshgrid(x, y)

            z_index = 0
            for stack in channel:
                fit_failed = False
                local_region = stack[(yCenter - window_radius):(yCenter + window_radius),
                                      (xCenter - window_radius):(xCenter + window_radius)]
                try:
                    parameters2D, covariance2D = opt.curve_fit(gaussian2D, (x, y),
                                                                  np.concatenate(local_region).ravel(),
                                                                  p0=initial2DGuess, maxfev=20000)
                    if not (0 <= parameters2D[1] <= window_radius * 2) or not \
                            (0 <= parameters2D[2] <= window_radius * 2) or (parameters2D[0] < 0):
                        fit_failed = True
                except (RuntimeError, ValueError) as e:
                    fit_failed = True
                if not fit_failed:
                    fit_parameters_per_slice.append(parameters2D)
                    slice_indices.append(z_index)
                    slice_intensities.append(local_region[min(window_radius*2 - 1, round(parameters2D[2]))]
                                    [min(window_radius*2 - 1, round(parameters2D[1]))])
                    peakFitted = gaussian2D((x, y), *parameters2D)
                    fig, ax = plt.subplots(1, 1)
                    ax.imshow(local_region, cmap=cm.binary, origin='lower',
                          extent=(x.min(), x.max(), y.min(), y.max()))
                    ax.contour(x, y, peakFitted.reshape((window_radius * 2), (window_radius * 2)), 8, colors='r')
                    plt.gca().invert_yaxis()
                    plt.title(str(point[channel_index]) + " - " + str(z_index))
                    plt.xlabel("X")
                    plt.ylabel("Y")
                    if saveOption:
                        plt.savefig(analysisFolder + '/' + str(image_index) + "-" + str([point[0], point[1]]) + "-" + str(
                        z_index) + "-" + str(channel_index))
                    plt.close()
                else:
                    fit_parameters_per_slice.append([local_region[window_radius][window_radius], window_radius, window_radius, 0, 0])
                    slice_indices.append(z_index)
                    slice_intensities.append(local_region[window_radius][window_radius])

                z_index += 1

            tempHolder = ([stack[0] for stack in fit_parameters_per_slice])
            indexMax = max(range(len(tempHolder)), key=tempHolder.__getitem__)
            z = np.linspace(indexMax - 4, indexMax + 3, 8)
            zTemp = np.linspace(indexMax - 4, indexMax + 3, 100)
            initial1DGuess = (initial_guess[1], indexMax, initial_guess[2])

            if indexMax < (4) or (indexMax + 4) > image_shape[2]:
                indexMax = 4
            if (indexMax + 4) > image_shape[2]:
                indexMax = image_shape[2] - 4

            try:
                parameters1D, covariance1D = opt.curve_fit(gaussian1D, z,
                                                           tempHolder[indexMax - 4:indexMax + 4],
                                                           p0=initial1DGuess, maxfev=20000)
                print(parameters1D)
            except (RuntimeError, ValueError) as e:
                fit_valid = False

                break

            if parameters1D[1] < 0:
                fit_valid = False
                break

            plt.plot(slice_indices, [stack[0] for stack in fit_parameters_per_slice], label='2D Amplitude')
            plt.plot(zTemp, gaussian1D(zTemp, *parameters1D), color='red', label='1D Gaussian Fit')
            plt.plot(slice_indices, slice_intensities, color='green', label='Actual Intensity')
            plt.scatter(parameters1D[1], parameters1D[0], color='red')
            plt.text(len(slice_indices)/2 - 1, 7000,
                     "Z-" + str(round(parameters1D[1], 3)) + " A-" + str(round(parameters1D[0], 3)),
                     fontsize='15')
            plt.title(str([point[0], point[1]]) + "-Amplitude-Z-" + str(channel_index))
            plt.legend(loc="upper left")
            plt.ylim(0, initial_guess[1]*4)
            plt.xlim(0, image_shape[2]-1)
            if saveOption:
                plt.savefig(analysisFolder + '/' + str(image_index) + "-" + str([point[0], point[1]]) + "-Amplitude-Z-" + str(channel_index))
            plt.close()

            try:
                xCoordinate = fit_parameters_per_slice[math.floor(parameters1D[1])][1]*(math.ceil(parameters1D[1]) - parameters1D[1]) + \
                          fit_parameters_per_slice[math.ceil(parameters1D[1])][1]*(1 - math.ceil(parameters1D[1]) + parameters1D[1])
                yCoordinate = fit_parameters_per_slice[math.floor(parameters1D[1])][2]*(math.ceil(parameters1D[1]) - parameters1D[1]) + \
                          fit_parameters_per_slice[math.ceil(parameters1D[1])][2]*(1 - math.ceil(parameters1D[1]) + parameters1D[1])
            except (IndexError, TypeError) as error:
                print("x/y coordinate index error")
                fit_valid = False
                break




            if channel_index == 0:
                primary_coordinates_temp.append(round(xCoordinate, 3) + xCenter - window_radius)
                primary_coordinates_temp.append(round(yCoordinate, 3) + yCenter - window_radius)
                primary_coordinates_temp.append(round(parameters1D[1], 3))
                primary_coordinates_temp.append(str(xCenter) + "," + str(yCenter))
                primary_coordinates_temp.append(str(round(xCoordinate + xCenter - window_radius, 3)) + "," +
                                   str(round(yCoordinate + yCenter - window_radius, 3)) + "," +
                                   str(round(parameters1D[1], 3)))
            if channel_index == 1:
                secondary_coordinates_temp.append(round(xCoordinate, 3) + xCenter - window_radius)
                secondary_coordinates_temp.append(round(yCoordinate, 3) + yCenter - window_radius)
                secondary_coordinates_temp.append(round(parameters1D[1], 3))
                secondary_coordinates_temp.append(str(xCenter) + "," + str(yCenter))
                secondary_coordinates_temp.append(str(round(xCoordinate + xCenter - window_radius, 3)) + "," +
                                   str(round(yCoordinate + yCenter - window_radius, 3)) + "," +
                                   str(round(parameters1D[1], 3)))
            channel_index += 1

        if fit_valid:
            primary_coordinates[0].append(primary_coordinates_temp[0])
            primary_coordinates[1].append(primary_coordinates_temp[1])
            primary_coordinates[2].append(primary_coordinates_temp[2])
            secondary_coordinates[0].append(secondary_coordinates_temp[0])
            secondary_coordinates[1].append(secondary_coordinates_temp[1])
            secondary_coordinates[2].append(secondary_coordinates_temp[2])
            distanceTemp = (((primary_coordinates_temp[0] - secondary_coordinates_temp[0]) * voxel_size[0]) ** 2 +
                                  ((primary_coordinates_temp[1] - secondary_coordinates_temp[1]) * voxel_size[1]) ** 2 +
                                  ((primary_coordinates_temp[2] - secondary_coordinates_temp[2]) * voxel_size[2]) ** 2) ** 0.5
            uncorrected_distances.append(distanceTemp)
            if saveOption:
                sheet.append([primary_coordinates_temp[3], primary_coordinates_temp[4], distanceTemp])
                sheet.append([secondary_coordinates_temp[3], secondary_coordinates_temp[4], distanceTemp])
        else:
            print("DOUBLECHECK FAILURE DOUBLECHECK FAILURE DOUBLECHECK FAILURE DOUBLECHECK FAILURE")
            if saveOption:
                for filename in os.listdir(analysisFolder):
                    if filename.startswith(str([point[0], point[1]])):
                        os.remove(os.path.join(analysisFolder, filename))
                for filename in os.listdir(analysisFolder):
                    if filename.startswith(str([point[0], point[1]])):
                        os.remove(os.path.join(analysisFolder, filename))
            del activeRNAHolder[pointHolder - 1]
        pointHolder += 1

    return uncorrected_distances, primary_coordinates, secondary_coordinates, activeRNAHolder

def localizationPSFDistanceAutomatic3(image_channels, voxel_size, image_shape, primaryList, secondaryList, tertiaryList,
                                      analysisFolder, saveOption, sheet, image_index, initial_guess, lab):
    window_radius = initial_guess[0]
    primary_coordinates = [[], [], []]
    secondary_coordinates = [[], [], []]
    coordinatesTertiary = [[], [], []]

    pointHolder = 0
    print("calling auto")
    for point in primaryList:
        channel_index = 0
        primary_coordinates_temp = []
        secondary_coordinates_temp = []
        tertiaryTemp = []
        tripleCheck = True
        for channel in image_channels:
            if channel_index == 0:
                xCenter = int(primaryList[pointHolder][1 if lab == "mike" else 0]) 
                yCenter = int(primaryList[pointHolder][0 if lab == "mike" else 1]) 
            if channel_index == 1:
                xCenter = int(secondaryList[pointHolder][1 if lab == "mike" else 0])
                yCenter = int(secondaryList[pointHolder][0 if lab == "mike" else 1])
            if channel_index == 2:
                xCenter = int(tertiaryList[pointHolder][1 if lab == "mike" else 0])
                yCenter = int(tertiaryList[pointHolder][0 if lab == "mike" else 1])

            if channel_index == 0:
                initial2DGuess = (initial_guess[1], window_radius, window_radius, initial_guess[2], initial_guess[2])
            if channel_index == 1:
                initial2DGuess = (initial_guess[1], window_radius, window_radius, initial_guess[2], initial_guess[2])
            if channel_index == 2:
                initial2DGuess = (initial_guess[1], window_radius, window_radius, initial_guess[2], initial_guess[2])

            fit_parameters_per_slice = []
            slice_indices = []
            slice_intensities = []
            x = np.linspace(0, window_radius * 2 - 1, (window_radius * 2))
            y = np.linspace(0, window_radius * 2 - 1, (window_radius * 2))
            x, y = np.meshgrid(x, y)

            z_index = 0
            for stack in channel:
                fit_failed = False
                local_region = stack[(yCenter - window_radius):(yCenter + window_radius),
                                      (xCenter - window_radius):(xCenter + window_radius)]
                try:
                    parameters2D, covariance2D = opt.curve_fit(gaussian2D, (x, y),
                                                                  np.concatenate(local_region).ravel(),
                                                                  p0=initial2DGuess, maxfev=20000)
                    if not (0 <= parameters2D[1] <= window_radius * 2) or not \
                            (0 <= parameters2D[2] <= window_radius * 2) or (parameters2D[0] < 0):
                        fit_failed = True
                except (RuntimeError, ValueError) as e:
                    fit_failed = True
                if not fit_failed:
                    fit_parameters_per_slice.append(parameters2D)
                    slice_indices.append(z_index)
                    slice_intensities.append(local_region[min(window_radius*2 - 1, round(parameters2D[2]))]
                                    [min(window_radius*2 - 1, round(parameters2D[1]))])
                    peakFitted = gaussian2D((x, y), *parameters2D)
                    fig, ax = plt.subplots(1, 1)
                    ax.imshow(local_region, cmap=cm.binary, origin='lower',
                          extent=(x.min(), x.max(), y.min(), y.max()))
                    ax.contour(x, y, peakFitted.reshape((window_radius * 2), (window_radius * 2)), 8, colors='r')
                    plt.gca().invert_yaxis()

                    plt.xlabel("X")
                    plt.ylabel("Y")
                    if saveOption:
                        plt.savefig(analysisFolder + '/' + str(image_index) + "-" + str([point[0], point[1]]) + "-" + str(
                        z_index) + "-" + str(channel_index))
                    plt.close()
                else:
                    fit_parameters_per_slice.append([local_region[window_radius][window_radius], "idiot", "dumbass", 0, 0])
                    slice_indices.append(z_index)
                    slice_intensities.append(local_region[window_radius][window_radius])

                z_index += 1

            tempHolder = ([stack[0] for stack in fit_parameters_per_slice])
            indexMax = max(range(len(tempHolder)), key=tempHolder.__getitem__)
            z = np.linspace(indexMax - 4, indexMax + 3, 8)
            zTemp = np.linspace(indexMax - 4, indexMax + 3, 100)
            initial1DGuess = (initial_guess[1], indexMax, initial_guess[2])

            if indexMax < (4) or (indexMax + 4) > image_shape[2]:
                indexMax = 4
            if (indexMax + 4) > image_shape[2]:
                indexMax = image_shape[2] - 4

            try:
                parameters1D, covariance1D = opt.curve_fit(gaussian1D, z,
                                                           tempHolder[indexMax - 4:indexMax + 4],
                                                           p0=initial1DGuess, maxfev=20000)
                print(parameters1D)
            except (RuntimeError, ValueError) as e:
                print("z error")
                tripleCheck = False

            if parameters1D[1] < 0:
                tripleCheck = False

            plt.plot(slice_indices, [stack[0] for stack in fit_parameters_per_slice], label='2D Amplitude')
            plt.plot(zTemp, gaussian1D(zTemp, *parameters1D), color='red', label='1D Gaussian Fit')
            plt.plot(slice_indices, slice_intensities, color='green', label='Actual Intensity')
            plt.scatter(parameters1D[1], parameters1D[0], color='red')
            plt.text(len(slice_indices)/2 - 1, 7000,
                     "Z-" + str(round(parameters1D[1], 3)) + " A-" + str(round(parameters1D[0], 3)),
                     fontsize='15')
            plt.title(str([point[0], point[1]]) + "-Amplitude-Z-" + str(channel_index))
            plt.legend(loc="upper left")
            plt.ylim(0, initial_guess[1]*4)
            plt.xlim(0, image_shape[2]-1)
            if saveOption:
                plt.savefig(analysisFolder + '/' + str(image_index) + "-" + str([point[0], point[1]]) + "-Amplitude-Z-" + str(channel_index))
            plt.close()

            try:
                xCoordinate = fit_parameters_per_slice[math.floor(parameters1D[1])][1]*(math.ceil(parameters1D[1]) - parameters1D[1]) + \
                          fit_parameters_per_slice[math.ceil(parameters1D[1])][1]*(1 - math.ceil(parameters1D[1]) + parameters1D[1])
                yCoordinate = fit_parameters_per_slice[math.floor(parameters1D[1])][2]*(math.ceil(parameters1D[1]) - parameters1D[1]) + \
                          fit_parameters_per_slice[math.ceil(parameters1D[1])][2]*(1 - math.ceil(parameters1D[1]) + parameters1D[1])
            except (IndexError, TypeError) as error:
                print("x/y coordinate index error")
                tripleCheck = False
                continue




            if channel_index == 0:
                primary_coordinates_temp.append(round(xCoordinate, 3) + xCenter - window_radius)
                primary_coordinates_temp.append(round(yCoordinate, 3) + yCenter - window_radius)
                primary_coordinates_temp.append(round(parameters1D[1], 3))
                primary_coordinates_temp.append(str(xCenter) + "," + str(yCenter))
                primary_coordinates_temp.append(str(round(xCoordinate + xCenter - window_radius, 3)) + "," +
                                   str(round(yCoordinate + yCenter - window_radius, 3)) + "," +
                                   str(round(parameters1D[1], 3)))
            if channel_index == 1:
                secondary_coordinates_temp.append(round(xCoordinate, 3) + xCenter - window_radius)
                secondary_coordinates_temp.append(round(yCoordinate, 3) + yCenter - window_radius)
                secondary_coordinates_temp.append(round(parameters1D[1], 3))
                secondary_coordinates_temp.append(str(xCenter) + "," + str(yCenter))
                secondary_coordinates_temp.append(str(round(xCoordinate + xCenter - window_radius, 3)) + "," +
                                   str(round(yCoordinate + yCenter - window_radius, 3)) + "," +
                                   str(round(parameters1D[1], 3)))
            if channel_index == 2:
                tertiaryTemp.append(round(xCoordinate, 3) + xCenter - window_radius)
                tertiaryTemp.append(round(yCoordinate, 3) + yCenter - window_radius)
                tertiaryTemp.append(round(parameters1D[1], 3))
                tertiaryTemp.append(str(xCenter) + "," + str(yCenter))
                tertiaryTemp.append(str(round(xCoordinate + xCenter - window_radius, 3)) + "," +
                                   str(round(yCoordinate + yCenter - window_radius, 3)) + "," +
                                   str(round(parameters1D[1], 3)))
            channel_index += 1

        if tripleCheck:
            primary_coordinates[0].append(primary_coordinates_temp[0])
            primary_coordinates[1].append(primary_coordinates_temp[1])
            primary_coordinates[2].append(primary_coordinates_temp[2])
            secondary_coordinates[0].append(secondary_coordinates_temp[0])
            secondary_coordinates[1].append(secondary_coordinates_temp[1])
            secondary_coordinates[2].append(secondary_coordinates_temp[2])
            coordinatesTertiary[0].append(tertiaryTemp[0])
            coordinatesTertiary[1].append(tertiaryTemp[1])
            coordinatesTertiary[2].append(tertiaryTemp[2])

        else:
            print("TRIPLECHECK FAILURE TRIPLECHECK FAILURE TRIPLECHECK FAILURE TRIPLECHECK FAILURE")
            if saveOption:
                for filename in os.listdir(analysisFolder):
                    if filename.startswith(str([point[0], point[1]])):
                        os.remove(os.path.join(analysisFolder, filename))
                for filename in os.listdir(analysisFolder):
                    if filename.startswith(str([point[0], point[1]])):
                        os.remove(os.path.join(analysisFolder, filename))
        pointHolder += 1

    return primary_coordinates, secondary_coordinates, coordinatesTertiary