import time

import h5py
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from sunpy.net.helioviewer import HelioviewerClient
from sunpy.map import Map
import datetime
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
import requests
import os

hv = HelioviewerClient()


def getMap(t, observatory, instrument, measurement):
    file = hv.download_jp2(t,
                           observatory=observatory,
                           instrument=instrument,
                           measurement=measurement)
    themap = Map(file)
    mapt = datetime.datetime.strptime(themap.date.value, '%Y-%m-%dT%H:%M:%S.%f')
    if abs(mapt - t) > datetime.timedelta(minutes=10):
        print("No map loaded, t={}, {}".format(t, measurement))
        return None
    return themap

def cmap2colorscale(cmap
                    # ,cmapAlpha
                    ):
    theCmapData = cmap._segmentdata
    # colorScale = [[theCmapData['red'][i][0].astype(np.float64),
    #                'rgba({},{},{},{})'.format(theCmapData['red'][i][1].astype(np.float64) * 255.,
    #                                           theCmapData['green'][i][1].astype(np.float64) * 255.,
    #                                           theCmapData['blue'][i][1].astype(np.float64) * 255.,
    #                                           # 0.5,
    #                                           cmapAlpha((theCmapData['red'][i][1].astype(np.float64) +
    #                                                      theCmapData['green'][i][1].astype(np.float64) +
    #                                                      theCmapData['blue'][i][1].astype(np.float64)) / 3.)
    #                                           )]
    #               for i in range(len(theCmapData['red']))]
    colorScale = [[theCmapData['red'][i][0].astype(np.float64),
                   'rgb({},{},{})'.format(theCmapData['red'][i][1].astype(np.float64) * 255.,
                                              theCmapData['green'][i][1].astype(np.float64) * 255.,
                                              theCmapData['blue'][i][1].astype(np.float64) * 255.)]
                  for i in range(len(theCmapData['red']))]
    return colorScale


testt = datetime.datetime.strptime('2020-01-01T00:00:00', '%Y-%m-%dT%H:%M:%S')
observatory = 'SDO'
instrument = 'AIA'
measurement = '193'
themap = getMap(testt,observatory,instrument,measurement)
# theCmapData = themap.cmap._segmentdata
# def cmapAlpha(grayValue):
#     if grayValue<=0.01:
#         return 0
#     elif grayValue<=0.05:
#         return 0.05
#     elif grayValue<=0.1:
#         return 0.1
#     # elif grayValue<=0.7:
#     #     return 0.4
#     else:
#         return 0.9+0.1*(grayValue)
# colorScale = [[theCmapData['red'][i][0].astype(np.float64),
#                'rgba({},{},{},{})'.format(theCmapData['red'][i][1].astype(np.float64)*255.,
#               theCmapData['green'][i][1].astype(np.float64)*255.,
#               theCmapData['blue'][i][1].astype(np.float64)*255.,
#               # 0.5,
#               cmapAlpha((theCmapData['red'][i][1].astype(np.float64)+
#               theCmapData['green'][i][1].astype(np.float64)+
#               theCmapData['blue'][i][1].astype(np.float64))/3.)
#               )]
#               for i in range(len(theCmapData['red']))]
colorScale = cmap2colorscale(themap.cmap)
# np.savez('data/colorScaleDataForPlotly/colorscales3',cmap193=colorScale)
print('testing')