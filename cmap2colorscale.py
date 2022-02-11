# 最后画图使用plotly而不是matplotlib，二者的colormap格式不同
# 如果希望在plotly里面绘制太阳图像，就需要自己写一个colorscale，画出来才和sunpy里的一样
# 本程序可以把sunpy的map的cmap提取出来，转换成plotly的colorscale（并且可以自己调整一下不透明度）

from sunpy.net.helioviewer import HelioviewerClient
from sunpy.map import Map
import datetime
import numpy as np

hv = HelioviewerClient()


def getMap(t, observatory, instrument, measurement):
    '''
    见loadData程序里的getMap
    '''
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

def cmap2colorscale(cmap,
                    cmapAlpha,
                    ):
    '''
    把matplotlib的cmap转换成plotly的colorscale。如果不需要对不同灰度设置不同的不透明度，可以把程序中的A部分注释，替换成B部分
    :param cmap:matplotlib的cmap
    :param cmapAlpha:一个函数，接受一个0-1的灰度值，返回该灰度值的不透明度a
    :return:plotly的colorscale
    '''
    theCmapData = cmap._segmentdata
    # ---------------A：要设置alpha的-----------------------
    colorScale = [[theCmapData['red'][i][0].astype(np.float64),
                   'rgba({},{},{},{})'.format(theCmapData['red'][i][1].astype(np.float64) * 255.,
                                              theCmapData['green'][i][1].astype(np.float64) * 255.,
                                              theCmapData['blue'][i][1].astype(np.float64) * 255.,
                                              # 0.5,
                                              cmapAlpha((theCmapData['red'][i][1].astype(np.float64) +
                                                         theCmapData['green'][i][1].astype(np.float64) +
                                                         theCmapData['blue'][i][1].astype(np.float64)) / 3.)
                                              )]
                  for i in range(len(theCmapData['red']))]
    # ---------------B：不需要设置alpha的-------------------
    # colorScale = [[theCmapData['red'][i][0].astype(np.float64),
    #                'rgb({},{},{})'.format(theCmapData['red'][i][1].astype(np.float64) * 255.,
    #                                           theCmapData['green'][i][1].astype(np.float64) * 255.,
    #                                           theCmapData['blue'][i][1].astype(np.float64) * 255.)]
    #               for i in range(len(theCmapData['red']))]
    return colorScale


testt = datetime.datetime.strptime('2020-01-01T00:00:00', '%Y-%m-%dT%H:%M:%S')
observatory = 'SDO'
instrument = 'AIA'
measurement = '193'
themap = getMap(testt,observatory,instrument,measurement)
# theCmapData = themap.cmap._segmentdata
def cmapAlpha(grayValue):
    if grayValue<=0.01:
        return 0
    elif grayValue<=0.05:
        return 0.05
    elif grayValue<=0.1:
        return 0.1
    # elif grayValue<=0.7:
    #     return 0.4
    else:
        return 0.9+0.1*(grayValue)
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