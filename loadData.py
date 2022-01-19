import time

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
# testt = datetime.datetime.strptime('2020-01-01T00:00:00', '%Y-%m-%dT%H:%M:%S')
# observatory = 'SDO'
# instrument = 'AIA'
# measurement = '193'
# themap = getMap(testt,observatory,instrument,measurement)
# plt.figure()
# plt.imshow(themap.data,origin='lower')
# r0 = 1630
# r = 1.1*r0
# thetas = np.arange(0,1.01,0.01)*np.pi*2
# xs = r*np.cos(thetas)+2048
# ys = r*np.sin(thetas)+2048
# plt.plot(xs,ys)
# plt.axis('equal')
# plt.show()
# print('testing')

def getEdge(xs,ys,amap,X,Y):
    points = [[xs[i],ys[i]] for i in range(len(xs))]
    f = RegularGridInterpolator((X,Y), amap.data.T)
    aEdge = f(points)
    return aEdge

# X = np.arange(4096)+0.5
# Y = np.arange(4096)+0.5
# r0 = 1630
# x0 = 2048
# y0 = 2048
# r = 0.8*r0
# lthetas = np.linspace(np.pi/2,3*np.pi/2,4096)
# rthetas = np.linspace(-np.pi/2,np.pi/2,4096)
# lxs = x0 + r*np.cos(lthetas)
# lys = y0 + r * np.sin(lthetas)
# rxs = x0 + r*np.cos(rthetas)
# rys = y0 + r * np.sin(rthetas)
# # xs,ys = np.meshgrid(np.arange(0,4096,8)+0.5,np.arange(0,4096,8)+0.5)
# # xs = xs.reshape(-1)
# # ys = ys.reshape(-1)
# testt = datetime.datetime.strptime('2020-01-01T00:00:00', '%Y-%m-%dT%H:%M:%S')
# observatory = 'SDO'
# instrument = 'AIA'
# measurement = '193'
# themap = getMap(testt,observatory,instrument,measurement)
# thelEdge = getEdge(lxs,lys,themap,X,Y)
# therEdge = getEdge(rxs,rys,themap,X,Y)
# plt.figure()
# #plt.plot(thetas,theEdge)
# plt.scatter(rxs,rys,c=therEdge,marker=',',s=1)
# plt.axis('equal')
# plt.show()

def keep_connect(url="https://baidu.com"):
    connected = False
    while not connected:
        try:
            r = requests.get(url, timeout=5)
            code = r.status_code
            if code == 200:
                connected = True
                return True
            else:
                print("未连接，等待5s")
                time.sleep(5)
                continue
        except:
            print("未连接，等待5s")
            time.sleep(5)
            continue

def getEdges(X,Y,r,x0,y0,lthetas,rthetas,ts,
             observatory = 'SDO',
             instrument = 'AIA',
             measurement = '193',
             ):
    lxs = x0 + r * np.cos(lthetas)
    lys = y0 + r * np.sin(lthetas)
    rxs = x0 + r * np.cos(rthetas)
    rys = y0 + r * np.sin(rthetas)
    ledges = np.zeros([len(lthetas),len(ts)])
    redges = np.zeros([len(rthetas),len(ts)])
    for i,t in enumerate(ts):
        done = False
        while not done:
            try:
                themap = getMap(t, observatory, instrument, measurement)
                if themap is None:
                    done = True
                    continue
                ledges[:, i] = getEdge(lxs, lys, themap, X, Y)
                redges[:, i] = getEdge(rxs, rys, themap, X, Y)
                done = True
            except (RuntimeError, IOError):
                print("RuntimeError/IOError 检查网络连接是否正常")
                intc = keep_connect()
                print("网络连接正常 检查Helioviewer网站连接是否正常")
                url = 'https://helioviewer.org'
                hvc = keep_connect(url=url)
                print('连接正常，重新运行程序')
                dirname = 'C:/Users/pjy/sunpy/data'
                # 把最近下载的文件删除（因为这个文件很可能是坏的）
                dir_list = os.listdir(dirname)
                if dir_list:
                    dir_list = sorted(dir_list,
                                      key=lambda x: os.path.getctime(os.path.join(dirname, x)))
                    os.remove(dirname + '/' + dir_list[-1])
                continue

    return ledges,redges

# X = np.arange(4096)+0.5
# Y = np.arange(4096)+0.5
# r0 = 1630
# x0 = 2048
# y0 = 2048
# Lr = 1.1
# r = Lr*r0
# t1 = datetime.datetime.strptime('2020-01-01T00:00:00', '%Y-%m-%dT%H:%M:%S')
# t2 = datetime.datetime.strptime('2020-01-28T00:00:00', '%Y-%m-%dT%H:%M:%S')
# freq = '1h'
# ts = list(pd.date_range(t1, t2, freq=freq))
# lthetas = np.linspace(np.pi/2,3*np.pi/2,4096)
# rthetas = np.linspace(-np.pi/2,np.pi/2,4096)
# ledges,redges = getEdges(X,Y,r,x0,y0,lthetas,rthetas,ts)
#
# plt.figure()
# plt.imshow(ledges,origin='lower')
# plt.gca().set_aspect(0.01)
# plt.title('R={}R0'.format(Lr))


print('testing')