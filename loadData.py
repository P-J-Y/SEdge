# 获取画图所需的数据
# 更改#210行以在本机上运行程序

# 像素4096*4096
# 角宽度（helioprojecting）2457*2457
# 太阳角半径960，对应像素1600.4
# 但是这个并不是1700图上的边缘（边缘是1630）

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
    '''
    从helioviewer获取所需的map，如果找不到合适的map，返回None
    :param t: 时间，为datetime数据类型
    :param observatory: 飞船，一般是'SDO'
    :param instrument: 仪器，‘AIA’或者‘HMI’
    :param measurement: 通道，例如‘1700’，‘193’
    :return:所需的map，是sunpy的map数据类型。
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
# 测试
# testt = datetime.datetime.strptime('2020-01-01T00:00:00', '%Y-%m-%dT%H:%M:%S')
# observatory = 'SDO'
# instrument = 'AIA'
# measurement = '1700'
# themap = getMap(testt,observatory,instrument,measurement)
# plt.figure()
# plt.imshow(themap.data,origin='lower')
# r0 = 1600.4
# r = 1.*r0
# thetas = np.arange(0,1.01,0.01)*np.pi*2
# xs = r*np.cos(thetas)+2048
# ys = r*np.sin(thetas)+2048
# plt.plot(xs,ys)
# plt.axis('equal')
# plt.show()
# print('testing')

def getEdge(xs, ys, amap, X, Y):
    '''
    获得一条轨迹（由xs、ys确定）上的map灰度值
    :param xs:list或者np.array数据类型，单位是像素
    :param ys:单位是像素
    :param amap:sunpy的map数据类型
    :param X:和amap.data大小一致的矩阵（meshgrid得到的X坐标），单位是像素
    :param Y:单位是像素
    :return:轨迹上的map灰度值
    '''
    points = [[xs[i], ys[i]] for i in range(len(xs))]
    f = RegularGridInterpolator((X, Y), amap.data.T)
    aEdge = f(points)
    return aEdge

# X = np.arange(4096)+0.5
# Y = np.arange(4096)+0.5
# r0 = 1600.4
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
    '''
    此程序可以避免下载数据时出现网络连接问题，导致程序崩溃。遇到网络问题，程序可以不断尝试连接。
    :param url:所需要连接的网站。默认是百度。
    '''
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


def getEdges(X, Y, r, x0, y0, lthetas, rthetas, ts,
             observatory='SDO',
             instrument='AIA',
             measurement='193',
             dirname='C:/Users/pjy/sunpy/data',
             ):
    '''
    这个函数只能获取一圈边缘，如果想获取边缘的3D图像，请使用函数getEdges3D
    :param X: meshgrid得到和sunpy的map对应的X坐标
    :param Y:
    :param r: 所需边缘的半径，单位是像素
    :param x0: 太阳中心坐标，单位是像素
    :param y0: 太阳中心坐标，单位是像素
    :param lthetas: 弧度制，左边缘的角度，例如 np.linspace(np.pi/2,3*np.pi/2,4096)
    :param rthetas: 弧度制，右边缘的角度，例如 np.linspace(-np.pi/2,np.pi/2,4096)
    :param ts: datetime数据类型，需要采集边缘的时间
    :param observatory: ‘SDO’
    :param instrument:‘AIA'或者’HMI‘
    :param measurement: ’1700‘’193‘等
    :return: 左边缘，右边缘
    :dirname: sunpy存放下载文件的地址，每台主机是不同的
    '''
    # 左边缘和右边缘的坐标，单位是像素
    lxs = x0 + r * np.cos(lthetas)
    lys = y0 + r * np.sin(lthetas)
    rxs = x0 + r * np.cos(rthetas)
    rys = y0 + r * np.sin(rthetas)
    ledges = np.zeros([len(lthetas), len(ts)])
    redges = np.zeros([len(rthetas), len(ts)])
    for i, t in enumerate(ts):
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
                intc = keep_connect() #网络连接
                print("网络连接正常 检查Helioviewer网站连接是否正常")
                url = 'https://helioviewer.org'
                hvc = keep_connect(url=url) #helioviewer网站连接
                print('连接正常，重新运行程序')
                # 把最近下载的文件删除（因为这个文件很可能是坏的）
                dir_list = os.listdir(dirname)
                if dir_list:
                    dir_list = sorted(dir_list,
                                      key=lambda x: os.path.getctime(os.path.join(dirname, x)))
                    os.remove(dirname + '/' + dir_list[-1])
                continue

    return ledges, redges

# X = np.arange(4096)+0.5
# Y = np.arange(4096)+0.5
# r0 = 1600.4
# x0 = 2048
# y0 = 2048
# Lr = 1.
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

def getEdges3D(X, Y, x0, y0, lthetas, rthetas, ts, rs,
               observatory='SDO',
               instrument='AIA',
               measurement='193',
               dirname='C:/Users/pjy/sunpy/data',
               ):
    '''

    :param X: meshgrid得到和sunpy的map对应的X坐标
    :param Y:
    :param x0: 太阳中心坐标，单位是像素
    :param y0:
    :param lthetas: 弧度制，左边缘的角度，例如 np.linspace(np.pi/2,3*np.pi/2,4096)
    :param rthetas: 弧度制，右边缘的角度，例如 np.linspace(-np.pi/2,np.pi/2,4096)
    :param ts: datetime数据类型，需要采集边缘的时间
    :param rs: 所需边缘的半径构成的list，单位是像素
    :param observatory: 见sunpy的说明
    :param instrument:
    :param measurement:
    :param dirname: sunpy存放下载文件的地址，每台主机是不同的
    :return: 左边缘和右边缘，维度：θ角向*时间*径向
    '''
    layerNum = len(rs)
    ledges = np.zeros([len(lthetas), len(ts), layerNum])
    redges = np.zeros([len(rthetas), len(ts), layerNum])


    for i, t in enumerate(ts):
        done = False
        while not done:
            try:
                themap = getMap(t, observatory, instrument, measurement)
                if themap is None:
                    done = True
                    continue
                for j,r in enumerate(rs):
                    lxs = x0 + r * np.cos(lthetas)
                    lys = y0 + r * np.sin(lthetas)
                    rxs = x0 + r * np.cos(rthetas)
                    rys = y0 + r * np.sin(rthetas)
                    ledges[:, i, j] = getEdge(lxs, lys, themap, X, Y)
                    redges[:, i, j] = getEdge(rxs, rys, themap, X, Y)
                    done = True
            except (RuntimeError, IOError):
                print("RuntimeError/IOError 检查网络连接是否正常")
                intc = keep_connect()
                print("网络连接正常 检查Helioviewer网站连接是否正常")
                url = 'https://helioviewer.org'
                hvc = keep_connect(url=url)
                print('连接正常，重新运行程序')
                # 把最近下载的文件删除（因为这个文件很可能是坏的）
                dir_list = os.listdir(dirname)
                if dir_list:
                    dir_list = sorted(dir_list,
                                      key=lambda x: os.path.getctime(os.path.join(dirname, x)))
                    os.remove(dirname + '/' + dir_list[-1])
                continue
    return ledges, redges

if __name__=='__main__':
    X = np.arange(4096) + 0.5
    Y = np.arange(4096) + 0.5
    r0 = 1600.4
    x0 = 2048
    y0 = 2048
    t1 = datetime.datetime.strptime('2020-01-01T00:00:00', '%Y-%m-%dT%H:%M:%S')
    t2 = datetime.datetime.strptime('2020-01-28T00:00:00', '%Y-%m-%dT%H:%M:%S')
    freq = '30min'
    ts = list(pd.date_range(t1, t2, freq=freq))
    lthetas = np.linspace(np.pi / 2, 3 * np.pi / 2, 4096)
    rthetas = np.linspace(-np.pi / 2, np.pi / 2, 4096)
    Lr1 = 1.0
    Lr2 = 1.2
    layerNum = 21
    rs = np.linspace(Lr1, Lr2, layerNum) * r0
    ledges, redges = getEdges3D(X, Y, x0, y0, lthetas, rthetas, ts, rs)
    file = h5py.File('data/lredgedata/edgedata6.h5', 'w')
    file.create_dataset('ledges', data=np.array(ledges))
    file.create_dataset('redges', data=np.array(redges))
    file.create_dataset('lthetas', data=np.array(lthetas))
    file.create_dataset('rthetas', data=np.array(rthetas))
    file.create_dataset('rs', data=np.array(rs))
    # file.create_dataset('ts',data=np.array(ts))
    file.close()
    plt.figure()
    plt.imshow(ledges[:, :, 0], origin='lower')
    plt.gca().set_aspect(1)
    print('testing')

