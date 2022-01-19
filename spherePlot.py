import datetime

import h5py
import pandas as pd
from matplotlib import cm, colors
import loadData
import numpy as np
import matplotlib.pyplot as plt

def plotSphere(edges,thetas,Lr,tlag=27,ttot=27):
    timepoint_num = edges.shape[1]
    tlag_num = timepoint_num*tlag//ttot
    phis = np.linspace(2*np.pi*(tlag_num-1)/(timepoint_num-1),0,tlag_num)
    p,t = np.meshgrid(phis,thetas)
    X = Lr*np.sin(t)*np.cos(p)
    Y = Lr*np.sin(t)*np.sin(p)
    Z = Lr*np.cos(t)
    colorfunction = edges[:,:tlag_num]
    # norm = colors.Normalize(vmin=0, vmax=255, clip=False)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(
        X, Y, Z,
        c=colorfunction,
        s=1,
    )
    plt.show()
    print('testing')

# X = np.arange(4096)+0.5
# Y = np.arange(4096)+0.5
# r0 = 1630
# x0 = 2048
# y0 = 2048
# Lr = 1.1
# r = Lr*r0
# t1 = datetime.datetime.strptime('2020-01-01T00:00:00', '%Y-%m-%dT%H:%M:%S')
# t2 = datetime.datetime.strptime('2020-01-28T00:00:00', '%Y-%m-%dT%H:%M:%S')
# freq = '2h'
# ts = list(pd.date_range(t1, t2, freq=freq))
# lthetas = np.linspace(np.pi/2,3*np.pi/2,256)
# rthetas = np.linspace(-np.pi/2,np.pi/2,256)
# ledges,redges = loadData.getEdges(X,Y,r,x0,y0,lthetas,rthetas,ts)
# file = h5py.File('C:/Users/pjy/Desktop/figure/lredgedata/edgedata3.h5','w')
# file.create_dataset('ledges',data=np.array(ledges))
# file.create_dataset('redges',data=np.array(redges))
# file.create_dataset('lthetas',data=np.array(lthetas))
# file.create_dataset('rthetas',data=np.array(rthetas))
# file.close()
file = h5py.File('C:/Users/pjy/Desktop/figure/lredgedata/edgedata3.h5','r')
lthetas = np.array(file['lthetas'])
rthetas = np.array(file['rthetas'])
ledges = np.array(file['ledges'])
redges = np.array(file['redges'])
plotSphere(ledges,lthetas-np.pi/2,1.1,tlag=20,ttot=27)
