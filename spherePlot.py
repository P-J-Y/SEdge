import datetime

import h5py
import pandas as pd
from matplotlib import cm, colors
from mpl_toolkits.mplot3d import Axes3D
import loadData
import numpy as np
import matplotlib.pyplot as plt
import plotly.offline as py
import plotly.graph_objects as go

def plotSphere(edges,thetas,Lr,tlag=27,ttot=27):
    timepoint_num = edges.shape[1]
    tlag_num = timepoint_num*tlag//ttot
    phis = np.linspace(2*np.pi*(tlag_num-1)/(timepoint_num-1),0,tlag_num)
    p,t = np.meshgrid(phis,thetas)
    X = Lr*np.sin(t)*np.cos(p)
    Y = Lr*np.sin(t)*np.sin(p)
    Z = Lr*np.cos(t)
    colorfunction = edges[:,:tlag_num]
    plot = go.Figure()
    plot.add_trace(go.Scatter3d(x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
                                mode='markers',
                                marker=dict(size=2,
                                            color=colorfunction.flatten(),
                                            colorscale='rainbow',
                                            showscale=True,
                                            colorbar=dict(title='R=1.1R0',
                                                          # tickvals=[3, 4, 5, 6],
                                                          # ticktext=['10^3', '10^4', '10^5', '10^6']
                                                          # dtick='log',
                                                          # exponentformat='power',
                                                          ),
                                            # cmax=6,
                                            # cmin=3,
                                            opacity=0.2),
                                name='193A',
                                ))
    py.plot(plot, filename='figure/test3.html', image='svg')
    # norm = colors.Normalize(vmin=0, vmax=255, clip=False)
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(
    #     X, Y, Z,
    #     c=colorfunction,
    #     s=1,
    # )
    # plt.show()
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
# file = h5py.File('data/lredgedata/edgedata3.h5','w')
# file.create_dataset('ledges',data=np.array(ledges))
# file.create_dataset('redges',data=np.array(redges))
# file.create_dataset('lthetas',data=np.array(lthetas))
# file.create_dataset('rthetas',data=np.array(rthetas))
# file.close()
# file = h5py.File('data/lredgedata/edgedata3.h5','r')
# lthetas = np.array(file['lthetas'])
# rthetas = np.array(file['rthetas'])
# ledges = np.array(file['ledges'])
# redges = np.array(file['redges'])
# plotSphere(ledges,lthetas-np.pi/2,1.1,tlag=27,ttot=27)

def plotSphere3D(edges,thetas,rs,tlag=27,ttot=27):
    timepoint_num = edges.shape[1]
    tlag_num = timepoint_num*tlag//ttot
    phis = np.linspace(2*np.pi*(tlag_num-1)/(timepoint_num-1),0,tlag_num)
    p,t,r = np.meshgrid(phis,thetas,rs[:10])
    X = r*np.sin(t)*np.cos(p)
    Y = r*np.sin(t)*np.sin(p)
    Z = r*np.cos(t)
    colorfunction = edges[:,:tlag_num,:]
    plot = go.Figure()
    plot.add_trace(go.Scatter3d(x=X[::2,::2,:].flatten(),
                                y=Y[::2,::2,:].flatten(),
                                z=Z[::2,::2,:].flatten(),
                                mode='markers',
                                marker=dict(size=2,
                                            color=colorfunction[::2,::2,:].flatten(),
                                            colorscale='rainbow',
                                            showscale=True,
                                            colorbar=dict(title='...',
                                                          # tickvals=[3, 4, 5, 6],
                                                          # ticktext=['10^3', '10^4', '10^5', '10^6']
                                                          # dtick='log',
                                                          # exponentformat='power',
                                                          ),
                                            # cmax=6,
                                            # cmin=3,
                                            opacity=0.1),
                                name='193A',
                                ))
    py.plot(plot, filename='figure/test4.html', image='svg')
    print('testing')

file = h5py.File('data/lredgedata/edgedata4.h5','r')
lthetas = np.array(file['lthetas'])
rthetas = np.array(file['rthetas'])
ledges = np.array(file['ledges'])
redges = np.array(file['redges'])
r0 = 1630
rs = np.array(file['rs'])/r0
plotSphere3D(ledges,lthetas-np.pi/2,rs)