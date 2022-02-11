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
# r0 = 1600.4
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

def plotSphere3D(edges,thetas,rs,tlag=27,ttot=27,cmap='rainbow'):
    timepoint_num = edges.shape[1]
    tlag_num = timepoint_num*tlag//ttot
    phis = np.linspace(2*np.pi*(tlag_num-1)/(timepoint_num-1),0,tlag_num)
    p,t,r = np.meshgrid(phis,thetas,rs)
    X = r*np.sin(t)*np.cos(p)
    Y = r*np.sin(t)*np.sin(p)
    Z = r*np.cos(t)
    colorfunction = edges[:,:tlag_num,:]
    plot = go.Figure()
    for i in range(np.shape(edges)[2]):
        if i==0:
            showscale = True
            opacity = 0.9
        else:
            showscale = False
            opacity = 0.3

        # opacity = (1-i/15)*0.3+0.1
        # opacity = 0.1
        ###########scatter###############
        # plot.add_trace(go.Scatter3d(x=X[:, :, i].flatten(),
        #                             y=Y[:, :, i].flatten(),
        #                             z=Z[:, :, i].flatten(),
        #                             mode='markers',
        #                             marker=dict(size=2,
        #                                         color=colorfunction[:, :, i].flatten(),
        #                                         colorscale=cmap,
        #                                         # colorscale='greys',
        #                                         showscale=showscale,
        #                                         # colorbar=dict(title='...',
        #                                         #               # tickvals=[3, 4, 5, 6],
        #                                         #               # ticktext=['10^3', '10^4', '10^5', '10^6']
        #                                         #               # dtick='log',
        #                                         #               # exponentformat='power',
        #                                         #               ),
        #                                         cmax=255,
        #                                         cmin=0,
        #                                         opacity=opacity,
        #                                         symbol='square',
        #                                         ),
        #                             showlegend=False,
        #                             # name='193A',
        #                             ))
        ############surface
        trace = go.Surface(x=X[::8, ::4, i],
                           y=Y[::8, ::4, i],
                           z=Z[::8, ::4, i],
                           surfacecolor=colorfunction[::8, ::4, i],
                           colorscale=cmap,
                           cmax=255,
                           cmin=0,
                           opacity=opacity,
                           )
        trace.update(showscale=showscale)
        plot.add_trace(trace,
                       )
        plot.update_layout(
            paper_bgcolor="black",
            template='plotly_dark',
        )


    py.plot(plot, filename='figure/test10.html', image='svg')
    print('testing')

file = h5py.File('data/lredgedata/edgedata6.h5','r')
lthetas = np.array(file['lthetas'])
rthetas = np.array(file['rthetas'])
ledges = np.array(file['ledges'])
redges = np.array(file['redges'])
r0 = 1600.4
rs = np.array(file['rs'])/r0

# plt.figure()
# plt.imshow(ledges[:,:,19])
cmaps = np.load('data/colorScaleDataForPlotly/colorscales2.npz')
cmap193 = list(cmaps['cmap193'])
thecmap = [list([cmap193[i][0].astype(np.float64),cmap193[i][1]]) for i in range(len(cmap193))]
plotSphere3D(ledges[:,:,:],lthetas-np.pi/2,rs,tlag=14,cmap=thecmap)
print('testing')