import numpy as np
import scipy.io
import os 
import math

import matplotlib.pyplot as plt

fs = 20
plt.rc('font', size=fs) #controls default text size
plt.rc('axes', titlesize=fs) #fontsize of the title
plt.rc('axes', labelsize=fs) #fontsize of the x and y labels
plt.rc('xtick', labelsize=fs) #fontsize of the x tick labels
plt.rc('ytick', labelsize=fs) #fontsize of the y tick labels
plt.rc('legend', fontsize=fs) #fontsize of the legend

data = []

with open('unsteadyCylinder.dat', 'r') as f:
    lines = f.readlines()
    for line in lines[3:3+2001*1501]:
        items = [float(item) for item in line.split()]
        # if not (items[2]==0 and items[3]==0):
        data.append(items)
    

print(data[:3])

data = np.array(data)
print(data.shape)

def d_x(f, x):
    dx = x[1,0]-x[0,0]
    f_x = np.zeros(shape=f.shape)
    for i in range(f.shape[1]):
        f_x[0,i] = (f[1,i]-f[0,i])/dx
        tmp = f.shape[0]-1
        f_x[tmp,i] = (f[tmp,i]-f[tmp-1,i])/dx
    for j in range(1,f.shape[0]-1):
        f_x[j,:] = (f[j+1,:]-f[j-1,:])/(2*dx)
    return f_x


def d_y(f, y):
    dy = y[0,1]-y[0,0]
    f_y = np.zeros(shape=f.shape)
    for i in range(f.shape[0]):
        f_y[i,0] = (f[i,1]-f[i,0])/dy
        tmp = f.shape[1]-1
        f_y[i,tmp] = (f[i,tmp]-f[i,tmp-1])/dy
    for j in range(1,f.shape[1]-1):
        f_y[:,j] = (f[:,j+1]-f[:,j-1])/(2*dy)
    return f_y

theta = np.linspace(0, 2*math.pi, 100)[:, None]
R = 0.5
cylinder_array = np.hstack((R*np.cos(theta), R*np.sin(theta)))


x = data[:,0]
y = data[:,1]
u = data[:,2]
v = data[:,3]
uu = data[:,5]
uv = data[:,6]
vv = data[:,7]

zero_index = (x < 0) & (x > 0)
zero_index = zero_index | ((u == 0) & (v == 0))
no_data_index = zero_index

x_ar = np.array([x[i] for i in range(x.shape[0]) if zero_index[i] and zero_index[i-2]])
y_ar = np.array([y[i] for i in range(x.shape[0]) if zero_index[i] and zero_index[i-2]])


x = x.reshape(2001,1501)
y = y.reshape(2001,1501)
u = u.reshape(2001,1501)
v = v.reshape(2001,1501)
uu = uu.reshape(2001,1501)
uv = uv.reshape(2001,1501)
vv = vv.reshape(2001,1501)

dx = 0.002
dy = 0.002

fx = -(d_x(uu, x) + d_y(uv, y))
fy = -(d_x(uv, x) + d_y(vv, y))


fx_min = np.percentile(fx, 1)
fx_max = np.percentile(fx, 99)
print(fx_min, fx_max)

fy_min = np.percentile(fy, 1)
fy_max = np.percentile(fy, 99)
print(fy_min, fy_max)


# fig,ax = plt.subplots(2,1,figsize=(8,10))
# p = ax[0].pcolor(x,y,fx, vmin=fx_min, vmax=fx_max)
# ax[0].title.set_text('$f_x$')
# ax[0].set_xlabel('x/c')
# ax[0].set_ylabel('y/c')
# plt.colorbar(p, ax=ax[0])
# ax[0].set_aspect('equal')
# p = ax[1].pcolor(x,y,fy, vmin=fy_min, vmax=fy_max)
# ax[1].title.set_text('$f_y$')
# ax[1].set_xlabel('x/c')
# ax[1].set_ylabel('y/c')
# plt.colorbar(p, ax=ax[1])
# ax[1].set_aspect('equal')
# plt.tight_layout()
# plt.savefig('forcing_xy.png', dpi=400)

f_curl = d_x(fy, x) - d_y(fx, y)

f_curl_min = np.percentile(f_curl, 0.1)
f_curl_max = np.percentile(f_curl, 99.9)
print(f_curl_min, f_curl_max)

plt.figure(figsize=(8,9))
plt.pcolor(x,y, -f_curl, vmin=f_curl_min, vmax=f_curl_max)
#plt.title('forcing curl')
plt.colorbar(label=r"$\nabla \times \mathbf{f}$", orientation='horizontal')
plt.scatter(x_ar, y_ar, s=0.05, c='white')
plt.plot(cylinder_array[:,0], cylinder_array[:,1], lw=1.5, c='black')
plt.xlabel('x/c')
plt.ylabel('y/c')
plt.text(-0.8, 1.2, r"true field")
axes=plt.gca()
axes.set_aspect(1)
plt.tight_layout()
plt.savefig('forcing.png', dpi=400)
