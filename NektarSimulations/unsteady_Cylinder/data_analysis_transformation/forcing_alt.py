import numpy as np
import scipy.io
import os 
import math

import matplotlib.pyplot as plt
fs = 27
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

def d_x2(f, x):
    dx = x[1,0]-x[0,0]
    f_x = np.zeros(shape=f.shape)
    for i in range(f.shape[1]):
        f_x[0,i] = (f[1,i]-f[0,i])/dx
        f_x[1,i] = (f[2,i]-f[1,i])/dx
        tmp = f.shape[0]-1
        f_x[tmp,i] = (f[tmp,i]-f[tmp-1,i])/dx
        f_x[tmp-1,i] = (f[tmp-1,i]-f[tmp-2,i])/dx
    for j in range(2,f.shape[0]-2):
        f_x[j,:] = (f[j+2,:]-f[j-2,:])/(4*dx)
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

def d_y2(f, y):
    dy = y[0,1]-y[0,0]
    f_y = np.zeros(shape=f.shape)
    for i in range(f.shape[0]):
        f_y[i,0] = (f[i,1]-f[i,0])/dy
        f_y[i,1] = (f[i,2]-f[i,1])/dy
        tmp = f.shape[1]-1
        f_y[i,tmp] = (f[i,tmp]-f[i,tmp-1])/dy
        f_y[i,tmp-1] = (f[i-1,tmp]-f[i-2,tmp-1])/dy
    for j in range(2,f.shape[1]-2):
        f_y[:,j] = (f[:,j+2]-f[:,j-2])/(4*dy)
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

du_y = d_y2(u, y)
du_x = d_x2(u, x)
du_yy = d_y2(du_y, y)
du_xy = d_x2(du_y, x)
du_yyy = d_y(du_yy, y)
du_xxy = d_x(du_xy, x)

dv_x = d_x2(v, x)
dv_y = d_y2(v, y)
dv_xx = d_x2(dv_x, x)
dv_xy = d_y2(dv_x, y)
dv_xxx = d_x(dv_xx, x)
dv_xyy = d_y(dv_xy, y)

# data_dict = {
#             "u": u,
#             "v": v,
#             "dux": du_x,
#             "duy": du_y,
#             "dvx": dv_x,
#             "dvy": dv_y,
#             "duxy": du_xy,
#             "duyy": du_yy,
#             "dvxx": dv_xx,
#             "dvxy": dv_xy
#         }
# scipy.io.savemat(f"velocity_and_derivatives.mat", data_dict)

# for key in data_dict.keys():
#     field = data_dict[key]
#     f_min = np.percentile(field, 0.5)
#     f_max = np.percentile(field, 99.5)
#     print(key, f_min, f_max)

#     plt.figure(figsize=(8,9))
#     plt.pcolor(x,y, field, vmin=f_min, vmax=f_max)
#     # plt.title('forcing curl')
#     plt.colorbar(label=key, orientation='horizontal')
#     plt.scatter(x_ar, y_ar, s=0.05, c='white')
#     plt.plot(cylinder_array[:,0], cylinder_array[:,1], lw=1.5, c='black')
#     # plt.text(-0.2, 0.4, r"true field")
#     plt.xlabel('x/c')
#     plt.ylabel('y/c')
#     axes=plt.gca()
#     axes.set_aspect(1)
#     plt.tight_layout()
#     plt.savefig(f'unsteadyCylinder_velos/truth_{key}.png', dpi=400)

curlf = -(u*du_xy+v*du_yy - 1/150*(du_xxy+du_yyy)) + (u*dv_xx+v*dv_xy - 1/150*(dv_xxx+dv_xyy)) 
curlf_1st =  -(u*du_xy+v*du_yy)  + (u*dv_xx+v*dv_xy)
curlf_2nd = (1/150*(du_xxy+du_yyy)) - (1/150*(dv_xxx+dv_xyy))


curlf_min = np.percentile(curlf, 1)
curlf_max = np.percentile(curlf, 99)
curlf_min = -2.1125
curlf_max = 2.1125
print(curlf_min, curlf_max)

plt.figure(figsize=(8,9))
plt.pcolor(x,y, -curlf, vmin=curlf_min, vmax=curlf_max)
# plt.title('forcing curl')
plt.colorbar(label=r"$\nabla \times \mathbf{f}$ from velocities", orientation='horizontal')
plt.scatter(x_ar, y_ar, s=0.05, c='white')
plt.plot(cylinder_array[:,0], cylinder_array[:,1], lw=1.5, c='black')
# plt.text(-0.2, 0.4, r"true field")
plt.xlabel('x/c')
plt.ylabel('y/c')
axes=plt.gca()
axes.set_aspect(1)
plt.tight_layout()
plt.savefig('forcing_alt.png', dpi=400)

curlf_1st_min = np.percentile(curlf_1st, 0.75)
curlf_1st_max = np.percentile(curlf_1st, 99.75)
print(curlf_1st_min, curlf_1st_max)

plt.figure(figsize=(8,9))
plt.pcolor(x,y, -curlf_1st, vmin=curlf_1st_min, vmax=curlf_1st_max)
# plt.title('forcing curl')
plt.colorbar(label=r"$\overline{u}\,\overline{v}_{xx}+\overline{u}\,\overline{v}_{xy}-\overline{u}\,\overline{u}_{xy}-\overline{v}\,\overline{u}_{yy}$", orientation='horizontal')
# plt.text(0.9, 0.65, r"truth")
plt.scatter(x_ar, y_ar, s=0.05, c='white')
plt.plot(cylinder_array[:,0], cylinder_array[:,1], lw=1.5, c='black')
# plt.text(-0.2, 0.4, r"true field")
plt.title('truth')
plt.xlabel('x/c')
plt.ylabel('y/c')
axes=plt.gca()
axes.yaxis.label.set_color('white')
[t.set_color('white') for t in axes.yaxis.get_ticklabels()]
axes.set_aspect(1)
plt.tight_layout()
plt.savefig('forcing_alt_1st.png', dpi=400)

curlf_2nd_min = np.percentile(curlf_2nd, 1)
curlf_2nd_max = np.percentile(curlf_2nd, 99)
print(curlf_2nd_min, curlf_2nd_max)

plt.figure(figsize=(8,9))
plt.pcolor(x,y, -curlf_2nd, vmin=curlf_2nd_min, vmax=curlf_2nd_max)
# plt.title('forcing curl')
plt.colorbar(label=r"Re$^{-1}(\overline{u}_{xxy}+\overline{u}_{yyy}-\overline{v}_{xyy}-\overline{v}_{xxx})$", orientation='horizontal')
plt.scatter(x_ar, y_ar, s=0.05, c='white')
plt.plot(cylinder_array[:,0], cylinder_array[:,1], lw=1.5, c='black')
# plt.text(-0.2, 0.4, r"true field")
plt.xlabel('x/c')
plt.ylabel('y/c')
axes=plt.gca()
axes.yaxis.label.set_color('white')
[t.set_color('white') for t in axes.yaxis.get_ticklabels()]
axes.set_aspect(1)
plt.tight_layout()
plt.savefig('forcing_alt_2nd.png', dpi=400)

data_dict = {
            "curlfalt_data": curlf.reshape(-1,1),
            "curlfalt1st_data": curlf_1st.reshape(-1,1),
            "curlfalt2nd_data": curlf_2nd.reshape(-1,1)
        }
scipy.io.savemat(f"forcing_velo.mat", data_dict)