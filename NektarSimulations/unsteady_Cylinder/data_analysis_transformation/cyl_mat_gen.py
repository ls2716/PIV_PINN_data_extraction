import numpy as np
import scipy.io
import os 

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


data_all = np.array(data)

# print(data_all.shape)

# data_dict = {
#     "x_data": data_all[:,0],
#     "y_data": data_all[:,1],
#     "u_data": data_all[:,2],
#     "v_data": data_all[:,3],
#     "p_data": data_all[:,4],
#     "uu_data": data_all[:,5],
#     "uv_data": data_all[:,6],
#     "vv_data": data_all[:,7],
# }

# scipy.io.savemat("unsteadyCylinder_full_field.mat", data_dict)


# new_data = [data[0]]
# for i in range(1, len(data)-1):
#     if data[i][2]!=0 or data[i][3]!=0 \
#         or data[i+1][2]!=0 or data[i+1][3]!=0 \
#         or data[i-1][2]!=0 or data[i-1][3]!=0:
#         new_data.append(data[i])
# new_data.append(data[-1])
# new_data = np.array(new_data)

# data_dict = {
#     "x_data": new_data[:,0],
#     "y_data": new_data[:,1],
#     "u_data": new_data[:,2],
#     "v_data": new_data[:,3],
#     "p_data": new_data[:,4],
#     "uu_data": new_data[:,5],
#     "uv_data": new_data[:,6],
#     "vv_data": new_data[:,7],
# }

# scipy.io.savemat("unsteadyCylinder_no_cylinder.mat", data_dict)



try:
    os.mkdir('unsteadyCylinder')
except:
    pass

plt_data = np.array(data_all)
u_tmp = plt_data[:,2].reshape(2001,1501).T
v_tmp = plt_data[:,3].reshape(2001,1501).T
p_tmp = plt_data[:,4].reshape(2001,1501).T
uu_tmp = plt_data[:,5].reshape(2001,1501).T
uv_tmp = plt_data[:,6].reshape(2001,1501).T
vv_tmp = plt_data[:,7].reshape(2001,1501).T

x = np.linspace(-1, 3, 2001)
y = np.linspace(-1.5, 1.5, 1501)
X, Y = np.meshgrid(x, y)


figsize= (8,5)

# plt.figure(figsize=(10,5))
# # plt.title('unsteady Cylinder alpha field truth')
# plt.pcolor(X, Y, uu_tmp-vv_tmp)
# plt.colorbar(label='alpha')
# plt.xlabel('x/c')
# plt.ylabel('y/c')
# axes=plt.gca()
# axes.set_aspect(1)
# plt.tight_layout()
# plt.savefig('unsteadyCylinder/truth_alpha.png')
# plt.close()
# plt.figure(figsize=(10,5))
# # plt.title('unsteady Cylinder beta field truth')
# plt.pcolor(X, Y, uv_tmp)
# plt.colorbar(label='alpha')
# plt.xlabel('x/c')
# plt.ylabel('y/c')
# axes=plt.gca()
# axes.set_aspect(1)
# plt.tight_layout()
# plt.savefig('unsteadyCylinder/truth_beta.png')
# plt.close()



plt.figure(figsize=figsize)
# plt.title('unsteady Cylinder u field truth')
plt.pcolor(X, Y, u_tmp)
plt.colorbar(label='$u/U_{inf}$')
plt.xlabel('x/c')
plt.ylabel('y/c')
axes=plt.gca()
axes.set_aspect(1)
plt.tight_layout()
plt.savefig('unsteadyCylinder/truth_u.png')
plt.close()

plt.figure(figsize=figsize)
# plt.title('unsteady Cylinder v field truth')
plt.pcolor(X, Y, v_tmp)
plt.colorbar(label='$v/U_{inf}$')
plt.xlabel('x/c')
plt.ylabel('y/c')
axes=plt.gca()
axes.set_aspect(1)
plt.tight_layout()
plt.savefig('unsteadyCylinder/truth_v.png')
plt.close()

plt.figure(figsize=figsize)
# plt.title('unsteady Cylinder p field truth')
plt.pcolor(X, Y, p_tmp)
plt.colorbar(label='p')
plt.xlabel('x/c')
plt.ylabel('y/c')
axes=plt.gca()
axes.set_aspect(1)
plt.tight_layout()
plt.savefig('unsteadyCylinder/truth_p.png')
plt.close()

plt.figure(figsize=figsize)
# plt.title('unsteady Cylinder uu field truth')
plt.pcolor(X, Y, uu_tmp)
plt.colorbar(label='uu')
plt.xlabel('x/c')
plt.ylabel('y/c')
axes=plt.gca()
axes.set_aspect(1)
plt.tight_layout()
plt.savefig('unsteadyCylinder/truth_uu.png')
plt.close()

plt.figure(figsize=figsize)
# plt.title('unsteady Cylinder uv field truth')
plt.pcolor(X, Y, uv_tmp)
plt.colorbar(label='uv')
plt.xlabel('x/c')
plt.ylabel('y/c')
axes=plt.gca()
axes.set_aspect(1)
plt.tight_layout()
plt.savefig('unsteadyCylinder/truth_uv.png')
plt.close()

plt.figure(figsize=figsize)
# plt.title('unsteady Cylinder vv field truth')
plt.pcolor(X, Y, vv_tmp)
plt.colorbar(label='vv')
plt.xlabel('x/c')
plt.ylabel('y/c')
axes=plt.gca()
axes.set_aspect(1)
plt.tight_layout()
plt.savefig('unsteadyCylinder/truth_vv.png')
plt.close()