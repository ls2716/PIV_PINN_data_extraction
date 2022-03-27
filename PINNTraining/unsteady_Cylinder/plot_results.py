from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os


import matplotlib.pyplot as plt


fs = 20
plt.rc('font', size=fs) #controls default text size
plt.rc('axes', titlesize=fs) #fontsize of the title
plt.rc('axes', labelsize=fs) #fontsize of the x and y labels
plt.rc('xtick', labelsize=fs) #fontsize of the x tick labels
plt.rc('ytick', labelsize=fs) #fontsize of the y tick labels
plt.rc('legend', fontsize=fs) #fontsize of the legend

import numpy as np
import math
import sys
import scipy.io
from copy import deepcopy

def rotate_points(x, y, x_0, y_0, dtheta):
    x = x - x_0
    y = y - y_0
    r = np.sqrt(x ** 2 + y ** 2)
    theta = np.arccos(x / r)
    theta[y < 0] = -theta[y < 0] + 2 * math.pi
    theta += dtheta
    x = r * np.cos(theta) + x_0
    y = r * np.sin(theta) + y_0
    return np.hstack((x[:, None], y[:, None]))


# airfoil geometry
def read_airfoil(filename):
    with open(filename, "r") as f:
        lines = f.readlines()
    points = [item.strip().split() for item in lines]
    points = [[float(item[0]), float(item[1])] for item in points]
    return points


def read_data():
    data = scipy.io.loadmat("./Data/unsteadyCylinder_full_field.mat")
    data_no_airfoil = scipy.io.loadmat("./Data/unsteadyCylinder_no_cylinder.mat")
    x = data["x_data"].T
    y = data["y_data"].T
    x_no_airfoil = data_no_airfoil["x_data"].T
    y_no_airfoil = data_no_airfoil["y_data"].T
    u = data["u_data"].T
    v = data["v_data"].T
    p = data["p_data"].T
    uu = data["uu_data"].T
    uv = data["uv_data"].T
    vv = data["vv_data"].T
    return x, y, u, v, p, uu, uv, vv, x_no_airfoil, y_no_airfoil


names_dict = {
    "u": r'$\overline{u}$',
    "v": r'$\overline{v}$',
    "p": r'$\overline{p}$',
    "uu": r"$\overline{u'u'}$",
    "uv": r"$\overline{u'v'}$",
    "vv": r"$\overline{v'v'}$",
    "fx": r"$f_x$",
    "fy": r"$f_x$",
    "curlf": r"$\nabla \times \mathbf{f}$",
    "curlfvar": r"$\nabla \times \mathbf{f}$",
    "curlfalt": r"$\nabla \times \mathbf{f}$",
    "curlfalt1st": r"convective term",
    "curlfalt2nd": r"viscous term",
}

ranges_dict = {
    "uu": (0,0.192686),
    "uv": (-0.106266, 0.106266),
    "vv": (0,0.391006),
    "curlf": (-2.1125,2.1125),
    "curlfalt": (-2.1125,2.1125),
    "curlfalt_1st": (-8.622697755466064, 17.40720578672134),
    "curlfalt_2nd": (-9.661458333333584, 9.893229166666492)
}

plot_error = ['p', 'u', 'v']#, 'curlfalt', 'curlfalt1st', 'curlfalt2nd']


if __name__=="__main__":
    
    case_name = sys.argv[1]

    x_data, y_data, u_data, v_data, p_data, uu_data, uv_data, vv_data, x_domain, y_domain = read_data()

    zero_index = (x_data < 0) & (x_data > 0)
    zero_index = zero_index | ((u_data == 0) & (v_data == 0))
    no_data_index = zero_index

    x_ar = np.array([x_data[i][0] for i in range(x_data.shape[0]) if zero_index[i] and zero_index[i-2]])
    y_ar = np.array([y_data[i][0] for i in range(x_data.shape[0]) if zero_index[i] and zero_index[i-2]])



    theta = np.linspace(0, 2*math.pi, 100)[:, None]
    print(theta.shape)
    R = 0.5
    cylinder_array = np.hstack((R*np.cos(theta), R*np.sin(theta)))


    #domain vertices
    v_ld = [-1, -1.5]
    v_ru = [3, 1.5]


    Nx = int((v_ru[0]-v_ld[0])*500)+1
    Ny = int((v_ru[1]-v_ld[1])*500)+1
    print('Nx', Nx, 'Ny', Ny)

    figsize = (8,5)

    predictions = scipy.io.loadmat(f"./{case_name}/results.mat")
    true_data = scipy.io.loadmat("./Data/unsteadyCylinder_full_field.mat")
    # true_velo_forcings = scipy.io.loadmat("./Data/forcing_velo.mat")
    # true_data.update(true_velo_forcings)

    x_piv_super = np.linspace(-1, 3, int(4/0.05)+1)
    y_piv_super = np.linspace(-1.5, 1.5, int(3/0.05)+1)

    x_piv_super = np.linspace(-1, 3, int(4/0.6)+1)
    y_piv_super = np.linspace(-1.5, 1.5, int(3/0.6)+1)

    x_piv_super = np.linspace(-1, 3, int(4/1)+1)
    y_piv_super = np.linspace(-1.5, 1.5, int(3/1)+1)

    super_points = []
    for x in x_piv_super:
        for y in y_piv_super:
            super_points.append([x,y])
    super_points = np.array(super_points)
    print(super_points)
    # plt.scatter(super_points[:,0], super_points[:,1])
    # plt.show()

    x_piv = np.linspace(-1,3, 500)
    y_piv = np.linspace(-1.5, 1.5, 300)
    x_m = np.linspace(-1,3, 21)
    y_m = np.linspace(-1.5,1.5, 16)
    uw_points = []
    wv_points = []
    for y in y_m:
        for x in x_piv:
            uw_points.append([x,y])
            uw_points.append([x,y])
    for x in x_m:
        for y in y_piv:
            wv_points.append([x,y])
            wv_points.append([x,y])
    uw_points = np.array(uw_points)
    wv_points = np.array(wv_points)
    

    # plt.figure(figsize=figsize)
    # plt.scatter(uw_points[:,0], uw_points[:,1], s=5, label=r"$\overline{u}$, $\overline{u'u'}$ PIV data")
    # plt.scatter(wv_points[:,0], wv_points[:,1], s=5, label=r"$\overline{v}$, $\overline{v'v'}$ PIV data")
    # plt.scatter(x_ar, y_ar, s=0.05, c='white', label='_')
    # plt.plot(cylinder_array[:,0], cylinder_array[:,1], lw=1.5, c='black', label='_')
    # plt.legend(loc=(1.05, 0.35))
    # plt.xlabel('x/c')
    # plt.ylabel('y/c')
    # axes=plt.gca()
    # axes.set_aspect(1)
    # plt.tight_layout()
    # plt.savefig(os.path.join(f'{case_name}','plots',
    #                             f'training_points.png'), dpi=400)
    # plt.close()
    # exit(0)

    x_plot = np.linspace(v_ld[0], v_ru[0], Nx)
    y_plot = np.linspace(v_ld[1], v_ru[1], Ny)

    X, Y = np.meshgrid(x_plot, y_plot)

    keys_to_plot = predictions.keys()
    keys_to_plot = ["u_star", "v_star"]
    # keys_to_plot = ["vv_star", "uv_star"]
    # keys_to_plot = ["curlf"]
    # keys_to_plot = ["curlfalt","curlfalt1st","curlfalt2nd"]

    for key in predictions.keys():
        if key[0] in ['_', 'x', 'y']:
            continue
        if key not in keys_to_plot:
            continue
        to_plot = deepcopy(predictions[key])
        to_plot[no_data_index] = to_plot[no_data_index]*0
        to_plot = to_plot.T.reshape(Nx, Ny).T
        plot_range = ranges_dict.get(key.split('_')[0], None)
        plt.figure(figsize=figsize)
        if plot_range is not None:
            plt.pcolor(X, Y, -to_plot, vmin=plot_range[0], vmax=plot_range[1])
        else:
            plt.pcolor(X, Y, -to_plot)
        # cb = plt.colorbar(label=names_dict.get(key.split('_')[0], 'not_found'))
        # cb.ax.set_yticklabels(["%.2f" %i for i in cb.get_ticks()])
        plt.scatter(x_ar, y_ar, s=0.05, c='white')
        plt.plot(cylinder_array[:,0], cylinder_array[:,1], lw=1.5, c='black')
        plt.xlabel('x/c')
        plt.ylabel('y/c')
        if key=='curlfalt1st':
            plt.title(r"(a) $\mathbf{f}$ form., no inlet BC")
        axes=plt.gca()
        axes.xaxis.label.set_color('white')
        [t.set_color('white') for t in axes.xaxis.get_ticklabels()]
        if key!='curlfalt':
            axes.yaxis.label.set_color('white')
            [t.set_color('white') for t in axes.yaxis.get_ticklabels()]
        axes.set_aspect(1)
        plt.tight_layout()
        plt.savefig(os.path.join(f'{case_name}','plots',
                                 f'{key}.png'), dpi=400)
        plt.close()

        if key.split('_')[0] in plot_error:
            true_plot = true_data[key.split('_')[0]+'_data'].T.reshape(Nx, Ny).T
            plt.figure(figsize=figsize)
            print('Max error', np.max(np.abs(to_plot-true_plot)))
            f_max = np.percentile(np.abs(to_plot-true_plot), 99)
            # plt.title(r"(a) $\mathbf{f}$ form., "+names_dict.get(key.split('_')[0]))
            plt.pcolor(X, Y, np.abs(to_plot-true_plot), vmin=0, vmax=f_max)
            # plt.pcolor(X, Y, np.abs(to_plot-true_plot))
            name = names_dict[key.split('_')[0]]
            plt.colorbar(label=f'{name} abs error')
            plt.scatter(super_points[:,0], super_points[:,1], s=40, c='red')
            plt.scatter(x_ar, y_ar, s=0.05, c='white')
            plt.plot(cylinder_array[:,0], cylinder_array[:,1], lw=1.5, c='black')
            plt.xlabel('x/c')
            plt.ylabel('y/c')
            axes=plt.gca()
            if key=="v_star":
                axes.yaxis.label.set_color('white')
                [t.set_color('white') for t in axes.yaxis.get_ticklabels()]
            # axes.xaxis.label.set_color('white')
            # [t.set_color('white') for t in axes.xaxis.get_ticklabels()]
            
            axes.set_aspect(1)
            plt.tight_layout()
            plt.savefig(os.path.join(f'{case_name}','plots',
                                    f'{key}_error.png'), dpi=400)
            plt.close()


