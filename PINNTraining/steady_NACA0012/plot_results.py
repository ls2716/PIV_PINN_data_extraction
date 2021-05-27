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
    data = scipy.io.loadmat("./Data/unsteadyNACA0012_full_field.mat")
    data_no_airfoil = scipy.io.loadmat("./Data/unsteadyNACA0012_no_airfoil.mat")
    x = data["x_data"].T
    y = data["y_data"].T
    x_no_airfoil = data_no_airfoil["x_data"].T
    y_no_airfoil = data_no_airfoil["y_data"].T
    u = data["u_data"].T
    v = data["v_data"].T
    p = data["p_data"].T
    return x, y, u, v, p, x_no_airfoil, y_no_airfoil


names_dict = {
    "u": r'$\overline{u}$',
    "v": r'$\overline{v}$',
    "p": r'$\overline{p}$',
    "uu": r"$\overline{u'u'}$",
    "uv": r"$\overline{u'v'}$",
    "vv": r"$\overline{v'v'}$",
}

ranges_dict = {
    "uu": (0,0.06),
    "uv": (-0.031, 0.045),
    "vv": (0,0.114),
}

plot_error = ['p', 'u', 'v']


if __name__=="__main__":
    
    case_name = sys.argv[1]

    x_data, y_data, u_data, v_data, p_data, x_domain, y_domain = read_data()

    zero_index = (x_data < 0) & (x_data > 0)
    zero_index = zero_index | ((u_data == 0) & (v_data == 0))
    no_data_index = zero_index

    x_ar = np.array([x_data[i][0] for i in range(x_data.shape[0]) if zero_index[i] and zero_index[i-2]])
    y_ar = np.array([y_data[i][0] for i in range(x_data.shape[0]) if zero_index[i] and zero_index[i-2]])



    airfoil_points = read_airfoil("Data/points_ok.dat")
    airfoil_array = np.array(airfoil_points)
    airfoil_array = rotate_points(
        airfoil_array[:, 0], airfoil_array[:, 1], 0.5, 0, -5 / 180 * math.pi
    )

    airfoil_points = airfoil_array.tolist()

    #domain vertices
    v_ld = [-0.3, -0.6]
    v_ru = [2.7, 0.6]


    Nx = int((v_ru[0]-v_ld[0])*500)+1
    Ny = int((v_ru[1]-v_ld[1])*500)+1
    print('Nx', Nx, 'Ny', Ny)

    figsize = (8,3)

    predictions = scipy.io.loadmat(f"./{case_name}/results.mat")
    true_data = scipy.io.loadmat("./Data/unsteadyNACA0012_full_field.mat")

    x_piv_super = np.linspace(-0.3, 2.7, 16)
    y_piv_super = np.linspace(-0.6, 0.6, 7)
    super_points = []
    for x in x_piv_super:
        for y in y_piv_super:
            super_points.append([x,y])
    super_points = np.array(super_points)

    x_plot = np.linspace(v_ld[0], v_ru[0], Nx)
    y_plot = np.linspace(v_ld[1], v_ru[1], Ny)

    X, Y = np.meshgrid(x_plot, y_plot)
    keys_to_plot = predictions.keys()
    keys_to_plot = ["u_star", "v_star"]
    keys_to_plot = ["p_star"]

    for key in predictions.keys():
        if key[0]=='_':
            continue
        if key not in keys_to_plot:
            continue
        
        to_plot = deepcopy(predictions[key])
        to_plot[no_data_index] = to_plot[no_data_index]*0
        to_plot = to_plot.T.reshape(Nx, Ny).T
        plot_range = ranges_dict.get(key.split('_')[0], None)
        plt.figure(figsize=figsize)
        if plot_range is not None:
            plt.pcolor(X, Y, to_plot, vmin=plot_range[0], vmax=plot_range[1])
        else:
            plt.pcolor(X, Y, to_plot)
        cb = plt.colorbar(label=names_dict.get(key.split('_')[0], 'not_found'))
        print(cb.get_ticks())
        cb.ax.set_yticklabels(["%.2f" %i for i in cb.get_ticks()[1::2]])
        plt.scatter(x_ar, y_ar, s=0.05, c='white')
        plt.plot(airfoil_array[:,0], airfoil_array[:,1], lw=1, c='black')
        plt.xlabel('x/c')
        plt.ylabel('y/c')
        axes=plt.gca()
        # axes.xaxis.label.set_color('white')
        # axes.yaxis.label.set_color('white')
        # [t.set_color('white') for t in axes.xaxis.get_ticklabels()]
        # [t.set_color('white') for t in axes.yaxis.get_ticklabels()]
        axes.set_aspect(1)
        plt.tight_layout()
        # plt.show()
        plt.savefig(os.path.join(f'{case_name}','plots',
                                 f'{key}.png'), dpi=400)
        plt.close()

        if key.split('_')[0] in plot_error:
            true_plot = true_data[key.split('_')[0]+'_data'].T.reshape(Nx, Ny).T
            plt.figure(figsize=figsize)
            print('Max error', np.max(np.abs(to_plot-true_plot)))
            plt.pcolor(X, Y, np.abs(to_plot-true_plot), vmin=0, vmax=0.065)
            cb2 = plt.colorbar(label=names_dict.get(key.split('_')[0], 'not_found')+' abs error')
            print(cb2.get_ticks())
            cb2.ax.set_yticklabels(["%.2f" %i for i in cb2.get_ticks()])
            plt.scatter(super_points[:,0], super_points[:,1], c='red', s=10)
            plt.scatter(x_ar, y_ar, s=0.05, c='white')
            plt.plot(airfoil_array[:,0], airfoil_array[:,1], lw=1, c='black')
            plt.xlabel('x/c')
            plt.ylabel('y/c')
            axes=plt.gca()
            axes.xaxis.label.set_color('white')
            axes.yaxis.label.set_color('white')
            [t.set_color('white') for t in axes.xaxis.get_ticklabels()]
            [t.set_color('white') for t in axes.yaxis.get_ticklabels()]
            axes.set_aspect(1)
            plt.tight_layout()
            plt.savefig(os.path.join(f'{case_name}','plots',
                                    f'{key}_error.png'), dpi=400)
            plt.close()

