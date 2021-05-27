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
    # uu = data["uu_data"].T
    # uv = data["uv_data"].T
    # vv = data["vv_data"].T
    return x, y, u, v, p, x_no_airfoil, y_no_airfoil


names_dict = {
    "u": r'$\overline{u}/U_{\infty}$',
    "v": r'$\overline{v}/U_{\infty}$',
    "p": r'$\overline{p}$',
    "uu": r"$\overline{u'u'}$",
    "uv": r"$\overline{u'v'}$",
    "vv": r"$\overline{v'v'}$",

}

if __name__=="__main__":
    
    try:
        os.mkdir('true_plots')
    except:
        pass

    x_data, y_data, u_data, v_data, p_data, x_domain, y_domain = read_data()

    zero_index = (x_data < 0) & (x_data > 0)
    zero_index = zero_index | ((u_data == 0) & (v_data == 0))
    no_data_index = zero_index

    x_ar = np.array([x_data[i][0] for i in range(x_data.shape[0]) if zero_index[i] and zero_index[i-2]])
    y_ar = np.array([y_data[i][0] for i in range(x_data.shape[0]) if zero_index[i] and zero_index[i-2]])



    airfoil_points = read_airfoil("Data/points_ok.dat")
    airfoil_array = np.array(airfoil_points)
    airfoil_array = rotate_points(
        airfoil_array[:, 0], airfoil_array[:, 1], 0.5, 0, -15 / 180 * math.pi
    )

    airfoil_points = airfoil_array.tolist()

    #domain vertices
    v_ld = [-0.3, -0.6]
    v_ru = [2.7, 0.6]


    Nx = int((v_ru[0]-v_ld[0])*500)+1
    Ny = int((v_ru[1]-v_ld[1])*500)+1
    print('Nx', Nx, 'Ny', Ny)


    Nx = int((v_ru[0]-v_ld[0])*500)+1
    Ny = int((v_ru[1]-v_ld[1])*500)+1
    print('Nx', Nx, 'Ny', Ny)

    figsize = (8,3)

    predictions = scipy.io.loadmat("./Data/unsteadyNACA0012_full_field.mat")

    x = predictions['x_data']
    y = predictions['y_data']

    x_plot = np.linspace(v_ld[0], v_ru[0], Nx)
    y_plot = np.linspace(v_ld[1], v_ru[1], Ny)

    X, Y = np.meshgrid(x_plot, y_plot)

    for key in predictions.keys():
        if key[0] in ['_', 'x', 'y']:
            continue
        to_plot = deepcopy(predictions[key]).T
        to_plot[no_data_index] = to_plot[no_data_index]*0
        to_plot = to_plot.T.reshape(Nx, Ny).T
        print(key)
        print(np.min(to_plot), np.max(to_plot))
        plt.figure(figsize=figsize)
        plt.pcolor(X, Y, to_plot)
        plt.colorbar(label=names_dict.get(key.split('_')[0], 'not_found'))
        plt.scatter(x_ar, y_ar, s=0.05, c='white')
        plt.plot(airfoil_array[:,0], airfoil_array[:,1], lw=1.5, c='black')
        plt.xlabel('x/c')
        plt.ylabel('y/c')
        axes=plt.gca()
        axes.set_aspect(1)
        plt.tight_layout()
        plt.savefig(os.path.join(f'true_plots',
                                 f'{key}.png'), dpi=400)
        plt.close()

