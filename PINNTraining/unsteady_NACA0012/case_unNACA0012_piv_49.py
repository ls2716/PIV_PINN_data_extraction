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


import deepxde as dde

# from deepxde.backend import tf

from equations import RANSf0var2D, func_zeros
from utilities import set_directory, plot_train_points

# Additional functions


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
    uu = data["uu_data"].T
    uv = data["uv_data"].T
    vv = data["vv_data"].T
    return x, y, u, v, p, uu, uv, vv, x_no_airfoil, y_no_airfoil


def generate_domain_points(x, y, geometry):
    points = []
    rs = []
    centre_x = 0
    centre_y = 0.12
    r = np.sqrt((x-centre_x)**2 + ((y-centre_y)*7)**2)
    r = r/(np.max(r)*1)
    r = r**0.3
    r = 1-r
    for i in range(x.shape[0]):
        tmp_u = np.random.random()
        tmp_r = np.random.random()
        if (tmp_r < r[i, 0]) and (tmp_u < 0.05) and geometry.inside([x[i, 0], y[i, 0]]):
            points.append([x[i, 0], y[i, 0]])
    print(f'Generated {len(points)} points in the domain')
    return points


def generate_PIV_points(x, y, u, v, p, x_stride, y_stride, v_ld, v_ru, geometry, plot=False):
    """ Generation of PIV points for training """
    x_p = deepcopy(x)
    y_p = deepcopy(y)
    u_p = deepcopy(u)
    v_p = deepcopy(v)
    p_p = deepcopy(p)
    x_p = x_p.reshape(1501, 601).T
    y_p = y_p.reshape(1501, 601).T
    u_p = u_p.reshape(1501, 601).T
    v_p = v_p.reshape(1501, 601).T
    p_p = p_p.reshape(1501, 601).T


    start_ind_x = int((x_p.shape[1] % x_stride)/2)
    start_ind_y = int((x_p.shape[0] % y_stride)/2)

    # start_ind_x=100
    # start_ind_y=175

    x_p = x_p[start_ind_y::y_stride, start_ind_x::x_stride]
    y_p = y_p[start_ind_y::y_stride, start_ind_x::x_stride]
    u_p = u_p[start_ind_y::y_stride, start_ind_x::x_stride]
    v_p = v_p[start_ind_y::y_stride, start_ind_x::x_stride]
    p_p = p_p[start_ind_y::y_stride, start_ind_x::x_stride]
    x_p = x_p.T.reshape(-1, 1)
    y_p = y_p.T.reshape(-1, 1)
    u_p = u_p.T.reshape(-1, 1)
    v_p = v_p.T.reshape(-1, 1)
    p_p = p_p.T.reshape(-1, 1)
    X = []
    for i in range(x_p.shape[0]):
        if geometry.inside([x_p[i, 0], y_p[i, 0]]) \
                and x_p[i, 0] >= v_ld[0] and x_p[i, 0] <= v_ru[0] \
                and y_p[i, 0] >= v_ld[1] and y_p[i, 0] <= v_ru[1]:
            X.append([x_p[i, 0], y_p[i, 0], u_p[i, 0], v_p[i, 0], p_p[i, 0]])
    X = np.array(X)
    # plt.scatter(X[:, 0], X[:, 1], s=0.1, color='red')
    # plt.show()
    return np.hsplit(X, 5)


def main(train=True, test=True):
    # case name
    case_name = "unNACA0012_fvar_superresolution"
    case_name_title = r'PIV stride $0.15 \times 0.15 fs=0 at airfoil$'

    set_directory(case_name)

    x_data, y_data, u_data, v_data, p_data, uu_data, uv_data, vv_data, x_domain, y_domain = read_data()

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

    figsize = (8,3)

    # geometry specification
    geom1 = dde.geometry.Polygon(airfoil_points)
    geom2 = dde.geometry.Rectangle(v_ld, v_ru)
    geom = geom2 - geom1

    [x_piv, y_piv, u_piv, v_piv, p_piv] = \
        generate_PIV_points(x_data, y_data, u_data, v_data,
                            p_data, 75, 75, v_ld, v_ru, geom, True)
    piv_points = np.hstack((x_piv, y_piv))

    for i in range(x_data.shape[0]):
        if x_data[i,0]==0.5 and y_data[i,0]==0.056:
            p1 = p_data[i,0]
            print(p1)
        elif x_data[i,0]==0.5 and y_data[i,0]==-0.054:
            p2 = p_data[i,0]
            print(p2)
    
    p_coors = np.array([[0.5, 0.056], [0.5,-0.054]])
    p_val = np.array([[p1], [p2]])

    # BC specification
    # boundaries functions
    def boundary_in(x, on_boundary):
        return on_boundary and np.isclose(x[0], -0.5)

    def boundary(x, on_boundary):
        return on_boundary and not (
            np.isclose(x[0], v_ld[0])
            or np.isclose(x[0], v_ru[0])
            or np.isclose(x[1], v_ld[1])
            or np.isclose(x[1], v_ru[1])
        )

    def boundary_left_free(x, on_boundary):
        return on_boundary and (np.isclose(x[0], v_ld[0])
                or (np.isclose(x[1], v_ru[1]) and x[0]<=-0.3)
                or (np.isclose(x[1], v_ld[1]) and x[0]<=-0.3))
    

    def boundary_left_full(x, on_boundary):
        return on_boundary and not (np.isclose(x[0], v_ru[0])
                or (np.isclose(x[1], v_ru[1]) and x[0]>-0.3)
                or (np.isclose(x[1], v_ld[1]) and x[0]>-0.3))
    

    # BC objects
    u_piv_points = dde.PointSetBC(piv_points, u_piv, component=0)
    v_piv_points = dde.PointSetBC(piv_points, v_piv, component=1)
    pressure_points = dde.PointSetBC(p_coors, p_val, component=2)
    bc_wall_u = dde.DirichletBC(geom, func_zeros, boundary, component=0)
    bc_wall_v = dde.DirichletBC(geom, func_zeros, boundary, component=1)
    bc_wall_fx = dde.DirichletBC(geom, func_zeros, boundary, component=3)
    bc_wall_fy = dde.DirichletBC(geom, func_zeros, boundary, component=4)

    # custom domain points
    domain_points = generate_domain_points(x_domain, y_domain, geometry=geom)

    # pde and physics compilation
    pde = RANSf0var2D(500)
    if train:
        data = dde.data.PDE(
            geom,
            pde,
            [bc_wall_u, bc_wall_v, bc_wall_fx, bc_wall_fy, u_piv_points, v_piv_points, pressure_points],
            100,
            1600,
            solution=None,
            num_test=100,
            train_distribution="custom",
            custom_train_points=domain_points,
        )
        plot_train_points(data, [4, 6, 7], ["wall BC", "PIV data", "pressure anchors"],
                          case_name, title=case_name_title, figsize=(10,3))
    else:
        data = dde.data.PDE(
            geom,
            pde,
            [bc_wall_u, bc_wall_v, bc_wall_fx, bc_wall_fy, u_piv_points, v_piv_points, pressure_points],
            100,
            100,
            solution=None,
            num_test=100
        )
    dde.backend.tf.logging.set_verbosity(20)
    print(dde.backend.tf.logging.get_verbosity())
    # NN model definition
    layer_size = [2] + [100] * 7 + [6]
    activation = "tanh"
    initializer = "Glorot uniform"
    net = dde.maps.FNN(layer_size, activation, initializer)

    # PINN definition
    model = dde.Model(data, net)

    if train:
        # Adam optimization
        loss_weights = [1, 1, 1, 1, 1, 10, 10, 10, 10, 10, 10, 10]
        model.compile("adam", lr=0.001, loss_weights=loss_weights)
        checkpointer = dde.callbacks.ModelCheckpoint(
            f"{case_name}/models/model_{case_name}.ckpt",
            verbose=1,
            save_better_only=True,
        )

        loss_update = dde.callbacks.LossUpdateCheckpoint(
            momentum=0.7,
            verbose=1, period=1, report_period=100,
            base_range=[0, 1, 2, 3 ,4],
            update_range=[ 5, 6, 7, 8, 9, 10, 11]
        )
        print('Training for 20000 epochs')
        losshistory, train_state = model.train(
            epochs=20000, callbacks=[checkpointer, loss_update], display_every=100
        )

        model.save(f"{case_name}/models/model-adam-last")

        # L-BFGS-B optimization
        model.compile("L-BFGS-B", loss_weights=loss_weights)
        losshistory, train_state = model.train()
        model.save(f"{case_name}/models/model-bfgs-last")

    if test:
        model.compile("adam", lr=0.001)
        model.compile("L-BFGS-B")
        last_epoch = model.train_state.epoch
        if not train:
            last_epoch = 60001
        model.restore(f"{case_name}/models/model-bfgs-last-{last_epoch}")

        x_plot = np.linspace(v_ld[0], v_ru[0], Nx)
        y_plot = np.linspace(v_ld[1], v_ru[1], Ny)
        # domain data
        x_data = x_data.reshape(1501, 601).T
        y_data = y_data.reshape(1501, 601).T
        u_data = u_data.reshape(1501, 601).T
        v_data = v_data.reshape(1501, 601).T
        p_data = p_data.reshape(1501, 601).T
        x_dom = np.linspace(-0.3, 2.7, 1501)
        y_dom = np.linspace(-0.6, 0.6, 601)
        x_min = np.argmin(np.abs(x_dom-v_ld[0]))
        x_max = np.argmin(np.abs(x_dom-v_ru[0]))
        y_min = np.argmin(np.abs(y_dom-v_ld[1]))
        y_max = np.argmin(np.abs(y_dom-v_ru[1]))
        print(x_min, x_max, y_min, y_max)
        x_data = x_data[y_min:y_max+1, x_min:x_max+1].T.reshape(-1,1)
        y_data = y_data[y_min:y_max+1, x_min:x_max+1].T.reshape(-1,1)
        u_data = u_data[y_min:y_max+1, x_min:x_max+1].T.reshape(-1,1)
        v_data = v_data[y_min:y_max+1, x_min:x_max+1].T.reshape(-1,1)
        p_data = p_data[y_min:y_max+1, x_min:x_max+1].T.reshape(-1,1)

        z = np.array([np.array([i, j]) for i in x_plot for j in y_plot])
        y = model.predict(z)
        u_star = y[:, 0][:, None]
        v_star = y[:, 1][:, None]
        p_star = y[:, 2][:, None]
        fx_star = y[:, 3][:,None]
        fy_star = y[:, 4][:,None]
        curl_f = y[:, 5][:,None]

        data_dict = {
            "u_star": u_star,
            "v_star": v_star,
            "p_star": p_star,
            "fx_star": fx_star,
            "fy_star": fy_star,
            "curlfvar_star": curl_f,
        }

        scipy.io.savemat(f"{case_name}/results.mat", data_dict)

        zero_index = (x_data < 0) & (x_data > 0)
        zero_index = zero_index | ((u_data == 0) & (v_data == 0))
        no_data_index = zero_index

        u_star_data = deepcopy(u_star)
        v_star_data = deepcopy(v_star)
        p_star_data = deepcopy(p_star)
        fx_star_data = deepcopy(fx_star)
        fy_star_data = deepcopy(fy_star)
        curl_f_data = deepcopy(curl_f)
        u_star_data[no_data_index] = u_star[no_data_index]*0
        v_star_data[no_data_index] = v_star[no_data_index]*0
        p_star_data[no_data_index] = p_star[no_data_index]*0
        fx_star_data[no_data_index] = fx_star[no_data_index]*0
        fy_star_data[no_data_index] = fy_star[no_data_index]*0
        curl_f_data[no_data_index] = curl_f[no_data_index]*0

        u_star_data = u_star_data.reshape(Nx, Ny).T
        v_star_data = v_star_data.reshape(Nx, Ny).T
        p_star_data = p_star_data.reshape(Nx, Ny).T
        fx_star_data = fx_star_data.reshape(Nx, Ny).T
        fy_star_data = fy_star_data.reshape(Nx, Ny).T
        curl_f_data = curl_f_data.reshape(Nx, Ny).T

        X, Y = np.meshgrid(x_plot, y_plot)
        plt.figure(figsize=figsize)
        # plt.title(f'regressed u field for {case_name_title}')
        plt.pcolor(X, Y, u_star_data)
        plt.colorbar(label='u')
        plt.xlabel('x/c')
        plt.ylabel('y/c')
        plt.tight_layout()
        plt.savefig(os.path.join(f'{case_name}',
                                 'plots', 'u_plot.png'), dpi=400)
        plt.close()
        plt.figure(figsize=figsize)
        # plt.title(f'regressed v field for {case_name_title}')
        plt.pcolor(X, Y, v_star_data)
        plt.colorbar(label='v')
        plt.xlabel('x/c')
        plt.ylabel('y/c')
        plt.tight_layout()
        plt.savefig(os.path.join(f'{case_name}',
                                 'plots', 'v_plot.png'), dpi=400)
        plt.close()
        plt.figure(figsize=figsize)
        # plt.title(f'regressed p field for {case_name_title}')
        plt.pcolor(X, Y, p_star_data)
        plt.colorbar(label='p')
        plt.xlabel('x/c')
        plt.ylabel('y/c')
        plt.tight_layout()
        plt.savefig(os.path.join(f'{case_name}',
                                 'plots', 'p_plot.png'), dpi=400)
        plt.close()
        plt.figure(figsize=figsize)
        # plt.title(f'regressed fx field for {case_name_title}')
        plt.pcolor(X, Y, fx_star_data)
        plt.colorbar(label='fx')
        plt.xlabel('x/c')
        plt.ylabel('y/c')
        plt.tight_layout()
        plt.savefig(os.path.join(f'{case_name}',
                                 'plots', 'fx_plot.png'), dpi=400)
        plt.close()
        plt.figure(figsize=figsize)
        # plt.title(f'regressed fy field for {case_name_title}')
        plt.pcolor(X, Y, fy_star_data)
        plt.colorbar(label='fy')
        plt.xlabel('x/c')
        plt.ylabel('y/c')
        plt.tight_layout()
        plt.savefig(os.path.join(f'{case_name}',
                                 'plots', 'fy_plot.png'), dpi=400)
        plt.close()
        plt.figure(figsize=figsize)
        # plt.title(f'regressed var curlf field for {case_name_title}')
        plt.pcolor(X, Y, -curl_f_data, vmin=-6.13125, vmax=6.26875)
        plt.colorbar(label=r"$\nabla \times \mathbf{f}$")
        plt.xlabel('x/c')
        plt.ylabel('y/c')
        plt.tight_layout()
        plt.savefig(os.path.join(f'{case_name}',
                                 'plots', 'curl_f_var_plot.png'), dpi=400)
        plt.close()

        # data error
        u_star_data = deepcopy(u_star)
        v_star_data = deepcopy(v_star)
        p_star_data = deepcopy(p_star)
        u_star_data[no_data_index] = u_star[no_data_index]*0
        v_star_data[no_data_index] = v_star[no_data_index]*0
        p_star_data[no_data_index] = p_star[no_data_index]*0

        u_star_data = u_star_data.reshape(Nx, Ny).T
        v_star_data = v_star_data.reshape(Nx, Ny).T
        p_star_data = p_star_data.reshape(Nx, Ny).T

        u_true = None
        v_true = None
        p_true = None

        u_true = deepcopy(u_data)
        v_true = deepcopy(v_data)
        p_true = deepcopy(p_data)

        u_true = u_true.reshape(Nx, Ny).T
        v_true = v_true.reshape(Nx, Ny).T
        p_true = p_true.reshape(Nx, Ny).T
        u_err = np.abs(u_true-u_star_data)
        v_err = np.abs(v_true-v_star_data)
        p_err = np.abs(p_true-p_star_data)

        plt.figure(figsize=figsize)
        # plt.title(f'u field abs error for {case_name_title}')
        plt.pcolor(X, Y, u_err)
        plt.colorbar(label='u')
        plt.xlabel('x/c')
        plt.ylabel('y/c')
        plt.tight_layout()
        plt.savefig(os.path.join(f'{case_name}',
                                 'plots', 'u_err_plot.png'), dpi=400)
        plt.close()
        plt.figure(figsize=figsize)
        # plt.title(f'v field abs error for {case_name_title}')
        plt.pcolor(X, Y, v_err)
        plt.colorbar(label='v')
        plt.xlabel('x/c')
        plt.ylabel('y/c')
        plt.tight_layout()
        plt.savefig(os.path.join(f'{case_name}',
                                 'plots', 'v_err_plot.png'), dpi=400)
        plt.close()
        plt.figure(figsize=figsize)
        # plt.title(f'p field abs error for {case_name_title}')
        plt.pcolor(X, Y, p_err)
        plt.colorbar(label='p')
        plt.xlabel('x/c')
        plt.ylabel('y/c')
        plt.tight_layout()
        plt.savefig(os.path.join(f'{case_name}',
                                 'plots', 'p_err_plot.png'), dpi=400)
        plt.close()

        e = model.predict(z, operator=pde)
        e_mass = e[0]
        e_u_momentum = e[1]
        e_v_momentum = e[2]
        f_divergence = e[3]
        f_curl_err = e[4]

        data_dict.update({
            "e_mass": e_mass,
            "e_u_momentum": e_u_momentum,
            "e_v_momentum": e_v_momentum,
            "f_divergence": f_divergence,
            "fvarerr_residual": f_curl_err
        })
        scipy.io.savemat(f"{case_name}/results.mat", data_dict)

        e_mass[no_data_index] = e_mass[no_data_index] * 0
        e_u_momentum[no_data_index] = e_u_momentum[no_data_index] * 0
        e_v_momentum[no_data_index] = e_v_momentum[no_data_index] * 0
        f_divergence[no_data_index] = f_divergence[no_data_index] * 0
        f_curl_err[no_data_index] = f_curl_err[no_data_index] * 0
        e_mass = e_mass.reshape(Nx, Ny).T
        e_u_momentum = e_u_momentum.reshape(Nx, Ny).T
        e_v_momentum = e_v_momentum.reshape(Nx, Ny).T
        f_divergence = f_divergence.reshape(Nx, Ny).T
        f_curl_err = f_curl_err.reshape(Nx, Ny).T

        plt.figure(figsize=figsize)
        # plt.title(f'mass conservation residual for {case_name_title}')
        plt.pcolor(X, Y, e_mass, vmin=-1, vmax=1)
        plt.colorbar(label='e_mass')
        plt.xlabel('x/c')
        plt.ylabel('y/c')
        plt.tight_layout()
        plt.savefig(os.path.join(f'{case_name}',
                                 'plots', 'e_mass_plot.png'), dpi=400)
        plt.close()
        plt.figure(figsize=figsize)
        # plt.title(f'u momentum conservation residual for {case_name_title}')
        plt.pcolor(X, Y, e_u_momentum, vmin=-1, vmax=1)
        plt.colorbar(label='e_u_momentum')
        plt.xlabel('x/c')
        plt.ylabel('y/c')
        plt.tight_layout()
        plt.savefig(os.path.join(
            f'{case_name}', 'plots', 'e_u_momentum_plot.png'), dpi=400)
        plt.close()
        plt.figure(figsize=figsize)
        # plt.title(f'v momentum conservation residual for {case_name_title}')
        plt.pcolor(X, Y, e_v_momentum, vmin=-1, vmax=1)
        plt.colorbar(label='e_v_momentum')
        plt.xlabel('x/c')
        plt.ylabel('y/c')
        plt.tight_layout()
        plt.savefig(os.path.join(
            f'{case_name}', 'plots', 'e_v_momentum_plot.png'), dpi=400)
        plt.close()
        plt.figure(figsize=figsize)
        # plt.title(f'fs divergence residual for {case_name_title}')
        plt.pcolor(X, Y, f_divergence, vmin=-1, vmax=1)
        plt.colorbar(label='f_divergence')
        plt.xlabel('x/c')
        plt.ylabel('y/c')
        plt.tight_layout()
        plt.savefig(os.path.join(
            f'{case_name}', 'plots', 'f_divergence_plot.png'), dpi=400)
        plt.close()
        plt.figure(figsize=figsize)
        # plt.title(f'f_curl equality residual for {case_name_title}')
        plt.pcolor(X, Y, f_curl_err, vmin=-1, vmax=1)
        plt.colorbar(label='f_curl err')
        plt.xlabel('x/c')
        plt.ylabel('y/c')
        plt.tight_layout()
        plt.savefig(os.path.join(
            f'{case_name}', 'plots', 'f_curl_eql_err_plot.png'), dpi=400)
        plt.close()

        
        def curl_f(X,V):
            dfsx_y = dde.grad.jacobian(V, X, i=3, j=1)
            dfsy_x = dde.grad.jacobian(V, X, i=4, j=0)
            return [dfsy_x - dfsx_y]
        
        e = model.predict(z, operator=curl_f)
        f_curl = e[0]

        data_dict.update({
            "f_curl_star": e[0]
        })
        scipy.io.savemat(f"{case_name}/results.mat", data_dict)

        f_curl[no_data_index] = f_curl[no_data_index] * 0

        f_curl = f_curl.reshape(Nx, Ny).T


        plt.figure(figsize=figsize)
        # plt.title(f'curl fs for {case_name_title}')
        plt.pcolor(X, Y, f_curl)
        plt.colorbar(label='f_curl')
        plt.xlabel('x/c')
        plt.ylabel('y/c')
        plt.tight_layout()
        plt.savefig(os.path.join(f'{case_name}',
                                 'plots', 'f_curl_plot.png'), dpi=400)
        plt.close()

        plt.figure(figsize=figsize)
        # plt.title(f'curl fs for {case_name_title}')
        plt.pcolor(X, Y, f_curl, vmin=-6.13125, vmax=6.26875)
        plt.colorbar(label=r"$\nabla \times \mathbf{f}$")
        plt.xlabel('x/c')
        plt.ylabel('y/c')
        plt.tight_layout()
        plt.savefig(os.path.join(f'{case_name}',
                                 'plots', 'f_curl_plot_rescaled.png'), dpi=400)
        plt.close()

if __name__ == "__main__":
    train = True
    test = True
    if "train" in sys.argv and "test" not in sys.argv:
        train = True
        test = False
    if "train" not in sys.argv and "test" in sys.argv:
        train = False
        test = True
    main(train, test)
