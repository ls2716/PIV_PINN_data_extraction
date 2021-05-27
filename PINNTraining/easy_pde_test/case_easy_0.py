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
plt.rc('lines', linewidth= 3.0)

import numpy as np
import math
import sys
import scipy.io
from copy import deepcopy


import deepxde as dde

# from deepxde.backend import tf

from equations import RANSf02D, func_zeros, easy_eq
from utilities import set_directory, plot_train_points


def main(train=True, test=True):
    # case name
    case_name = "easy_slope_05"
    case_name_title = r'_'

    set_directory(case_name)


    #domain vertices
    ends = [0, 1]



    Nx = int((ends[1]-ends[0])*500)+1
    print('Nx', Nx)

    figsize = (7,5)

    # geometry specification
    geom = dde.geometry.Interval(ends[0], ends[1])

    # BC specification
    # boundaries functions
    def boundary_in(x, on_boundary):
        return on_boundary and np.isclose(x[0], -0.5)

    # def boundary(x, on_boundary):
    #     return on_boundary and not (
    #         np.isclose(x[0], v_ld[0])
    #         or np.isclose(x[0], v_ru[0])
    #         or np.isclose(x[1], v_ld[1])
    #         or np.isclose(x[1], v_ru[1])
    #     )
    

    # BC objects
    left_point = dde.PointSetBC(np.array([0]).reshape(-1,1), np.array([0]).reshape(-1,1), component=0)

    # pde and physics compilation
    pde = easy_eq(0.5)
    if train:
        data = dde.data.PDE(
            geom,
            pde,
            [left_point],
            1000,
            1,
            solution=None,
            num_test=100,
            train_distribution="sobol"
        )
        plot_train_points(data, [1], ["left"],
                          case_name, title=case_name_title, figsize=figsize)
    else:
        data = dde.data.PDE(
            geom,
            pde,
            [left_point],
            100,
            100,
            solution=None,
            num_test=100
        )
    # NN model definition
    layer_size = [1] + [100] * 5 + [1]
    activation = "tanh"
    initializer = "Glorot uniform"
    net = dde.maps.FNN(layer_size, activation, initializer)

    # PINN definition
    model = dde.Model(data, net)

    if train:
        # Adam optimization
        loss_weights = [100, 1000]
        model.compile("adam", lr=0.001, loss_weights=loss_weights)
        checkpointer = dde.callbacks.ModelCheckpoint(
            f"{case_name}/models/model_{case_name}.ckpt",
            verbose=1,
            save_better_only=True,
        )

        # loss_update = dde.callbacks.LossUpdateCheckpoint(
        #     momentum=0.7,
        #     verbose=1, period=1, report_period=100,
        #     base_range=[0],
        #     update_range=[1]
        # )
        print('Training for 1000 epochs')
        losshistory, train_state = model.train(
            epochs=1000, callbacks=[checkpointer], display_every=100
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
            last_epoch=1047
        model.restore(f"{case_name}/models/model-bfgs-last-{last_epoch}")


        x_plot = np.linspace(ends[0], ends[1], Nx)
        f_plot = 0.5*x_plot

        y = model.predict(x_plot.reshape(-1,1))

        f_star = y[:, 0]

        print(f_star.shape)

        data_dict = {
            "f_star": f_star,
        }
        scipy.io.savemat(f"{case_name}/results.mat", data_dict)


        plt.figure(figsize=figsize)
        # plt.title(f'regressed u field for {case_name_title}')
        plt.plot(x_plot, f_star, label='prediction')
        plt.plot(x_plot, f_plot, ':', label='true')
        plt.xlabel('x/c')
        plt.title(r'$u$')
        plt.legend()
        axes=plt.gca()
        plt.grid()
        axes.set_aspect(1)
        print("limits", axes.get_ylim())
        plt.ylim([-0.025000017881393433, 2.933020759374183e-05*20000])
        plt.tight_layout()
        plt.savefig(os.path.join(f'{case_name}',
                                 'plots', 'f_plot.png'), dpi=400)
        plt.close()

        plt.figure(figsize=figsize)
        # plt.title(f'regressed u field for {case_name_title}')
        plt.plot(x_plot, np.abs(f_star-f_plot))
        plt.xlabel('x/c')
        plt.title(r'$u$ abs error')
        axes=plt.gca()
        plt.grid()
        axes.set_aspect(20000)
        print("limits", axes.get_ylim())
        # plt.ylim([-0.025000017881393433/20000, 0.525000375509262/20000])
        plt.tight_layout()
        plt.savefig(os.path.join(f'{case_name}',
                                 'plots', 'f_err.png'), dpi=400)
        plt.close()
        

        e = model.predict(x_plot.reshape(-1,1), operator=pde)
        e_err = e[0]

        plt.figure(figsize=figsize)
        # plt.title(f'regressed u field for {case_name_title}')
        plt.plot(x_plot, e_err)
        plt.xlabel('x/c')
        plt.ylabel('pde abs error')
        plt.grid()
        axes=plt.gca()
        # axes.set_aspect(1)
        plt.tight_layout()
        plt.savefig(os.path.join(f'{case_name}',
                                 'plots', 'pde_err.png'), dpi=400)
        plt.close()
        
        def derivatives(X,V):
            u_x = dde.grad.jacobian(V,X,i=0,j=0)
            u_xx = dde.grad.jacobian(u_x,X,i=0,j=0)
            u_xxx = dde.grad.jacobian(u_xx,X,i=0,j=0)
            return [u_x, u_xx, u_xxx]
        
        e = model.predict(x_plot.reshape(-1,1), operator=derivatives)
        f_x = e[0]
        f_xx = e[1]
        f_xxx = e[2]


        data_dict.update({
            "dfx": f_x,
            "dfxx": f_xx,
            "dfxxx": f_xxx
        })
        scipy.io.savemat(f"{case_name}/results.mat", data_dict)


        plt.figure(figsize=(12,5))
        # plt.title(f'regressed u field for {case_name_title}')
        plt.plot(x_plot, f_x-0.5, label=r'$u_x-0.5$')
        plt.plot(x_plot, f_xx, label=r'$u_{xx}$')
        plt.plot(x_plot, f_xxx, label=r'$u_{xxx}$')
        plt.legend(loc=(1.05,0.4))
        plt.grid()
        plt.xlabel('x/c')
        # plt.ylabel(r'$u_x$')
        axes=plt.gca()
        # axes.set_aspect(1)
        plt.tight_layout()
        plt.savefig(os.path.join(f'{case_name}',
                                 'plots', 'derivatives.png'), dpi=400)
        plt.close()

        # plt.figure(figsize=figsize)
        # # plt.title(f'regressed u field for {case_name_title}')
        # plt.plot(x_plot, f_x)
        # plt.xlabel('x/c')
        # plt.ylabel(r'$u_x$')
        # axes=plt.gca()
        # # axes.set_aspect(1)
        # plt.tight_layout()
        # plt.savefig(os.path.join(f'{case_name}',
        #                          'plots', 'fx.png'), dpi=400)
        # plt.close()
        
        # plt.figure(figsize=figsize)
        # # plt.title(f'regressed u field for {case_name_title}')
        # plt.plot(x_plot, f_xx)
        # plt.xlabel('x/c')
        # plt.ylabel(r'$u_{xx}$')
        # axes=plt.gca()
        # # axes.set_aspect(1)
        # plt.tight_layout()
        # plt.savefig(os.path.join(f'{case_name}',
        #                          'plots', 'fxx.png'), dpi=400)
        # plt.close()
        
        # plt.figure(figsize=figsize)
        # # plt.title(f'regressed u field for {case_name_title}')
        # plt.plot(x_plot, f_xxx)
        # plt.xlabel('x/c')
        # plt.ylabel(r'$u_{xxx}$')
        # axes=plt.gca()
        # # axes.set_aspect(1)
        # plt.tight_layout()
        # plt.savefig(os.path.join(f'{case_name}',
        #                          'plots', 'fxxx.png'), dpi=400)
        # plt.close()
        


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
