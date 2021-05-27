import deepxde as dde


def easy_eq(slope):
    def pde(X, V):
        u = V[:, 0:1]
        # v = V[:, 1:2]
        # p = V[:,2:3]
        du_x = dde.grad.jacobian(V, X, i=0, j=0)
        # dv_y = dde.grad.jacobian(V, X, i=1, j=1)
        # du_y = dde.grad.jacobian(V, X, i=0, j=1)
        # dv_x = dde.grad.jacobian(V, X, i=1, j=0)
        # dp_x = dde.grad.jacobian(V, X, i=2, j=0)
        # dp_y = dde.grad.jacobian(V, X, i=2, j=1)
        # du_xx = dde.grad.hessian(u, X, i=0, j=0)
        # dv_xx = dde.grad.hessian(v, X, i=0, j=0)
        # du_yy = dde.grad.hessian(u, X, i=1, j=1)
        # dv_yy = dde.grad.hessian(v, X, i=1, j=1)

        return [
            du_x -slope
        ]

    return pde


def NS2D(Rey):
    def pde(X, V):
        u = V[:, 0:1]
        v = V[:, 1:2]
        # p = V[:,2:3]
        du_x = dde.grad.jacobian(V, X, i=0, j=0)
        dv_y = dde.grad.jacobian(V, X, i=1, j=1)
        du_y = dde.grad.jacobian(V, X, i=0, j=1)
        dv_x = dde.grad.jacobian(V, X, i=1, j=0)
        dp_x = dde.grad.jacobian(V, X, i=2, j=0)
        dp_y = dde.grad.jacobian(V, X, i=2, j=1)
        du_xx = dde.grad.hessian(u, X, i=0, j=0)
        dv_xx = dde.grad.hessian(v, X, i=0, j=0)
        du_yy = dde.grad.hessian(u, X, i=1, j=1)
        dv_yy = dde.grad.hessian(v, X, i=1, j=1)

        return [
            du_x + dv_y,
            u * du_x + v * du_y + dp_x - 1.0 / Rey * (du_xx + du_yy),
            u * dv_x + v * dv_y + dp_y - 1.0 / Rey * (dv_xx + dv_yy),
        ]

    return pde


def RANS2D(Rey):
    def pde(X, V):
        u = V[:, 0:1]
        v = V[:, 1:2]
        
        # p = V[:,2:3]
        du_x = dde.grad.jacobian(V, X, i=0, j=0)
        dv_y = dde.grad.jacobian(V, X, i=1, j=1)
        du_y = dde.grad.jacobian(V, X, i=0, j=1)
        dv_x = dde.grad.jacobian(V, X, i=1, j=0)
        dp_x = dde.grad.jacobian(V, X, i=2, j=0)
        dp_y = dde.grad.jacobian(V, X, i=2, j=1)
        du_xx = dde.grad.hessian(u, X, i=0, j=0)
        dv_xx = dde.grad.hessian(v, X, i=0, j=0)
        du_yy = dde.grad.hessian(u, X, i=1, j=1)
        dv_yy = dde.grad.hessian(v, X, i=1, j=1)

        # Reynolds stresses
        duu_x = dde.grad.jacobian(V, X, i=3, j=0)
        duv_x = dde.grad.jacobian(V, X, i=5, j=0)
        dvv_y = dde.grad.jacobian(V, X, i=4, j=1)
        duv_y = dde.grad.jacobian(V, X, i=5, j=1)

        return [
            du_x + dv_y,
            u * du_x + v * du_y + dp_x - 1.0 / Rey * (du_xx + du_yy) - duu_x - duv_x,
            u * dv_x + v * dv_y + dp_y - 1.0 / Rey * (dv_xx + dv_yy) - dvv_y - duv_y,
        ]

    return pde

def RANSf2D(Rey):
    def pde(X, V):
        u = V[:, 0:1]
        v = V[:, 1:2]
        p = V[:, 2:3]
        fsx = V[:, 3:4]
        fsy = V[:, 4:5]
        psi = V[:, 5:6]
        ppsi = p+psi
        
        du_x = dde.grad.jacobian(V, X, i=0, j=0)
        dv_y = dde.grad.jacobian(V, X, i=1, j=1)
        du_y = dde.grad.jacobian(V, X, i=0, j=1)
        dv_x = dde.grad.jacobian(V, X, i=1, j=0)
        dppsi_x = dde.grad.jacobian(ppsi[:,None], X, i=0, j=0)
        dppsi_y = dde.grad.jacobian(ppsi[:,None], X, i=0, j=1)
        du_xx = dde.grad.hessian(u, X, i=0, j=0)
        dv_xx = dde.grad.hessian(v, X, i=0, j=0)
        du_yy = dde.grad.hessian(u, X, i=1, j=1)
        dv_yy = dde.grad.hessian(v, X, i=1, j=1)

        dfsx_x = dde.grad.jacobian(V, X, i=3, j=0)
        dfsy_y = dde.grad.jacobian(V, X, i=4, j=1)

        return [
            du_x + dv_y,
            u * du_x + v * du_y + dppsi_x - 1.0 / Rey * (du_xx + du_yy) + fsx,
            u * dv_x + v * dv_y + dppsi_y - 1.0 / Rey * (dv_xx + dv_yy) + fsy,
            dfsx_x + dfsy_y
        ]
    return pde


def RANSf02D(Rey):
    def pde(X, V):
        u = V[:, 0:1]
        v = V[:, 1:2]
        p = V[:, 2:3]
        fsx = V[:, 3:4]
        fsy = V[:, 4:5]
        
        # p = V[:,2:3]
        du_x = dde.grad.jacobian(V, X, i=0, j=0)
        dv_y = dde.grad.jacobian(V, X, i=1, j=1)
        du_y = dde.grad.jacobian(V, X, i=0, j=1)
        dv_x = dde.grad.jacobian(V, X, i=1, j=0)
        dp_x = dde.grad.jacobian(V, X, i=2, j=0)
        dp_y = dde.grad.jacobian(V, X, i=2, j=1)
        du_xx = dde.grad.hessian(u, X, i=0, j=0)
        dv_xx = dde.grad.hessian(v, X, i=0, j=0)
        du_yy = dde.grad.hessian(u, X, i=1, j=1)
        dv_yy = dde.grad.hessian(v, X, i=1, j=1)

        dfsx_x = dde.grad.jacobian(V, X, i=3, j=0)
        dfsy_y = dde.grad.jacobian(V, X, i=4, j=1)

        return [
            du_x + dv_y,
            u * du_x + v * du_y + dp_x - 1.0 / Rey * (du_xx + du_yy) + fsx,
            u * dv_x + v * dv_y + dp_y - 1.0 / Rey * (dv_xx + dv_yy) + fsy,
            dfsx_x + dfsy_y
        ]

    return pde

def RANSf0var2D(Rey):
    def pde(X, V):
        u = V[:, 0:1]
        v = V[:, 1:2]
        p = V[:, 2:3]
        fsx = V[:, 3:4]
        fsy = V[:, 4:5]
        curlf = V[:,5:6]
        
        # p = V[:,2:3]
        du_x = dde.grad.jacobian(V, X, i=0, j=0)
        dv_y = dde.grad.jacobian(V, X, i=1, j=1)
        du_y = dde.grad.jacobian(V, X, i=0, j=1)
        dv_x = dde.grad.jacobian(V, X, i=1, j=0)
        dp_x = dde.grad.jacobian(V, X, i=2, j=0)
        dp_y = dde.grad.jacobian(V, X, i=2, j=1)
        du_xx = dde.grad.hessian(u, X, i=0, j=0)
        dv_xx = dde.grad.hessian(v, X, i=0, j=0)
        du_yy = dde.grad.hessian(u, X, i=1, j=1)
        dv_yy = dde.grad.hessian(v, X, i=1, j=1)

        dfsx_x = dde.grad.jacobian(V, X, i=3, j=0)
        dfsy_y = dde.grad.jacobian(V, X, i=4, j=1)
        dfsx_y = dde.grad.jacobian(V, X, i=3, j=1)
        dfsy_x = dde.grad.jacobian(V, X, i=4, j=0)

        return [
            du_x + dv_y,
            u * du_x + v * du_y + dp_x - 1.0 / Rey * (du_xx + du_yy) + fsx,
            u * dv_x + v * dv_y + dp_y - 1.0 / Rey * (dv_xx + dv_yy) + fsy,
            dfsx_x + dfsy_y,
            curlf - (dfsy_x-dfsx_y)
        ]

    return pde

def RANSpknown2D(Rey):
    def pde(X, V):
        u = V[:, 0:1]
        v = V[:, 1:2]
        p = V[:, 2:3]
        fsx = V[:, 3:4]
        fsy = V[:, 4:5]
        
        # p = V[:,2:3]
        du_x = dde.grad.jacobian(V, X, i=0, j=0)
        dv_y = dde.grad.jacobian(V, X, i=1, j=1)
        du_y = dde.grad.jacobian(V, X, i=0, j=1)
        dv_x = dde.grad.jacobian(V, X, i=1, j=0)
        dp_x = dde.grad.jacobian(V, X, i=2, j=0)
        dp_y = dde.grad.jacobian(V, X, i=2, j=1)
        du_xx = dde.grad.hessian(u, X, i=0, j=0)
        dv_xx = dde.grad.hessian(v, X, i=0, j=0)
        du_yy = dde.grad.hessian(u, X, i=1, j=1)
        dv_yy = dde.grad.hessian(v, X, i=1, j=1)

        dfsx_x = dde.grad.jacobian(V, X, i=3, j=0)
        dfsy_y = dde.grad.jacobian(V, X, i=4, j=1)

        return [
            du_x + dv_y,
            u * du_x + v * du_y + dp_x - 1.0 / Rey * (du_xx + du_yy) + fsx,
            u * dv_x + v * dv_y + dp_y - 1.0 / Rey * (dv_xx + dv_yy) + fsy
        ]

    return pde


def RANSalphabeta2D(Rey):
    def pde(X, V):
        u = V[:, 0:1]
        v = V[:, 1:2]
        p = V[:, 2:3]
        alpha = V[:, 3:4]
        beta = V[:, 4:5]
        
        
        # p = V[:,2:3]
        du_x = dde.grad.jacobian(V, X, i=0, j=0)
        dv_y = dde.grad.jacobian(V, X, i=1, j=1)
        du_y = dde.grad.jacobian(V, X, i=0, j=1)
        dv_x = dde.grad.jacobian(V, X, i=1, j=0)
        dp_x = dde.grad.jacobian(V, X, i=2, j=0)
        dp_y = dde.grad.jacobian(V, X, i=2, j=1)
        du_xx = dde.grad.hessian(u, X, i=0, j=0)
        dv_xx = dde.grad.hessian(v, X, i=0, j=0)
        du_yy = dde.grad.hessian(u, X, i=1, j=1)
        dv_yy = dde.grad.hessian(v, X, i=1, j=1)

        dalpha_x = dde.grad.jacobian(V, X, i=3, j=0)
        dalpha_y = dde.grad.jacobian(V, X, i=3, j=1)
        dbeta_x = dde.grad.jacobian(V, X, i=4, j=0)
        dbeta_y = dde.grad.jacobian(V, X, i=4, j=1)
        dalpha_xx = dde.grad.hessian(alpha, X, i=0, j=0)
        dbeta_xy = dde.grad.hessian(beta, X, i=0, j=1)
        dalpha_yy = dde.grad.hessian(alpha, X, i=1, j=1)

        return [
            du_x + dv_y,
            u * du_x + v * du_y + dp_x - 1.0 / Rey * (du_xx + du_yy) + dalpha_x+dbeta_y,
            u * dv_x + v * dv_y + dp_y - 1.0 / Rey * (dv_xx + dv_yy) + dbeta_x-dalpha_y,
            dalpha_xx+dbeta_xy*2-dalpha_yy
        ]

    return pde


def RANSalphabetafvar2D(Rey):
    def pde(X, V):
        u = V[:, 0:1]
        v = V[:, 1:2]
        p = V[:, 2:3]
        alpha = V[:, 3:4]
        beta = V[:, 4:5]
        curl_f = V[:,5:6]
        
        # p = V[:,2:3]
        du_x = dde.grad.jacobian(V, X, i=0, j=0)
        dv_y = dde.grad.jacobian(V, X, i=1, j=1)
        du_y = dde.grad.jacobian(V, X, i=0, j=1)
        dv_x = dde.grad.jacobian(V, X, i=1, j=0)
        dp_x = dde.grad.jacobian(V, X, i=2, j=0)
        dp_y = dde.grad.jacobian(V, X, i=2, j=1)
        du_xx = dde.grad.hessian(u, X, i=0, j=0)
        dv_xx = dde.grad.hessian(v, X, i=0, j=0)
        du_yy = dde.grad.hessian(u, X, i=1, j=1)
        dv_yy = dde.grad.hessian(v, X, i=1, j=1)

        dalpha_x = dde.grad.jacobian(V, X, i=3, j=0)
        dalpha_y = dde.grad.jacobian(V, X, i=3, j=1)
        dbeta_x = dde.grad.jacobian(V, X, i=4, j=0)
        dbeta_y = dde.grad.jacobian(V, X, i=4, j=1)
        dalpha_xx = dde.grad.hessian(alpha, X, i=0, j=0)
        dbeta_xy = dde.grad.hessian(beta, X, i=0, j=1)
        dalpha_yy = dde.grad.hessian(alpha, X, i=1, j=1)
        dalpha_xy = dde.grad.hessian(alpha, X, i=0, j=1)
        dbeta_xx = dde.grad.hessian(beta, X, i=0, j=0)
        dbeta_yy = dde.grad.hessian(beta, X, i=1, j=1)

        return [
            du_x + dv_y,
            u * du_x + v * du_y + dp_x - 1.0 / Rey * (du_xx + du_yy) + dalpha_x+dbeta_y,
            u * dv_x + v * dv_y + dp_y - 1.0 / Rey * (dv_xx + dv_yy) + dbeta_x-dalpha_y,
            dalpha_xx+dbeta_xy*2-dalpha_yy,
            curl_f - (dbeta_xx-dbeta_yy-2*dalpha_xy)
        ]

    return pde

def ab_equation():
    def pde(X,V):
        a = V[:, 0:1]
        b = V[:, 1:2]
        a_x = dde.grad.jacobian(V,X,i=0,j=0)
        a_y = dde.grad.jacobian(V,X,i=0,j=1)
        b_x = dde.grad.jacobian(V,X,i=1,j=0)
        b_y = dde.grad.jacobian(V,X,i=1,j=1)
        fx = V[:, 2:3]
        fy = V[:, 3:4]
        return [
            fx - (a_x+b_y),
            fy - (b_x-a_y)
        ]
    return pde

def RANSReStresses2D(Rey):
    def pde(X, V):
        u = V[:, 0:1]
        v = V[:, 1:2]
        p = V[:, 2:3]
        uu = V[:, 3:4]
        uv = V[:, 4:5]
        vv = V[:, 5:6]
        
        # p = V[:,2:3]
        du_x = dde.grad.jacobian(V, X, i=0, j=0)
        dv_y = dde.grad.jacobian(V, X, i=1, j=1)
        du_y = dde.grad.jacobian(V, X, i=0, j=1)
        dv_x = dde.grad.jacobian(V, X, i=1, j=0)
        dp_x = dde.grad.jacobian(V, X, i=2, j=0)
        dp_y = dde.grad.jacobian(V, X, i=2, j=1)
        du_xx = dde.grad.hessian(u, X, i=0, j=0)
        dv_xx = dde.grad.hessian(v, X, i=0, j=0)
        du_yy = dde.grad.hessian(u, X, i=1, j=1)
        dv_yy = dde.grad.hessian(v, X, i=1, j=1)

        duu_x = dde.grad.jacobian(V, X, i=3, j=0)
        duv_y = dde.grad.jacobian(V, X, i=4, j=1)
        duv_x = dde.grad.jacobian(V, X, i=4, j=0)
        dvv_y = dde.grad.jacobian(V, X, i=5, j=1)
        

        return [
            du_x + dv_y,
            u * du_x + v * du_y + dp_x - 1.0 / Rey * (du_xx + du_yy) + duu_x+duv_y,
            u * dv_x + v * dv_y + dp_y - 1.0 / Rey * (dv_xx + dv_yy) + duv_x+dvv_y,
            dde.backend.tf.nn.relu(-uu) + dde.backend.tf.nn.relu(-vv)
        ]

    return pde

def func_ones(X):
    x = X[:, 0:1]
    return x * 0 + 1


def func_zeros(X):
    x = X[:, 0:1]
    return x * 0


def Blasius():
    def pde(eta, f):
        df = dde.grad.jacobian(f, eta)
        ddf = dde.grad.jacobian(df, eta)
        dddf = dde.grad.jacobian(ddf, eta)
        return 2*dddf + f*ddf
    return pde
