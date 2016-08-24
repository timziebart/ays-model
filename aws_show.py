
import aws_model as aws

import numpy as np

import scipy.integrate as integ

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as plt3d
import matplotlib.ticker as ticker

# patch to remove padding at ends of axes:
###patch start###
from mpl_toolkits.mplot3d.axis3d import Axis
if not hasattr(Axis, "_get_coord_info_old"):
    def _get_coord_info_new(self, renderer):
        mins, maxs, centers, deltas, tc, highs = self._get_coord_info_old(renderer)
        mins += deltas / 4
        maxs -= deltas / 4
        return mins, maxs, centers, deltas, tc, highs
    Axis._get_coord_info_old = Axis._get_coord_info
    Axis._get_coord_info = _get_coord_info_new
###patch end###



def create_figure(A_max, W_mid, S_mid):
    fig = plt.figure(figsize=(16,9))
    ax3d = plt3d.Axes3D(fig)
    ax3d.set_xlabel("\nexcess atmospheric carbon stock A [GtC]")
    ax3d.set_ylabel("\nwelfare W [1e12 USD/yr]")
    ax3d.set_zlabel("\n\nrenewable knowledge\nstock S [1e9 GJ]")

    # make proper tickmarks:
    Aticks = np.linspace(0,A_max,11)
    Wticks = np.concatenate((np.linspace(0,1e14,11)[:-1],np.linspace(0,.5e15,6)[1:]))
    Sticks = np.concatenate((np.linspace(0,1e10,11)[:-1],np.linspace(0,1e11,11)[1:]))
    ax3d.w_xaxis.set_major_locator(ticker.FixedLocator(Aticks))
    ax3d.w_xaxis.set_major_formatter(ticker.FixedFormatter(Aticks.astype("int")))
    ax3d.set_xlim(Aticks[0],Aticks[-1])
    ax3d.w_yaxis.set_major_locator(ticker.FixedLocator(np.concatenate((Wticks/(W_mid+Wticks),[1]))))
    ax3d.w_yaxis.set_major_formatter(ticker.FixedFormatter(np.concatenate(((Wticks/1e12).astype("int"),["inf"]))))
    ax3d.set_ylim(0,1)
    ax3d.w_zaxis.set_major_locator(ticker.FixedLocator(np.concatenate((Sticks/(S_mid+Sticks),[1]))))
    ax3d.w_zaxis.set_major_formatter(ticker.FixedFormatter(np.concatenate(((Sticks/1e9).astype("int"),["inf"]))))
    ax3d.set_zlim(0,1)

    return fig, ax3d


def add_boundary(ax3d, A_PB, boundary= "PB", add_outer=False):
    # show boundaries of undesirable region:
    if boundary == "PB":
        boundary_surface_PB = plt3d.art3d.Poly3DCollection([[[A_PB,0,0],[A_PB,1,0],[A_PB,1,1],[A_PB,0,1]]])
        boundary_surface_PB.set_color("gray"); boundary_surface_PB.set_edgecolor("gray"); boundary_surface_PB.set_alpha(0.25)
        ax3d.add_collection3d(boundary_surface_PB)
    elif boundary == "both":
        boundary_surface_both = plt3d.art3d.Poly3DCollection([[[0,.5,0],[0,.5,1],[A_PB,.5,1],[A_PB,.5,0]],
                                                        [[A_PB,.5,0],[A_PB,1,0],[A_PB,1,1],[A_PB,.5,1]]])
        boundary_surface_both.set_color("gray"); boundary_surface_both.set_edgecolor("gray"); boundary_surface_both.set_alpha(0.25)
        ax3d.add_collection3d(boundary_surface_both)
    else:
        raise NameError("Unkown boundary {!r}".format(boundary))

    if add_outer:
        # add outer limits of undesirable view from standard view perspective:
        undesirable_outer_stdview = plt3d.art3d.Poly3DCollection([[[0,0,0],[0,0,1],[0,.5,1],[0,.5,0]],
                                            [[A_PB,1,0],[aws.A_max,1,0],[aws.A_max,1,1],[A_PB,1,1]],
                                            [[0,0,0],[0,.5,0],[A_PB,.5,0],[A_PB,1,0],[aws.A_max,1,0],[aws.A_max,0,0]]])
        undesirable_outer_stdview.set_color("gray"); undesirable_outer_stdview.set_edgecolor("gray"); undesirable_outer_stdview.set_alpha(0.25)
        ax3d.add_collection3d(undesirable_outer_stdview)


if __name__ == "__main__":

    # a small hack to make all the parameters available as global variables
    aws.globalize_dictionary(aws.AWS_parameters, module=aws)
    aws.globalize_dictionary(aws.grid_parameters, module=aws)

    num = 1000
    sample = np.random.rand(num,3)
    AWS_0 = np.zeros((num,3))
    # sample whole space:
    #AWS_0[:,0] = sample[:,0] * aws.A_max
    #AWS_0[:,1:] = sample[:,1:]/(1-sample[:,1:]) * WS_mid
    # sample DG + ET + PB lake candidates:
    AWS_0[:,0] = sample[:,0] * aws.A_PB
    AWS_0[:,1] = sample[:,1]/(1-sample[:,1]) * aws.W_mid
    AWS_0[:,2] = sample[:,2]/(1-sample[:,2]) * aws.S_mid

    fig, ax3d = create_figure(aws.A_max, aws.W_mid, aws.S_mid)

    for i in range(num):
        x0 = AWS_0[i]
        time = np.linspace(0, 1000, 1000)
        # management trajectory with degrowth:
        beta = aws.beta_DG
        aws.sigma = aws.sigma_default
        traj = integ.odeint(aws.AWS_rhs, x0, time, args=(beta,))
        # alternative trajectory with tax/subsidy:
        beta = aws.beta_default
        aws.sigma = aws.sigma_ET
        traj2 = integ.odeint(aws.AWS_rhs, x0, time, args=(beta,))
        if traj2[:,0].max() > aws.A_PB > traj[:,0].max() and traj[-1,2] < 1e10 and traj2[-1,2] > 1e10: # lake candidate!
            # JH: transform so that W_SF,sigma_default go to 1/2 and infinity goes to 1:
            ax3d.plot3D(xs=traj[:,0], ys=traj[:,1]/(aws.W_mid+traj[:,1]), zs=traj[:,2]/(aws.S_mid+traj[:,2]),
                        color="red" if traj[-1,2]<1000 else "blue", alpha=.7)
            ax3d.plot3D(xs=traj2[:,0], ys=traj2[:,1]/(aws.W_mid+traj2[:,1]), zs=traj2[:,2]/(aws.S_mid+traj2[:,2]),
                        color="orange" if traj2[-1,2]<1000 else "cyan", alpha=.7)
            #print(traj2[:,0].max() - traj[:,0].max())


    add_boundary(ax3d, aws.A_PB)


    plt.show()




