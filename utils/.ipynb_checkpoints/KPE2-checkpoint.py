# import glob
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import cmocean
# import seawater as sw
from scipy import linalg
# import gsw
import sys
# sys.path.append('/Users/paban23/Research/PhD/S-MODE/S-MODE-IOP/IOP_codes')
# sys.path.append('/Users/paban23/Research/PhD/S-MODE/SD_ADCP_Analyses')
#sys.path.insert(0,'/home/pab21003/PhD/Practice/NRT/Maps-main/tools/')
# from itertools import combinations
# from utils import *
# from utils import *
from KPE2 import *
# from KPE_functions import *
# from spectra_utils import *
# from scipy import signal
# from scipy.special import gammainc
from scipy.stats import kurtosis, skew
from scipy import stats
from scipy import linalg
# from scipy.signal import fftconvolve
# #%matplotlib widget
# import math


def rotate_coords(ds):
    for i in range(len(ds.legs.values)):
        if i==0:
            ds_nnull = ds.isel(legs=i).where(~ds.isel(legs=i).isnull(), drop=True)
            _,_,coefs = least_squares(ds_nnull.y.values.flatten(),ds_nnull.x.values.flatten()) #get the mean track using LS
            slope = coefs[1]
            x_m,y_m = ds_nnull.isel(trajectory=0).x.values, coefs[0] + slope*ds_nnull.x.median(dim='trajectory').values  # try median instead of mean
            theta_radians = np.arctan(slope)
            coords = x_m +1j*y_m
            rot_coords = coords*np.exp(-1j*theta_radians)

            leg_coord = ds_nnull.x + 1j*ds_nnull.y
            rotated_leg = leg_coord*np.exp(-1j*theta_radians)
            ds_nnull['x_rot'] = rotated_leg.real                 #Add the rotated x,y coordinate for each saildrone to the data
            ds_nnull['y_rot'] = rotated_leg.imag

            ds_nnull['mean_rotated_x'] = (('time',), rotated_leg.real.mean(dim='trajectory').values)
            ds_nnull['mean_rotated_y'] = (('time',), rotated_leg.imag.mean(dim='trajectory').values)
        else:
            ds_add = ds.isel(legs=i).where(~ds.isel(legs=i).isnull(), drop=True)
            _,_,coefs = least_squares(ds_add.y.values.flatten(),ds_add.x.values.flatten()) #get the mean track using LS
            slope = coefs[1]
            x_m,y_m = ds_add.isel(trajectory=0).x.values, coefs[0] + slope*ds_add.isel(trajectory=0).x.values
            theta_radians = np.arctan(slope)
            coords = x_m +1j*y_m
            rot_coords = coords*np.exp(-1j*theta_radians)

            leg_coord = ds_add.x + 1j*ds_add.y
            rotated_leg = leg_coord*np.exp(-1j*theta_radians)
            ds_add['x_rot'] = rotated_leg.real                 #Add the rotated x,y coordinate for each saildrone to the data
            ds_add['y_rot'] = rotated_leg.imag

            ds_add['mean_rotated_x'] = (('time',), rotated_leg.real.mean(dim='trajectory').values)
            ds_add['mean_rotated_y'] = (('time',), rotated_leg.imag.mean(dim='trajectory').values)
            ds_nnull = xr.concat([ds_nnull,ds_add],dim='legs')
            
            
    return ds_nnull

# Calculate the weighted uncertainity of kinematic properties (Vorticity, Divergence, and Strain-rate)
def KPE_uncertainity(A,c_u,c_v,B_u,B_v,z_u,z_v,n,strain_rate,u_uncert,v_uncert,sig2u,sig2v): # We provide the position(distance matrix), the velocities, and the least-square coefficients that includes mean velocities and velocity gradients
    # provide only 5 u's i.e., u_all[:,i]
    #calculate residual sum square SSres
    ux = c_u[1]
    uy = c_u[2]
    vx = c_v[1] 
    vy = c_v[2]
#     SSres_u = z_u.T@z_u - c_u.T@B_u.T@z_u
#     SSres_v = z_v.T@z_v - c_v.T@B_v.T@z_v
    
#     # caclulate residual mean square MSres = SSres/(n-p) = sigma square 
#     sigma_square_u = SSres_u/(n-3)
#     sigma_square_v = SSres_v/(n-3)
    #print(u_uncert)
    sigma_square_u_net = sig2u #+ (np.mean(u_uncert)**2)  #total = misfit sigma^2 + measurement sigma^2
    sigma_square_v_net = sig2v #+ (np.mean(v_uncert)**2)
    
    # Calculate the diagonal elements of Cjj = A.T@A, which is correlation?
    #Cjj = linalg.inv(A.T@A)
    #r_xy = np.corrcoef(xi,yi)[0,1]
    Z_u = linalg.inv(A.T@linalg.inv(sigma_square_u_net)@A)   #@linalg.inv(V_u)
    Z_v = linalg.inv(A.T@linalg.inv(sigma_square_v_net)@A)   #@linalg.inv(V_v)
    
    Ux_uncert = 2*np.sqrt(np.diag(Z_u)[1])
    Uy_uncert = 2*np.sqrt(np.diag(Z_u)[2])
    
    Vx_uncert = 2*np.sqrt(np.diag(Z_v)[1])
    Vy_uncert = 2*np.sqrt(np.diag(Z_v)[2])
    
    vort_uncert = np.sqrt(Vx_uncert**2 + Uy_uncert**2)
    div_uncert = np.sqrt(Ux_uncert**2 + Vy_uncert**2)
    
    strain_rate_uncert = np.sqrt((((ux-vy)*Ux_uncert)**2 + ((ux-vy)*Vy_uncert)**2 + ((vx+uy)*Vx_uncert)**2 +((vx+uy)*Uy_uncert)**2)/(strain_rate**2))
    
    return vort_uncert, div_uncert, strain_rate_uncert
    
    

    
def create_box_data(ds_names):
    combined_ds1 = ds_names
    # Calculate the mean longitude
    central_x = combined_ds1.mean_rotated_x
    central_y = combined_ds1.mean_rotated_y
    
    box_size = 2  # 1 km in meters
    box_half_size = box_size / 2
    #combined_ds1 = combined_ds1.set_coords(['longitude', 'latitude','x', 'y','u','v','u_uncertainty','v_uncertainty']) # Add all the variables and coordinates you need here 
                                                                                    #otherwise the dataset does not crop corretly for the variable not included here
    count = 0
    for j in range(len(combined_ds1.legs.values)):
        #print(j)
        combined_ds =  combined_ds1.isel(legs=j).where(~combined_ds1.isel(legs=j).isnull(), drop=True)# finish coding here
        #combined_ds = combined_ds.set_coords(['longitude', 'latitude','x', 'y','u','v',\      #we can comment this part no need to convert change the variables to coordiates, it works fine
                     #                         'u_uncertainty','v_uncertainty','x_rot',\
                  #                            'y_rot','mean_rotated_x','mean_rotated_y'])
        for i in range(len(combined_ds.time)):
            k = i+count
            #print(midpoint_lon[k].values)
            x_min = central_x.isel(legs=j)[k] - box_half_size 
            x_max = central_x.isel(legs=j)[k] + box_half_size
            y_min = central_y.isel(legs=j)[k] - box_half_size   # for some reason the y constarin was not working properly and for now the y constain does not matter that much
            y_max = central_y.isel(legs=j)[k] + box_half_size

            box = ((combined_ds.y_rot > y_min) & (combined_ds.y_rot < y_max) & (combined_ds.x_rot > x_min) & (combined_ds.x_rot < x_max) )
            # try: 
                            # Generalize for any number of trajectories
            all_trajectories = []
            for traj in range(len(combined_ds.trajectory)):
                trajectory_ds = combined_ds.isel(trajectory=traj).where(box.isel(trajectory=traj), drop=True)
                all_trajectories.append(trajectory_ds)
            
            concatenated_trajectories = xr.concat(all_trajectories, dim='trajectory')
            
            x_prime = (concatenated_trajectories.x - np.mean(concatenated_trajectories.x))
            y_prime = (concatenated_trajectories.y - np.mean(concatenated_trajectories.y))
            #print(x_prime.values.flatten())
            points = np.array(list(zip(x_prime.values[~np.isnan(x_prime.values)].flatten(),y_prime.values[~np.isnan(y_prime.values)].flatten())))
            
            if len(points) < 4:
                # print(concatenated_trajectories.longitude.count().values)
                # print(x_prime.values[~np.isnan(x_prime.values)].flatten())
                eig = np.nan
            else:
                eig = eig_ratio(points)

            concatenated_trajectories['x_prime'] = x_prime
            concatenated_trajectories['y_prime'] = y_prime
            concatenated_trajectories['sigX'] = np.nanstd(x_prime)
            concatenated_trajectories['sigY'] = np.nanstd(y_prime)
            concatenated_trajectories['eig'] = eig

            if k == 0:
                box_ds = concatenated_trajectories
            else:
                #print(k)
                box_ds = xr.concat([box_ds, concatenated_trajectories], dim='box')

        count +=i
    
    return box_ds

def calculate_values(A,u,v,B_u,B_v,z_u,z_v,f,p,u_uncert,v_uncert,sig2u,sig2v):
    wi_u = linalg.inv(sig2u)
    wi_v = linalg.inv(sig2v)
    c_u= linalg.inv(A.T@wi_u@A)@A.T@wi_u@u
    c_v= linalg.inv(A.T@wi_v@A)@A.T@wi_v@v

    vorticity = c_v[1] - c_u[2]
    divergence = c_u[1] + c_v[2]
    sigmaS = c_v[1] + c_u[2]
    sigmaN = c_u[1] - c_v[2]
    strain_rate = np.sqrt(sigmaS**2 + sigmaN**2)
    
    vort_uncert, div_uncert, strain_uncert = KPE_uncertainity(A,c_u,c_v,B_u,B_v,z_u,z_v,p,strain_rate,u_uncert,v_uncert,sig2u,sig2v)

    return vorticity / f, divergence / f, strain_rate / f, vort_uncert/f, div_uncert/f, strain_uncert/f

def vort_div(x, y, u_all, v_all, u_un_all, v_un_all, eig):
    omega = 2*np.pi/(3600*24) + 2*np.pi/(365*86400)
    f = 2*omega*np.sin((37.4*np.pi)/180)

    vort, div, strain, vort_uncert, div_uncert, strain_uncert,no_of_data_points = [], [], [], [], [], [], []

    for i in range(len(x[:, 0, 0])):
        valid_indices = ~np.isnan(u_all[i, :, :])
        #print(valid_indices)

        # Continue only if the eig value satisfies the condition
        if abs(eig[i]) > 0.2:
            p = np.sum(valid_indices) 
            if p > 6:
                A = np.ones((p, 3))
                A[:, 1] = x[i, :, :][valid_indices]
                A[:, 2] = y[i, :, :][valid_indices]
                u_uncert = u_un_all[i,:,:][valid_indices]
                v_uncert = v_un_all[i,:,:][valid_indices]
                sig2u = np.diag(u_uncert**2)
                sig2v = np.diag(v_uncert**2)
                w_u = np.sqrt(1/u_uncert**2)   # 1/sigma = w
                w_v = np.sqrt(1/v_uncert**2)
                #print(len(u_uncert),p)
                B_u = A*w_u[:, np.newaxis]
                B_v = A*w_v[:, np.newaxis]
                # print(i)
                # print('number of points:{}',p)
                
                u = u_all[i,:,:][valid_indices]
                v = v_all[i,:,:][valid_indices]
                z_u = u*w_u
                z_v = v*w_v

                # u_all_nonan = u_all[i, :, :][valid_indices]
                # v_all_nonan = v_all[i, :, :][valid_indices]

                vorticity, divergence, strain_rate, vorticity_uncert, divergence_uncert, strainrate_uncert = calculate_values(A,u,v,B_u,B_v,z_u,z_v,f,p,u_uncert,v_uncert,sig2u,sig2v)
                N = p 
            else:
                vorticity, divergence, strain_rate, vorticity_uncert, divergence_uncert, strainrate_uncert,N = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,np.nan
        else:
            vorticity, divergence, strain_rate, vorticity_uncert, divergence_uncert, strainrate_uncert,N = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,np.nan

        vort.append(vorticity)
        div.append(divergence)
        strain.append(strain_rate)
        vort_uncert.append(vorticity_uncert)
        div_uncert.append(divergence_uncert)
        strain_uncert.append(strainrate_uncert)
        no_of_data_points.append(N)

    return div, vort, strain, vort_uncert, div_uncert, strain_uncert,no_of_data_points



# SD parameters required to calculate kinematic properties
def number_of_sd_parameters(c,ds_names): #n was there in place of c 
    #c = np.arange(n)
    k=0
    #ds = create_box_data(ds_names)  
    xp = ds_names.x_prime*1000
    yp = ds_names.y_prime*1000
    u_all = ds_names.u
    v_all = ds_names.v
    u_un_all = ds_names.u_uncertainty
    v_un_all = ds_names.v_uncertainty
    eig = ds_names.eig
    #print(ds.u[:,0].values)
    
    return xp,yp,u_all,v_all,u_un_all,v_un_all,eig

def calculate_kinematic_properties(n,r,ds_names):
    if r==n:
        c = np.arange(n)
        xp_c,yp_c,u_all_c,v_all_c,u_un_all_c,v_un_all_c,eig = number_of_sd_parameters(c,ds_names)
        kinematic_prop_c = [vort_div(xp_c.values,yp_c.values,u_all_c.isel(cell_depth=i).values,\
                                              v_all_c.isel(cell_depth=i).values,\
                                              u_un_all_c.isel(cell_depth=i).values,\
                                              v_un_all_c.isel(cell_depth=i).values,eig) for i in range(0,24)]                  # should start from 0 instead of 1
        div, vort, strain, vort_uncert, div_uncert, strain_uncert,N = np.array(kinematic_prop_c)[:,0,:], np.array(kinematic_prop_c)[:,1,:],\
                                                                    np.array(kinematic_prop_c)[:,2,:], np.array(kinematic_prop_c)[:,3,:],\
                                                                    np.array(kinematic_prop_c)[:,4,:], np.array(kinematic_prop_c)[:,5,:],\
                                                                    np.array(kinematic_prop_c)[:,6,:]

                
    return xr.DataArray(div),xr.DataArray(vort),xr.DataArray(strain), xr.DataArray(vort_uncert),xr.DataArray(div_uncert),xr.DataArray(strain_uncert),xr.DataArray(N)
        
        
    

def least_squares(y,x):
    
    
    N = len(y)
    A = np.ones((N,2)) + 0*1j
    A[:,1] = x
    coefs = linalg.inv(A.T@A)@A.T@y
    y_ls = coefs[0] + coefs[1]*x
    return x,y_ls,coefs

#adcp_oceanus.time.values[3278:3280]


def earth_radius(lat=37):
    
    """ 
    Calculates sea-level Earth's radius in kilometers
        at a given latitude (lat).
    """
    
    Re = 6378.137  # Earth's equatorial radius in km
    Rp = 6356.752  # Earth's polar radius in km
    
    theta = lat*np.pi/180
    
    a = ((Re**2)*np.cos(theta))**2
    b = ((Rp**2)*np.sin(theta))**2
    c = (Re*np.cos(theta))**2
    d = (Rp*np.sin(theta))**2

    R = np.sqrt((a+b)/(c+d))

    return R




def ll2xy(lon,
          lat,
          lon0=-124,
          lat0=37.25):
    
    """ 
    Converts longitude and latitude (lon,lat)
        arrays to Cartesian distance (x,y) about a 
        reference coordinate (lon0, lat0)    
    """
    
    lat2km = earth_radius(lat0)*np.pi/180
    
    x = (lon - lon0)*lat2km * np.cos(np.pi*lat0/180)
    y = (lat - lat0)*lat2km     
    
    return x,y 


def normal_dist(delta, mask=None):
    
    if mask is None:
        data_to_process = np.ravel(delta)
    else:
        data_to_process = np.ravel(delta[mask])
    
    mean = np.nanmean(data_to_process)
    std = np.nanstd(data_to_process)
    
    x = np.linspace(mean - 3*std, mean + 3*std, 50)
    A = 1/std/np.sqrt(2*np.pi)
    f = A*np.exp(-(x-mean)**2/(2*std**2))
    
    return f, x




def basic_stats(ds):
    mean = np.nanmean(ds)
    std = np.nanstd(ds)
    skewness = np.nanmean(skew(ds[pd.notna(ds)]))
    kurt = np.nanmean(kurtosis(ds[pd.notna(ds)]))
    median = np.nanmedian(ds)
    
    return mean, std, skewness, kurt, median 
  
def relative_dis(ds1,ds2):
    dis = []
    for i in range(len(ds1.time.values)):
        lon = [ds1.longitude.isel(time=i),ds2.longitude.isel(time=i)]
        lat = [ds1.latitude.isel(time=i), ds2.latitude.isel(time=i)]
        rdis  = gsw.distance(lon,lat)/1000
        dis =np.append(dis,rdis)
    return dis

def eig_ratio(points):
    # Calculate the variances and covariance
    var_x = np.var(points[:, 0])
    var_y = np.var(points[:, 1])
    cov_xy = np.cov(points[:, 0], points[:, 1])[0, 1]
    cov_mat = np.array([[var_x, cov_xy], [cov_xy, var_y]])

    # Compute the eigenvalues and eigenvectors
    eigvals, eigvecs = np.linalg.eig(cov_mat)

    # Compute the ratio of the smallest eigenvalue to the largest eigenvalue
    eig_ratio = np.abs(np.min(eigvals) / np.max(eigvals))
    return eig_ratio
 
  