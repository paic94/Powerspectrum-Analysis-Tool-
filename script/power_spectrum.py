from __future__ import division
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import numpy as np
import os,sys
import timeit
import multiprocessing
from functools import partial
import pickle as pkl
import pycs
from astropy.io import fits
from scipy.signal import lombscargle, periodogram
from scipy.stats import multivariate_normal
#import astroML.time_series as amlt
import glob
import matplotlib
from mpl_toolkits.axes_grid.inset_locator import (inset_axes, InsetPosition, mark_inset)
import pycs3.gen.lc_func
import pycs3.gen.mrg
import pycs3.spl.topopt
import pycs3.regdiff.rslc
import pycs3.gen.util
from pycs3.gen.util import datetimefromjd

import matplotlib.gridspec as gridspec
from pylab import *
import matplotlib.tri as tri

from matplotlib import ticker
from matplotlib.ticker import LogLocator, LogFormatterSciNotation as LogFormatter
from astropy.modeling.functional_models import Gaussian2D
#execfile('useful_functions.py')

from matplotlib.patches import Ellipse
from scipy.stats import chi2
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


#Useful paths :
datadir = "/home/ericpaic/Documents/PHD/data/"
scriptdir = "/home/ericpaic/Documents/PHD/script/"
resultdir = "/home/ericpaic/Documents/PHD/results/powerspectrum/"
plotdir = resultdir+"plots/"
pkldir = resultdir+"pkl/"
storagedir = "/home/ericpaic/Documents/PHD/Powerspectrum-Analysis-Tool-/maps/with_macromag/"
drwdir = "/home/ericpaic/Documents/PHD/data/DRW/"
#Plot options
font = {'family' : 'normal',
        'size'   : 12}
matplotlib.rc('xtick', labelsize = 12)
matplotlib.rc('ytick', labelsize = 12)
matplotlib.rc('font', **font)


einstein_r_1131= 2.5e16 #Einstein ring of RXJ1131 in cm assuming <M> = 0.3 M_0
einstein_r_03 = 3.414e16 #Einstein ring of QJ0158 in cm assuming <M> = 0.3 M_0
einstein_r_01 = einstein_r_03/np.sqrt(3) #Einstein ring of QJ0158 in cm assuming <M> = 0.1 M_0
einstein_r_001 = einstein_r_03/np.sqrt(30) #Einstein ring of QJ0158 in cm assuming <M> = 0.01 M_0

cm_per_pxl = 20*einstein_r_03/8192 #Find pixel scale assuming the map is 20R_E x 20R_E and 8192 pxl x 8192pxl
ld_per_pxl = cm_per_pxl/(30000000000*3600*24) #Light-day per pixel

A = np.array([0.0,0.0])
B = np.array([-1.156, -0.398])
G = np.array([-0.780,-0.234])

A -= G
B -= G
G -= G

gamma_A = np.arctan(A[1]/A[0])#+np.pi/2.
gamma_B = np.arctan(B[1]/B[0])#+np.pi/2.

#v_source = 31
#day_per_pxl = cm_per_pxl/(100000*v_source*3600*24)


def nice_colorbar(mappable,ax, position='right', pad=0.1, size='5%', label=None, fontsize=12,
                  invisible=False, divider_kwargs={}, colorbar_kwargs={}, label_kwargs={}):
    divider_kwargs.update({'position': position, 'pad': pad, 'size': size})
    divider = make_axes_locatable(ax)
    cax = divider.append_axes(**divider_kwargs)
    if invisible:
        cax.axis('off')
        return None
    cb = plt.colorbar(mappable, cax=cax, **colorbar_kwargs)
    if label is not None:
        colorbar_kwargs.pop('label', None)
        cb.set_label(label, fontsize=fontsize, **label_kwargs)
    return cb

def LSperiodogram(data,time, errdata, frequencies):
    # A periodogram supposed to work on not uniformly sampled data but the tests showed lack of robustness
    power = amlt.lomb_scargle(time, data, errdata, frequencies)
    return power

def pvalue(sim, data, errdata):
    chi = chi2_custom(sim, data, errdata)
    return 1 - chi2.cdf(chi, 1)

def likelyhood(chi):
    return np.exp(-chi/2)

def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    '''
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero.

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower offset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax / (vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highest point in the colormap's range.
          Defaults to 1.0 (no upper offset). Should be between
          `midpoint` and 1.0.
    '''
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False),
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap

def good_draw_LC(params, map, time, err_data,spline_fit=False):
    x_start = params[0]
    y_start = params[1]
    v = params[2]
    angle = params[3]

    v_x = np.multiply(v, np.cos(angle))
    v_x = np.divide(np.multiply(100000 * 3600 * 24, v_x), cm_per_pxl)

    v_y = np.multiply(v, np.sin(angle))
    v_y = np.divide(np.multiply(100000 * 3600 * 24, v_y), cm_per_pxl)

    if x_start + (time[-1] - time[0]) * v_x <= len(map) and y_start + (time[-1] - time[0]) * v_y <= len(
            map) and x_start + (time[-1] - time[0]) * v_x >= 0 and y_start + (time[-1] - time[0]) * v_y >= 0:
        if v_x == 0:
            path_x = x_start * np.ones(len(time))
        else:
            path_x = np.add(np.multiply(np.add(time, -time[0]), v_x), x_start)
        if v_y == 0:
            path_y = y_start * np.ones(len(time))
        else:
            path_y = np.add(np.multiply(np.add(time, -time[0]), v_y), y_start)

        path_x = path_x.astype(int)
        path_y = path_y.astype(int)

        lc = map[path_y, path_x]

        if spline_fit==True:
            curve = pycs.gen.lc.factory(time, lc, magerrs=err_data)
            spline = pycs.gen.spl.fit([curve], knotstep=300, bokeps=20, verbose=False)
            lc = spline.eval(time)
        #temp = np.add(np.multiply(-2.5, np.log10(map[path_y, path_x])),
        #              np.random.normal(0, np.mean(err_data), len(path_y)))  # -2.5 log() to convert flux into mag
        #lc = temp - temp[0] * np.ones(len(temp))
        return lc

def draw_LC_withreverberation(params, map_A, map_B, time, err_data, f,Mc=1.,fBLR=0.43,muBLR=1.):
    f_cut = 0.1

    whichcurve = 1
    lc_c = lc_c_list[whichcurve]
    lc_BLR = lc_BLR_list[whichcurve]
    ml_A = good_draw_LC(params[0], map_A, time, err_data)
    ml_B = good_draw_LC(params[1], map_B, time, err_data)

    Mc_A = 2.24
    Mc_B = 0.84
    # lc_c -= np.mean(lc_c)
    # lc_BLR -= np.mean(lc_BLR)

    # print "fBLR : %s"%(fBLR)

    if ml_A is not None and ml_B is not None:
        ml_A = np.abs(ml_A)
        ml_B = np.abs(ml_B)
        # print Mc * fBLR * muBLR * lc_BLR

        lc_A = -2.5 * np.log10((1.-fBLR)*Mc * ml_A * lc_c + Mc * fBLR * muBLR * lc_BLR)
        lc_B = -2.5 * np.log10((1.-fBLR)*Mc * ml_B * lc_c + Mc * fBLR * muBLR * lc_BLR)
        lc_reverberation = lc_A - lc_B
        lc_reverberation -= lc_reverberation[0]

    return lc_reverberation


def schuster_periodogram(t, mag, freq):
    # Classic Fourrier transform. Requires uniformly sampled data and not robust enough.
    t, mag, freq = map(np.asarray, (t, mag, freq))
    return abs(np.dot(mag, np.exp(-2j * np.pi * freq * t[:, None])) / np.sqrt(len(t))) ** 2



def power_spectrum(param, map,time,err_data, cm_per_pxl, f, detrend = 'constant', window = 'flattop'):
    # Uses good_draw_LC to draw a lightcurve and then computes its powerspectrum using scipy.signal.periodogram function
    # for f, detrend and window arguments check https://docs.scipy.org/doc/scipy-0.13.0/reference/generated/scipy.signal.periodogram.html

    temp = good_draw_LC(param, map, time, err_data, cm_per_pxl)

    if temp is not None:
        lc = temp
        f_cut = 0.01 # highest frequency plotted in days^-1, this is limited by the sampling of the original lightcurve.
        frequency, power = periodogram(lc, f, window=window, detrend=detrend)
        frequency = np.array(frequency[1:])
        power = np.array(power[1:])
        #ax[0].plot(time, lc,alpha=0.7)
        #ax[1].plot(frequency[frequency<f_cut], power[frequency<f_cut], alpha = 0.7)
        return [power, frequency]

def chi2_custom(sim,err_sim, data, err_data):
    # Measure the quality of fit of the simulted powerspectrum. In this function, a classic chi square formula is implemented
    err_tot = err_sim +err_data**2
    chi2 = np.sum(np.divide(np.abs(sim - data), err_tot))
    return chi2 / len(sim)

def weighted_mean(a, err_a, b, err_b):
    weights_a = 1/(err_a**2)
    weights_b = 1/(err_b**2)
    weights = np.array([weights_a, weights_b])
    data = np.array([a,b])
    return np.average(data,weights=weights, axis=0)

def PS_2maps(params,map_A, map_B, time, err_data, f):

    params_A = params[0]
    params_B = params[1]
    f_cut=0.1
    ml_A = good_draw_LC(params_A, map_A, time,err_data)
    ml_B = good_draw_LC(params_B, map_B, time,err_data)

    Mc_A = 2.24
    Mc_B = 0.84
    #lc_c -= np.mean(lc_c)
    #lc_BLR -= np.mean(lc_BLR)

    #print "fBLR : %s"%(fBLR)

    if ml_A is not None and ml_B is not None:
        ml_A = np.abs(ml_A)
        ml_B = np.abs(ml_B)

        lc = -2.5*np.log10(ml_A*Mc_B/(ml_B*Mc_A) )
        lc-=lc[0]

        frequency, power = periodogram(lc, f, window="flattop")
        frequency = np.array(frequency)
        power=np.array(power)


        return frequency[frequency<f_cut], power[frequency<f_cut]

def twoD_Gaussian((x, y), amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    xo = float(xo)
    yo = float(yo)
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo)
                            + c*((y-yo)**2)))
    return g.ravel()

def fmt(x, pos):
    return round(np.log10(x),3)




z_QSO = 1.29
#----------------------------------Import Data-----------------------------------------------
f = open(datadir+"microlensing/data/J0158_Euler_microlensing_upsampled_B-A.rdb","r")
f2 = open(datadir+"microlensing/data/RXJ1131_ALL_microlensing_upsampled_B-A.rdb","r")

f= f.read()
f=f.split("\n")
data = f[2:]
window = 'flattop'
mjhd = np.array([]) # Time
mag_ml = np.array([]) # Magnitude od microlensing
err_mag_ml = np.array([]) # Error on the magnitude

for i,elem in enumerate(data):
    mjhd = np.append(mjhd,float(elem.split("\t")[0]))
    mag_ml = np.append(mag_ml, float(elem.split("\t")[1]))
    temp = elem.split("\t")[2]
    err_mag_ml= np.append(err_mag_ml,float(temp.split("\r")[0]))

magshift_ml= 0.45
mag_ml +=magshift_ml
f2= f2.read()
f2=f2.split("\n")
data_2 = f2[2:]

mjhd_2 = np.array([])
mag_ml_2 = np.array([])
err_mag_ml_2 = np.array([])

for i,elem2 in enumerate(data_2):
    mjhd_2 = np.append(mjhd_2,float(elem2.split("\t")[0]))
    mag_ml_2= np.append(mag_ml_2,float(elem2.split("\t")[1]))
    temp = elem2.split("\t")[2]
    err_mag_ml_2= np.append(err_mag_ml_2,float(temp.split("\r")[0]))

sampling = 1
print max(mjhd/(1+z_QSO))-min(mjhd/(1+z_QSO))
new_mjhd_QSO = np.arange(min(mjhd), min(mjhd)+int(max(mjhd/(1+z_QSO))-min(mjhd/(1+z_QSO))),sampling)
new_mjhd = np.arange(min(mjhd), max(mjhd),sampling)
vfile = open('/home/ericpaic/Documents/PHD/data/velocities_QJ0158.dat', "r")
vfile = vfile.read()
vfile = vfile.split("\n")

v1 = np.array([])
angle1 = np.array([])
v2 = np.array([])
angle2 = np.array([])
v3 = np.array([])
angle3 = np.array([])
v4 = np.array([])
angle4 = np.array([])
for i, elem in enumerate(vfile):
    temp = elem.split(' ')
    while '' in temp:
        temp.remove('')
    # if i%1000==0:
    #    print i
    if len(temp)>1:
        v1 = np.append(v1, float(temp[0]))
        v2 = np.append(v2, float(temp[2]))
        v3 = np.append(v3, float(temp[4]))
        v4 = np.append(v4, float(temp[6]))
        angle1 = np.append(angle1, np.radians(float(temp[1])))
        angle2 = np.append(angle2, np.radians(float(temp[3])))
        angle3 = np.append(angle3, np.radians(float(temp[5])))
        angle4 = np.append(angle4, np.radians(float(temp[7])))

sampled_v1, bin_edges, patches =plt.hist(v1,bins = 50, range = (100,5100),density = True, label='Data', alpha = 0.3)
#bin_middles = 0.5 * (bin_edges[1:] + bin_edges[:-1])
#print len(bin_edges)
#print bin_edges
#print len(sampled_v1)
#print bin_middles
#plt.show()


if 0 :
    # Plot example lc
    # Creating mock lightcurves using a given magnification map for a given velocity of the source


    list_r0 = [2,15,100]
    list_r0cm = [r'5$\cdot 10^{14}$ cm',r'2$\cdot 10^{15}$ cm',r'2$\cdot 10^{16}$ cm']
    n_curves = 1  # number of generated curves
    colors = pl.cm.viridis(np.linspace(0, 1, len(list_r0)))
    v_source = 700  # in km.s^-1

    x = 5484
    y = 4500
    angle = 3.14/2
    fig, ax = plt.subplots(1,1,figsize = (5,5))
    params = [x, y, v_source, angle]
    for i,r0 in enumerate(list_r0):
        print i
        map = storagedir + "mapA-B_R%s_thin_disk.fits" % (r0)
        final_map = fits.open(map)[0]
        final_map = final_map.data[:, :]

        #final_map = final_map * (np.mean(final_mapB)/np.mean(final_mapA))




        temp = good_draw_LC(params, final_map, new_mjhd, err_mag_ml)
        lc =-2.5*np.log10(temp)
        ax.plot(new_mjhd-new_mjhd[0], lc, label =r"$R_0$ = %s"%(list_r0cm[i]), linewidth=4)

    ax.set(title=r'$v_e$ = 700km$\cdot s^{-1}$',xlabel='Time [days]',ylabel=r'$m_A$ - $m_B$ [mag]')
    fig.suptitle('Simulated differential microlensing lightcurve')
    ax.legend()
    ax.invert_yaxis()
    plt.show()
    fig.savefig(plotdir+'ex_lc.png',dpi=100)
    sys.exit()

if 0 :
    #Check spectral leakage
    f_cut = 0.01
    lc = pycs.gen.lc.factory(mjhd, mag_ml, magerrs=err_mag_ml)
    spline = pycs.gen.spl.fit([lc], knotstep=70, verbose=False)
    new_magml = spline.eval(new_mjhd)
    spline_long = pycs.gen.spl.fit([lc], knotstep=1200, verbose=False)
    long_magml = spline_long.eval(new_mjhd)
    short_magml = new_magml-long_magml

    frequency_flattop_whole, power_flattop_whole = periodogram(new_magml, 1, window='flattop')
    print len(frequency_flattop_whole)
    print len(power_flattop_whole)
    power_flattop_whole= power_flattop_whole[1:len(frequency_flattop_whole[frequency_flattop_whole < f_cut])]
    frequency_flattop_whole= frequency_flattop_whole[1:len(frequency_flattop_whole[frequency_flattop_whole < f_cut])]

    frequency_flattop_short, power_flattop_short = periodogram(short_magml, 1, window='flattop')
    power_flattop_short = power_flattop_short[1:len(frequency_flattop_short[frequency_flattop_short< f_cut])]
    frequency_flattop_short = frequency_flattop_short[1:len(frequency_flattop_short[frequency_flattop_short < f_cut])]

    frequency_flattop_long, power_flattop_long = periodogram(long_magml, 1, window='flattop')
    power_flattop_long = power_flattop_long[1:len(frequency_flattop_long[frequency_flattop_long < f_cut])]
    frequency_flattop_long = frequency_flattop_long[1:len(frequency_flattop_long[frequency_flattop_long < f_cut])]

    frequency_hamming_whole, power_hamming_whole = periodogram(new_magml, 1, window='hamming')
    power_hamming_whole = power_hamming_whole[1:len(frequency_hamming_whole[frequency_hamming_whole < f_cut])]
    frequency_hamming_whole = frequency_hamming_whole[1:len(frequency_hamming_whole[frequency_hamming_whole < f_cut])]

    frequency_hamming_short, power_hamming_short = periodogram(short_magml, 1, window='hamming')
    power_hamming_short = power_hamming_short[1:len(frequency_hamming_short[frequency_hamming_short < f_cut])]
    frequency_hamming_short = frequency_hamming_short[1:len(frequency_hamming_short[frequency_hamming_short < f_cut])]

    frequency_hamming_long, power_hamming_long = periodogram(long_magml, 1, window='hamming')
    power_hamming_long = power_hamming_long[1:len(frequency_hamming_long[frequency_hamming_long < f_cut])]
    frequency_hamming_long = frequency_hamming_long[1:len(frequency_hamming_long[frequency_hamming_long < f_cut])]

    gs = gridspec.GridSpec(2,2)
    fig = plt.figure(figsize = (15,15))
    ax1 = plt.subplot(gs[:, 0])
    ax1.plot(new_mjhd, new_magml, 'r', label= 'Spline example')
    ax1.plot(new_mjhd, short_magml, label='Short variations')
    ax1.plot(new_mjhd, long_magml, 'g', label='Long variations')
    ax1.set(ylabel = r'$m_A$-$m_B$ (mag)', xlabel = 'HJD-2400000.5 (day)')
    ax1.legend()

    ax2 = plt.subplot(gs[0, 1])
    ax2.plot(frequency_flattop_whole, power_flattop_whole, 'r')
    ax2.plot(frequency_flattop_short, power_flattop_short)
    ax2.plot(frequency_flattop_long, power_flattop_long, 'g')
    ax2.text(0.001, power_flattop_whole[0], 'Flat top window')
    ax2.set(xscale = 'log', yscale = 'log', ylabel = 'Power (A.U) ', xlim = (np.min(frequency_flattop_whole),0.01) )
    #idxflattop = 4
    #ax2.fill_between(frequency_flattop_whole[:idxflattop], power_flattop_whole[:idxflattop], power_flattop_long[:idxflattop], alpha = 0.3, facecolor= 'r')
    #ax2.fill_between(frequency_flattop_whole[:idxflattop], power_flattop_whole[:idxflattop],
    #                 power_flattop_short[:idxflattop], alpha = 0.3, facecolor = 'r')
    ax2.xaxis.set_visible(False)

    ax3 = plt.subplot(gs[1, 1], sharex=ax2)
    ax3.plot(frequency_hamming_whole, power_hamming_whole, 'r')
    ax3.plot(frequency_hamming_short, power_hamming_short)
    ax3.plot(frequency_hamming_long, power_hamming_long, 'g')
    ax3.text(0.001, np.max(power_flattop_whole), 'Hamming window')
    ax3.set(xscale = 'log', yscale = 'log', ylabel = 'Power (A.U) ', xlabel = r'Frequency (days$^{-1}$)', xlim = (np.min(frequency_flattop_whole),0.01 ))
    #idxhamming = 5
    #ax3.fill_between(frequency_hamming_whole[:idxhamming],
    #                 power_hamming_whole[:idxhamming],
    #                 power_hamming_long[:idxhamming], alpha=0.3, facecolor='r')

    #ax3.fill_between(frequency_hamming_whole[idxhamming:],
    #                 power_hamming_whole[idxhamming:],
    #                 power_hamming_short[idxhamming:], alpha=0.3, facecolor='r')

    plt.subplots_adjust(hspace=.0)

    #locs = ax3.get_xticks()

    #locs[0] = frequency_flattop_whole[0]

    #temp_lab = ax3.get_xticklabels()
    #lab = np.divide(1, locs).astype(int)
    #print lab
    #labels = []
    #for label in lab[1:-1]:
    #    labels.append('$\\frac{{{:2.0f}}}{{{:2.0f}}}$'.format(1, int(label)))
    #print labels
    # labels[0]='0'
    #ax3.set_xticks(locs[1:-2], minor=False)
    #ax3.set_xticklabels(labels, minor=False)

    plt.show()
    fig.savefig(resultdir +"spectral_leakage.pdf")
    sys.exit()

if 0:
    # Testing the robustness of the powerspectrum for different sampling of the curve
    lc = pycs.gen.lc.factory(mjhd, mag_ml, magerrs=err_mag_ml)


    fig, ax = plt.subplots(1, 1, figsize=(15, 15))
    list_ks = np.arange(20,120,1)
    chi = []
    for ks in list_ks:
        spline = pycs.gen.spl.fit([lc], knotstep=ks, verbose=False)
        new_magml = spline.eval(mjhd)
        new_err_mag_ml = np.random.normal(np.mean(err_mag_ml), np.var(err_mag_ml), len(new_magml))
        chi.append(chi2_custom(new_magml, mag_ml, err_mag_ml))

    ax.plot(list_ks, chi)
    plt.show()
    #plt.savefig(resultdir +"robustness_sampling.png")
    sys.exit()

if 0:
    # Power spectrum of the data

    rdbfile = "lightcurves/J0158_Euler.csv"

    datadir = "/home/ericpaic/Documents/PHD/data/"
    n_realisation = 1
    fig, ax = plt.subplots(2, 1, sharex=True)
    fig.subplots_adjust(hspace=0)
    lcs = [
        pycs3.gen.lc_func.flexibleimport(rdbfile, jdcol=1, magcol=2, errcol=3, startline=2),
        pycs3.gen.lc_func.flexibleimport(rdbfile, jdcol=1, magcol=4, errcol=5, startline=2)
    ]
    list_ks = np.arange(30, 105, 5)
    for ks in list_ks:
        print ks
        splineA = pycs.gen.spl.fit([lcs[0]], knotstep=ks, verbose=False)
        splineB = pycs.gen.spl.fit([lcs[1]], knotstep=ks, verbose=False)
        new_magmlA = splineA.eval(new_mjhd)
        new_err_mag_mlA = np.random.normal(np.mean(err_mag_ml), np.std(err_mag_ml), len(new_magmlA))
        new_magmlB = splineB.eval(new_mjhd)
        new_err_mag_mlB = np.random.normal(np.mean(err_mag_ml), np.std(err_mag_ml), len(new_magmlB))
        plt.errorbar(new_mjhd, new_magmlA, yerr=new_err_mag_mlA)
        plt.errorbar(new_mjhd, new_magmlB, yerr=new_err_mag_mlB)

        plt.show()
        with open(resultdir + 'splines4dom/splinefit_eta%s_2images.pkl' % (ks), 'wb') as handle:
            pkl.dump((new_magmlA, new_err_mag_mlA,new_magmlB, new_err_mag_mlB), handle, protocol=pkl.HIGHEST_PROTOCOL)
    sys.exit()

if 0:
    # Power spectrum of the data
    rdbfile = "lightcurves/J0158_Euler.csv"

    datadir = "/home/ericpaic/Documents/PHD/data/"
    n_realisation = 1
    fig, ax = plt.subplots(3, 1, sharex = True,gridspec_kw={'height_ratios':[3,3,1]})
    fig.subplots_adjust(hspace =0)
    lcs = [
        pycs3.gen.lc_func.flexibleimport(rdbfile, jdcol=1, magcol=2, errcol=3, startline=2),
        pycs3.gen.lc_func.flexibleimport(rdbfile, jdcol=1, magcol=4, errcol=5, startline=2)
    ]

    disptext = []

    shiftA = 0.0
    shiftB = 0.2

    lcs[0].shiftmag(+30)
    lcs[1].shiftmag(-shiftB + 30)

    jdrange = [53300, 58500]
    lcsnames = [r"$S_{\rm A}$", r"$S_{\rm B}$"]
    lcsaddtxt = ["", "-" + str(shiftB)]

    magrange = [19, 17.2]

    pycs3.gen.mrg.colourise(lcs)  # Gives each curve a different colour.

    for ind, lc in enumerate(lcs):
        firstpt = (lc.getjds()[20], lc.getmags()[20])
        print(firstpt)
        xcoord = (firstpt[0] - 400 - jdrange[0]) / (jdrange[1] - jdrange[0])
        if ind == 0:
            ycoord = (firstpt[1] - 0.02 - magrange[0]) / (magrange[1] - magrange[0])
        else:
            ycoord = (firstpt[1] - 0.02 - magrange[0]) / (magrange[1] - magrange[0])
        txt = lcsnames[ind]
        colour = lc.plotcolour
        kwargs = {"fontsize": 20, "color": lc.plotcolour}
        disptext.append((xcoord, ycoord, txt, kwargs))

        xcoordadd = (firstpt[0] - 400 - jdrange[0]) / (jdrange[1] - jdrange[0])
        ycoordadd = (firstpt[1] + 0.1 - magrange[0]) / (magrange[1] - magrange[0])
        txtadd = lcsaddtxt[ind]
        colouradd = lc.plotcolour
        kwargsadd = {"fontsize": 12, "color": lc.plotcolour}
        disptext.append((xcoordadd, ycoordadd, txtadd, kwargsadd))

    timeshift = [0, 22.7]
    magshift = [0, 0]
    # pycs3.gen.lc_func.applyshifts(lcs, timeshift, magshift)

    #BonA = interpolate(lcs[0], lcs[1])
    #AonB = interpolate(lcs[1], lcs[0])
    #ml_AonB = difflc(lcs[1], AonB)
    #ml = difflc(BonA, lcs[0])

    # ml.merge(ml_AonB)

    # lcs.append(AonB)#
    # lcs.append(BonA)
    # lcs.append(ml)

    pycs.gen.lc.display(lcs, jdrange=jdrange, magrange=magrange, text=disptext,
                        filename=resultdir + "lc_0158.pdf", transparent=False, style='homepagepdfnologo', ax = ax[0])

    f_cut=1

    power= []
    inset = plt.axes([0, 0, 1, 1])
    ip = InsetPosition(ax[1], [0.5,0.1,0.4,0.4])
    inset.set_axes_locator(ip)
    mark_inset(ax[1], inset, loc1=1, loc2=2, fc="w", ec='0.5')

    list_ks = np.arange(30, 105, 5)
    #list_ks = np.flip(list_ks)
    ax[1].errorbar(mjhd, mag_ml, yerr=err_mag_ml, c='r', marker='.', ls='', alpha=0.3, label = r'$S_A - S_B$')
    inset.errorbar(mjhd, mag_ml, yerr=err_mag_ml, c='r', marker='.', ls='', alpha=0.3)
    #plt.setp(inset.get_xticklabels(), visible=True)

    list_plot_ks = [30,70,100]
    list_color_text = ['tab:blue', 'tab:orange','tab:green']
    for ks in list_ks:
        print ks
        for j in range(n_realisation):
            magml2 = mag_ml + np.random.normal(0, np.mean(err_mag_ml), len(mag_ml))
            lc = pycs.gen.lc.factory(mjhd, magml2, magerrs=err_mag_ml)
            spline = pycs.gen.spl.fit([lc], knotstep=ks, verbose=False)
            new_magml = spline.eval(new_mjhd)

            new_err_mag_ml = np.random.normal(np.mean(err_mag_ml), np.std(err_mag_ml), len(new_magml))
            frequency_spline, power_spline = periodogram(new_magml, window = 'flattop')
            power.append(power_spline)
            #ax[1].plot(frequency_spline[1:len(frequency_spline[frequency_spline<f_cut])], power_spline[1:len(frequency_spline[frequency_spline<f_cut])], label="%s knotstep"% (ks))
            if j ==0 and ks in list_plot_ks:
                ax[1].plot(new_mjhd, new_magml, label="$\eta$ =%s "% (ks))
                inset.plot(new_mjhd, new_magml)
                ax[2].errorbar(mjhd, (spline.eval(mjhd)-mag_ml)+(0.1*list_plot_ks.index(ks)-0.1), label="$\eta$ =%s " % (ks),yerr=err_mag_ml, marker='.', ls='', alpha=0.3 )
                ax[2].text(mjhd[0]-200, 0.1*list_plot_ks.index(ks)-0.12, '%s'%(0.1*list_plot_ks.index(ks)-0.1), color=list_color_text[list_plot_ks.index(ks)])

    minjd = ax[1].get_xlim()[0]
    maxjd = ax[1].get_xlim()[1]
    # axes.set_xlim(minjd, maxjd)
    yearsx = ax[1].twiny()
    yearxmin = datetimefromjd(minjd + 2400000.5)
    yearxmax = datetimefromjd(maxjd + 2400000.5)
    yearsx.set_xlim(yearxmin, yearxmax)
    yearsx.xaxis.set_minor_locator(matplotlib.dates.MonthLocator())
    yearsx.xaxis.set_major_locator(matplotlib.dates.YearLocator())
    #ax[1].xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%Y'))

    #yearx.xaxis.tick_top()
    yearsx.grid( linestyle='dotted')
    yearsx.set_xticklabels('')


    # axes.set_xlim(minjd, maxjd)
    yearsx_2 = ax[2].twiny()
    yearxmin_2 = datetimefromjd(minjd + 2400000.5)
    yearxmax_2 = datetimefromjd(maxjd + 2400000.5)
    yearsx_2.set_xlim(yearxmin_2, yearxmax_2)
    yearsx_2.xaxis.set_minor_locator(matplotlib.dates.MonthLocator())
    yearsx_2.xaxis.set_major_locator(matplotlib.dates.YearLocator())
    # ax[1].xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%Y'))

    # yearx.xaxis.tick_top()
    yearsx_2.grid(linestyle='dotted')
    yearsx_2.set_xticklabels('')

    print len(power)
    power = np.array(power)

    mean = []
    std = []
    for i in range(len(power[0])):
        mean.append(np.median(power[:, i]))
        std.append(np.std(power[:, i]))

    pklfile = pkldir+"data_%s_contracted.pkl"%(n_realisation)
    with open(
            pklfile, 'wb') as handle:
        pkl.dump((mean, std, frequency_spline), handle,
                 protocol=pkl.HIGHEST_PROTOCOL)

    #ax[1].plot(frequency_lin[1:len(frequency_lin[frequency_lin<f_cut])], power_lin[1:len(frequency_lin[frequency_lin<f_cut])], label = 'Linear interp')


    ax[1].legend()
    #ax[2].legend()
    ax[1].set(ylabel = r'$m_A$-$m_B$ +$\mathcal{M}_A$-$\mathcal{M}_B$ (mag)', xlabel = 'HJD-2400000.5 (day)')
    ax[2].set(ylabel=r'Residuals', xlabel='HJD-2400000.5 (day)')
    inset.set(xlim=(56400,56800),ylim=(0.9+magshift_ml,1.12+magshift_ml))
    inset.set_facecolor("white")

    #inset.set_xticklabels([])
    #inset.set_yticklabels([], minor=False)
    #inset.set_ylim(min(power_spline[10:]), max(power_spline))



    plt.show()
    fig.savefig(resultdir + "lc_0158_withml.pdf", dpi = 100)
    sys.exit()

if 0:
    power_data, std_data, freq_data = pkl.load(open(pkldir + 'data.pkl', 'rb'))
    new_power_spline = power_data[:]
    f_data = freq_data[1:]
    std_data = np.array(std_data[:])

    f_boundary = 1/100

    print len(f_data[f_data<f_boundary])
    print len(f_data)

    f_high = f_data[len(f_data[f_data<f_boundary]):]
    power_high = new_power_spline[len(f_data[f_data<f_boundary]):]
    std_high = std_data[len(f_data[f_data<f_boundary]):]
    print len(f_high)

    fig, ax = plt.subplots(1, 1, sharex=True)
    ax.plot(f_data, new_power_spline, "-r", label="Data power spectrum")
    ax.fill_between(f_data[:len(f_data[f_data<f_boundary])], np.add(new_power_spline[:len(f_data[f_data<f_boundary])], std_data[:len(f_data[f_data<f_boundary])]),
                                         np.subtract(new_power_spline[:len(f_data[f_data<f_boundary])], std_data[:len(f_data[f_data<f_boundary])]), color='r', alpha=0.3, label=r'1-$\sigma$ envelope')

    subsampling = 20
    ax.fill_between(f_high[::subsampling],
                    np.add(power_high[::subsampling], std_high[::subsampling]),
                    np.subtract(power_high[::subsampling], std_high[::subsampling]), color='r',
                    alpha=0.3)

    ax.fill_between([0.01, 10], [0, 0], [1000, 1000], alpha=0.1, facecolor='k')
    ax.axvline(1 / 750, 0, 1000, ls='--', c='k')

    #labels = [item.get_text() for item in ax.get_xticklabels()]
    #print labels
    #new_labels = [float(l)/365. for l in labels]

    #invyearsx.set_xticklabels(new_labels)

    ax.legend()
    ax.set(yscale='log', xscale='log', xlabel=r'Frequency (days$^{-1}$)', ylabel='Power (A.U.)',
            xlim=(np.min(f_data), np.max(f_data)), ylim=(np.min(new_power_spline), np.max(new_power_spline) + 10))
    #invyearsx = ax.twiny()

    #miny = ax.get_xlim()[0]
    #maxy = ax.get_xlim()[1]
    # axes.set_xlim(minjd, maxjd)
    #yearsx = ax[1].twiny()
    #invyearsx.set_xscale('log')
    #invyearsx.set_xlim(miny, maxy)
    #invyearsx.xaxis.set_minor_locator(ticker.LogLocator(base=10.0, numticks=15))
    #invyearsx.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=3))

    #labels = [item for item in ax.get_xticks()]
    #print labels
    #new_labels = [1/3*365,1/2*365,1/365]
    #invyearsx.set_xticklabels(new_labels)
    #print labels
    #new_labels = [float(l)/365. for l in labels]

    #invyearsx.xaxis.set_major_locator([1/365, 2/365, 3/365])
    #ax.grid(True, ls = '--')


    plt.show()
    fig.savefig(resultdir + "powerspectrum_data.pdf", dpi = 100)


    #locs = ax[1].get_xticks()
    #print locs
    #locs[0] = frequency_spline[0]

    #temp_lab = ax[1].get_xticklabels()
    #lab = np.divide(1, locs).astype(int)
    #print lab
    #labels = []
    #for label in lab[1:-1]:
    #    labels.append('$\\frac{{{:2.0f}}}{{{:2.0f}}}$'.format(1, int(label)))
    #print labels
        # labels[0]='0'
    #ax[1].set_xticks(locs[1:-2], minor=False)
    #ax[1].set_xticklabels(labels, minor=False)
    sys.exit()


if 0:
    # Creating mock lightcurves using a given magnification map for a given velocity of the source

    list_r0 = [2,4,10,15,20,30,40,60,80,100] #radii of the source in pxl
    n_curves = 100000 #number of generated curves
    n_good_curves = 5000 #minimum number of curves that are not flat
    r0_pkl = np.ones(5000*len(list_r0))
    for i,r0 in enumerate(list_r0):
        print r0
        r0_pkl[5000*i:5000*(i+1)]= r0*np.ones(5000)
        map = storagedir + "FML0.9/M0,3/mapA-B_fml09_R%s.fits" % (r0)
        img = fits.open(map)[0]
        final_map = img.data[:, :]
        v_source = 500 #in km.s^-1
        v = v_source * np.ones(n_curves)

        x = np.random.random_integers(200, len(final_map) - 200, n_curves)
        y = np.random.random_integers(200, len(final_map) - 200, n_curves)
        angle = np.random.uniform(0, 2 * np.pi, n_curves)
        params = []
        for i, elem in enumerate(x):
            params.append([x[i], y[i], v[i], angle[i]])

        lc = []
        i = 0
        j=0
        while i<n_good_curves:
            temp= good_draw_LC(params[j], final_map,mjhd,(20 * einstein_r_03) / 8192, err_mag_ml)
            j += 1
            if temp is not None:
                temp = temp[0]
                if np.amax(temp)-np.amin(temp) > 1: # we consider a curve is "not flat" if the difference between it min and max is over 1
                    lc.append(temp)
                    i+=1
                    if i%1000 ==0:
                        print i


        with open(resultdir + 'LCmocks/simLC_A-B_n%s_v%s_R%s_M0,3_gaps.pkl' % (
         n_good_curves, v_source,r0), 'wb') as handle:
            pkl.dump((lc), handle, protocol=pkl.HIGHEST_PROTOCOL)

    sys.exit()

if 0:
    #Generating powerspectra of simulated curve
    n_spectrum = 50000
    f = open('/home/ericpaic/Documents/PHD/data/velocities_QJ0158.dat', "r")

    f = f.read()
    f = f.split("\n")

    v1 = np.array([])
    angle1 = np.array([])
    v2 = np.array([])
    angle2 = np.array([])
    v3 = np.array([])
    angle3 = np.array([])
    v4 = np.array([])
    angle4 = np.array([])
    for i, elem in enumerate(f[:n_spectrum]):
        temp = elem.split(' ')

        while '' in temp:
            temp.remove('')
        # if i%1000==0:
        #    print i
        v1 = np.append(v1, float(temp[0]))
        # v2 = np.append(v2, float(temp[2]))
        # v3 = np.append(v3, float(temp[4]))
        # v4 = np.append(v4, float(temp[6]))
        angle1 = np.append(angle1, np.radians(float(temp[1])))
        # angle2 = np.append(angle2, np.radians(float(temp[3])))
        # angle3 = np.append(angle3, np.radians(float(temp[5])))
        # angle4 = np.append(angle4, np.radians(float(temp[7])))
    start2 = timeit.default_timer()


    list_comb=[('A','B')] #name of the maps used
    list_r0 = [2,4,10,15,20,30,40,60,80,100,3,7,12,17,25,35,45,50,55,65,70,75,90] #Radii tested in pxl
    list_v_source = np.arange(100,1500,100)
    for comb in list_comb:
        print comb
        for r0 in list_r0:
            print r0
            map_A = storagedir + "FML0.9/M0,3/with_macromag/convolved_map_%s_fft_thin-disk_%s.fits" % (comb[0], r0)
            img_A = fits.open(map_A)[0]
            final_map_A = img_A.data[:, :]

            map_B = storagedir + "FML0.9/M0,3/with_macromag/convolved_map_%s_fft_thin-disk_%s.fits" % (comb[1], r0)
            img_B = fits.open(map_B)[0]
            final_map_B = img_B.data[:, :]

            for v_source in list_v_source:
                #generating random starting positions and directions

                x_A = np.random.random_integers(0, len(final_map_A) - 1, len(angle1))
                y_A = np.random.random_integers(0, len(final_map_A) - 1, len(angle1))
                #angle_A = np.random.uniform(0, 2 * np.pi, len(angle1))

                angle_A = np.radians(angle1)

                x_B = np.random.random_integers(0, len(final_map_B) - 1, len(angle1))
                y_B = np.random.random_integers(0, len(final_map_B) - 1, len(angle1))
                angle_B = angle_A - (gamma_B - gamma_A)

                v = v_source* np.ones(len(angle1))
                params = []
                for i, elem in enumerate(x_A):
                    params.append([[x_A[i], y_A[i], v[i], angle_A[i]], [x_B[i], y_B[i], v[i], angle_B[i]]])
                    #PS_2maps([[x_A[i], y_A[i], v[i], angle_A[i]], [x_B[i], y_B[i], v[i], angle_B[i]]],map_A = final_map_A, map_B = final_map_B, time = new_mjhd, err_data = new_err_mag_ml, f = 1)

                    #sys.exit()

                parrallel_power_spectrum = partial(PS_2maps,map_A= final_map_A,map_B= final_map_B, time=new_mjhd, err_data=new_err_mag_ml, f=1)

                pool = multiprocessing.Pool(12)
                res = pool.map(parrallel_power_spectrum, params)
                pool.close()
                pool.join()
                #extracting the non None results
                res = filter(None, res)
                res = np.array(res)
                print "@@@@@@@@@@@@@@@@"
                print res.shape
                power = res[:,1]
                print power.shape
                freq = res[:, 0]
                print freq.shape
                mean_power = []
                var_power = []
                power=np.array(power)
                for i in range(len(power[0])):
                    mean_power.append(np.mean(power[:,i]))
                    var_power.append(np.var(power[:,i]))


                stop = timeit.default_timer()
                print stop - start2
                #storing the data in pkl file
                with open(resultdir + 'pkl/angledistrib/spectrum_%s-%s_%s_v%s_R%s_M0,3_1.pkl'%(comb[0], comb[1],n_spectrum,v_source, r0), 'wb') as handle:
                    pkl.dump((mean_power,var_power, freq[0]), handle, protocol=pkl.HIGHEST_PROTOCOL)


    sys.exit()

if 0: #Grid of PS slide
    #Grid plot of powerspectra, each portion contains the powerspectrum for all the velocities for a single radius
    list_comb = [('A3', 'B2')] #name of the maps used to generate the powerspectra
    f_cut = 0.01 #higher frequency considered in days^-1
    r_ref = 15 # Reference radius found in litterature in pxl
    list_r0 = [2,10, 20,30]
    #list_r0 = [5, 10, 24, 36, 49, 73, 146, 195]
    all_v = [400, 700,1100] #Velocities in km.s^-1
    #all_v =[100]
    #list_r0 = [20]
    resultdir ='/home/ericpaic/Documents/PHD/results/powerspectrum/pkl/reverberation_study/'

#    resultdir ='/media/ericpaic/TOSHIBA EXT/TPIVb/results/powerspectrum/'

    #os.chdir(resultdir+'pkl/M0.3/A3-B2_flattop') #path to the pkl containing the powerspectra
    os.chdir(resultdir )
    #Selecting the frequencies of interest of the powerspectrum
    power_data, std_data, freq_data = pkl.load(open(pkldir+'data.pkl', 'rb'))
    new_power_spline = power_data[1:len(freq_data[freq_data < f_cut])]
    f_data = freq_data[1:len(freq_data[freq_data < f_cut])]
    std_data = np.array(std_data[1:len(freq_data[freq_data < f_cut])])

    for comb in list_comb:
        ncol = 2 #number of column in the plot
        fig,ax = plt.subplots(int(len(list_r0)/ncol),ncol)
        for i,r in enumerate(list_r0):

            ax[i//ncol][i%ncol].plot(f_data, new_power_spline, "-r", label="Data")
            ax[i//ncol][i%ncol].fill_between(f_data, np.add(new_power_spline, std_data), np.subtract(new_power_spline, std_data), color = 'r',alpha=0.3)

            for v in all_v:
                list_pkl = glob.glob("spectrum_A4-B4_100000_%s_R%s_thin-disk_2maps-withmacroM.pkl"%(v,r))
                #list_pkl = glob.glob("spectrum_A4-B4_100000_%s_R%s_thin-disk_2maps-withmacroM_BLRt65_s65_tau810_noshiftDRW_fBLRdistrib.pkl" % (v, r))
                print list_pkl
                all_power = []
                all_var = []
                for ii,elem in enumerate(list_pkl):
                    print elem
                    mean_power, var_power, freq = pkl.load(open(elem, 'rb'))

                    all_power.append(np.array(mean_power[1:]))
                    all_var.append(np.array(var_power[1:]))
                    freq = np.array(freq[1:])
                    print type(mean_power)
                mean_power = np.array([])
                var_power = np.array([])
                all_power = np.array(all_power)
                all_var = np.array(all_var)
                for j in range(len(all_power[0])):
                    mean_power=np.append(mean_power, np.mean(all_power[:, j]))
                    var_power=np.append(var_power, max(all_var[:, j]))

                new_f = freq[:len(freq_data[freq_data < f_cut])]
                new_mean_power = mean_power[:len(freq_data[freq_data < f_cut])]
                new_var_power = var_power[:len(freq_data[freq_data < f_cut])]
                ax[i//ncol][i%ncol].plot(new_f, new_mean_power, "-", label=r"$v_e$ = %s km/s"%(v))

                ax[i//ncol][i%ncol].fill_between(new_f, np.add(new_mean_power, np.sqrt(new_var_power)), new_mean_power, alpha = 0.3)
                ax[i//ncol][i%ncol].set(yscale='log',xscale='log', ylim = (0.00001, 1000))#, ylim = (min(power_spline[10:]), max(power_spline)))
                ax[i//ncol][i%ncol].text(0.003, 100, r"$R_0$ = %s $R_{MK11}$"%(round(r/r_ref,1)))

                ax[i//ncol][i%ncol].axvline(1/750, 0,1000, ls ='--', c = 'k')

                if i//2==1:
                    plt.setp(ax[i//ncol][i%ncol].get_xticklabels(), visible=True)
               #     locs = ax[i//ncol][i%ncol].get_xticks()
               #     locs[1] = frequency_spline[0]#

               #     temp_lab = ax[i//ncol][i%ncol].get_xticklabels()
               #     lab = np.divide(1, locs).astype(int)
               #     labels = []
               #     for label in lab[1:-1]:
               #         labels.append('$\\frac{{{:2.0f}}}{{{:2.0f}}}$'.format(1, int(label)))

#                    ax[i//ncol][i%ncol].set_xticks(locs[1:-2], minor=False)
 #                   ax[i//ncol][i%ncol].set_xticklabels(labels, minor = False)

                else:

                    plt.setp(ax[i//ncol][i%ncol].get_xticklabels(), visible=False)

                if i % 2 == 0:
                    plt.setp(ax[i // ncol][i % ncol].get_yticklabels(), visible=True)
                #     locs = ax[i//ncol][i%ncol].get_xticks()
                #     locs[1] = frequency_spline[0]#

                #     temp_lab = ax[i//ncol][i%ncol].get_xticklabels()
                #     lab = np.divide(1, locs).astype(int)
                #     labels = []
                #     for label in lab[1:-1]:
                #         labels.append('$\\frac{{{:2.0f}}}{{{:2.0f}}}$'.format(1, int(label)))

                #                    ax[i//ncol][i%ncol].set_xticks(locs[1:-2], minor=False)
                #                   ax[i//ncol][i%ncol].set_xticklabels(labels, minor = False)

                else:
                    plt.setp(ax[i // ncol][i % ncol].get_yticklabels(), visible=False)

                #if i%ncol != 0:
                #    plt.setp(ax[i//ncol][i%ncol].get_yticklabels(), visible=False)
        #handles, labels = ax[0][1].get_legend_handles_labels()

        #ax[0][ncol-1].legend(handles[::-1],labels[::-1],bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        ax[0][0].legend()
        ax[1][1].set(xlabel=r'Frequency (days$^{-1}$)')
        ax[1][0].set(ylabel=r'Power')
        ax[int(len(list_r0)/ncol)-1][ncol-1].xaxis.set_label_coords(-0.05, -0.15)
        ax[1][0].yaxis.set_label_coords(-0.1, 1.03)
        #ax[0][0].set_title(, y=1.5, pad=5 )

        fig.subplots_adjust(hspace =0.05, wspace = 0.05)
        fig.suptitle(r'Microlensing only, $\left<M\right> = 0.3 M_{\odot}$')
        plt.show()
        fig.savefig(plotdir + "spectrumvsV_%s-%s_thin-disk_slides.png"%(comb[0],comb[1]), dpi = 100)
    sys.exit()

if 0: #Grid of PS paper
    #Grid plot of powerspectra, each portion contains the powerspectrum for all the velocities for a single radius
    list_comb = [('A3', 'B2')] #name of the maps used to generate the powerspectra
    f_cut = 0.01 #higher frequency considered in days^-1
    r_ref = 15 # Reference radius found in litterature in pxl
    list_r0 = [2,10, 20,30]
    #list_r0 = [5, 10, 24, 36, 49, 73, 146, 195]
    all_v = [400, 700,1100] #Velocities in km.s^-1
    #all_v =[100]
    #list_r0 = [20]
    resultdir ='/home/ericpaic/Documents/PHD/results/powerspectrum/pkl/reverberation_study/'

#    resultdir ='/media/ericpaic/TOSHIBA EXT/TPIVb/results/powerspectrum/'

    #os.chdir(resultdir+'pkl/M0.3/A3-B2_flattop') #path to the pkl containing the powerspectra
    os.chdir(resultdir )
    #Selecting the frequencies of interest of the powerspectrum
    power_data, std_data, freq_data = pkl.load(open(pkldir+'data.pkl', 'rb'))
    new_power_spline = power_data[1:len(freq_data[freq_data < f_cut])]
    f_data = freq_data[1:len(freq_data[freq_data < f_cut])]
    std_data = np.array(std_data[1:len(freq_data[freq_data < f_cut])])

    for comb in list_comb:
        ncol = 1 #number of column in the plot
        fig,ax = plt.subplots(int(len(list_r0)/ncol),ncol)
        for i,r in enumerate(list_r0):

            ax[i].plot(f_data, new_power_spline, "-r", label="Data")
            ax[i].fill_between(f_data, np.add(new_power_spline, std_data), np.subtract(new_power_spline, std_data), color = 'r',alpha=0.3)

            for v in all_v:
                list_pkl = glob.glob("spectrum_A4-B4_100000_%s_R%s_thin-disk_2maps-withmacroM.pkl"%(v,r))
                #list_pkl = glob.glob("spectrum_A4-B4_100000_%s_R%s_thin-disk_2maps-withmacroM_BLRt65_s65_tau810_noshiftDRW_fBLRdistrib.pkl" % (v, r))
                print list_pkl
                all_power = []
                all_var = []
                for ii,elem in enumerate(list_pkl):
                    print elem
                    mean_power, var_power, freq = pkl.load(open(elem, 'rb'))

                    all_power.append(np.array(mean_power[1:]))
                    all_var.append(np.array(var_power[1:]))
                    freq = np.array(freq[1:])
                    print type(mean_power)
                mean_power = np.array([])
                var_power = np.array([])
                all_power = np.array(all_power)
                all_var = np.array(all_var)
                for j in range(len(all_power[0])):
                    mean_power=np.append(mean_power, np.mean(all_power[:, j]))
                    var_power=np.append(var_power, max(all_var[:, j]))

                new_f = freq[:len(freq_data[freq_data < f_cut])]
                new_mean_power = mean_power[:len(freq_data[freq_data < f_cut])]
                new_var_power = var_power[:len(freq_data[freq_data < f_cut])]
                ax[i].plot(new_f, new_mean_power, "-", label=r"$v_e$ = %s km/s"%(v))

                ax[i].fill_between(new_f, np.add(new_mean_power, np.sqrt(new_var_power)), new_mean_power, alpha = 0.3)
                ax[i].set(yscale='log',xscale='log', ylim = (0.00001, 1000))#, ylim = (min(power_spline[10:]), max(power_spline)))
                ax[i].text(0.003, 100, r"$R_0$ = %s $R_{MK11}$"%(round(r/r_ref,1)))

                ax[i].axvline(1/750, 0,1000, ls ='--', c = 'k')

                if i==3:
                    plt.setp(ax[i].get_xticklabels(), visible=True)
               #     locs = ax[i//ncol][i%ncol].get_xticks()
               #     locs[1] = frequency_spline[0]#

               #     temp_lab = ax[i//ncol][i%ncol].get_xticklabels()
               #     lab = np.divide(1, locs).astype(int)
               #     labels = []
               #     for label in lab[1:-1]:
               #         labels.append('$\\frac{{{:2.0f}}}{{{:2.0f}}}$'.format(1, int(label)))

#                    ax[i//ncol][i%ncol].set_xticks(locs[1:-2], minor=False)
 #                   ax[i//ncol][i%ncol].set_xticklabels(labels, minor = False)

                else:
                    plt.setp(ax[i].get_xticklabels(), visible=False)

                #if i%ncol != 0:
                #    plt.setp(ax[i//ncol][i%ncol].get_yticklabels(), visible=False)
        #handles, labels = ax[0][1].get_legend_handles_labels()

        #ax[0][ncol-1].legend(handles[::-1],labels[::-1],bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        ax[0].legend()
        ax[3].set(xlabel=r'Frequency (days$^{-1}$)')
        ax[2].set(ylabel=r'Power')
        #ax[int(len(list_r0)/ncol)-1][ncol-1].xaxis.set_label_coords(-0.05, -0.15)
        ax[2].yaxis.set_label_coords(-0.1, 1.03)
        ax[0].set(title = r'$\left<M\right> = 0.3 M_{\odot}$')

        fig.subplots_adjust(hspace =0.08, wspace = 0.05)
        plt.show()
        fig.savefig(plotdir + "spectrumvsV_%s-%s_thin-disk.pdf"%(comb[0],comb[1]), dpi = 500)
    sys.exit()

if 0: #PS
    # Plot of the power spectrum for only one radius at a time.
    list_comb = [('A3','B2')]
    f_cut = 0.01
    list_r0 = [2]
    list_v = [5000,1000,500,100]
    os.chdir(resultdir+'powerspectrum/pkl/M0.3/A3-B2_flattop')
    for comb in list_comb:
       for r in list_r0:
           fig = plt.figure(figsize=(10, 10))
           gs = gridspec.GridSpec(2, 1, height_ratios=[3.5, 1], hspace=0.0)
           ax0 = plt.subplot(gs[0])
           ax1 = plt.subplot(gs[1])
           for v_source in list_v:
                list_pkl = glob.glob("spectrum_%s-%s_*_%s_R%s_thin-disk_flattop_2.pkl"%(comb[0], comb[1],v_source,r))
                n_spectrum = int(list_pkl[0].split('_')[2])
                #v_source = []
                #for elem in list_pkl:
                #    v_source.append(int(elem.split('_')[3]))
                #sort = np.argsort(v_source)
                #v_source = np.sort(v_source)
                #list_pkl = [list_pkl[iii] for iii in sort]
                all_pow = []
                all_var = []
                for ii,elem in enumerate(list_pkl):
                    print elem
                    mean_power, var_power, freq = pkl.load(open(elem, 'rb'))
                    var_power = np.array(var_power[1:])
                    mean_power= np.array(mean_power[1:])
                    freq = np.array(freq[1:])
                    new_f = freq[freq<= f_cut]
                    new_mean_power = mean_power[freq<= f_cut]
                    new_var_power = var_power[freq<= f_cut]
                    all_pow.append(new_mean_power)
                    all_var.append(new_var_power)
                    ax0.plot(new_f, new_mean_power, "--", label=r"v = %s km/s, "%(v_source))
                    ax0.fill_between(new_f, np.add(new_mean_power, np.sqrt(new_var_power)),
                                    new_mean_power, alpha = 0.3)

                print all_pow
                residuals = all_pow[1]-all_pow[0]
                print residuals.shape
                ax1.plot(new_f, residuals, "--")
                ax1.fill_between(new_f, np.add(residuals, np.sqrt(all_var[0])-np.sqrt(all_var[1])), residuals, alpha=0.3)

           ax0.set(yscale='log',xscale='log')
           ax1.set(yscale='log',xscale='log', ylabel = 'Residuals')
           plt.setp(ax0.get_xticklabels(), visible=False)
           ax0.set_title(r"$R_0$ = %s $R_{MK11}$"%(round(r/15,1)), fontdict={'fontsize':18})
           ax0.legend()

           locs = ax1.get_xticks()
           locs[1] = frequency_spline[0]

           temp_lab = ax1.get_xticklabels()
           lab = np.divide(1, locs).astype(int)
           labels = []
           for label in lab[1:-1]:
               labels.append('$\\frac{{{:2.0f}}}{{{:2.0f}}}$'.format(1, int(label)))

                #labels[0]='0'
           ax1.set_xticks(locs[1:-2], minor=False)
           ax1.set_xticklabels(labels, minor = False)

           ax1.legend()
           ax1.set(xlabel=r'Frequency (days$^{-1}$)')
           ax0.set(ylabel=r'Power')
           plt.show()
           fig.savefig(resultdir + "powerspectrum/png/M0.3/spectrumvsV_R%s_A3-B2_flattop.png"%(r),dpi =300)
    sys.exit()




if 0:
    sigma = 30
    BLRtime = 65
    list_pklfile = glob.glob(drwdir + 'DRW_sigma%s_tau810_BLRt%s_tfshapetophat_*.pkl' % (sigma, BLRtime))
    print len(list_pklfile)

    lc_c_list = []
    lc_BLR_list = []

    for pklfile in list_pklfile[:50]:

        new_time, lc_c, lc_BLR = pkl.load(open(pklfile, 'rb'))
        if (lc_c > 0).all() and (lc_BLR > 0).all():
            lc_c_list.append(lc_c)
            lc_BLR_list.append(lc_BLR[:])
    #Necessary but not sufficient
    n_spectrum = 2 #number of curves to simulate
    print len(lc_BLR_list)
    print len(new_time)
    print len(lc_BLR_list[0])
    print len(lc_c_list[0])
    new_err_mag_ml = np.random.normal(np.mean(err_mag_ml), np.std(err_mag_ml), len(new_time))

    start2 = timeit.default_timer()

    v_source = 1500
    r0 = 10
    f_cut = 0.01

    power_data, std_data, freq_data = pkl.load(open(pkldir + 'data.pkl', 'rb'))
    new_power_spline = power_data[:len(freq_data[freq_data < f_cut])]
    new_f = freq_data[:len(freq_data[freq_data < f_cut])]
    std_data = np.array(std_data[:len(freq_data[freq_data < f_cut])])



    print r0
    mapA = "/home/ericpaic/Documents/PHD/Powerspectrum-Analysis-Tool-/maps/with_macromag/convolved_map_A_fft_thin-disk_%s.fits" % (r0)
    imgA = fits.open(mapA)[0]
    final_map_A = imgA.data

    mapB = "/home/ericpaic/Documents/PHD/Powerspectrum-Analysis-Tool-/maps/with_macromag/convolved_map_B_fft_thin-disk_%s.fits" % (r0)
    imgB = fits.open(mapB)[0]
    final_map_B = imgB.data

    print v_source
    v = v_source*np.ones(n_spectrum)
#    x_A = np.append(np.random.random_integers(1000, len(final_map_A) - 1000, n_spectrum-1),893)
#
#    y_A =  np.append(np.random.random_integers(1000, len(final_map_A) - 1000, n_spectrum-1), 6692)
#    angle_A =np.append(np.random.uniform(0, 2 * np.pi, n_spectrum-1), 5.06093509)
#
#    x_B = np.append(np.random.random_integers(1000, len(final_map_A) - 1000, n_spectrum-1), 264)
#
#    y_B = np.append(np.random.random_integers(1000, len(final_map_A) - 1000, n_spectrum-1), 687)
#    angle_B = np.append(np.random.uniform(0, 2 * np.pi, n_spectrum-1), 5.01157114)

    x_A = np.array([3789, 3641])

    y_A = np.array([1067, 6179])
   # np.random.random_integers(0, len(final_map_A) - 1, n_spectrum)
    angle_A = np.array([4.63711, 3.817])
   # np.random.uniform(0, 2 * np.pi, n_spectrum)

    x_B = np.array([3418, 2385])

    y_B = np.array([6621, 6155])
   # np.random.random_integers(0, len(final_map_B) - 1, n_spectrum)
    angle_B = np.array([0.3279, 3.439])
#np.random.uniform(0, 2 * np.pi, n_spectrum)

    print x_A
    print x_B
    print y_A
    print y_B
    print angle_A
    print angle_B
    params_A = np.stack((x_A, y_A, v, angle_A), axis=-1)

    params_B = np.stack((x_B, y_B, v, angle_B), axis=-1)
    params = np.stack((params_A, params_B), axis=1)

    fig, ax = plt.subplots(2, 1, figsize=(10, 5))
    ax[0].errorbar(mjhd, mag_ml, yerr=err_mag_ml, label='Data', c='r', marker='.', ls='', alpha=0.3)
    ax[1].plot(new_f,
               new_power_spline, '-', label='Data', c = 'r')

    ax[1].fill_between(new_f, np.add(new_power_spline, std_data),
                       np.subtract(new_power_spline, std_data), color = 'r', alpha=0.3)

    Mc_A = 2.24
    Mc_B = 0.84
    whichcurve = 1
    lc_c = lc_c_list[whichcurve]
    lc_BLR = lc_BLR_list[whichcurve]

    for j,param in enumerate(params):
        ml_A = good_draw_LC(param[0], final_map_A, new_time, err_data=new_err_mag_ml)
        ml_B = good_draw_LC(param[1], final_map_B, new_time, err_data=new_err_mag_ml)
        ml_A = np.abs(ml_A)
        ml_B = np.abs(ml_B)

        lc = -2.5 * np.log10(ml_A * Mc_B*lc_c / (ml_B * Mc_A*lc_c))
        lc -= lc[0]

        lc_reverberation = draw_LC_withreverberation(param,final_map_A,final_map_B, time=new_time, err_data=new_err_mag_ml, fBLR=0.43, f=1)

        print lc_reverberation
        frequency_BLR, power_BLR = periodogram(lc_reverberation, window="flattop")

        temp = PS_2maps(param,final_map_A,final_map_B, time=new_time, err_data=new_err_mag_ml, f=1)
        freq = temp[0]
        power = temp[1]
        if j==0:
            ax[0].plot(new_time, lc,c='blue', label = 'Simulation 1')
            ax[0].plot(new_time, lc_reverberation, c='blue',ls=':' )
            ax[1].plot(freq[freq < f_cut], power[freq < f_cut], label = 'Simulation 1', c='blue')
            ax[1].plot(frequency_BLR[frequency_BLR < f_cut], power_BLR[frequency_BLR < f_cut], c='blue',ls=':')

        else:
            ax[0].plot(new_time, lc, label = 'Simulation 2', c = 'green')
            ax[0].plot(new_time, lc_reverberation, c='green',ls=':' )

            ax[1].plot(freq[freq<f_cut], power[freq<f_cut], label = 'Simulation 2', c='green')
            ax[1].plot(frequency_BLR[frequency_BLR < f_cut], power_BLR[frequency_BLR < f_cut], c='green',ls=':')



    #ax[1].plot(frequency_spl2[frequency_spl < f_cut], power_spl2[frequency_spl < f_cut],
    #           label="Long time scale variations")
    #ax[1].plot(frequency_spl[frequency_spl < f_cut], power_spl[frequency_spl < f_cut],
    #           label="Short time scale variations")

    #ax[1].fill_between(freq[freq<f_cut], np.add(mean_power[freq<f_cut], np.sqrt(var_power[freq<f_cut])),
    #                  np.add(mean_power[freq < f_cut], -np.sqrt(var_power[freq < f_cut])), alpha=0.3)
    #ax[2].plot(frequency_spline[frequency_spline < 0.02], np.log10(power_spline[frequency_spline < 0.02]), 'r')
    #ax[2].fill_between(freq[freq < 0.02], np.log10(np.add(mean_power[freq < 0.02], np.sqrt(var_power[freq < 0.02]))),
    #                  np.log10(np.add(mean_power[freq < 0.02], -np.sqrt(var_power[freq < 0.02]))), alpha=0.3)
    #ax[2].plot(np.abs(deriv_data),'k', label='data')
    ax[0].set(xlabel=r'MJHD (days)', ylabel=r'$m_A$ - $m_B$ (mag)')
    ax[0].xaxis.tick_top()
    ax[0].xaxis.set_label_position('top')
    ax[1].set(xlabel=r'Frequency (days$^{-1}$)', ylabel='Power', yscale='log', xscale='log')
    ax[0].legend()
    ax[1].legend()

    plt.subplots_adjust(hspace=.05)

    plt.show()
    fig.savefig(plotdir+'necessarynotsufficient.pdf',dpi=100)
    stop = timeit.default_timer()
    print stop - start2

    #with open(resultdir + 'powerspectrum/pkl/spectrum_%s-%s_%s_%s_R%s_M0,01_1.pkl'%(comb[0], comb[1],n_spectrum, v_source, r0), 'wb') as handle:
    #    pkl.dump((mean_power,var_power, freq[0]), handle, protocol=pkl.HIGHEST_PROTOCOL)


    sys.exit()


if 0:
    #Calculating mean and standard deviation for the powerspectrum of the data using a high nunmber of noise realisation
    power = []
    fig, ax = plt.subplots(1, 1, figsize=(15, 15))
    f_cut = 0.1
    for i in range(100000):
        magml2 = mag_ml + np.random.normal(0, np.mean(err_mag_ml), len(mag_ml))

        lc2 = pycs.gen.lc.factory(mjhd, magml2, magerrs=err_mag_ml)
        spline = pycs.gen.spl.fit([lc2], knotstep=70, bokeps=20, verbose=False)
        new_magml = spline.eval(new_mjhd)
        frequency_spline, power_spline = periodogram(new_magml, 1, window='flattop')
        power.append(power_spline)
        #plt.plot(mjhd, magml2)
        #plt.plot(new_mjhd, new_magml)

    #plt.show()
        #power.append(lombscargle(new_mjhd, new_magml_2, frequency_spline))

    power = np.array(power)
    mean_p = []
    var_p = []

    for ii in range(len(power[0])):
        mean_p.append(np.mean(power[:,ii]))
        var_p.append(np.var(power[:, ii]))

    print var_p
    print mean_p
    frequency_spline = np.array(frequency_spline[1:])
    mean_p = np.array(mean_p[1:])
    var_p = np.array(var_p[1:])
    ax.set_xlim(1 / 4546, 1 / 10)
    ax.plot(frequency_spline[frequency_spline<f_cut], mean_p[frequency_spline<f_cut], "--")
    ax.fill_between(frequency_spline[frequency_spline<f_cut], np.add(mean_p[frequency_spline<f_cut], np.sqrt(var_p[frequency_spline<f_cut])),
                           mean_p[frequency_spline<f_cut], alpha = 0.3)
    ax.set(xlabel=r'Frequency (days$^{-1}$)',ylabel = 'Power', yscale='log', xscale='log')

    #ax.set_title(r"Mean spectrum of the data curve affected by 10000 different photometric noise", fontdict={'fontsize':16})
    #ax.set_title("Same curve with 100000 different realisation of the noise")
    ax.legend(prop={'size':16})
    locs = ax.get_xticks()
    locs[1] = frequency_spline[0]
    locs[-1] = 0.1
    temp_lab = ax.get_xticklabels()
    lab = np.divide(1, locs).astype(int)
    labels = []
    for i, elem in enumerate(lab[1:-1]):
        labels.append('$\\frac{{{:2.0f}}}{{{:2.0f}}}$'.format(1, int(elem)))

    labels[-1] = '1'
    ax.set_xticks(locs[1:-1], minor=False)
    ax.set_xticklabels(labels, minor = False)
    plt.show()

    fig.savefig(resultdir + "png/spectrum_uncertainty_flattop.png")
    sys.exit()

#### STUDY RES
if 0:
    # Comparing the same trajectories generated by 2 maps with a different resolution. Just a test but not useful anymore
    n_spectrum = 20000
    pool = multiprocessing.Pool(2)
    start2 = timeit.default_timer()
    list_v = [100,250,500]

    map = storagedir+ "FML0.9/M0,3/mapA2_Re5-B2_Re5_fml09_R40.fits"
    img = fits.open(map)[0]
    final_map = img.data[:, :]
    map2 = storagedir + "FML0.9/M0,3/mapA2-B2_fml09_R10.fits"
    img2 = fits.open(map2)[0]
    final_map_2 = img2.data[:, :]

    fig, ax = plt.subplots(1, 1, figsize=(15, 15))
    inset = plt.axes([0.1, 1, 0.5, 1])
    ip = InsetPosition(ax, [0.2, 0.55, 0.45, 0.45])
    inset.set_axes_locator(ip)
    mark_inset(ax, inset, loc1=2, loc2=4, fc="none", ec='0.5')

    for v_source in list_v:
        print v_source
        v = v_source*np.ones(n_spectrum)

        x = np.random.random_integers(0, len(final_map) - 1, n_spectrum)
        y = np.random.random_integers(0, len(final_map) - 1, n_spectrum)
        x_dezoom = x/4+int(len(final_map)*1.5/4)
        y_dezoom = y/4+int(len(final_map)*1.5/4)
        angle = np.random.uniform(0, 2 * np.pi, n_spectrum)


        lc1, arrow1=good_draw_LC([x[0],y[0],v[0],angle[0]], final_map, mjhd, err_mag_ml, (5*einstein_r_03)/8192 )
        lc2, arrow2 = good_draw_LC([x_dezoom[0], y_dezoom[0], v[0], angle[0]], final_map_2, mjhd, err_mag_ml, (20 * einstein_r_03) / 8192)

        #Display of the trajectory in the map
        display_multiple_trajectory([arrow1,arrow2], map, map2)
        plt.plot(mjhd, lc1,label="zoom")
        plt.plot(mjhd, lc2, label = "dezoom")
        plt.legend()
        plt.show()
        sys.exit()


        params = []
        params_2 = []
        for i, elem in enumerate(x):
            params.append([x[i], y[i], v[i], angle[i]])
            params_2.append([x_dezoom[i], y_dezoom[i], v[i], angle[i]])

        parrallel_power_spectrum = partial(power_spectrum,map= final_map, time=new_mjhd, err_data=new_err_mag_ml, f=1, cm_per_pxl = (5*einstein_r_03)/8192 )
        parrallel_power_spectrum_2 = partial(power_spectrum, map=final_map, time=new_mjhd, err_data=new_err_mag_ml, f=1,
                                           cm_per_pxl=(20 * einstein_r_03) / 8192)

        res = pool.map(parrallel_power_spectrum, params)
        res_2 = pool.map(parrallel_power_spectrum_2, params_2)
        res = filter(None, res)
        res = np.array(res)
        power = res[:,0]
        freq = np.array(res[0, 1])
        print freq

        res_2 = filter(None, res_2)
        res_2 = np.array(res_2)
        power_2 = res_2[:, 0]
        freq_2 = res_2[0, 1]
        print freq_2
        mean_power = np.array([])
        var_power = np.array([])
        mean_power_2 = np.array([])
        var_power_2 = np.array([])

        power_2 = np.array(power_2)
        power=np.array(power)
        for i in range(len(power[0])):
            mean_power=np.append(mean_power,np.mean(power[:,i]))
            var_power=np.append(var_power,np.var(power[:,i]))
            mean_power_2=np.append(mean_power_2,np.mean(power_2[:, i]))
            var_power_2=np.append(var_power_2,np.var(power_2[:, i]))



        new_f = freq[freq<= 0.2]
        new_mean_power = mean_power[freq<= 0.2]
        new_var_power = var_power[freq<= 0.2]

        new_f_2 = freq_2[freq_2 <= 0.2]
        new_mean_power_2 = mean_power_2[freq_2 <= 0.2]
        new_var_power_2 = var_power_2[freq_2 <= 0.2]


        ax.plot(new_f, new_mean_power, "--",
                label=r"Zoom ; v = %s km/s" % (v_source))
        ax.fill_between(new_f, np.add(new_mean_power, np.sqrt(new_var_power)),
                        new_mean_power, alpha=0.3)

        ax.plot(new_f_2, new_mean_power_2, "--",
                label=r"Dezoom ; v = %s km/s" % (v_source))
        ax.fill_between(new_f_2, np.add(new_mean_power_2, np.sqrt(new_var_power_2)),
                        new_mean_power_2, alpha=0.3)

        f_cut = freq[freq<= 0.02]
        p_cut = mean_power[freq<= 0.02]
        var_cut = var_power[freq<= 0.02]

        f_cut_2 = freq_2[freq_2 <= 0.02]
        p_cut_2 = mean_power_2[freq_2 <= 0.02]
        var_cut_2 = var_power_2[freq_2 <= 0.02]

        inset.plot(f_cut, p_cut, '--')
        inset.fill_between(f_cut, np.add(p_cut, np.sqrt(var_cut)),
                           p_cut, alpha=0.3)

        inset.plot(f_cut_2, p_cut_2, '--')
        inset.fill_between(f_cut_2, np.add(p_cut_2, np.sqrt(var_cut_2)),
                           p_cut_2, alpha=0.3)


    inset.set(yscale='log')

    ax.set(xlabel=r'Frequency (days$^{-1}$)', ylabel='Power', yscale='log',
           ylim=(min(power_spline[10:]), max(power_spline)))
    ax.legend(prop={'size': 16})
    locs = ax.get_xticks()
    # locs[1] = frequency_spline[0]

    temp_lab = ax.get_xticklabels()
    lab = np.divide(1, locs).astype(int)
    labels = []
    for i, elem in enumerate(lab[1:-1]):
        labels.append('$\\frac{{{:2.0f}}}{{{:2.0f}}}$'.format(1, int(elem)))

    labels[0] = '0'
    ax.set_xticks(locs[1:-1], minor=False)
    ax.set_xticklabels(labels, minor=False)

    locs2 = inset.get_xticks()
    # locs2[1] = frequency_spline[0]
    temp_lab = inset.get_xticklabels()
    lab2 = np.divide(1, locs2).astype(int)
    labels = []
    for i, elem2 in enumerate(lab2[1:-1]):
        labels.append('$\\frac{{{:2.0f}}}{{{:2.0f}}}$'.format(1, int(elem2)))

    labels[0] = '0'
    inset.set_xticks(locs2[1:-1], minor=False)
    inset.set_xticklabels(labels, minor=False)
    inset.set_ylim(min(power_spline[10:]), max(power_spline))
    fig.savefig(resultdir + "PowerSvsResolution.png")
    sys.exit()

if 0:
    power = []
    power_2 = []
    fig, ax = plt.subplots(1, 1, figsize=(15, 15))
    inset = plt.axes([0.1, 1, 0.5, 1])
    ip = InsetPosition(ax, [0.2, 0.55, 0.45, 0.45])
    inset.set_axes_locator(ip)
    mark_inset(ax, inset, loc1=2, loc2=4, fc="none", ec='0.5')

    print "youhou"
    for i in range(10000):
        new_magml_3 = new_magml + np.random.normal(0, np.mean(err_mag_ml), len(new_magml))
        new_magml_4 = new_magml_2 + np.random.normal(0, np.mean(err_mag_ml_2), len(new_magml_2))
        freq, power_spline = periodogram(new_magml_3, 1)
        power.append(power_spline[1:])
        freq_2, power_spline_2 = periodogram(new_magml_4, 1)
        power_2.append(power_spline_2[1:])
        #power.append(lombscargle(new_mjhd, new_magml_2, frequency_spline))

    power = np.array(power)
    mean_p = []
    var_p = []
    freq = freq[1:]

    for ii in range(len(power[0])):
        mean_p.append(np.mean(power[:,ii]))
        var_p.append(np.var(power[:, ii]))

    mean_p = np.array(mean_p)
    var_p = np.array(var_p)

    power_2 = np.array(power_2)
    mean_p_2 = []
    var_p_2 = []
    freq_2 = freq_2[1:]

    for ii in range(len(power_2[0])):
        mean_p_2.append(np.mean(power_2[:, ii]))
        var_p_2.append(np.var(power_2[:, ii]))

    mean_p_2 = np.array(mean_p_2)
    var_p_2 = np.array(var_p_2)

    ax.plot(freq[freq<0.2], mean_p[freq<0.2], "--", label = 'Q0158')
    ax.fill_between(freq[freq<0.2], np.add(mean_p[freq<0.2], np.sqrt(var_p[freq<0.2])),
                           mean_p[freq<0.2], alpha = 0.3)
    f_cut = freq[freq<=0.02]
    p_cut = mean_p[freq<=0.02]
    var_cut = var_p[freq<=0.02]

    inset.plot(f_cut, p_cut, '-')
    inset.fill_between(f_cut, np.add(p_cut, np.sqrt(var_cut)),
                           p_cut, alpha=0.3)

    ax.plot(freq_2[freq_2 < 0.2], mean_p_2[freq_2 < 0.2], "--",label='RXJ1131')
    ax.fill_between(freq_2[freq_2 < 0.2], np.add(mean_p_2[freq_2 < 0.2], np.sqrt(var_p_2[freq_2 < 0.2])),
                    mean_p_2[freq_2 < 0.2], alpha=0.3)
    f_cut_2 = freq_2[freq_2 <= 0.02]
    p_cut_2 = mean_p_2[freq_2 <= 0.02]
    var_cut_2 = var_p_2[freq_2 <= 0.02]

    inset.plot(f_cut_2, p_cut_2, '-')
    inset.fill_between(f_cut_2, np.add(p_cut_2, np.sqrt(var_cut_2)),
                       p_cut_2, alpha=0.3)

    inset.set(yscale = 'log')


    ax.set(xlabel=r'Frequency (days$^{-1}$)',ylabel = 'Power', yscale='log')
    ax.set_title(r"Powerspectrum of the data", fontdict={'fontsize':16})
    #ax.set_title("Same curve with 100000 different realisation of the noise")
    ax.legend(prop={'size':16})
    locs = ax.get_xticks()
    locs[1] = freq[0]
    temp_lab = ax.get_xticklabels()
    lab = np.divide(1, locs).astype(int)
    labels = []
    for i, elem in enumerate(lab[1:-1]):
        labels.append('$\\frac{{{:2.0f}}}{{{:2.0f}}}$'.format(1, int(elem)))

    ax.set_xticks(locs[1:-1], minor=False)
    ax.set_xticklabels(labels, minor = False)





    locs2 = inset.get_xticks()
    locs2[1] = freq[0]
    temp_lab = inset.get_xticklabels()
    lab2 = np.divide(1, locs2).astype(int)
    labels = []
    for i, elem2 in enumerate(lab2[1:-1]):
        labels.append('$\\frac{{{:2.0f}}}{{{:2.0f}}}$'.format(1, int(elem2)))

    inset.set_xticks(locs2[1:-1], minor=False)
    inset.set_xticklabels(labels, minor=False)

    plt.show()

    fig.savefig(resultdir + "powerspectrum/png/spectrum_uncertainty_schuster.png")
    sys.exit()

if 0:
    # Plot of the map of proba as function of radius and velocity for 1 model
    resultdir ='/media/ericpaic/TOSHIBA EXT/TPIVb/results/powerspectrum/pkl/'
    os.chdir(resultdir + 'M0.3/A3-B2_flattop')
    list_pkl = glob.glob("*.pkl")
    r_ref =15 #Reference radius in pxl

    # All the variables that will be extracted from the name of the pkl file
    v_source = []
    reject_v_source=[]
    r0 = np.array([])
    reject_r0 = np.array([])
    chi = []
    reject_chi = []

    better_v = []
    better_r0 = []
    all_l = np.array([])
    all_prior = np.array([])

    all_r0 = np.array([])
    all_v = []

    maxchi = 10
    f_cut_chi = 1/750
    length = np.array([])
    data_power, std_data, freq_data = pkl.load(open(pkldir+'data.pkl', 'rb'))
    data_power = np.array(data_power[1:len(freq_data[freq_data < f_cut_chi])])
    std_data = np.array(std_data[1:len(freq_data[freq_data < f_cut_chi])])
    freq_data = np.array(freq_data[1:len(freq_data[freq_data < f_cut_chi])])

    for i, elem in enumerate(list_pkl):
        print elem
        mean_power, var_power, freq = pkl.load(open(elem, 'rb'))
        #print lenpower
        #length=np.append(length,lenpower)
        var_power = np.array(var_power[1:])
        mean_power = np.array(mean_power[1:])

        v = int(elem.split('_')[3])
        temp = chi2_custom(mean_power[np.where(frequency_spline<=f_cut_chi)],var_power[np.where(frequency_spline<=f_cut_chi)],data_power, std_data)
        l = likelyhood(temp)
        #print temp
        all_l= np.append(all_l, l)
        print "v : %s"%(v)
        print "prior : %s"%(sampled_v1[np.where(bin_edges == v)])
        all_prior= np.append(all_prior, sampled_v1[np.where(bin_edges == v)])
        all_r0= np.append(all_r0,int(elem.split('.')[0].split('_')[4].split('R')[1]))
        all_v.append(v)



    z = zip(all_r0, all_v)
    z_unique,inv = np.unique(z, return_inverse=True, axis =0)
    inv_unique = np.unique(inv)
    print z_unique
    print inv_unique

    #for i,elem in enumerate(inv_unique):
    #    temp = np.average(all_chi[np.where(inv == elem)])
    #    if temp < maxchi:
    #        chi.append(temp)
    #        v_source.append(z_unique[i,1])
    #        r0= np.append(r0,z_unique[i,0])
    #        if temp <1:
    #            print "+++++++++++++++"
    #            print temp
    #            print z_unique[i, 1]
    #            print z_unique[i, 0]
    #            better_v.append(z_unique[i, 1])
    #            better_r0 = np.append(better_r0, z_unique[i, 0])
    #    else :
    #        reject_v_source.append(z_unique[i, 1])
    #        reject_r0= np.append(reject_r0,z_unique[i, 0])

    print chi
    print v_source
    print r0
    print better_v
    print better_r0

    new_v = np.linspace(min(all_v), max(all_v), 10000)
    new_r0 = np.linspace(min(all_r0), max(all_r0), 10000)
    # print xx
    # print yy
    # triang = tri.Triangulation(better_sigma, better_BLRt)
    # interpolator = tri.LinearTriInterpolator(triang, chi)

    proba = all_l*all_prior
    triang = tri.Triangulation(all_v, all_r0)
    interpolator = mtri.CubicTriInterpolator(triang, all_l)
    interpolator_prior = mtri.CubicTriInterpolator(triang, all_prior)
    interpolator_proba = mtri.CubicTriInterpolator(triang, proba)
    xx, yy = np.meshgrid(new_v, new_r0)
    zi = interpolator(xx, yy)
    zi_prior = interpolator_prior(xx,yy)
    zi_proba = interpolator_proba(xx, yy)/((new_v[1]-new_v[0])*(new_r0[1]-new_r0[0]))

    #plt.plot(zi)
    #plt.show()
    from matplotlib import patches
    el = Ellipse((np.mean(653), 4.2), width=358, height=2.5, angle=0, alpha = 0.7, ec='w', lw='3', fill = False, ls ='--')

    fig, ax = plt.subplots(1, 1, figsize = (5,5))
    #levels = np.logspace(np.min(np.log10(all_chi)),np.max(np.log10(all_chi)),20)

    #orig_cmap = matplotlib.cm.RdYlGn_r
    #shifted_cmap = shiftedColorMap(orig_cmap, midpoint=1 / np.max(np.log10(zi)), name='shifted')
    #levels = np.logspace(np.min(np.log10(all_chi)), np.max(np.log10(all_chi)), 20)
    sc = ax.contourf(new_v,new_r0/r_ref,zi_proba, 20, cmap = 'viridis')#,locator=ticker.LogLocator(), cmap=shifted_cmap, alpha = 0.3,levels =levels)
    ax.add_patch(el)
    ax.set(ylabel=r'$R_0$  in units of $R_{MK11}$', xlabel=r'$v_e$ [$km\cdot s^{-1}}$]', xlim=(0,1500))


    cb =plt.colorbar(sc, label=r"Probability density")
    #locator = LogLocator()
    #formatter = LogFormatter()
    #cb.locator = locator
    #cb.formatter = formatter
    #cb.update_normal(sc)
    #ax.text(1000, 1.5, "Best fit : v %s ; R %s" %(v_source[idx], round(r0[idx]/20,1)))

    #plt.legend()
    plt.show()
    fig.savefig(plotdir+"RvsVvsposterior_fml09_M03_lowf.pdf", dpi = 100)
    sys.exit()

if 0:
    import matplotlib.tri as mtri

    # Plot of the map of proba as function of radius and velocity for low, high freq and reverberation one mean M
    resultdir ='/home/ericpaic/Documents/PHD/results/powerspectrum/pkl/reverberation_study/'
    os.chdir(resultdir)
    r_ref =15 #Reference radius in pxl
    list_pklname = ['M01/M01spectrum_A-B_100000_*_R*_thin-disk_2maps-withmacroM.pkl', 'M01/M01spectrum_A-B_100000_*_R*_thin-disk_2maps-withmacroM.pkl','M01/M01spectrum_A-B_100000_*_R*_thin-disk_2maps-withmacroM_BLRt65_s65_tau810_noshiftDRW_fBLRdistrib.pkl']
    list_f_cut = [1./800.,1./100.,1./100.]


    #sampled_v1 = sampled_v1 / np.sum(sampled_v1)
    fig, ax = plt.subplots(1, 3)
    for j,pklname in enumerate(list_pklname):
        data_power, std_data, freq_data = pkl.load(open(pkldir + 'data.pkl', 'rb'))
        print len(data_power)
        list_pkl = glob.glob(pklname)

        # All the variables that will be extracted from the name of the pkl file
        all_l = np.array([])
        all_prior = np.array([])
        all_r0 = np.array([])
        all_v = []

        f_cut_chi = list_f_cut[j]
        print f_cut_chi
        print freq_data[freq_data < f_cut_chi]

        data_power = np.array(data_power[1:len(freq_data[freq_data < f_cut_chi])])
        std_data = np.array(std_data[1:len(freq_data[freq_data < f_cut_chi])])
        freq_data = np.array(freq_data[1:len(freq_data[freq_data < f_cut_chi])])
        print len(data_power)


        for i, elem in enumerate(list_pkl):
            #print elem
            mean_power, var_power, freq = pkl.load(open(elem, 'rb'))
            #print lenpower
    #s      print mean_power
            #length=np.append(length,lenpower)
            var_power = np.array(var_power[1:])
            mean_power = np.array(mean_power[1:])

            v = int(elem.split('_')[3])
            #print len(var_power[np.where(freq_data<=f_cut_chi)])
            #print len(std_data)
            temp = chi2_custom(mean_power[np.where(freq_data<=f_cut_chi)],var_power[np.where(freq_data<=f_cut_chi)],data_power, std_data)
            l = likelyhood(temp)


            #print temp
            all_l= np.append(all_l, l)
            #print "v : %s"%(v)
            #print "prior : %s"%(sampled_v1[np.where(bin_edges == v)])
            all_prior= np.append(all_prior, sampled_v1[np.where(bin_edges == v)])
            all_r0= np.append(all_r0,int(elem.split('.')[0].split('_')[4].split('R')[1]))
            all_v.append(v)

        print f_cut_chi

        new_v = np.linspace(min(all_v), max(all_v), 100)
        new_r0 = np.linspace(min(all_r0), max(all_r0), 100)
        # print xx
        # print yy
        # triang = tri.Triangulation(better_sigma, better_BLRt)
        # interpolator = tri.LinearTriInterpolator(triang, chi)

        proba = all_l*all_prior

        triang = tri.Triangulation(all_v, all_r0)
        interpolator = mtri.CubicTriInterpolator(triang, all_l)
        interpolator_prior = mtri.CubicTriInterpolator(triang, all_prior)
        interpolator_proba = mtri.CubicTriInterpolator(triang, proba)
        xx, yy = np.meshgrid(new_v, new_r0)
        zi = interpolator(xx, yy)
        zi_prior = interpolator_prior(xx,yy)
        zi_proba = interpolator_proba(xx, yy)#/((new_v[1]-new_v[0])*(new_r0[1]-new_r0[0]))

        #print zi_proba
        #plt.plot(zi)
        #plt.show()
        from matplotlib import patches
        el = Ellipse((np.mean(653), 4.2), width=358, height=2.5, angle=0, alpha = 0.7, ec='w', lw='3', fill = False, ls ='-')


        #levels = np.logspace(np.min(np.log10(all_chi)),np.max(np.log10(all_chi)),20)

        #orig_cmap = matplotlib.cm.RdYlGn_r
        #shifted_cmap = shiftedColorMap(orig_cmap, midpoint=1 / np.max(np.log10(zi)), name='shifted')
        levels = np.linspace(np.min(zi_proba), 0.00135, 10)
        sc = ax[j].contourf(new_v,new_r0/r_ref,zi_proba,cmap = 'viridis', levels=levels)#,locator=ticker.LogLocator(), cmap=shifted_cmap, alpha = 0.3,levels =levels)
        ax[j].add_patch(el)
        ax[j].set(ylabel=r'$R_0$  in units of $R_{MK11}$', xlabel=r'$v_e$ [$km\cdot s^{-1}}$]', xlim=(0,1500))
    cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
    ax[1].yaxis.set_visible(False)
    ax[2].yaxis.set_visible(False)

    cb = fig.colorbar(sc, label="Posterior Probability", cax=cbar_ax)
    plt.subplots_adjust(wspace = 0.1, hspace = 0.0)

    ax[0].set(title =r"Up to 1/750 days$^{-1}$")
    ax[1].set(title =r"Up to 1/100 days$^{-1}$")
    ax[2].set(title =r"Up to 1/100 days$^{-1}$, including reverberation")


    #locator = LogLocator()
    #formatter = LogFormatter()
    #cb.locator = locator
    #cb.formatter = formatter
    #cb.update_normal(sc)
    #ax.text(1000, 1.5, "Best fit : v %s ; R %s" %(v_source[idx], round(r0[idx]/20,1)))
    plt.suptitle(r"$f_{M/L}$ =0.9, $\left<M\right>$= 0.1 $M_{\odot}$")
    #ax[0][0].text(0, 0.2, r'$\left<M\right>$= 0.3 $M_{\odot}$',
    #        bbox=dict(edgecolor='red', linestyle='-', facecolor='w'))
    #ax[1][0].text(0, 0.4, r'$\left<M\right>$= 0.1 $M_{\odot}$',
    #        bbox=dict(edgecolor='red', linestyle='-', facecolor='w'))

    #ax[2][0].text(0, 0.6, r'$\left<M\right>$= 0.01 $M_{\odot}$',
    #        bbox=dict(edgecolor='red', linestyle='-', facecolor='w'))

    #plt.legend()
    plt.show()
    fig.savefig(plotdir+"RvsVvsposterior_lowf_highf_reverb_M01.pdf", dpi = 100)
    sys.exit()

if 0:
    import matplotlib.tri as mtri
    from matplotlib import patches
    from scipy import interpolate
    from copy import deepcopy

    #matplotlib.rc('text', usetex=True)  # use latex for text

    # add amsmath to the preamble
    matplotlib.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]

    # Plot of the map of proba as function of radius and velocity for low, high freq and reverberation all mean M
    resultdir ='/home/ericpaic/Documents/PHD/results/powerspectrum/pkl/reverberation_study/'
    os.chdir(resultdir)
    r_ref =15 #Reference radius in pxl
    fml = 0.9


    if fml==0.9:
        list_pklname = [ 'M001/spectrum_A-B_50000_*_R*_thin-disk_2maps-withmacroM.pkl',
                         'M001/spectrum_A-B_50000_*_R*_thin-disk_2maps-withmacroM.pkl',
                         'M001/spectrum_A-B_50000_*_R*_thin-disk_2maps-withmacroM_BLRt65_s65_tau810_noshiftDRW_fBLRdistrib.pkl',
                         'M01/spectrum_A-B_100000_*_R*_thin-disk_2maps-withmacroM.pkl',
                         'M01/spectrum_A-B_100000_*_R*_thin-disk_2maps-withmacroM.pkl',
                         'M01/spectrum_A-B_*_*_R*_thin-disk_2maps-withmacroM_BLRt65_s65_tau810_noshiftDRW_fBLRdistrib.pkl',
                         'spectrum_A4-B4_100000_*_R*_thin-disk_2maps-withmacroM.pkl',
                         'spectrum_A4-B4_100000_*_R*_thin-disk_2maps-withmacroM.pkl',
                         'spectrum_A4-B4_100000_*_R*_thin-disk_2maps-withmacroM_BLRt65_s65_tau810_noshiftDRW_fBLRdistrib.pkl']



        list_f_cut = [ 1. / 750., 1. / 100., 1. / 100., 1. / 750., 1. / 100., 1. / 100.,1. / 800., 1. / 100., 1. / 100.]
        #list_pklname.reverse()
        #list_f_cut.reverse()

    else:
        list_pklname = ['FML1/M03/spectrum_A-B_*_*_R*_thin-disk_2maps-withmacroM.pkl',
                        'FML1/M03/spectrum_A-B_*_*_R*_thin-disk_2maps-withmacroM.pkl',
                        'FML1/M03/spectrum_A-B_100000_*_R*_thin-disk_2maps-withmacroM_BLRt65_s65_tau810_noshiftDRW_fBLRdistrib.pkl',
                        'FML1/M01/spectrum_A-B_50000_*_R*_thin-disk_2maps-withmacroM.pkl',
                        'FML1/M01/spectrum_A-B_50000_*_R*_thin-disk_2maps-withmacroM.pkl',
                        'FML1/M01/spectrum_A-B_50000_*_R*_thin-disk_2maps-withmacroM_BLRt65_s65_tau810_noshiftDRW_fBLRdistrib.pkl',
                        'FML1/M001/spectrum_A-B_50000_*_R*_thin-disk_2maps-withmacroM.pkl',
                        'FML1/M001/spectrum_A-B_50000_*_R*_thin-disk_2maps-withmacroM.pkl',
                        'FML1/M001/spectrum_A-B_50000_*_R*_thin-disk_2maps-withmacroM_BLRt65_s65_tau810_noshiftDRW_fBLRdistrib.pkl']
        list_f_cut = [1./1000.,1./100.,1./50.,1./2000.,1./100.,1./200.,1./800.,1./100.,1./100.]

    list_maxzi = []
    #sampled_v1 = sampled_v1 / np.sum(sampled_v1)
    fig, ax = plt.subplots(3, 3, figsize=(5, 5))
    list_zi = []
    for j,pklname in enumerate(list_pklname):
        data_power, std_data, freq_data = pkl.load(open(pkldir + 'data.pkl', 'rb'))
        #print len(data_power)
        list_pkl = glob.glob(pklname)
        #print len(list_pkl)
        # All the variables that will be extracted from the name of the pkl file
        all_l = np.array([])
        all_prior = np.array([])
        all_r0 = np.array([])
        all_v = []

        f_cut_chi = list_f_cut[j]
        #print f_cut_chi
        #print freq_data[freq_data < f_cut_chi]

        data_power = np.array(data_power[1:len(freq_data[freq_data < f_cut_chi])])
        std_data = np.array(std_data[1:len(freq_data[freq_data < f_cut_chi])])
        freq_data = np.array(freq_data[1:len(freq_data[freq_data < f_cut_chi])])
        print len(data_power)


        for i, elem in enumerate(list_pkl):
            #print elem
            mean_power, var_power, freq = pkl.load(open(elem, 'rb'))
            #print lenpower
    #s      print mean_power
            #length=np.append(length,lenpower)
            var_power = np.array(var_power[1:])
            mean_power = np.array(mean_power[1:])

            v = int(elem.split('_')[3])
            #print len(var_power[np.where(freq_data<=f_cut_chi)])
            #print len(std_data)
            temp = chi2_custom(mean_power[np.where(freq_data<=f_cut_chi)],var_power[np.where(freq_data<=f_cut_chi)],data_power, std_data)
            l = likelyhood(temp)

            #print temp
            #if l>=0.0002:
            all_l= np.append(all_l, l)
            #else :
            #    print "2222222222222222222222222222222222222222222"
            #    all_l = np.append(all_l, 0.0)
            #print "v : %s"%(v)
            #print "prior : %s"%(sampled_v1[np.where(bin_edges == v)])
            all_prior= np.append(all_prior, sampled_v1[np.where(bin_edges == v)])
            all_r0= np.append(all_r0,int(elem.split('.')[0].split('_')[4].split('R')[1]))
            all_v.append(v)

        print f_cut_chi

        new_v = np.linspace(min(all_v), max(all_v), 100)
        new_r0 = np.linspace(min(all_r0), max(all_r0), 100)
        # print xx
        # print yy
        # triang = tri.Triangulation(better_sigma, better_BLRt)
        # interpolator = tri.LinearTriInterpolator(triang, chi)

        proba = all_l*all_prior

        triang = tri.Triangulation(all_v, all_r0)
        interpolator = mtri.CubicTriInterpolator(triang, all_l)
        interpolator_prior = mtri.CubicTriInterpolator(triang, all_prior)
        interpolator_proba = mtri.CubicTriInterpolator(triang, proba)

        xx, yy = np.meshgrid(new_v, new_r0)

        #zi = interpolator(xx, yy)
        #zi_prior = interpolator_prior(xx,yy)
        zi_proba = interpolator_proba(xx, yy)#/((new_v[1]-new_v[0])*(new_r0[1]-new_r0[0]))
        #zi_proba = interpolator(xx,yy)*interpolator_prior(xx,yy)
        #zi_proba = zi_proba/np.sum(zi_proba)
        zi_proba[zi_proba<0] = 0
        list_maxzi.append(np.max(zi_proba))
        list_zi.append(zi_proba)
        #print zi_proba
        #plt.plot(zi)
        #plt.show()

    list_zi = np.array(list_zi)

    for j, zi in enumerate(list_zi):

        el = Ellipse((np.mean(653), 4.2), width=358, height=2.5, angle=0, alpha = 0.7, ec='w', lw='3', fill = False, ls ='-')


        #levels = np.logspace(np.min(np.log10(all_chi)),np.max(np.log10(all_chi)),20)

        #orig_cmap = matplotlib.cm.RdYlGn_r
        #shifted_cmap = shiftedColorMap(orig_cmap, midpoint=1 / np.max(np.log10(zi)), name='shifted')
        weight03= (0.3-0.2)#*((new_v[1]-new_v[0])*(new_r0[1]-new_r0[0]))
        weight01 = (0.2 - (0.1-0.01)/2)#*((new_v[1]-new_v[0])*(new_r0[1]-new_r0[0]))
        weight001 = (0.1-0.01)/2#*((new_v[1]-new_v[0])*(new_r0[1]-new_r0[0]))
        list_weights = [weight001, weight01,weight03]
        new_list_zi = deepcopy(list_zi)
        lvlshift = 0.0
        if j//3 ==0:
#
#
            #normalization = np.sum(list_zi[0::3])

            normalization = np.sum(new_list_zi[0])*weight001+np.sum(new_list_zi[3])*weight01+np.sum(new_list_zi[6])*weight03
            #print "max :%s, %s" %(np.max(list_zi[0::3]/normalization), np.max(list_zi[0::3]))
            #print normalization
            #levels = np.linspace(np.min(list_zi[0::3]/normalization), np.max(list_zi[0::3]/normalization), 10)
            levels = np.round(np.linspace(0.0, np.max(list_zi[0::3]*list_weights[j//3] / normalization+lvlshift), 10),6)
            #levels = np.linspace(0.0, 0.00094, 10)
            max = np.max(list_zi[0::3])
            #levels = np.linspace(0, np.max(list_zi[0::3] / normalization) + 0.0001, 10)
        elif j//3 ==1:
#
            #normalization = np.sum(list_zi[1::3])
            normalization = np.sum(new_list_zi[1])*weight001+np.sum(new_list_zi[4])*weight01+np.sum(new_list_zi[7])*weight03

            #print "max :%s, %s" % (np.max(list_zi[1::3] / normalization), np.max(list_zi[1::3]))
            #print normalization
            #levels = np.linspace(np.min(list_zi[1::3]/normalization), np.max(list_zi[1::3]/normalization), 10)
            levels = np.round(np.linspace(0.0, np.max(list_zi[1::3]*list_weights[j//3] / normalization+lvlshift), 10),4)
            #levels = np.linspace(0.0, 0.00087, 10)

            max = np.max(list_zi[1::3])
#
        elif j//3 == 2:
            #normalization = np.sum(list_zi[2::3])
            normalization = np.sum(new_list_zi[2])*weight001+np.sum(new_list_zi[5])*weight01+np.sum(new_list_zi[8])*weight03
            print "%s max :%s, %s, %s" % (j,np.max(list_zi[2::3] / normalization), np.max(list_zi[2::3]),np.max(list_zi[j]/normalization))
            #print normalization
            #levels = np.linspace(np.min(list_zi[2::3]/normalization), np.max(list_zi[2::3]/normalization), 10)
            levels = np.round(np.linspace(0.0, 0.00006, 10),6)
            #levels = np.linspace(0.0, 0.00096, 10)

            max = np.max(list_zi[2::3])
        else:
            print "No idea of what I'm doing here"
            print j
            sys.exit()

        #levels = np.linspace(np.min(zi), np.max(zi), 10)
        proba = deepcopy(zi)
        #proba = proba/proba.sum()
        print "++++++++++++++++++++++++++++++++++++++++++++"
        print np.max(zi)
        print list_weights[j//3]
        zi = list_weights[j//3]*zi/normalization
        print np.max(zi)
        n = 10
        t = np.linspace(0, proba.max(), n)
        integral = ((proba >= t[:, None, None]) * proba).sum(axis=(1, 2))

        f = interpolate.interp1d(integral, t, kind='zero')
        t_contours = f(np.array([0.95,0.68])*proba.sum())
        #print "shape"
        #t_contours = np.insert(t_contours,0,np.min(proba))
        #t_contours = np.append(t_contours, np.max(proba))


        sc = ax[j % 3][j // 3].contourf(new_v, new_r0 / r_ref, zi,cmap='viridis', extend= 'max',levels =levels)  # ,locator=ticker.LogLocator(), cmap=shifted_cmap, alpha = 0.3)
        CS =ax[j % 3][j // 3].contour(new_v, new_r0 / r_ref, proba ,levels= [t_contours],colors='w', linestyles = '--' )

        #fmt = {}
        #strs = [r'2 $\sigma$',r'1 $\sigma$']
        #for l, s in zip(CS.levels, strs):
        #    fmt[l] = s
        #ax[j % 3][j // 3].clabel(CS, fmt=fmt, inline=1)#,
        #          #manual=manual_locations)

        #if j//3 ==0:
        #    sc = ax[j % 3][j // 3].contourf(new_v, new_r0 / r_ref, zi, cmap='viridis')  # ,locator=ticker.LogLocator(), cmap=shifted_cmap, alpha = 0.3,levels =levels)
        #else:
        #    sc = ax[j%3][j//3].contourf(new_v,new_r0/r_ref,zi,cmap = 'viridis', levels=levels)#,locator=ticker.LogLocator(), cmap=shifted_cmap, alpha = 0.3,levels =levels)

        ax[j%3][j//3].add_patch(el)
        ax[1][0].set(ylabel=r'$R_0$  in units of $R_{\rm MK11}$')#, xlim=(0,1500), ylim=(0,np.max(all_r0)/r_ref))
        ax[2][1].set(xlabel=r'$v_e$ [$km\cdot s^{-1}}$]')
        #ax[1][1].text(200, 5.5, 'Light curve fitting measurement', color='w')
        #ax[1][1].text(200, 1.2, 'Flux estimate',color='w')
        #ax[1][2].set_xticks([100,500,1000,1500])
        ax[j % 3][j // 3].set_yticks([0.1,2,4,6])
        ax[j%3][j//3].axhline(1, 0,1500, ls ='-',color='w')
        if not j%3==0:
            ax[j//3][j%3].yaxis.set_visible(False)
        if not (j//3==2):
            ax[j // 3][j % 3].xaxis.set_visible(False)
        print "levels : "
        print levels
        #levels = np.log10(levels)
        if j // 3 == 0 and j%3==2:
            cbar_ax = fig.add_axes([0.91, 0.66, 0.02, 0.2])
            cb = fig.colorbar(sc, cax=cbar_ax)
            #cb.ax.set_yticklabels(['{:.2e}'.format(x) for x in levels.tolist()])
            cb.ax.text(0.,1.1,r'$\times 10^{-5}$')
            levels = levels*100000
            cb.ax.set_yticklabels(['{:.3}'.format(x) for x in levels.tolist()])
            #cb.ax.set_yticklabels(['0.0', r'2.5$\sigma$', r'2$\sigma$', r'1.5$\sigma$', r'1$\sigma$', r'0.5$\sigma$'])
        elif j // 3 == 1 and j%3==2:
            cbar_ax = fig.add_axes([0.91, 0.4, 0.02, 0.2])
            cb = fig.colorbar(sc, label="Posterior Probability", cax=cbar_ax)
            cb.ax.text(0.,1.1,r'$\times 10^{-3}$')


            levels = levels * 1000
            cb.ax.set_yticklabels(['{:.3}'.format(x) for x in levels.tolist()])
            #cb.ax.set_yticklabels(['0.0', r'2.5$\sigma$', r'2$\sigma$', r'1.5$\sigma$', r'1$\sigma$', r'0.5$\sigma$'])

        elif j // 3 == 2 and j%3==2:
            cbar_ax = fig.add_axes([0.91, 0.12, 0.02, 0.2])
            cb = fig.colorbar(sc, cax=cbar_ax)
            cb.ax.text(0., 1.1, r'$\times 10^{-5}$')
            levels = levels * 100000
            cb.ax.set_yticklabels(['{:.3}'.format(x) for x in levels.tolist()])
            #cb.ax.set_yticklabels(['0.0', r'2.5$\sigma$', r'2$\sigma$', r'1.5$\sigma$', r'1$\sigma$', r'0.5$\sigma$'])



    plt.subplots_adjust(wspace = 0.01, hspace = 0.1)

    ax[0][2].set(title =r'$\left<M\right>$= 0.3 $M_{\odot}$')
    ax[0][1].set(title = r'$\left<M\right>$= 0.1 $M_{\odot}$')
    ax[0][0].set(title =r'$\left<M\right>$= 0.01 $M_{\odot}$')


    #locator = LogLocator()
    #formatter = LogFormatter()
    #cb.locator = locator
    #cb.formatter = formatter
    #cb.update_normal(sc)
    #ax.text(1000, 1.5, "Best fit : v %s ; R %s" %(v_source[idx], round(r0[idx]/20,1)))
    #plt.suptitle(r"$f_{M/L}$ =%s"%(fml))
    xshift =-450
    yshift = 3.4
    xshift2 = -330
    yshift2 = 3.4
    xshift3 = -230
    yshift3 = 3.4
    ax[0][0].text(xshift, yshift, r"Model 1",rotation = 'vertical',va="center",ha='center',
            bbox=dict(edgecolor='red', linestyle='-', facecolor='w'))
    ax[1][0].text(xshift, yshift,r"Model 2",rotation = 'vertical',va="center",ha='center',
            bbox=dict(edgecolor='red', linestyle='-', facecolor='w'))

    ax[2][0].text(xshift, yshift, r"Model 3",rotation = 'vertical',ha='center',va = 'center',
            bbox=dict(edgecolor='red', linestyle='-', facecolor='w'))
    ax[0][0].text(xshift2, yshift2,'Low f',
                  rotation='vertical', va="center", ha="center",color='grey',

                  )
    ax[1][0].text(xshift2, yshift2, r"Low and high f", rotation='vertical', va="center",
                  ha="center",color='grey'
                  )

    ax[2][0].text(xshift2, yshift2, r"Low f and high f", rotation='vertical',va = 'center', ha = 'center',color='grey'

                  )
    ax[0][0].text(xshift3, yshift3,' No reverberation',
                  rotation='vertical', va="center", ha="center",color='grey'

                  )
    ax[1][0].text(xshift3, yshift3, r" No reverberation", rotation='vertical', va="center",ha='center',color='grey'
                  )

    ax[2][0].text(xshift3, yshift3, r"With reverberation", rotation='vertical',va='center',color='grey',
                  ha='center',
                  )

    #plt.legend()
    plt.show()
    fig.savefig(plotdir+"RvsVvsposterior_lowf_highf_reverb_all_fml%s_normalized.pdf"%(float(fml)), dpi = 100)
    sys.exit()

if 0:
    import matplotlib.tri as mtri
    from matplotlib import patches
    from scipy import interpolate
    from copy import deepcopy

    # Plot of the map of proba as function of radius and velocity for low, high freq and reverberation collapsed on M
    resultdir ='/home/ericpaic/Documents/PHD/results/powerspectrum/pkl/reverberation_study/'
    os.chdir(resultdir)
    r_ref =15 #Reference radius in pxl
    fml = 0.9


    if fml==0.9:
        list_pklname = ['spectrum_A4-B4_100000_*_R*_thin-disk_2maps-withmacroM.pkl','spectrum_A4-B4_100000_*_R*_thin-disk_2maps-withmacroM.pkl','spectrum_A4-B4_100000_*_R*_thin-disk_2maps-withmacroM_BLRt65_s65_tau810_noshiftDRW_fBLRdistrib.pkl', 'M01/spectrum_A-B_100000_*_R*_thin-disk_2maps-withmacroM.pkl', 'M01/spectrum_A-B_100000_*_R*_thin-disk_2maps-withmacroM.pkl','M01/spectrum_A-B_*_*_R*_thin-disk_2maps-withmacroM_BLRt65_s65_tau810_noshiftDRW_fBLRdistrib.pkl','M001/spectrum_A-B_50000_*_R*_thin-disk_2maps-withmacroM.pkl','M001/spectrum_A-B_50000_*_R*_thin-disk_2maps-withmacroM.pkl', 'M001/spectrum_A-B_50000_*_R*_thin-disk_2maps-withmacroM_BLRt65_s65_tau810_noshiftDRW_fBLRdistrib.pkl']
        list_f_cut = [1. / 800., 1. / 100., 1. / 100., 1. / 750., 1. / 100., 1. / 100., 1. / 750., 1. / 100., 1. / 100.]

    else:
        list_pklname = ['FML1/M03/spectrum_A-B_*_*_R*_thin-disk_2maps-withmacroM.pkl',
                        'FML1/M03/spectrum_A-B_*_*_R*_thin-disk_2maps-withmacroM.pkl',
                        'FML1/M03/spectrum_A-B_100000_*_R*_thin-disk_2maps-withmacroM_BLRt65_s65_tau810_noshiftDRW_fBLRdistrib.pkl',
                        'FML1/M01/spectrum_A-B_50000_*_R*_thin-disk_2maps-withmacroM.pkl',
                        'FML1/M01/spectrum_A-B_50000_*_R*_thin-disk_2maps-withmacroM.pkl',
                        'FML1/M01/spectrum_A-B_50000_*_R*_thin-disk_2maps-withmacroM_BLRt65_s65_tau810_noshiftDRW_fBLRdistrib.pkl',
                        'FML1/M001/spectrum_A-B_50000_*_R*_thin-disk_2maps-withmacroM.pkl',
                        'FML1/M001/spectrum_A-B_50000_*_R*_thin-disk_2maps-withmacroM.pkl',
                        'FML1/M001/spectrum_A-B_50000_*_R*_thin-disk_2maps-withmacroM_BLRt65_s65_tau810_noshiftDRW_fBLRdistrib.pkl']
        list_f_cut = [1./1000.,1./100.,1./50.,1./2000.,1./100.,1./200.,1./800.,1./100.,1./100.]

    list_maxzi = []
    #sampled_v1 = sampled_v1 / np.sum(sampled_v1)
    #fig, ax = plt.subplots(3, 1, constrained_layout=True)
    fig = plt.figure()

    gs = GridSpec(3, 2, width_ratios=[20, 1])  # 2 rows, 3 columns

    ax1 = fig.add_subplot(gs[0, 0])  # First row, first column
    ax2 = fig.add_subplot(gs[0, 1])  # First row, second column
    ax3 = fig.add_subplot(gs[1, 0])  # First row, third column
    ax4 = fig.add_subplot(gs[1, 1])  # Second row, span all columns
    ax5 = fig.add_subplot(gs[2, 0])  # First row, third column
    ax6 = fig.add_subplot(gs[2, 1])
    ax=[ax1,ax2,ax3,ax4,ax5,ax6]
    list_zi = []
    for j,pklname in enumerate(list_pklname):
        data_power, std_data, freq_data = pkl.load(open(pkldir + 'data.pkl', 'rb'))
        print len(data_power)
        list_pkl = glob.glob(pklname)
        print len(list_pkl)
        # All the variables that will be extracted from the name of the pkl file
        all_l = np.array([])
        all_prior = np.array([])
        all_r0 = np.array([])
        all_v = []

        f_cut_chi = list_f_cut[j]
        print f_cut_chi
        print freq_data[freq_data < f_cut_chi]

        data_power = np.array(data_power[1:len(freq_data[freq_data < f_cut_chi])])
        std_data = np.array(std_data[1:len(freq_data[freq_data < f_cut_chi])])
        freq_data = np.array(freq_data[1:len(freq_data[freq_data < f_cut_chi])])
        print len(data_power)


        for i, elem in enumerate(list_pkl):
            #print elem
            mean_power, var_power, freq = pkl.load(open(elem, 'rb'))
            #print lenpower
    #s      print mean_power
            #length=np.append(length,lenpower)
            var_power = np.array(var_power[1:])
            mean_power = np.array(mean_power[1:])

            v = int(elem.split('_')[3])
            #print len(var_power[np.where(freq_data<=f_cut_chi)])
            #print len(std_data)
            temp = chi2_custom(mean_power[np.where(freq_data<=f_cut_chi)],var_power[np.where(freq_data<=f_cut_chi)],data_power, std_data)
            l = likelyhood(temp)

            #print temp
            #if l>=0.0002:
            all_l= np.append(all_l, l)
            #else :
            #    print "2222222222222222222222222222222222222222222"
            #    all_l = np.append(all_l, 0.0)
            #print "v : %s"%(v)
            #print "prior : %s"%(sampled_v1[np.where(bin_edges == v)])
            all_prior= np.append(all_prior, sampled_v1[np.where(bin_edges == v)])
            all_r0= np.append(all_r0,int(elem.split('.')[0].split('_')[4].split('R')[1]))
            all_v.append(v)

        print f_cut_chi

        new_v = np.linspace(min(all_v), max(all_v), 100)
        new_r0 = np.linspace(min(all_r0), max(all_r0), 100)
        # print xx
        # print yy
        # triang = tri.Triangulation(better_sigma, better_BLRt)
        # interpolator = tri.LinearTriInterpolator(triang, chi)

        proba = all_l*all_prior

        triang = tri.Triangulation(all_v, all_r0)
        interpolator = mtri.CubicTriInterpolator(triang, all_l)
        interpolator_prior = mtri.CubicTriInterpolator(triang, all_prior)
        print triang
        print proba.shape
        interpolator_proba = mtri.CubicTriInterpolator(triang, proba)

        xx, yy = np.meshgrid(new_v, new_r0)

        #zi = interpolator(xx, yy)
        #zi_prior = interpolator_prior(xx,yy)
        zi_proba = interpolator_proba(xx, yy)#/((new_v[1]-new_v[0])*(new_r0[1]-new_r0[0]))
        #zi_proba = interpolator(xx,yy)*interpolator_prior(xx,yy)
        #zi_proba = zi_proba/np.sum(zi_proba)
        zi_proba[zi_proba<0] = 0
        list_maxzi.append(np.max(zi_proba))
        list_zi.append(zi_proba)
        #print zi_proba
        #plt.plot(zi)
        #plt.show()

    list_zi = np.array(list_zi)
    print list_maxzi

    for j, zi in enumerate(list_zi):


        el_collapsed = Ellipse((np.mean(653), 4.2), width=358, height=2.5, angle=0, alpha=0.2, ec='w', lw='3', fill=False, ls='-')

        #levels = np.logspace(np.min(np.log10(all_chi)),np.max(np.log10(all_chi)),20)

        #orig_cmap = matplotlib.cm.RdYlGn_r
        #shifted_cmap = shiftedColorMap(orig_cmap, midpoint=1 / np.max(np.log10(zi)), name='shifted')
        weight03= (0.3-0.2)*((new_v[1]-new_v[0])*(new_r0[1]-new_r0[0]))
        weight01 = (0.2 - (0.1-0.01)/2)*((new_v[1]-new_v[0])*(new_r0[1]-new_r0[0]))
        weight001 = (0.1-0.01)/2*((new_v[1]-new_v[0])*(new_r0[1]-new_r0[0]))
        print "weights : "
        print weight03
        print weight01
        print weight001
        new_list_zi = deepcopy(list_zi)
        lvlshift = 0.
        if j//3 ==0:
#
#
            #normalization = np.sum(list_zi[0::3])

            normalization = np.sum(new_list_zi[0])*weight03+np.sum(new_list_zi[3])*weight01+np.sum(new_list_zi[6])*weight001
            print "max :%s, %s" %(np.max(list_zi[0::3]/normalization), np.max(list_zi[0::3]))
            print normalization
            #levels = np.linspace(np.min(list_zi[0::3]/normalization), np.max(list_zi[0::3]/normalization), 10)
            levels = np.linspace(0.0, np.max(list_zi[0::3] / normalization+lvlshift), 10)
            #levels = np.linspace(0.0, 0.00094, 10)
            max = np.max(list_zi[0::3])
            #levels = np.linspace(0, np.max(list_zi[0::3] / normalization) + 0.0001, 10)
            collapsed_zi = new_list_zi[0]*weight03+new_list_zi[3]*weight01+new_list_zi[6]*weight001
        elif j//3 ==1:
#
            #normalization = np.sum(list_zi[1::3])
            normalization = np.sum(new_list_zi[1])*weight03+np.sum(new_list_zi[4])*weight01+np.sum(new_list_zi[7])*weight001

            print "max :%s, %s" % (np.max(list_zi[1::3] / normalization), np.max(list_zi[1::3]))
            print normalization
            #levels = np.linspace(np.min(list_zi[1::3]/normalization), np.max(list_zi[1::3]/normalization), 10)
            levels = np.linspace(0.0, np.max(list_zi[1::3] / normalization+lvlshift), 10)
            #levels = np.linspace(0.0, 0.00087, 10)

            max = np.max(list_zi[1::3])
            collapsed_zi = new_list_zi[1]*weight03+new_list_zi[4]*weight01+new_list_zi[7]*weight001
#
        elif j//3 == 2:
            #normalization = np.sum(list_zi[2::3])
            normalization = np.sum(new_list_zi[2])*weight03+np.sum(new_list_zi[5])*weight01+np.sum(new_list_zi[8])*weight001
            print "max :%s, %s" % (np.max(list_zi[2::3] / normalization), np.max(list_zi[2::3]))
            print normalization
            #levels = np.linspace(np.min(list_zi[2::3]/normalization), np.max(list_zi[2::3]/normalization), 10)
            levels = np.linspace(0.0, np.max(list_zi[2::3] / normalization+lvlshift), 10)
            #levels = np.linspace(0.0, 0.00096, 10)

            max = np.max(list_zi[2::3])
            collapsed_zi = new_list_zi[2] * weight03 + new_list_zi[5] * weight01 + new_list_zi[8] * weight001
        else:
            print "No idea of what I'm doing here"
            print j
            sys.exit()


        proba_collapsed = deepcopy(collapsed_zi)
        # proba = proba/proba.sum()

        collapsed_zi = collapsed_zi / normalization


        n = 10
        t_collapsed = np.linspace(0, collapsed_zi.max(), n)
        integral_collapsed = ((collapsed_zi >= t_collapsed[:, None, None]) * collapsed_zi).sum(axis=(1, 2))

        f_collapsed = interpolate.interp1d(integral_collapsed, t_collapsed, kind='linear')
        t_contours_collapsed = f_collapsed(np.array([0.95,0.68]) * collapsed_zi.sum())
        print "contours :"
        print t_contours_collapsed
        # print "shape"
        # t_contours = np.insert(t_contours,0,np.min(proba))
        # t_contours = np.append(t_contours, np.max(proba))
        sc_collapsed = ax[(j//3)*2].contourf(new_v, new_r0 / r_ref, collapsed_zi, cmap='viridis',
                                        extend='max')# ,levels =levels)  # ,locator=ticker.LogLocator(), cmap=shifted_cmap, alpha = 0.3)
        CS_collapsed = ax[(j//3)*2].contour(new_v, new_r0 / r_ref, collapsed_zi, levels=[t_contours_collapsed], colors='w', linestyles='--')
        ax[0].xaxis.set_visible(False)
        ax[2].xaxis.set_visible(False)

        #fmt = {}
        #strs = [r'2 $\sigma$',r'1 $\sigma$']
        #for l, s in zip(CS.levels, strs):
        #    fmt[l] = s
        #ax[j % 3][j // 3].clabel(CS, fmt=fmt, inline=1)#,
        #          #manual=manual_locations)

        #if j//3 ==0:
        #    sc = ax[j % 3][j // 3].contourf(new_v, new_r0 / r_ref, zi, cmap='viridis')  # ,locator=ticker.LogLocator(), cmap=shifted_cmap, alpha = 0.3,levels =levels)
        #else:
        #    sc = ax[j%3][j//3].contourf(new_v,new_r0/r_ref,zi,cmap = 'viridis', levels=levels)#,locator=ticker.LogLocator(), cmap=shifted_cmap, alpha = 0.3,levels =levels)


        ax[(j//3)*2].add_patch(el_collapsed)
        ax[2].set(ylabel=r'$R_0$  in units of $R_{MK11}$')#, xlim=(0,1500), ylim=(0,np.max(all_r0)/r_ref))
        ax[4].set(xlabel=r'$v_e$ [$km\cdot s^{-1}}$]')
        #ax[1][2].set_xticks([100,500,1000,1500])
        ax[(j//3)*2].set_yticks([0.1,2,4,6])

        ax[(j//3)*2].axhline(1, 0, 1500, ls='-', color='w')

        if j // 3 == 0:
            #divider = make_axes_locatable(ax[j//3])
            #cbar_ax = divider.append_axes("right", size="50%", pad=0.05)

            cb = fig.colorbar(sc_collapsed, cax=ax[(j // 3)*2+1], format = '%.2e')
            cb.ax.set_yticklabels(['{:.2e}'.format(x) for x in levels.tolist()])
            #cb.ax.set_yticklabels(['0.0', r'2.5$\sigma$', r'2$\sigma$', r'1.5$\sigma$', r'1$\sigma$', r'0.5$\sigma$'])
        elif j // 3 == 1:
            #cbar_ax = fig.add_axes([0.7, 0.4, 0.02, 0.2])
            #divider = make_axes_locatable(ax[j//3])
            #cbar_ax = divider.append_axes("right", size="50%", pad=0.05)
            cb = fig.colorbar(sc_collapsed, label="Posterior Probability", cax=ax[(j // 3)*2+1], format = '%.2e')
            cb.ax.set_yticklabels(['{:.2e}'.format(x) for x in levels.tolist()])
            #cb.ax.set_yticklabels(['0.0', r'2.5$\sigma$', r'2$\sigma$', r'1.5$\sigma$', r'1$\sigma$', r'0.5$\sigma$'])
#
        elif j // 3 == 2:
            #cbar_ax = fig.add_axes([0.7, 0.15, 0.02, 0.2])
            #divider = make_axes_locatable(ax[j//3])
            #cbar_ax = divider.append_axes("right", size="50%", pad=0.05)
            cb = fig.colorbar(sc_collapsed, cax=ax[(j // 3)*2+1], format = '%.2e')
            cb.ax.set_yticklabels(['{:.2e}'.format(x) for x in levels.tolist()])
            #cb.ax.set_yticklabels(['0.0', r'2.5$\sigma$', r'2$\sigma$', r'1.5$\sigma$', r'1$\sigma$', r'0.5$\sigma$'])



    #plt.subplots_adjust(hspace = 0.1)

    #ax[0][0].set(title =r'$\left<M\right>$= 0.3 $M_{\odot}$')
    #ax[0][1].set(title = r'$\left<M\right>$= 0.1 $M_{\odot}$')
    #ax[0][2].set(title =r'$\left<M\right>$= 0.01 $M_{\odot}$')
    ax[0].set(title ='Collapsed probability')


    #locator = LogLocator()
    #formatter = LogFormatter()
    #cb.locator = locator
    #cb.formatter = formatter
    #cb.update_normal(sc)
    #ax.text(1000, 1.5, "Best fit : v %s ; R %s" %(v_source[idx], round(r0[idx]/20,1)))
    #plt.suptitle(r"$f_{M/L}$ =%s"%(fml))
    xshift =-400
    yshift = 3
    #ax[0].text(xshift, yshift, r"Model 1",
    #        bbox=dict(edgecolor='red', linestyle='-', facecolor='w'))
    #ax[1].text(xshift, yshift,r"Model 2",
    #        bbox=dict(edgecolor='red', linestyle='-', facecolor='w'))
    #ax[2].text(xshift, yshift, r"Model 3",
    #        bbox=dict(edgecolor='red', linestyle='-', facecolor='w'))


    #plt.legend()
    plt.tight_layout()
    plt.show()
    fig.savefig(plotdir+"RvsVvsposterior_lowf_highf_reverb_all_fml%s_normalized_onlycollapsed.pdf"%(float(fml)), dpi = 100)
    sys.exit()

if 0:
    # Plot of the map of proba as function of radius and velocity. Comparing PS to delta m likelyhoods
    resultdir ='/media/ericpaic/TOSHIBA EXT/TPIVb/results/powerspectrum/pkl/'
    os.chdir(resultdir + 'M0.3/A3-B2_flattop')
    list_pkl = glob.glob("*.pkl")
    r_ref =15 #Reference radius in pxl

    # All the variables taht will be extracted from the name of the pkl file
    v_source = []
    reject_v_source=[]
    r0 = np.array([])
    reject_r0 = np.array([])
    chi = []
    reject_chi = []

    better_v = []
    better_r0 = []
    all_l = np.array([])
    all_prior = np.array([])

    all_r0 = np.array([])
    all_v = []

    maxchi = 10
    f_cut_chi = 0.01
    length = np.array([])
    data_power, std_data, freq_data = pkl.load(open(pkldir+'data.pkl', 'rb'))
    data_power = np.array(data_power[1:len(freq_data[freq_data < f_cut_chi])])
    std_data = np.array(std_data[1:len(freq_data[freq_data < f_cut_chi])])
    freq_data = np.array(freq_data[1:len(freq_data[freq_data < f_cut_chi])])

    for i, elem in enumerate(list_pkl):
        print elem
        mean_power, var_power, freq = pkl.load(open(elem, 'rb'))
        #print lenpower
        #length=np.append(length,lenpower)
        var_power = np.array(var_power[1:])
        mean_power = np.array(mean_power[1:])

        v = int(elem.split('_')[3])
        temp = chi2_custom(mean_power[np.where(frequency_spline<=f_cut_chi)],var_power[np.where(frequency_spline<=f_cut_chi)],data_power, std_data)
        l = likelyhood(temp)
        #print temp
        all_l= np.append(all_l, l)
        print "v : %s"%(v)
        print "prior : %s"%(sampled_v1[np.where(bin_edges == v)])
        all_prior= np.append(all_prior, sampled_v1[np.where(bin_edges == v)])
        all_r0= np.append(all_r0,int(elem.split('.')[0].split('_')[4].split('R')[1]))
        all_v.append(v)



    z = zip(all_r0, all_v)
    z_unique,inv = np.unique(z, return_inverse=True, axis =0)
    inv_unique = np.unique(inv)
    print z_unique
    print inv_unique

    #for i,elem in enumerate(inv_unique):
    #    temp = np.average(all_chi[np.where(inv == elem)])
    #    if temp < maxchi:
    #        chi.append(temp)
    #        v_source.append(z_unique[i,1])
    #        r0= np.append(r0,z_unique[i,0])
    #        if temp <1:
    #            print "+++++++++++++++"
    #            print temp
    #            print z_unique[i, 1]
    #            print z_unique[i, 0]
    #            better_v.append(z_unique[i, 1])
    #            better_r0 = np.append(better_r0, z_unique[i, 0])
    #    else :
    #        reject_v_source.append(z_unique[i, 1])
    #        reject_r0= np.append(reject_r0,z_unique[i, 0])

    print chi
    print v_source
    print r0
    print better_v
    print better_r0

    new_v = np.linspace(min(all_v), max(all_v), 10000)
    new_r0 = np.linspace(min(all_r0), max(all_r0), 10000)
    # print xx
    # print yy
    # triang = tri.Triangulation(better_sigma, better_BLRt)
    # interpolator = tri.LinearTriInterpolator(triang, chi)

    proba = all_l*all_prior
    triang = tri.Triangulation(all_v, all_r0)
    interpolator = tri.LinearTriInterpolator(triang, all_l)
    interpolator_prior = tri.LinearTriInterpolator(triang, all_prior)
    interpolator_proba = tri.LinearTriInterpolator(triang, proba)
    xx, yy = np.meshgrid(new_v, new_r0)
    zi = interpolator(xx, yy)
    zi_prior = interpolator_prior(xx,yy)
    zi_proba = interpolator_proba(xx, yy)

    #plt.plot(zi)
    #plt.show()
    from matplotlib import patches
    el = Ellipse((480, 4.2), width=370, height=2.5, angle=0, alpha = 0.3)

    fig, ax = plt.subplots(1,3, figsize = (13,8))
    #levels = np.logspace(np.min(np.log10(all_chi)),np.max(np.log10(all_chi)),20)

    #orig_cmap = matplotlib.cm.RdYlGn_r
    #shifted_cmap = shiftedColorMap(orig_cmap, midpoint=1 / np.max(np.log10(zi)), name='shifted')
    #levels = np.logspace(np.min(np.log10(all_chi)), np.max(np.log10(all_chi)), 20)
    sc = ax[0].contourf(new_v,new_r0/r_ref,zi, 20, cmap = 'RdYlGn')#,locator=ticker.LogLocator(), cmap=shifted_cmap, alpha = 0.3,levels =levels)
    ax[1].contourf(new_v, new_r0 / r_ref, zi_prior, 20, cmap = 'RdYlGn')#, locator=ticker.LogLocator(), cmap=shifted_cmap, alpha=0.3,levels=levels)
    ax[2].contourf(new_v, new_r0 / r_ref, zi_proba, 20, cmap='RdYlGn')
    #ax.contourf(x, y, multivariate_normal.pdf(pos,[485, 4.2], [[375, 0.0], [0.0, 2.5]]), alpha=0.3)
    #ax.plot(reject_v_source, reject_r0/r_ref, 'or', label = r"$\chi^2 > %s $"%(maxchi))
    #ax.fill_between([110, 750], [1.7, 1.7], [6.7, 6.7], alpha = 0.3, label='Morgan et al.(2012)')
   # ax.scatter(better_v, better_r0/r_ref, s = 90,facecolor='None',edgecolors = "c")
    ax[0].add_patch(el)
    ax[0].set(ylabel=r'$R_0$  in units of $R_{MK11}$', xlabel=r'$v_e$ [$km\cdot s^{-1}}$]', xlim=(0,1500))
    ax[1].set(ylabel=r'$R_0$  in units of $R_{MK11}$', xlabel=r'$v_e$ [$km\cdot s^{-1}}$]', xlim=(0, 1500))

    ax[2].set(ylabel=r'$R_0$  in units of $R_{MK11}$', xlabel=r'$v_e$ [$km\cdot s^{-1}}$]', xlim=(0, 1500))


    #cb =plt.colorbar(sc, label=r"$\chi^2$")
    #locator = LogLocator()
    #formatter = LogFormatter()
    #cb.locator = locator
    #cb.formatter = formatter
    #cb.update_normal(sc)
    #ax.text(1000, 1.5, "Best fit : v %s ; R %s" %(v_source[idx], round(r0[idx]/20,1)))

    #plt.legend()
    plt.show()
    #fig.savefig(resultdir+"powerspectrum/png/r0vsVvsChi2_FML0,9M0,3_wavy_flattop_A7-B3.png")
    sys.exit()

if 0:
    # Plot of the map of p value as function of radius and velocity

    # os.chdir(resultdir + 'pkl/angledistrib/')
    r_ref = 15  # Reference radius in pxl

    resultdir ='/media/ericpaic/TOSHIBA EXT/TPIVb/results/powerspectrum/pkl/'
    list_subdir = ['M0.3/A3-B2_flattop','FML1M0.3/flattop', 'M0.1/A2-B3_flattop', 'FML1M0.1', 'M0.01/flattop', 'FML1M0.01']
    list_fml = [0.9,1,0.9,1,0.9,1]
    list_M = [0.3,0.3,0.1,0.1,0.01,0.01]
    from matplotlib import patches

    f_cut_chi = 1 / 100


    power_data, std_data, freq_data = pkl.load(open(pkldir + 'data.pkl', 'rb'))
    new_power_spline = power_data[1:len(freq_data[freq_data < f_cut_chi])]
    new_f = freq_data[1:len(freq_data[freq_data < f_cut_chi])]
    std_data = np.array(std_data[1:len(freq_data[freq_data < f_cut_chi])])

    #levels = np.logspace(np.min(np.log10(all_chi)),np.max(np.log10(all_chi)),20)

    orig_cmap = matplotlib.cm.RdYlGn
    shifted_cmap = shiftedColorMap(orig_cmap, midpoint=0.15, name='shifted')
    levels = np.arange(0,1.05,0.05)
    # levels = np.logspace(np.min(np.log10(all_chi)),np.max(np.log10(all_chi)),20)

    fig, ax = plt.subplots(int(len(list_subdir)/2), 2, figsize=(15, 10))

    for j,subdir in enumerate(list_subdir):
        os.chdir(resultdir+subdir)
        #list_pkl = glob.glob("spectrum_A3-B2_*_flattop.pkl")
        list_pkl = glob.glob("*.pkl")


        # All the variables taht will be extracted from the name of the pkl file
        v_source = []
        reject_v_source=[]
        r0 = np.array([])
        reject_r0 = np.array([])
        chi = []
        reject_chi = []

        all_l = np.array([])
        all_r0 = np.array([])
        all_v = []
        all_prior= np.array([])

        length = np.array([])
        sampled_v1 = sampled_v1/np.sum(sampled_v1)
        for i, elem in enumerate(list_pkl):
            print elem
            mean_power, var_power, freq = pkl.load(open(elem, 'rb'))
            #print lenpower
            #length=np.append(length,lenpower)
            var_power = np.array(var_power[1:])
            mean_power = np.array(mean_power[1:])
            v = int(elem.split('_')[3])
            #print mean_power
            #print var_power
            temp = chi2_custom(mean_power[np.where(frequency_spline<=f_cut_chi)],var_power[np.where(frequency_spline<=f_cut_chi)],new_power_spline, std_data)
            l = likelyhood(temp)
            #print temp
            all_l= np.append(all_l, l)
            print "v : %s"%(v)
            print "prior : %s"%(sampled_v1[np.where(bin_edges == v)])
            all_prior= np.append(all_prior, sampled_v1[np.where(bin_edges == v)])
            all_r0= np.append(all_r0,int(elem.split('.')[0].split('_')[4].split('R')[1]))
            all_v.append(v)#.split('v')[1]))



        new_v = np.linspace(min(all_v), max(all_v), 1000)
        new_r0 = np.linspace(min(all_r0), max(all_r0), 1000)

        #all_prior = all_prior/np.sum(all_prior)
        proba = all_l*all_prior
        triang = tri.Triangulation(all_v, all_r0)
        interpolator = tri.LinearTriInterpolator(triang, all_l)
        interpolator_prior = tri.LinearTriInterpolator(triang, all_prior)
        interpolator_proba = tri.LinearTriInterpolator(triang, proba)
        xx, yy = np.meshgrid(new_v, new_r0)

        zi = interpolator(xx, yy)
        zi_prior = interpolator_prior(xx,yy)
        zi_proba = interpolator_proba(xx, yy)/((new_v[1]-new_v[0])*(new_r0[1]-new_r0[0]))
        #to_fit = np.ravel(zi_proba)
        #print type(zi)
        #print type(zi_prior)
        #import scipy.optimize as opt
        #initial_guess = (0.8, 700, 0.1, 50, 1, 0,0)
        #print "shape :"
        #print xx.shape
        #print to_fit.shape
        #popt, pcov = opt.curve_fit(twoD_Gaussian, (xx, yy), to_fit, p0=initial_guess)
        #data_fitted = twoD_Gaussian((xx, yy), *popt)
        #plt.plot(zi)
        #plt.show()

        #sys.exit()

        alpha = 1
        el = Ellipse((653, 3.15), width=358, height=2.5, angle=0, alpha=0.5)
        sc = ax[j//2][j%2].contourf(new_v,new_r0/r_ref,zi_proba, 20, alpha = alpha, cmap= 'Greens')#, levels=levels)
        #ax[j // 2][j % 2].contour(new_v, new_r0 / r_ref, data_fitted.reshape(len(new_v),len(new_r0)), 20, alpha=alpha,
        #                           colors = 'w')  # , levels=levels)
        #ax.contourf(x, y, multivariate_normal.pdf(pos,[485, 4.2], [[375, 0.0], [0.0, 2.5]]), alpha=0.3)
        #ax.plot(reject_v_source, reject_r0/r_ref, 'or', label = r"$\chi^2 > %s $"%(maxchi))
        #ax.fill_between([110, 750], [1.7, 1.7], [6.7, 6.7], alpha = 0.3, label='Morgan et al.(2012)')
       # ax.scatter(better_v, better_r0/r_ref, s = 90,facecolor='None',edgecolors = "c")
        ax[j//2][j%2].add_patch(el)
        el.set()
        ax[j//2][j%2].text(600, 5, r'$f_{M/L}$= %s, $\left<M\right>$ = %s $M_{\odot}$'%(list_fml[j], list_M[j]), color = 'white')
        ax[j//2][j%2].axhline(y=1, ls = '--')
        ax[j//2][j%2].set(xlim=(0,1500), ylim = (0,max(all_r0)/r_ref))
        if j//2 != 2:
            ax[j//2][j%2].xaxis.set_visible(False)
        #if j%2 !=0 :
        #    ax[j // 2][j % 2].yaxis.set_visible(False)
        #ax[j//2][j%2].legend(loc = 1)
    plt.subplots_adjust(hspace=.0, wspace = 0.1)
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.91,0.15,0.02,0.7])
    cb = fig.colorbar(sc, label="Posterior Probability", cax = cbar_ax, alpha = alpha)
    #, ticks = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

    ax[1][0].set(ylabel=r'$R_0$  in units of $R_{MK11}$')
    #ax[1][1].set(ylabel=r'$R_0$  in units of $R_{MK11}$')
    ax[2][0].set(xlabel=r'$v_e$ [$km\cdot s^{-1}}$]')
    ax[2][1].set(xlabel=r'$v_e$ [$km\cdot s^{-1}}$]')

    plt.show()
    fig.savefig(plotdir+"RvsVvsposterior_all.pdf", dpi = 100)

    sys.exit()

if 0:
    # Plot proba vs PS

    # os.chdir(resultdir + 'pkl/angledistrib/')
    r_ref = 15  # Reference radius in pxl

    resultdir = '/media/ericpaic/TOSHIBA EXT/TPIVb/results/powerspectrum/pkl/'
    list_probadir = ['/home/ericpaic/Documents/PHD/results/proba/M0.3/pkl',
                     '/home/ericpaic/Documents/PHD/results/proba/M0.1/pkl',
                     '/home/ericpaic/Documents/PHD/results/proba/M0.01/pkl']
    list_subdir = ['M0.3/A3-B2_flattop', 'M0.1/A2-B3_flattop', 'M0.01/flattop']

    list_v = np.arange(100, 1600, 100)

    f_cut_chi = 1 / 100

    power_data, std_data, freq_data = pkl.load(open(pkldir + 'data.pkl', 'rb'))
    new_power_spline = power_data[1:len(freq_data[freq_data < f_cut_chi])]
    new_f = freq_data[1:len(freq_data[freq_data < f_cut_chi])]
    std_data = np.array(std_data[1:len(freq_data[freq_data < f_cut_chi])])

    fig, ax = plt.subplots(3, int(len(list_probadir)), figsize=(15, 15))

    for j, (probadir, subdir) in enumerate(zip(list_probadir,list_subdir)):
        #fig, ax = plt.subplots(1,3, figsize=(15, 15))

        print probadir
        print subdir
        os.chdir(probadir)
        list_proba = glob.glob("*.pkl")

        all_v_proba = np.array([])
        all_r0_proba = np.array([])
        all_proba = np.array([])

        for elem in list_proba:
            proba = pkl.load(open(elem, 'rb'))
            all_v_proba = np.append(all_v_proba, list_v)
            all_r0_proba = np.append(all_r0_proba,
                                     int(elem.split('.')[0].split('_')[3].split('R')[1]) * np.ones(len(list_v)))
            all_proba = np.append(all_proba, proba)

        new_v_proba = np.linspace(min(all_v_proba), max(all_v_proba), 1000)
        new_r0_proba = np.linspace(min(all_r0_proba), max(all_r0_proba), 1000)

        triang = tri.Triangulation(all_v_proba, all_r0_proba)
        interpolator_deltam = tri.LinearTriInterpolator(triang, all_proba)

        xx, yy = np.meshgrid(new_v_proba, new_r0_proba)

        zi_deltam = interpolator_deltam(xx, yy)/((new_v_proba[1]-new_v_proba[0])*(new_r0_proba[1]-new_r0_proba[0]))


        os.chdir(resultdir + subdir)
        # list_pkl = glob.glob("spectrum_A3-B2_*_flattop.pkl")
        list_pkl = glob.glob("*.pkl")

        all_l = np.array([])
        all_r0 = np.array([])
        all_v = []
        all_prior = np.array([])

        length = np.array([])
        sampled_v1 = sampled_v1 / np.sum(sampled_v1)
        for i, elem in enumerate(list_pkl):
            #print elem
            mean_power, var_power, freq = pkl.load(open(elem, 'rb'))
            # print lenpower
            # length=np.append(length,lenpower)
            var_power = np.array(var_power[1:])
            mean_power = np.array(mean_power[1:])
            v = int(elem.split('_')[3])
            # print mean_power
            # print var_power
            temp = chi2_custom(mean_power[np.where(frequency_spline <= f_cut_chi)],
                               var_power[np.where(frequency_spline <= f_cut_chi)], new_power_spline, std_data)
            l = likelyhood(temp)
            # print temp
            all_l = np.append(all_l, l)

            all_prior = np.append(all_prior, sampled_v1[np.where(bin_edges == v)])
            all_r0 = np.append(all_r0, int(elem.split('.')[0].split('_')[4].split('R')[1]))
            all_v.append(v)  # .split('v')[1]))

        new_v = np.linspace(min(all_v), 1500, 1000)
        new_r0 = np.linspace(min(all_r0), max(all_r0), 1000)
        all_prior = all_prior
        print all_prior


        triang = tri.Triangulation(all_v, all_r0)
        interpolator = tri.LinearTriInterpolator(triang, all_l)
        interpolator_prior = tri.LinearTriInterpolator(triang, all_prior)
        #interpolator_proba = tri.LinearTriInterpolator(triang, proba)
        xx, yy = np.meshgrid(new_v, new_r0)

        zi = interpolator(xx, yy)/((new_v[1]-new_v[0])*(new_r0[1]-new_r0[0]))
        zi_prior = interpolator_prior(xx, yy)/((new_v[1]-new_v[0])*(new_r0[1]-new_r0[0]))
        #zi_proba = interpolator_proba(xx, yy)/((new_v[1]-new_v[0])*(new_r0[1]-new_r0[0]))
        zi = zi/np.sum(zi)
        #zi_proba = zi_proba/np.sum(zi_proba)
        zi_deltam = zi_deltam/np.sum(zi_deltam)
        # print zi_proba
        # to_fit = np.ravel(zi_proba)
        # print type(zi)
        # print type(zi_prior)
        # import scipy.optimize as opt
        # initial_guess = (0.8, 700, 0.1, 50, 1, 0,0)
        # print "shape :"
        # print xx.shape
        # print to_fit.shape
        # popt, pcov = opt.curve_fit(twoD_Gaussian, (xx, yy), to_fit, p0=initial_guess)
        # data_fitted = twoD_Gaussian((xx, yy), *popt)
        # plt.plot(zi)
        # plt.show()

        # sys.exit()

        alpha = 0.5
        el = Ellipse((653, 3.15), width=358, height=2.5, angle=0, alpha=alpha)
        sc = ax[0][j].contourf(new_v_proba, new_r0_proba / r_ref, zi_deltam*zi_prior/np.sum(zi_deltam*zi_prior), 20, cmap='RdYlGn')  #
        sc2 = ax[1][j].contourf(new_v, new_r0 / r_ref, zi*zi_prior/np.sum(zi*zi_prior) , 20,
                                        cmap='RdYlGn')  # , levels=levels)

        sc3 = ax[2][j].contourf(new_v, new_r0 / r_ref, zi*zi_deltam*zi_prior/np.sum(zi*zi_deltam*zi_prior), 20,
                                   cmap='RdYlGn')  # , levels=levels)
        # ax[j // 2][j % 2].contour(new_v, new_r0 / r_ref, data_fitted.reshape(len(new_v),len(new_r0)), 20, alpha=alpha,
        #                           colors = 'w')  # , levels=levels)
        # ax.contourf(x, y, multivariate_normal.pdf(pos,[485, 4.2], [[375, 0.0], [0.0, 2.5]]), alpha=0.3)
        # ax.plot(reject_v_source, reject_r0/r_ref, 'or', label = r"$\chi^2 > %s $"%(maxchi))
        # ax.fill_between([110, 750], [1.7, 1.7], [6.7, 6.7], alpha = 0.3, label='Morgan et al.(2012)')
        # ax.scatter(better_v, better_r0/r_ref, s = 90,facecolor='None',edgecolors = "c")
        ax[1][j].add_patch(el)
        el.set()

        ax[1][j].axhline(y=1, ls='--')
        ax[1][j].set(xlim=(0, 1500), ylim=(0, max(all_r0) / r_ref))

        el = Ellipse((653, 3.15), width=358, height=2.5, angle=0, alpha=alpha)
        ax[0][j].add_patch(el)
        el.set()

        ax[0][j].axhline(y=1, ls='--')
        ax[0][j].set(xlim=(0, 1500), ylim=(0, max(all_r0) / r_ref))

        el = Ellipse((653, 3.15), width=358, height=2.5, angle=0, alpha=alpha)
        ax[2][j].add_patch(el)
        el.set()

        ax[2][j].axhline(y=1, ls='--')
        ax[2][j].set(xlim=(0, 1500), ylim=(0, max(all_r0) / r_ref))
        #if j // 2 != 2:
        #    ax[j // 2][j % 2].xaxis.set_visible(False)
        # if j%2 !=0 :
        #    ax[j // 2][j % 2].yaxis.set_visible(False)
        # ax[j//2][j%2].legend(loc = 1)
    #if j ==0:
    #    cb=nice_colorbar(sc,ax[j][0], position='right', label=r"$L_{\Delta m}$", fontsize=12, invisible=False, divider_kwargs={}, colorbar_kwargs={}, label_kwargs={})
    #    cb2= nice_colorbar(sc2,ax[j][1], position='right', label=r"$L_{PS}$", fontsize = 12,invisible = False, divider_kwargs = {}, colorbar_kwargs = {}, label_kwargs = {})
    #    cb3=nice_colorbar(sc3,ax[j][2], position='right', label = r"$P_{\Delta m} \cdot P_{\chi^2}$", fontsize = 12,invisible = False, divider_kwargs = {}, colorbar_kwargs = {}, label_kwargs = {})
    #else:
    #    cb = nice_colorbar(sc,ax[j][0], position='bottom', label=r"$L_{\Delta m}$", fontsize=12, invisible=True,
    #                       divider_kwargs={}, colorbar_kwargs={}, label_kwargs={})
    #    cb2 = nice_colorbar(sc2,ax[j][1], position='bottom', label=r"$L_{PS}$", fontsize=12, invisible=True,
    #                        divider_kwargs={}, colorbar_kwargs={}, label_kwargs={})
    #    cb3 = nice_colorbar(sc3,ax[j][2], position='bottom', label=r"$P_{\Delta m} \cdot P_{\chi^2}$", fontsize=12,
    #                        invisible=True, divider_kwargs={}, colorbar_kwargs={}, label_kwargs={})

    cb = fig.colorbar(sc, ax =ax[0][:], label=r"$P_{\Delta m}$")#, shrink = 0.8, orientation = 'vertical')
    cb = fig.colorbar(sc2,ax =ax[1][:], label=r"$P_{PS}$")#, shrink = 0.8, orientation = 'vertical')
    cb = fig.colorbar(sc3,ax =ax[2][:], label=r"$P_{\Delta m} \cdot P_{PS}$")#, shrink = 0.8, orientation = 'vertical')
    ax[0][0].text(600, 5,r'$\left<M\right>$ = 0.3 $M_{\odot}$')
    ax[0][1].text(600, 5, r'$\left<M\right>$ = 0.1 $M_{\odot}$')
    ax[0][2].text(600, 5, r'$\left<M\right>$ = 0.01 $M_{\odot}$')

    plt.show()
    fig.savefig(plotdir+"deltamvsPS.pdf", dpi = 100)



    sys.exit()


if 0:
    #Plot prior

    from matplotlib import patches

    # levels = np.logspace(np.min(np.log10(all_chi)),np.max(np.log10(all_chi)),20)

    orig_cmap = matplotlib.cm.viridis
    shifted_cmap = shiftedColorMap(orig_cmap, midpoint=0.15, name='shifted')
    levels = np.arange(0, 1.05, 0.05)
    # levels = np.logspace(np.min(np.log10(all_chi)),np.max(np.log10(all_chi)),20)
    r_ref = 15
    fig, ax = plt.subplots(1, 1, figsize=(15, 15))

    list_r0 = np.array([2,4,8, 10,12,15,20,35,30,40,60,70,80,90,100])
    list_v = np.arange(100,1600,100)
    all_r0 = []
    all_v = []
    all_prior = np.array([])

    length = np.array([])
    sampled_v1 = sampled_v1 / np.sum(sampled_v1)
    for r0 in list_r0:
        for v in list_v:
            all_prior = np.append(all_prior, sampled_v1[np.where(bin_edges == v)])
            all_r0.append(r0)
            all_v.append(v)

    new_v = np.linspace(min(all_v), max(all_v), 1000)
    new_r0 = np.linspace(min(all_r0), max(all_r0), 1000)

    print len(all_v)
    print len(all_r0)
    print len(all_prior)
    triang = tri.Triangulation(all_v, all_r0)
    interpolator_prior = tri.LinearTriInterpolator(triang, all_prior)
    xx, yy = np.meshgrid(new_v, new_r0)

    zi_prior = interpolator_prior(xx, yy)/((new_v[0]-new_v[1])*(new_r0[0]-new_r0[1]))


    sc = ax.contourf(new_v, new_r0 / r_ref, zi_prior, 10,cmap='viridis')  # , levels=levels)
    ax.set(xlim=(0, 1500), ylim=(0, max(all_r0) / r_ref))

    # if j%2 !=0 :
    #    ax[j // 2][j % 2].yaxis.set_visible(False)
    # ax[j//2][j%2].legend(loc = 1)
    plt.subplots_adjust(hspace=.0, wspace=0.1)
    fig.subplots_adjust(right=0.9)
    #cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
    cb = fig.colorbar(sc, label = 'Probability density')
    # , ticks = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

    ax.set(ylabel=r'$R_0$  in units of $R_{MK11}$')
    ax.set(xlabel=r'$v_e$ [$km\cdot s^{-1}}$]')

    plt.show()
    fig.savefig(plotdir + "prior.pdf",dpi = 100)

    sys.exit()