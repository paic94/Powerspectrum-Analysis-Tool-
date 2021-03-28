from __future__ import division
import numpy as np
import os, sys
import timeit
import multiprocessing
from functools import partial
import pickle as pkl
from astropy.io import fits
import scipy.signal
import glob
import pycs
from scipy.signal import periodogram
import argparse as ap
import psutil
import matplotlib

#execfile('diflcv_4eric.py')

datadir = "/home/astro/paic/BLRstudy/data/"
scriptdir = "/home/astro/paic/BLRstudy/script/"
resultdir = "/home/astro/paic/BLRstudy/results/"
pkldir = resultdir+"FML0.9/M0.3/withBLR_extended/"
mapdir = datadir+"maps/with_macromag/FML09M03/"
#pkldir = resultdir+"FML1/M001/"
#mapdir = datadir+"maps/with_macromag/FML1M03/"

drwdir = datadir +"DRW/regenerate_DRW/"

print "Starting !"
print mapdir

einstein_r_1131 = 2.5e16  # cm
einstein_r_03 = 3.41e16
einstein_r_01 = einstein_r_03 / np.sqrt(3)
einstein_r_001 = einstein_r_03 / np.sqrt(30)

cm_per_pxl = (20 * einstein_r_03) / 8192
ld_per_pxl = cm_per_pxl / (30000000000 * 3600 * 24)


# day_per_pxl = cm_per_pxl/(100000*v_source*3600*24)

A = np.array([0.0,0.0])
B = np.array([-1.156, -0.398])
G = np.array([-0.780,-0.234])

A -= G
B -= G
G -= G

gamma_A = np.arctan(A[1]/A[0])
gamma_B = np.arctan(B[1]/B[0])



def good_draw_LC(params, map, time, err_data):
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
        #temp = np.add(np.multiply(-2.5, np.log10(map[path_y, path_x])),
        #              np.random.normal(0, np.mean(err_data), len(path_y)))  # -2.5 log() to convert flux into mag
        #lc = temp - temp[0] * np.ones(len(temp))
        return lc


def PS_compare_BLR(params, map_A, map_B,lc_c,lc_BLR, time, err_data, f,Mc=1.,fBLR=0.8,muBLR=1.):

    
    f_cut=0.1
    ml_A = good_draw_LC(params[0], map_A, time, err_data)
    ml_B = good_draw_LC(params[1], map_B, time, err_data)

    Mc_A = 2.24
    Mc_B = 0.84
    #lc_c -= np.mean(lc_c)
    #lc_BLR -= np.mean(lc_BLR)

    #print "fBLR : %s"%(fBLR)

    if ml_A is not None and ml_B is not None:
        ml_A = np.abs(ml_A)
        ml_B = np.abs(ml_B)
        #print Mc * fBLR * muBLR * lc_BLR

        lc_A = -2.5 * np.log10(Mc * ml_A * lc_c + Mc * fBLR * muBLR * lc_BLR)
        lc_B = -2.5 * np.log10(Mc * ml_B * lc_c + Mc * fBLR * muBLR * lc_BLR)
        lc_reverberation = lc_A - lc_B
        lc_reverberation -= lc_reverberation[0]
        lc = -2.5*np.log10(ml_A*Mc_B/(ml_B*Mc_A) )
        lc-=lc[0]

        #new_lcBLR = pycs.gen.lc.factory(time, lc_BLR, magerrs=err_data)
        #spline = pycs.gen.spl.fit([new_lcBLR], knotstep=1600, bokeps=20, verbose=False)
        #spline_BLR = spline.eval(time)

        frequency, power = periodogram(lc, f, window="flattop")
        frequency_BLR, power_BLR = periodogram(lc_reverberation, f, window = "flattop")
        #frequency_BLR_short, power_BLR_short = periodogram(lc_BLR-spline_BLR, f, window="flattop")
        #frequency_BLR_long, power_BLR_long = periodogram(spline_BLR, f, window="flattop")
        frequency = np.array(frequency)
        power=np.array(power)
        frequency_BLR = np.array(frequency_BLR)

       
        return frequency[frequency<f_cut], power[frequency<f_cut], frequency_BLR[frequency_BLR<f_cut], power_BLR[frequency_BLR<f_cut]
	
def PS_BLR(params, map_A, map_B, err_data, f,Mc,muBLR=1.):
    global lc_c_list
    global lc_BLR_list
    global new_time
    
    params_A = params[0]
    params_B = params[1]
   


    whichcurve = int(np.random.uniform(0,len(lc_c_list)))
    lc_c = lc_c_list[whichcurve]
    lc_BLR = lc_BLR_list[whichcurve]
    
    f_cut=0.1
    ml_A = good_draw_LC(params_A, map_A, new_time, err_data)
    ml_B = good_draw_LC(params_B, map_B, new_time, err_data)
    fBLR = params_A[-1]
    	
    Mc_A = 2.24
    Mc_B = 0.84
    #lc_c -= np.min(lc_c)
    #lc_BLR -= np.min(lc_BLR)
    #lc_c +=1
    #lc_BLR +=1
    

    #print "fBLR : %s"%(fBLR)
    #print ml_A
    #print lc_c 
    #print lc_BLR

    if ml_A is not None and ml_B is not None:
        ml_A = np.abs(ml_A)
        ml_B = np.abs(ml_B)
        #print Mc * fBLR * muBLR * lc_BLR
	#if (Mc * ml_B * lc_c + Mc * fBLR * muBLR * lc_BLR).any() < 0 :
	#    print ml_B
	#    print lc_c
	#    print lc_BLR 
	
        lc_A = -2.5 * np.log10(Mc * ml_A * lc_c + Mc * fBLR * muBLR * lc_BLR)
        lc_B = -2.5 * np.log10(Mc * ml_B * lc_c + Mc * fBLR * muBLR * lc_BLR)
        lc_reverberation = lc_A - lc_B
        #lc_reverberation -= lc_reverberation[0]
        
	#print 
        frequency_BLR, power_BLR = periodogram(lc_reverberation, f, window = "flattop")
        frequency_BLR = np.array(frequency_BLR)

        #if not np.isnan(power_BLR.any()):
	    #print  power_BLR[frequency_BLR<f_cut]power_BLR[frequency_BLR<f_cut]
        return frequency_BLR[frequency_BLR<f_cut], power_BLR[frequency_BLR<f_cut]

def PS_noBLR(params, map_A, map_B, err_data, f,Mc,muBLR=1.):
    global new_time
    
    params_A = params[0]
    params_B = params[1]
    
    f_cut=0.1
    ml_A = good_draw_LC(params_A, map_A, new_time, err_data)
    ml_B = good_draw_LC(params_B, map_B, new_time, err_data)
    	
    Mc_A = 2.24
    Mc_B = 0.84
    
    if ml_A is not None and ml_B is not None:
        ml_A = np.abs(ml_A)
        ml_B = np.abs(ml_B)
    
        
	lc = -2.5*np.log10(ml_A*Mc_B/(ml_B*Mc_A) )
        #lc_reverberation -= lc_reverberation[0]
        
	#print 
        frequency, power = periodogram(lc, f, window = "flattop")
        frequency = np.array(frequency)

        #if not np.isnan(power_BLR.any()):
	    #print  power_BLR[frequency_BLR<f_cut]power_BLR[frequency_BLR<f_cut]
        return frequency[frequency<f_cut], power[frequency<f_cut]

def PS_compare_BLR_regenerateDRW(params, map_A, map_B, err_data, f,Mc,fBLR,muBLR,tau,sigma,len_time,timelag,width):

    params_A = params[:4]
    params_B = params[4:]
    print params_A
    print params_B
    sigma = float(sigma.replace(',', '.'))


    DRW = crealcv(ltype =4,lngth=len_time, BLRtime=timelag, sigma=sigma, tau=tau, flmean=100., wid=width, sfx='', mdir='')
    new_time = DRW[0]
    lc_c = DRW[1]
    lc_BLR = DRW[2]
    f_cut=0.1
    #plt.plot(new_time,lc_c)
    #plt.plot(new_time, lc_BLR)
    #plt.show()
    ml_A = good_draw_LC(params_A, map_A, new_time, err_data)
    ml_B = good_draw_LC(params_B, map_B, new_time, err_data)
    
    print fBLR
    Mc_A = 2.24
    Mc_B = 0.84
    #lc_c -= np.mean(lc_c)
    #lc_BLR -= np.mean(lc_BLR)

    #print "fBLR : %s"%(fBLR)

    if ml_A is not None and ml_B is not None:
        ml_A = np.abs(ml_A)
        ml_B = np.abs(ml_B)
        #print Mc * fBLR * muBLR * lc_BLR
	print 
        lc_A = -2.5 * np.log10(Mc * ml_A * lc_c + Mc * fBLR * muBLR * lc_BLR)
        lc_B = -2.5 * np.log10(Mc * ml_B * lc_c + Mc * fBLR * muBLR * lc_BLR)
        lc_reverberation = lc_A - lc_B
        lc_reverberation -= lc_reverberation[0]
        lc = -2.5*np.log10(ml_A*Mc_B/(ml_B*Mc_A) )
        lc-=lc[0]

        #new_lcBLR = pycs.gen.lc.factory(time, lc_BLR, magerrs=err_data)
        #spline = pycs.gen.spl.fit([new_lcBLR], knotstep=1600, bokeps=20, verbose=False)
        #spline_BLR = spline.eval(time)

        frequency, power = periodogram(lc, f, window="flattop")
        frequency_BLR, power_BLR = periodogram(lc_reverberation, f, window = "flattop")
        #frequency_BLR_short, power_BLR_short = periodogram(lc_BLR-spline_BLR, f, window="flattop")
        #frequency_BLR_long, power_BLR_long = periodogram(spline_BLR, f, window="flattop")
        frequency = np.array(frequency)
        power=np.array(power)
        frequency_BLR = np.array(frequency_BLR)

        
        return frequency[frequency<f_cut], power[frequency<f_cut], frequency_BLR[frequency_BLR<f_cut], power_BLR[frequency_BLR<f_cut]



sampling = 1
new_mjhd = np.arange(53601, 58147, sampling)
new_time = new_mjhd-new_mjhd[0]
new_err_mag_ml = np.random.normal(0.008653884816753927, 5.583092113856527e-05, len(new_mjhd))
window = "flattop"
detrend = "constant"


tf_shape = 'tophat'

parser = ap.ArgumentParser(description='Computing of the power spectrum')
#parser.add_argument('R0', metavar='R0', type=int,help='Size of accretion disk')
parser.add_argument('ve', metavar='ve', type=int,help='effective velocity')
#parser.add_argument('fBLR', metavar='fBLR', type=float,help='Fraction of flux reverberated by the BLR')




#compare PS with and without BLR
sigma = 65
BLRtime = 65
#list_fBLR =[0.1,0.2,0.3,0.4]



n_spectrum = 50000
start = timeit.default_timer()


n_cpu = multiprocessing.cpu_count()
print(n_cpu)

#list_v = np.arange(100,1600,100)
# list_comb = [('A2', 'B3'), ('A4', 'B4'), ('A3', 'B2'), ('A5', 'B5'), ('A', 'B2'), ('A2', 'B4'), ('A3', 'B5'),('A', 'B'),('A5','B') ]
# list_comb = [('A3', 'B2'), ('A3', 'B3'), ('A4', 'B2'), ('A4', 'B3'), ('A5', 'B2'), ('A5', 'B3'), ('A6', 'B2'),('A6', 'B3'),('A7', 'B3'),('A8', 'B2'),('A8', 'B3')]
list_comb = [('A', 'B')]
list_r0 = [2,10,20,30,40,60,80,100]


args = parser.parse_args()
#r0 = args.R0
v_source = args.ve
#fBLR = args.fBLR

print "BLRt : %s, sigma : %s "%(BLRtime, sigma)
print v_source 
f_cut = 0.01

v_distrib = np.fromfile(datadir+'velocities_QJ0158.dat', sep = " ")
v1 = v_distrib[0::8]
v2 = v_distrib[2::8]
v3 = v_distrib[4::8]
v4 = v_distrib[6::8]

angle1 = v_distrib[1::8]
angle2 = v_distrib[3::8]
angle3 = v_distrib[5::8]
angle4 = v_distrib[7::8]

if 0:
	print "Reverberation included"
	fBLR_list = np.random.normal(0.432, 0.036,n_spectrum)

	#fBLR_list = 0*np.ones(n_spectrum)    

	list_pklfile = glob.glob(drwdir+'DRW_sigma%s_tau810_BLRt%s_tfshapetophat_*.pkl' % (sigma,BLRtime))
	print len(list_pklfile)

	lc_c_list = []
	lc_BLR_list = []

	for pklfile in list_pklfile:
    		new_time, lc_c, lc_BLR = pkl.load(open(pklfile, 'rb'))
    		if (lc_c>0).all() and (lc_BLR>0).all():
    			lc_c_list.append(lc_c)
    			lc_BLR_list.append(lc_BLR)
	
	print len(lc_BLR_list)

	for comb in list_comb:
    		for r0 in list_r0:
        		print comb
        		print r0
        		final_map_A = []
        		map_A = mapdir + "convolved_map_%s_fft_thin_disk_%s.fits" % (comb[0], r0)
        		img_A = fits.open(map_A, memmap=True)[0]
        		final_map_A = img_A.data[:,:]
	
        		final_map_B = []
        		map_B = mapdir + "convolved_map_%s_fft_thin_disk_%s.fits" % (comb[1], r0)
        		img_B = fits.open(map_B, memmap=True)[0]
        		final_map_B = img_B.data[:,:]


        		if not os.path.isfile(pkldir + 'spectrum_%s-%s_%s_%s_R%s_thin-disk_2maps-withmacroM_BLRt%s_s%s_tau810_noshiftDRW_fBLRdistrib.pkl' % (comb[0], comb[1], n_spectrum, v_source, r0,BLRtime,str(sigma).replace('.',','))):
     #   if not os.path.isfile(pkldir + 'spectrum_%s-%s_%s_%s_R%s_thin-disk_2maps-withmacroM.pkl' % (comb[0], comb[1], n_spectrum, v_source, r0)):
    	    			print "11111111111111111111111"
    	    			print v_source

    	    			res = []
    	    			v = v_source * np.ones(n_spectrum)
    	    			x_A = np.random.random_integers(0, len(final_map_A) - 1, n_spectrum)
    	    			y_A = np.random.random_integers(0, len(final_map_A) - 1, n_spectrum)
    	    
    	    			angle_A = (angle1[:n_spectrum]+angle2[:n_spectrum]+angle3[:n_spectrum]+angle4[:n_spectrum])/(2*np.pi)


    	    			x_B = np.random.random_integers(0, len(final_map_B) - 1, n_spectrum)
    	    			y_B = np.random.random_integers(0, len(final_map_B) - 1, n_spectrum)
    	    			angle_B = angle_A - (gamma_B-gamma_A)



    	    			params_A = np.stack((x_A, y_A, v, angle_A, fBLR_list),axis = -1)

    	    			params_B = np.stack((x_B, y_B, v, angle_B, fBLR_list),axis = -1)
    	    			params = np.stack((params_A, params_B), axis = 1)
    
    
    
    	    			parallel_compare_PS_BLR = partial(PS_BLR, map_A=final_map_A,map_B=final_map_B, err_data=None,f=1 / sampling,Mc=1.,muBLR=1.)

    	    			res = map(parallel_compare_PS_BLR, params)      
    	 
    	
    	    			res = filter(None, res)
    	    			res = np.array(res)
    	    			print res.shape

    #freq = res[:, 0]
    #power = res[:, 1]

    	    			freq_BLR = res[:, 0]
    	    			power_BLR = res[:, 1]
    	    #power_BLR = power_BLR[~np.isnan(power_BLR)]
    	    			print power_BLR.shape
    	    			res = []
    #freq_BLR = np.array(freq_BLR[0,1:])

    #freq = np.array(freq[0,1:])

    #mean_power = []
    #var_power = []
    #power = np.array(power)
    #print "111111111111111111"
    	    			mean_power_BLR = []
    	    			var_power_BLR = []
    	    
    	    
    	    			for i in range(len(power_BLR[0])):
    					mean_power_BLR.append(np.mean(power_BLR[:, i]))
    					var_power_BLR.append(np.var(power_BLR[:, i]))

    	    			
    #for i in range(len(power_BLR[0])):
    #	 mean_power_BLR.append(np.mean(power_BLR[:, i]))
    #	 var_power_BLR.append(np.var(power_BLR[:, i]))
    #	 mean_power.append(np.mean(power[:, i]))
    #	 var_power.append(np.var(power[:, i]))

    	    			print "Finished, here are some results :"

    	    			print mean_power_BLR
    #mean_power = np.array(mean_power[1:])
    	    			mean_power_BLR = np.array(mean_power_BLR[1:])
    #var_power = np.array(var_power[1:])
    	    			var_power_BLR = np.array(var_power_BLR[1:])

    	    			stop = timeit.default_timer()
    	    			print(stop - start)
    #print len(mean_power)
    #print len(var_power)
    	    			print len(freq_BLR)
    	    
	    			print pkldir + 'spectrum_%s-%s_%s_%s_R%s_thin-disk_2maps-withmacroM_BLRt%s_s%s_tau810_noshiftDRW_fBLRdistrib.pkl' % (comb[0], comb[1], n_spectrum, v_source, r0,BLRtime,str(sigma).replace('.',','))
    	    			with open(pkldir + 'spectrum_%s-%s_%s_%s_R%s_thin-disk_2maps-withmacroM_BLRt%s_s%s_tau810_noshiftDRW_fBLRdistrib.pkl' % (comb[0], comb[1], n_spectrum, v_source, r0,BLRtime,str(sigma).replace('.',',')), 'wb') as handle:
    					pkl.dump((mean_power_BLR, var_power_BLR, freq_BLR[0]), handle, protocol=pkl.HIGHEST_PROTOCOL)
        			mean_power_BLR = []
    	    			var_power_BLR = []
				freq_BLR = []
    	    			power_BLR = []
			else:
    	    
    	    			print "Already did this one : %s"%(v_source)

	print "Analysis completed !"
	
if 1:
	print "Reverberation not included"
	for comb in list_comb:
    		for r0 in list_r0:
        		print comb
        		print r0
        		final_map_A = []
        		map_A = mapdir + "convolved_map_%s_fft_thin_disk_%s.fits" % (comb[0], r0)
        		img_A = fits.open(map_A, memmap=True)[0]
        		final_map_A = img_A.data[:,:]
	
        		final_map_B = []
        		map_B = mapdir + "convolved_map_%s_fft_thin_disk_%s.fits" % (comb[1], r0)
        		img_B = fits.open(map_B, memmap=True)[0]
        		final_map_B = img_B.data[:,:]


        		if not os.path.isfile(pkldir + 'spectrum_%s-%s_%s_%s_R%s_thin-disk_2maps-withmacroM.pkl' % (comb[0], comb[1], n_spectrum, v_source, r0)):
    	    			print "11111111111111111111111"
    	    			print v_source

    	    			res = []
    	    			v = v_source * np.ones(n_spectrum)
    	    			x_A = np.random.random_integers(0, len(final_map_A) - 1, n_spectrum)
    	    			y_A = np.random.random_integers(0, len(final_map_A) - 1, n_spectrum)
    	    
    	    			angle_A = (angle1[:n_spectrum]+angle2[:n_spectrum]+angle3[:n_spectrum]+angle4[:n_spectrum])/(2*np.pi)


    	    			x_B = np.random.random_integers(0, len(final_map_B) - 1, n_spectrum)
    	    			y_B = np.random.random_integers(0, len(final_map_B) - 1, n_spectrum)
    	    			angle_B = angle_A - (gamma_B-gamma_A)



    	    			params_A = np.stack((x_A, y_A, v, angle_A),axis = -1)

    	    			params_B = np.stack((x_B, y_B, v, angle_B),axis = -1)
    	    			params = np.stack((params_A, params_B), axis = 1)
    
    
    
    	    			parallel_PSnoBLR = partial(PS_noBLR, map_A=final_map_A,map_B=final_map_B, err_data=None,f=1 / sampling,Mc=1.,muBLR=1.)

    	    			res = map(parallel_PSnoBLR, params)      
    	 
    	
    	    			res = filter(None, res)
    	    			res = np.array(res)
    	    			print res.shape

    #freq = res[:, 0]
    #power = res[:, 1]

    	    			freq_BLR = res[:, 0]
    	    			power_BLR = res[:, 1]
    	    #power_BLR = power_BLR[~np.isnan(power_BLR)]
    	    			print power_BLR.shape
    	    			res = []
    #freq_BLR = np.array(freq_BLR[0,1:])

    #freq = np.array(freq[0,1:])

    #mean_power = []
    #var_power = []
    #power = np.array(power)
    #print "111111111111111111"
    	    			mean_power_BLR = []
    	    			var_power_BLR = []
    	    
    	    
    	    			for i in range(len(power_BLR[0])):
    					mean_power_BLR.append(np.mean(power_BLR[:, i]))
    					var_power_BLR.append(np.var(power_BLR[:, i]))

    	    			
    #for i in range(len(power_BLR[0])):
    #	 mean_power_BLR.append(np.mean(power_BLR[:, i]))
    #	 var_power_BLR.append(np.var(power_BLR[:, i]))
    #	 mean_power.append(np.mean(power[:, i]))
    #	 var_power.append(np.var(power[:, i]))

    	    			print "Finished, here are some results :"

    	    			print mean_power_BLR
    #mean_power = np.array(mean_power[1:])
    	    			mean_power_BLR = np.array(mean_power_BLR[1:])
    #var_power = np.array(var_power[1:])
    	    			var_power_BLR = np.array(var_power_BLR[1:])

    	    			stop = timeit.default_timer()
    	    			print(stop - start)
    #print len(mean_power)
    #print len(var_power)
    	    			print len(freq_BLR)
    	    
	    			
	    
    	    			print pkldir + 'spectrum_%s-%s_%s_%s_R%s_thin-disk_2maps-withmacroM.pkl' % (comb[0], comb[1], n_spectrum, v_source, r0)
    	    			with open(pkldir + 'spectrum_%s-%s_%s_%s_R%s_thin-disk_2maps-withmacroM.pkl' % (comb[0], comb[1], n_spectrum, v_source, r0), 'wb') as handle:
    					pkl.dump((mean_power_BLR, var_power_BLR, freq_BLR[0]), handle, protocol=pkl.HIGHEST_PROTOCOL)
        		else:
    	    
    	    			print "Already did this one : %s"%(v_source)

	print "Analysis completed !"
		    

            
