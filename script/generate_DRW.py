import pickle as pkl
import sys,os
import numpy as np
import javelin
import matplotlib.pyplot as plt
import matplotlib

font = {'family' : 'normal',
       'size'   : 22}

matplotlib.rc('font', **font)
datadir = "/home/ericpaic/Documents/PHD/data/"

R_lambda = 15 #in pxl
R_max = 1000
einstein_r_03 = 3.41e16
einstein_r_01 = einstein_r_03 / np.sqrt(3)
einstein_r_001 = einstein_r_03 / np.sqrt(30)

cm_per_pxl = (20 * einstein_r_03) / 8192
ld_per_pxl = cm_per_pxl / (30000000000 * 3600 * 24)
print ld_per_pxl
z_QSO = 1.29


def create_lc(time,tf_shape,R0=0,BLRtime=60, sigma=25.4 / 2., tau=79.8, flmean=100., wid=None):
    # Create continuum lightcurve
    PS = javelin.predict.PredictSignal(lcmean=flmean, covfunc='drw', sigma=sigma, tau=tau)
    yzero = np.zeros_like(time)
    y = PS.generate(time, ewant=yzero)  # ewant = errors for the simulated lightcurves
    mytimenew, ynew, y2new = reverberated_lc(time, y, BLRtime, tf_shape,R0,flmean=flmean,
                                    wid=wid)  # flmean is based on what we are using for the lightcurve of J1131 if sigma=0.049

    return [mytimenew, ynew, y2new, time, y]



def reverberated_lc(mytime, y, BLRtime,tf_shape,R0, flmean=100., wid=None):

    # Convolve with the top-hat kernel to get line light curve, however, the input continuum signal has to be dense enough and regularly sampled.
    # wid= width of transfer function, scale = ratio btw cont and line
    if tf_shape=='top-hat':
        if wid == None:
            wid = 2. * BLRtime  # This top-hat transfer function is representative of a thin shell geometry; see Peterson 1993; Pancoast 2011
        lcvBLR = javelin.predict.generateLine(mytime, y, lag=BLRtime, scale=1, wid=int(wid), mc_mean=0, ml_mean=0)
        tcmin, tcmax, tBmax = np.searchsorted(mytime, np.min(lcvBLR[0])), mytime.__len__(), np.searchsorted(lcvBLR[0],
                                                                                                            np.max(mytime))
        y2 = lcvBLR[1][0:tBmax + 1]
        return [mytime[tcmin:tcmax], y[tcmin:tcmax], y2]
    if tf_shape=='james':
        mean = np.mean(y)
        y-= mean
        time_sampling = 10
        integral_interval = 10
        R = np.arange(1, R_max, 1. / (time_sampling * ld_per_pxl))

        time = np.arange(0, len(R) / time_sampling, 1. / time_sampling)
        xi = (R / R0) ** (3. / 4.)
        G_lambda = xi * np.exp(xi) / ((np.exp(xi) - 1) ** 2)
        transfer_function_dt = G_lambda * R / (np.sum(G_lambda * R))
        transfer_function = []
        for i, dt in enumerate(transfer_function_dt):
            transfer_function.append(np.sum(transfer_function_dt[i-integral_interval:i+integral_interval]))

        transfer_function/=np.max(transfer_function)

        #transfer_function = np.ones(len(transfer_function))
        #transfer_function[30:]=np.zeros(len(transfer_function[30:]))

        #x = y
        window_len = int(np.where(transfer_function == np.max(transfer_function))[0])
        #window_len=15
        #s = np.r_[x[window_len - 1:0:-1], x, x[-1:-window_len:-1]]


        reverberated_signal = np.convolve(transfer_function/np.sum(transfer_function),y)

        tcmin, tcmax, tBmax = np.searchsorted(mytime, np.min(mytime)), mytime.__len__(), np.searchsorted(mytime,np.max(mytime))
        reverberated_signal = reverberated_signal[0:tBmax + 1]
        #print "Finished convolving"
        #fig,ax = plt.subplots(1,2,figsize =(20,10))
        #ax[0].plot(time,transfer_function[:-1])
        #ax[0].set(xlabel='Days',ylabel = 'Scaled probability')
        #ax[1].plot(mytime[tcmin:tcmax], y[tcmin:tcmax],label='Continuum')
        #ax[1].plot(mytime[tcmin+window_len:tcmax-window_len],reverberated_signal[window_len:-window_len],label='Reverberated')
        #ax[1].set(xlabel='Days',ylabel = 'Flux')
        #plt.legend()
        #plt.show()
        #sys.exit()


        return mytime[tcmin+window_len:tcmax-window_len],reverberated_signal[window_len:-window_len]+mean,y[tcmin+window_len:tcmax-window_len]+mean



#time_sampling = 1
#R = np.arange(1,R_max,1./(time_sampling*ld_per_pxl))
#
#time = np.arange(0,len(R)/time_sampling,1./time_sampling)
#xi = (R/R_lambda)**(3./4.)
#G_lambda = xi*np.exp(xi)/((np.exp(xi)-1)**2)
#transfer_function_dt = G_lambda*R/(np.sum(G_lambda*R))
#transfer_function=[]
#for i,dt in enumerate(transfer_function_dt):
#    transfer_function.append(np.sum(transfer_function_dt[i-3:i+3]))
#plt.plot(time,transfer_function[:])
#plt.plot()
#plt.show()



sampling = 1
new_mjhd = np.arange(53601, 58147, sampling)
contracted_mjhd = new_mjhd/(1+z_QSO)
#new_mjhd = np.arange(1,365*20,sampling)
#new_err_mag_ml = np.random.normal(0.008653884816753927, 5.583092113856527e-05, len(new_mjhd))

#sigma generated 6, 13, 26 , 40,50,60,70,80,90,100
#list_sigma = np.arange(1,30,2.9)
list_sigma=[9,15,20, 30 , 45,55,65,75,85,95]
#list_BLRtime = np.arange(10,100,10)
#blrt generated 2,15,35,5,25,50,55,65,75,85,100,95,2,15,35,5,25,50,55,65,75,85,100
#5, 10, 20, 30, 40, 45, 55, 65, 75, 85, 95,
list_timelag = [5, 10, 20, 30, 40, 45, 55, 65, 75, 85, 95,100,110,120,130,140,150]
#list_width = [1,10,20,30,40]
list_width = [None]
list_r0 = [15,60]
n_real=1000
for sigma in list_sigma:
    print "sigma :%s"%(sigma)
    for timelag in list_timelag:
        print timelag
    #for r0 in list_r0:
        for i in range(n_real):
            if not os.path.isfile(datadir + 'DRW_contracted/DRW_sigma%s_tau810_BLRt%s_tfshapetophat_%i.pkl' % ( str(sigma).replace('.',','),timelag,i)):
                DRW = create_lc(time=contracted_mjhd,BLRtime=timelag, tf_shape= 'top-hat', sigma=sigma, tau = 810, flmean=100., wid=None)
                new_time = DRW[0]
                lc_c = DRW[1]
                lc_BLR = DRW[2]
                if i%100==0:
                    print "realisation %s of %s"%(i,n_real)

                #print "==================================="
                #print lc_BLR
                #plt.plot(new_time,lc_c)
                #plt.plot(new_time,lc_BLR)
                #plt.show()
                #lc_DRW = (lc_c + fBLR*lc_BLR)/(1+fBLR)

                #with open(datadir + 'DRW/DRW_t%s_sigma%s_tau810_tfshapetophat_%i.pkl' % (timelag, str(sigma).replace('.',','),i), 'wb') as handle:
                #    pkl.dump((new_time, lc_c, lc_BLR), handle, protocol=pkl.HIGHEST_PROTOCOL)
                with open(datadir + 'DRW_contracted/DRW_sigma%s_tau810_BLRt%s_tfshapetophat_%i.pkl' % ( str(sigma).replace('.',','),timelag,i), 'wb') as handle:
                    pkl.dump((new_time, lc_c, lc_BLR), handle, protocol=pkl.HIGHEST_PROTOCOL)

