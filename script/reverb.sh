#!/usr/bin/env bash
#!/bin/bash -l
##10 20 30 40 45 55 65 75 85 95

##list_r0=(2 10 15 20 30 40 60 80 100)

## 15 20 30 45 55 65 75 85 95
list_BLRt =(100 110 120 130 140 150)

list_sigma =(9 15 20 30 45 55 65 75 85 95)

##list_ve=(100)
##0.2 0.3 0.4 
##list_fBLR=(0.1 0.2 0.3 0.4)

##list_r0=${list_r0[*]}
list_BLRt=${list_BLRt[*]}
list_sigma=${list_sigma[*]}
start_file='reverb.run'


for tBLR in ${list_tBLR[*]};
    do
    for sigma in ${list_sigma[*]};
	##for fBLR in ${list_fBLR}; 
	    ##do 
    		##export fBLR=$fBLR
    	export tBLR=$tBLR
    	export sigma = $sigma
    	echo "Launching $start_file with tBLR = $tBLR and sigma = $sigma"
    		sbatch $start_file $tBLR $sigma
    done

     
