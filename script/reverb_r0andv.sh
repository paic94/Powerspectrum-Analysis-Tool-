#!/usr/bin/env bash
#!/bin/bash -l
##10 20 30 40 45 55 65 75 85 95

##list_r0=(2 10 15 20 30 40 60 80 100)

## 15 20 30 45 55 65 75 85 95
list_ve=(100 200 300 400 500 600 700 800 900 1000 1100 1200 1300 1400 1500)

##list_ve=(100)
##0.2 0.3 0.4 
##list_fBLR=(0.1 0.2 0.3 0.4)

##list_r0=${list_r0[*]}
list_ve=${list_ve[*]}
##list_fBLR=${list_fBLR[*]}
start_file='reverb_r0andv.run'


for ve in ${list_ve[*]}; 
    do
	##for fBLR in ${list_fBLR}; 
	    ##do 
    		##export fBLR=$fBLR
    	export ve=$ve
    	echo "Launching $start_file with ve = $ve"
    		sbatch $start_file $ve 
    done

     
