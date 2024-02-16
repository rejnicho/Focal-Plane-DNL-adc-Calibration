## This file will create a focal plane view of results for the DNL calibration of the ADCs 

# Note: you must update the file directories in getRaftsData,  getRaftsDataDifference,  and variablesPlottable to use this 

# This file has 3 runable functions: variablesPlottable, plotFocalPlane, plotFocalPlaneDiffernece 

#variablesPlottable() 
#plotFocalPlane(variabletoplotasstring, labelforcolorbar, mincolorbar, maxcolorbar)
#plotFocalPlaneDifference("plot name", "colorbar label", paramtoplot, mincolorbar, maxcolorbar)

from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import pickle as pkl
import scipy.stats as stats
import numpy as np
import scipy.interpolate
import csv  
import pandas
import os 
import os.path
import glob

ampsplotted = 0 
ampsplotted2 = 0

def combineAmpsData(datadict): 
    # the top row is 3*8 = 24 units across S20, S21, S22 
    combineddata = [] # the proper order of the values 
    data07 = [] # amplifiers 00-07 
    data1017 = [] # amplifiers 10-17 
    
    for sensor in datadict: 
        ampdata = datadict[sensor] 
        data07.extend(ampdata[:8]) #first 8 amps are the 00-07 
        data1017.extend(ampdata[8:]) #second 8 amps are 10-17 

    ## now we want to have the order be correct for the full, so 3*8* = 24 elements per row (that is 16 amps per sensor, and 9 sensors per raft), total 144 elements
    combineddata.extend(data07[:24])
    combineddata.extend(data1017[:24])    
    combineddata.extend(data07[24:48])
    combineddata.extend(data1017[24:48])
    combineddata.extend(data07[48:])
    combineddata.extend(data1017[48:])
    ## reshape the data so that it is an array that is 8*3 = 24 x 6 
    reshapeddata = np.array(combineddata).reshape(6, 24)

    return reshapeddata
    
def getRaftData(raft, varplot): #generalizes so you can plot any value you desire for the entire focal plane 
    global ampsplotted

    #pull the data from the correct CSV file 
    csvfile = f"/home/r/rejnicho/adc_investigation/adcbinning/focalplane/raftcsvfiles/{raft}allsensors.csv"
    
    if os.path.isfile(csvfile):
        df = pandas.read_csv(csvfile)
    
        # we want the sensors to be in the order they are typically viewed in 
        sensordorder = ["S20", "S21", "S22", 
                        "S10", "S11", "S12", 
                        "S00", "S01", "S02"]
        indexdict = [[0,0], [0,1], [0,2], 
                     [1,0], [1,1], [1,2], 
                     [2,0], [2,1], [2,2]]

        # amp order we want to realize that the order of the amp matters 
        amporder = ["C00", "C01", "C02", "C03", "C04", "C05", "C06", "C07",
                   "C10", "C11", "C12", "C13", "C14", "C15", "C16", "C17"]

        # what you truly want is 0-7 for S20, 0-7 S21, 0-7 S22 then 10-17 S20, 10-17 S21, ... 
        # so make 2 data subparts, then combine them together 
        
        datadict = {} #dictionary of the values to be plotted, in order of of amplifiers 
        for x in sensordorder: # iterates sensors in the order in which this will be plotted 
            data = [] # values of the desired value 0-7             
            
            for ampname in amporder: # iterates in the way to be plotted 
                fullname = raft+"_"+x+"_"+ampname 
                
                if fullname in df['Amplifier'].values: # determine whether it is in the df 
                    location = df[df['Amplifier'] == fullname].index                    
                    data.append(df[varplot][location[0]])
                    ampsplotted += 1 

                elif fullname not in df['Amplifier'].values: # if the amplifier not in csv leave it blank 
                    data.append(np.nan) 
                    
            datadict[x] = data 

        # order the datadict such that you get an array in the desired order 
        dataset = combineAmpsData(datadict)
    else: # if the file does not exist, skip the entire raft 
        nandataset = [np.nan] * (16*9)
        dataset = np.array(nandataset).reshape(6, 24)
        
    return dataset 

def plotFocalPlane(varplot, labelword, low, high): #plots a focal plane, and saves it 
    # create subplots which are all the rafts on the focal plane 
    fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(10, 10))
    
    # Loop through subplots and plot blocks of color according to the way the focal plane is viewed 
    raftorder = ["R41", "R42", "R43", 
                "R30", "R31", "R32", "R33", "R34",
                "R20", "R21", "R22", "R23", "R24",
                "R10", "R11", "R12", "R13", "R14",
                "R01", "R02", "R03"]
    a= 0 # tracks which Raft is being plotted so it can be labeled 
    
    for i in range(5): 
        for j in range(5):
            skip = [(0, 4), (0,0), (4,4), (4,0)] # skip the corner rafts 
            if (i, j) in skip: # leave the corner rafts blank 
                ax = axes[i, j]
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                continue
            
            else: # All other rafts we want to pull the data in proper order 
                ax = axes[i, j] 
                raft = raftorder[a] # yields the name of the raft to be plotted 
                data = getRaftData(raft, varplot) # get the requested information in the proper order 
                im = ax.imshow(data, cmap='viridis', interpolation='none', aspect='auto', vmin=low, vmax=high)
                ax.set_title(raft)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_xticklabels([])
                ax.set_yticklabels([])
            a+=1 

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7]) 
    fig.colorbar(im, cax=cbar_ax, label=f'{labelword}')
    fig.suptitle(f'{varplot}', fontsize=16)
    
    print("the number of amps plotted is", ampsplotted, "out of 3024 total amps")
    plt.savefig(f'{varplot}.png')

    
def getRaftDataDifference(raft, param): #generalizes so you can plot any value you desire for the entire focal plane 
    global ampsplotted2

    csvfile = f"/home/r/rejnicho/adc_investigation/adcbinning/focalplane/raftcsvfiles/{raft}allsensors.csv"
    if os.path.isfile(csvfile):
        df = pandas.read_csv(csvfile)
    
        ## we want the sensors to be in the order they are typically viewed in 
        sensordorder = ["S20", "S21", "S22", 
                        "S10", "S11", "S12", 
                        "S00", "S01", "S02"]
        indexdict = [[0,0], [0,1], [0,2], 
                     [1,0], [1,1], [1,2], 
                     [2,0], [2,1], [2,2]]

        ## amp order we want to realize that the order of the amp matters 
        amporder = ["C00", "C01", "C02", "C03", "C04", "C05", "C06", "C07",
                   "C10", "C11", "C12", "C13", "C14", "C15", "C16", "C17"]

        # what you truly want is 0-7 for S20, 0-7 S21, 0-7 S22 then 10-17 S20, 10-17 S21, ... 
        # so make 2 data subparts, then combine them together 
        
        datadict = {} # dictionary of the values to be plotted, in order of of amplifiers 
        for x in sensordorder: # iterates in the order in which this will be plotted 
            data = [] # values of the desired value 0-7             
            
            for ampname in amporder: # iterates in the way to be plotted 
                fullname = raft+"_"+x+"_"+ampname 
                #print(ampname)
                
                if fullname in df['Amplifier'].values: # determine wheter it is in the df 
                    location = df[df['Amplifier'] == fullname].index 

                    value13144 = df[f"13144 {param}"][location[0]]
                    value13549 = df[f"13549 {param}"][location[0]]
                    difference = value13549 - value13144
                    
                    data.append(difference)
                    ampsplotted2 += 1 
                    
                elif fullname not in df['Amplifier'].values: # if the amplifier not in csv leave it blank
                    data.append(np.nan) 
                    
            datadict[x] = data 
        
        # order the datadict such that you get an array in the desired order 
        dataset = combineAmpsData(datadict)
    else: # if the file does not exist, skip the entire raft
        nandataset = [np.nan] * (16*9)
        dataset = np.array(nandataset).reshape(6, 24)
        
    return dataset 

def plotFocalPlaneDifference(titlename, labelword, param, low, high):
    # create subplots which are all the rafts on the focal plane 
    fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(10, 10))
    # Loop through subplots and plot blocks of color according to the way the focal plane is viewed 
    
    raftorder = ["R41", "R42", "R43",
                "R30", "R31", "R32", "R33", "R34",
                "R20", "R21", "R22", "R23", "R24",
                "R10", "R11", "R12", "R13", "R14",
                "R01", "R02", "R03"]
    a= 0 # tracks which Raft is being plotted so it can be labeled 
    for i in range(5):
        for j in range(5):
            skip = [(0, 4), (0,0), (4,4), (4,0)] # skip the corner rafts 
            if (i, j) in skip: # leave the corner rafts blank 
                ax = axes[i, j]
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                continue
            else: # All other rafts we want to pull the data in proper order 
                ax = axes[i, j]
                raft = raftorder[a] # yields the name of the raft to be plotted 
                data = getRaftDataDifference(raft, param) # get the requested information in the proper order 
                im = ax.imshow(data, cmap='viridis', interpolation='none', aspect='auto', vmin=low, vmax=high)
                ax.set_title(raft)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_xticklabels([])
                ax.set_yticklabels([])
            a+=1 

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    fig.colorbar(im, cax=cbar_ax, label=f'{labelword}')
    fig.suptitle(f'{titlename}', fontsize=16)
    plt.savefig(f'{param} difference.png')
    plt.show()
    print("the number of amps plotted is", ampsplotted2, "out of 3024 total amps")

def variablesPlottable(): #returns all the names of the variables you can plot, based generically on R01 
    csvfile = f"/home/r/rejnicho/adc_investigation/adcbinning/focalplane/raftcsvfiles/R01allsensors.csv" 
    with open(csvfile, 'r') as file:
        csv_reader = csv.reader(file)
        # Assuming the first row contains column names
        column_names = next(csv_reader)
        print(column_names)


#variablesPlottable() 
#plotFocalPlane(variabletoplotasstring, labelforcolorbar, mincolorbar, maxcolorbar)
#plotFocalPlaneDifference("plot name", "colorbar label", paramtoplot, mincolorbar, maxcolorbar)
