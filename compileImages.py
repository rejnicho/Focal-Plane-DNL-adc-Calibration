# This file will run the Butler and pull and compile the relevant images 

# Runable functions: startRun 

# Note you need to change the directory in: runAll13549, runAll13144

from lsst.daf.butler import Butler
from scipy.signal import savgol_filter
from operator import add
from multiprocessing import Pool 
import scipy.stats as stats
import matplotlib.pyplot as plt
import pickle as pkl
import numpy as np 
import scipy.interpolate
import statistics 
import random 
import csv  
import os 
import pathlib
import copy

def getDatarefs(butler, detectornum, runnum, rtype): # gets the Datarefs 
    where = f"exposure.science_program='{runnum}' and exposure.observation_type ='{rtype}'"
    collections = 'LSSTCam/raw/all'
    dataId = {'detector': detectornum}
    datarefs = list(butler.registry.queryDatasets(datasetType='raw', collections=collections, where=where, dataId=dataId))
    exp = butler.get(datarefs[0]) #use just the first image to get the order of amps
    det = exp.getDetector()
    amps_list = ["C00", "C01", "C02","C03","C04","C05","C06", "C07", "C10","C11","C12", "C13", "C14", "C15", "C16", "C17"]
    ampNames = [amp.getName() for amp in det if amp.getName() in amps_list] #the order in which the channels are in the dataset 
    return datarefs, amps_list, ampNames

def checkSmoothedBins(pcounts, pbins):
    ## there is an issue for my amp that is causing the two to be different lengths.. 
    issueindex = []
    for x in range(len(pbins)-1):
        if pbins[x] != pbins[x-1] +1:
            issueindex.append(x)            
    print("the index of issues list is", issueindex[:5])
    
    if len(issueindex) > 2: #use a suitable index
        diffs = []             
        for x in range(len(issueindex)-1):
            diff = issueindex[x+1] - issueindex[x] 
            diffs.append(diff)
        
        if len(diffs) != 0: 
            largestregion = max(diffs)
            loc = diffs.index(max(diffs)) 
            
            if issueindex[-1] < 7e4: ## have some way to exit if index issues only are low values... 
                print("We satified the beast") 
                
                st = issueindex[loc]
                en = len(pbins) -1 
                bins = pbins[st:en]
                counts = pcounts[st:en]
            
            else:
                st = issueindex[loc]
                en = issueindex[loc+1]
                bins = pbins[st:en]
                counts = pcounts[st:en]  
        
        else: 
            st = issueindex[0] 
            en = len(pbins)-1 #it is to be the INDEX
            bins = pbins[st:en]
            counts = pcounts[st:en]
    else: 
        counts = pcounts
        bins = pbins 
        st = 0 #it is to be the INDEX
        en = len(bins)-1 #it is to be the INDEX
    return counts, bins, st, en


def getFilter(counts, bins): 
    abins = bins[:-1]     
    mincounts  = np.array(counts) > 100 #trim function 
    counts150  = np.array(counts)[mincounts]
    bins150 = abins[mincounts]
    smoothed = savgol_filter(counts150, 33, 3, mode='constant', cval=0) #use the padding, smooth the distribution 
    
    filtered, filtbins, st, en = checkSmoothedBins(smoothed, bins150) # check that the filter doesnt jump

    validcounts = counts150[st:en] #match length array to that of the filter 
    cumulativesum = np.cumsum(filtered)
              
    cs = scipy.interpolate.CubicSpline(filtbins +1, cumulativesum) #have right edge
    return cs, filtbins, validcounts 

def defineCountsBins(datarefs, amps_list, ampNames, butler): #write a function that iterates over all the images 
    bins = np.arange(2e4, 2e5)
    countsdict = {} 
    for dataref in datarefs:
        exp = butler.get(dataref) 
        det = exp.getDetector()
        trimmed_ims = [exp.getMaskedImage()[amp.getRawDataBBox()].getImage() for amp in det if amp.getName() in amps_list]
        im_arrs = [trimmed_im.getArray().flatten().astype(int) for trimmed_im in trimmed_ims]
        comp_hist = {ampName: np.histogram(im_arrs, bins=bins)[0] for ampName, im_arrs in zip(ampNames, im_arrs)} 
        
        if len(countsdict) == 0: 
            countsdict = comp_hist
        else: 
            for key in countsdict: 
                countsdict[key] = list(map(add, countsdict[key], comp_hist[key]))
    
    return countsdict, bins

def determinePrescale(countsdict, prescaleregions):
    prescaleregions = np.arange(20000, 200000, 25) #every 25 bins
    prescalesdict = {}
    # create prescale for each amplifier 
    for key in countsdict: 
        counts = countsdict[key] 
        compcounts = copy.deepcopy(counts)
        compcounts[compcounts == 0] = np.nan # make any region without counts have no prescale 
        dynamicprescales = [] 
        for x in range(len(prescaleregions)-1):
            regionmean = np.nanmean(compcounts[25*x:25*(x+1)]) #exclude values that are zero, mostly for first/last bins 
            prescalevalue = 1.5e3/ regionmean
            if prescalevalue > 1: #remove any prescale that would inflate
                prescalevalue = 1
            if regionmean == 0: # if the region mean is 0, keep all the data 
                prescalevalue = 1
            dynamicprescales.append(prescalevalue)
        prescalesdict[key] = dynamicprescales
    return prescalesdict 


def selectImages(datarefs, butler, countsdict, amps_list, ampNames):
    prescaleregions = np.arange(20000, 200000, 25) 
    prescalesdict = determinePrescale(countsdict, prescaleregions) # write a function to Prescale the data to flatten  
    countsdictionary = {} 
    for dataref in datarefs:
        exp = butler.get(dataref) 
        det = exp.getDetector()
        # get the images 
        trimmed_ims = [exp.getMaskedImage()[amp.getRawDataBBox()].getImage() for amp in det if amp.getName() in amps_list]
        # flatten the array 
        im_arrs = [trimmed_im.getArray().flatten().astype(int) for trimmed_im in trimmed_ims]
        # sort the arrays by amplifier 
        imagearr = {ampName: im_arrs for ampName, im_arrs in zip(ampNames, im_arrs)}
        for key in imagearr: 
            # find the mode to determine what the prescale will be 
            arraymode = statistics.mode(imagearr[key])
            region = [element for element in prescaleregions if element > arraymode][0]
            index = np.argwhere(prescaleregions == region)[0][0]
            prescale = prescalesdict[key][index]
            image = imagearr[key]
            # select the counts randomly, the number as required by the prescale 
            randomlyselectedadu = np.random.choice(image, size=int(len(image)*prescale), replace=False)
            # sort into a histogram to save 
            counts, bins = np.histogram(randomlyselectedadu, bins=np.arange(20000, 200000))            
            if key not in countsdictionary.keys(): 
                countsdictionary[key] = counts  
            else: 
                countsdictionary[key] = list(map(add, countsdictionary[key], counts ))
    return countsdictionary

def runAll13144(detectornum, butler): #write this function afterwards 
    datarefs, amps_list, ampNames = getDatarefs(butler, detectornum, 13144, 'flat')
    countsdict, bins = defineCountsBins(datarefs, amps_list, ampNames, butler)
    # use the super-histogram to pass for the prescale 
    countsdictionary = selectImages(datarefs, butler, countsdict, amps_list, ampNames)
    makeedgedict = {}

    # write edge dict for each amp in the sensor 
    for key in countsdictionary:
        cs, filtbins, validcounts  = getFilter(countsdictionary[key], bins) 
        makeedgedict[key] = [validcounts, filtbins, cs]

    filename = f'sensor{detectornum}.pkl'
    directory = '/sdf/data/rubin/user/rejnicho/edgedicts/dicts13144'
    fullfilename = os.path.join(directory, filename)
    with open(fullfilename, 'wb') as f:
        pkl.dump(makeedgedict, f)

def runAll13549(detectornum, butler): 
    datarefs, amps_list, ampNames = getDatarefs(butler, detectornum, 13549, 'ramp')
    countsdict, bins = defineCountsBins(datarefs, amps_list, ampNames, butler)
    makeedgedict = {}
    
    # go by amplifier and write the edge dict for each amp in the sensor 
    for key in countsdict: 
        print(key)
        cs, filtbins, validcounts = getFilter(countsdict[key], bins)
        makeedgedict[key] = [validcounts, filtbins, cs]
    
    filename = f'sensor{detectornum}.pkl'
    directory = '/sdf/data/rubin/user/rejnicho/edgedicts/dicts13549'
    fullfilename = os.path.join(directory, filename)
    with open(fullfilename, 'wb') as f:
        pkl.dump(makeedgedict, f)

def defineSensorAttributes(detectornum): 
    repo_path = "/repo/ir2"
    butler = Butler(repo_path, collections=['LSSTCam/photodiode','LSSTCam/raw/all'], instrument='LSSTCam')
    registry = butler.registry
    recordClasses = butler.registry.queryDimensionRecords('detector', where="instrument='LSSTCam'")
    det_raft_pairs = sorted([(rc.id, rc.full_name) for rc in recordClasses])
    sensorname = det_raft_pairs[detectornum][1]
    print(sensorname)
    
    # make the edge dict for 13549
    edgedict13549 = runAll13549(detectornum, butler)
    print("finished 13549") #verifies that this run has pkl file created for it 
    
    # make the edge dict for 13144
    edgedict13144 = runAll13144(detectornum, butler)
    print("finished 13144")

def startRun(start, end): #run all the sensors on the focal plane 
    detectors = np.arange(start, end)
    for x in detectors: 
        defineSensorAttributes(x) 
    print("you have finished running all the sensors on the focal plane, congrats!") 
