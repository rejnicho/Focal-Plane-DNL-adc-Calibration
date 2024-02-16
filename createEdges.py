# this will read pkl files by each detector and create CSV files 

# runable functions: runAllRafts

# Note: you will have to change directories in: readAllRaftFiles, recordEdgeProblem, problemAmplifiers, makeGeneralLn,
# and change: getdnl, getinl 

from lsst.daf.butler import Butler
from scipy.signal import savgol_filter
import pickle as pkl
import numpy as np 
import scipy.stats as stats
import matplotlib.pyplot as plt
import scipy.interpolate
import scipy.signal
import csv  
import os 
import pathlib

def checkstartingbin(start, bins): #verifies data exists for starting bin 
    if start < bins[0]:
        print("binning cannot start at specified value")
        print("binning can start at", bins[0]) 
        raise KeyboardInterrupt

def makenarrow(bcvar, right, left):
    width = right - left
    width *= (1-bcvar) #reduce the width of the bin 
    right = width + left 
    bcvar *= 0.85 #make the change amount reduced by 85% 
    return right, bcvar
    
def makewide(bcvar, right, left): #function increases the bin width 
    width = right - left
    width *= (1+bcvar) #increase the width of the bin 
    right = width + left 
    bcvar *= 0.85 
    return right, bcvar


def checkSkippedBins(start, bins13549, bins13144): # look for skipped bins 
    issues13549 = [] # next bin after the skip 13549
    issues13144 = [] 
    
    # look for skipped bins in 13549 below 30k - where it is common for skipped bins 
    for x in range(len(bins13549) -1):
        if bins13549[x] +1  != bins13549[x+1]:
            if 30000 > bins13549[x +1] > start: 
                issues13549.append(bins13549[x+1])

    for x in range(len(bins13144) -1): 
        if bins13144[x] +1  != bins13144[x+1]:
            if 30000 > bins13144[x +1] > start: 
                issues13144.append(bins13549[x+1])
    
    # look at the values in the list, recommend starting the binning after you have the skipped bin 
    if len(issues13549) != 0: 
        recstart13549 = issues13549[-1] +1 

    elif len(issues13549) == 0: 
        recstart13549 = start 
        
    if len(issues13144) != 0: 
        recstart13144 = issues13144[-1] +1 

    elif len(issues13144) == 0: 
        recstart13144 = start

    recommendedstart = max(recstart13144, recstart13549)

    return recommendedstart

def determineStartEnd(counts13549, bins13549, counts13144, bins13144): ## Editted 
    # look for 13549 peaks which is a distribution we observed peaks in 
    mins13549 = scipy.signal.find_peaks(counts13549, prominence=[3e4, None], width=5)
    boi13549 = bins13549[mins13549[0]]

    #if peaks are detected, treat them properly 
    if len(boi13549) !=0: 
        lowpeaksboolean1 = [bin<3e4 for bin in boi13549] 
        lowpeaks1 = boi13549[lowpeaksboolean1] # locations of peaks in the low end of the adc
        highpeaksboolean1 = [bin > 5e4 for bin in boi13549] 
        highpeaks1 = boi13549[highpeaksboolean1] # locations of peaks in the high end of the adc

        if len(lowpeaks1) != 0 and len(highpeaks1) != 0: 
            s1 = lowpeaks1[-1]+ 50 # have some buffer because there is width to the peak, start binning after the peak  
            e1 = highpeaks1[0] - 50 

        elif len(lowpeaks1) == 0 and len(highpeaks1) != 0:
            s1 = bins13549[0]
            e1 = highpeaks1[0] - 50 

        elif len(lowpeaks1) != 0 and len(highpeaks1) == 0:
            s1 = lowpeaks1[-1]+ 50 
            e1 = bins13549[-1]

    # if no peaks, just use generic start and end of the distribution as start and end 
    else: 
        s1 = bins13549[0]
        e1 = bins13549[-1]

    # 13144 is prescaled so there are no peaks in the distribution to worry about by construction 
    s2 = bins13144[0]
    e2 = bins13144[-1]
    
    # now determine which are the proper ones to use
    start = max(s1, s2)
    end1 = min(e1, e2)
    
    # eliminate some issues where making more edges than necessary for exercise, this is a generic, arbitrary choice
    if end1 > 9e4: 
        end = 9e4
    else: 
        end = end1 

    # check that there is are no skipped bins in the distribution beyond where you start 
    checkedstart = checkSkippedBins(start, bins13549, bins13144) 

    # return the optomized start, which excludes any dramatic peaks and skipped bins 
    return checkedstart, end 

def makeedges2(counts, bins, spline, un, start, end, ampname, runnum):     
    checkstartingbin(start, bins)
    
    #edges start at leftmost bin, and force it to start when we determined 
    edges = [start] 

    #verify that the length of the count array is not terminated prematurely due to processing of images 
    if len(counts) < 200: 
        print("this amp has a terminated dataset", ampname) 
        edges = [] 
        widths = [] 
        return edges, widths 
    
    residualcumsum = spline(start) # the cumsum of the bins not included in the binning 
    firstcount = np.argwhere(bins == start) # start using values where specified     
    rightgoals = np.cumsum(counts[firstcount[0][0]:]) # the cumsum of the counts to be achieved; ie the right edge
    goals = rightgoals + residualcumsum # add in the ped of the spline discarded 

    for a in range(len(goals)): 
        left = edges[-1]
        if left > end: #stop the binning at some specified bin 
            break
        right = left + 1 # standard bin width = 1
        cv = spline(right) # current value, based on current right edge
        bcvar = 0.25 # bin change variable (start with 25% change)  
        x = 0 #iteration counter
        b = 0 #reset bcvar counter 
        while len(edges) != a+2: 
            if goals[a]-un < cv < goals[a] + un: #specify the requirement
                edges.append(right)
                
            elif cv < goals[a]: #bin is too small, widen right edge
                right, bcvar = makewide(bcvar, right, left)
                cv = spline(right)
                
                if x >1500: #reset bcvar 
                    if b <5: 
                        bcvar = 0.25 #gets stuck w too small, increase to help get closer
                        b +=1
                    else: 
                        #raise KeyboardInterrupt   ## this was the original method 
                        # we want to check how far along this error is computed, if its more than 50k from start, call "sufficient"
                        if len(edges) > 5e4: 
                            print("index", a, "edge cannot be computed")
                            print("we pass the edges, although termination occurred", runnum, ampname) 
                            # pass the edges and widths along 
                            widths = np.array(edges[1:]) - np.array(edges[:-1])
                            return edges, widths 
                        
                        else: 
                            print("we have an issue where less than 50k edges were made") 
                            print("index", a, "edge cannot be computed")
                            print("the issue is with run", runnum, "amplifier", ampname)
                            edges = [] 
                            widths = [] 
                            return edges, widths  

            elif cv > goals[a]:  # bin is too big, reduce right edge
                right, bcvar = makenarrow(bcvar, right, left)
                cv = spline(right)
                if x >1500: # reset bcvar
                    if b <5: # only let bcvar reset so many times
                        bcvar = 0.25 # gets stuck w too small, increase to help get closer
                        b+=1 
                    else: 
                        #raise KeyboardInterrupt   ## this was the original method 
                        if len(edges) > 5e4: 
                            print("index", a, "edge cannot be computed")
                            print("we pass the edges, although termination occurred", runnum, ampname) 
                            widths = np.array(edges[1:]) - np.array(edges[:-1])
                            return edges, widths 
                        
                        else: 
                            print("we have an issue where less than 50k edges were made") 
                            print("index", a, "edge cannot be computed")
                            print("the issue is with run", runnum, "amplifier", ampname)
                            edges = [] 
                            widths = [] 
                            return edges, widths  
                        
            x+= 1 #count iterations for the reset of bcvar
    widths = np.array(edges[1:]) - np.array(edges[:-1])
    return edges, widths

def determineDowntick(validcounts, filtbins): # examine if the source histogram has a downtick 
    # find peaks by looking at mins of inverse distribution
    mins= scipy.signal.find_peaks(-validcounts, distance=70, prominence=[1e4, None], width=2)
    boi = filtbins[mins[0]]
    if len(boi)> 0:
        downflag = True
    else:
        downflag = False
    return downflag 
    
def getinl(edges, ampname): #look at intergal nonlinearity 
    inlbins = np.arange(edges[0], edges[0]+len(edges)-1)
    idealmids = 0.5 + inlbins
    adcmids = [sum(i) for i in zip(edges[1:], edges[:-1])] #add together elements
    adcmids[:] = [x / 2 for x in adcmids] #find their mean, verified 3 entires, sufficient
    arraymids = np.array(adcmids)*(-1) 
    inl = [sum(i) for i in zip(idealmids, arraymids)]
    
    avinl = np.mean(inl)
    stdinl = np.std(inl)
    maxinl = max(inl)
    mininl = min(inl)
    
    maginl = max(maxinl, abs(mininl))
    rmsinl = np.sqrt(np.mean(np.square(inl))) 

    # use the specifications sheet to determine whether to flag this amplifier 
    if maginl > 1: #this amplifier is above the typical level 
        
        noposav = sum(1 for element in inl if element > 1) # number inl postive above average 
        noposhigh = sum(1 for element in inl if element > 2) # number inl positive above maximum 
        
        nonegav = sum(1 for element in inl if element < -1) # number inl negative below av
        noneglow = sum(1 for element in inl if element < -2) # number inl negative below minimum 

        # write a csv file to report this 
        specs = [ampname, noposav/len(inl), noposhigh/len(inl), nonegav/len(inl), noneglow/len(inl)]
        
        filename = f"/home/r/rejnicho/adc_investigation/adcbinning/focalplane/atypicalINL.csv"
        if os.path.isfile(filename) == False: 
            columns = ['Amplifier', 'inl higher than 1', 'inl higher than 2', 'inl lower than -1', 'inl lower than -2'] 

            with open(filename, 'w') as csvfile:  
                csvwriter = csv.writer(csvfile)  
                csvwriter.writerow(columns)  
                csvwriter.writerow(specs) 
        else: 
            with open(filename, 'a', newline='') as csvfile:  
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(specs)
    
    return avinl, stdinl, maginl, rmsinl

def getdnl(edges, ampname): #look at the differential nonlinearity 
    widths = np.array(edges[1:]) - np.array(edges[:-1])
    dnl = np.array(widths) -1  #ideal width =1 
    
    avdnl = np.mean(dnl)
    stddnl = np.std(dnl)
    maxdnl = max(dnl)
    mindnl = min(dnl)
    magdnl = max(maxdnl, abs(mindnl))
    rmsdnl = np.sqrt(np.mean(np.square(dnl))) 

    ## use the specifications sheet to determine whether to flag this amplifier 
    if magdnl > 0.5: ## if there are any elements that are above or below average report the amplifier 
        # determine the amount that above typical 
        noposav = sum(1 for element in dnl if element > 0.5) # number dnl postive above average 
        noposhigh = sum(1 for element in dnl if element > 1.5) # number dnl positive above maximum 
        
        nonegav = sum(1 for element in dnl if element < -0.5) # number dnl negative below av
        noneglow = sum(1 for element in dnl if element < -0.85) # number dnl negative below minimum 

        # write a csv file to report this 
        specs = [ampname, noposav/len(dnl), noposhigh/len(dnl), nonegav/len(dnl), noneglow/len(dnl)]
        
        filename = f"/home/r/rejnicho/adc_investigation/adcbinning/focalplane/atypicalDNL.csv"
        if os.path.isfile(filename) == False: 
            columns = ['Amplifier', '% dnl higher than 0.5', '% dnl higher than 1.5', '% dnl lower than -0.5', '% dnl lower than -0.85'] 

            with open(filename, 'w') as csvfile:  
                csvwriter = csv.writer(csvfile)  
                csvwriter.writerow(columns)  
                csvwriter.writerow(specs) 
        else: 
            with open(filename, 'a', newline='') as csvfile:  
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(specs)
    
    return avdnl, stddnl, magdnl, rmsdnl 

def makeGeneralLn(specs, raft): # make a log of important things for each run you analyze
    filename = f"/home/r/rejnicho/adc_investigation/adcbinning/focalplane/raftcsvfiles5/{raft}allsensors.csv"
    if os.path.isfile(filename) == False: 
        columns = ['Amplifier', 'starting bin', 'width diff p value', 'left diff p value', 
                      '13144 av. dnl', '13144 dnl std', '13144 max dnl', '13144 rms dnl', 
                      '13144 av. inl', '13144 inl std', '13144 max inl', '13144 rms inl',
                      '13549 av. dnl', '13549 dnl std', '13549 max dnl', '13549 rms dnl',
                      '13549 av. inl', '13549 inl std', '13549 max inl', '13549 rms inl',
                      'mean width diff', 'std width diff', 'mean left diff', 'std left diff', 
                      'min width 13549', 'max width 13549', 'min width 13144', 'max width 13144', 
                      'last bin 13549', 'last bin 13144'] 

        with open(filename, 'w') as csvfile:  
            csvwriter = csv.writer(csvfile)  
            csvwriter.writerow(columns)  
            csvwriter.writerow(specs) 
    else: 
        with open(filename, 'a', newline='') as csvfile:  
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(specs)

def getPvalues(e13549, w13549, e13144, w13144):
    # force there to be the same number edges in each method
    if len(e13144) != len(e13549): 
        lengthnec = min(len(e13144), len(e13549))
        e13144 = e13144[:lengthnec -1]
        w13144 = w13144[:lengthnec -1]
        e13549 = e13549[:lengthnec -1]
        w13549 = w13549[:lengthnec -1]

    # look at the range of the bin widths 
    minw9 = min(w13549)
    maxw9 = max(w13549)

    minw4 = min(w13144)
    maxw4 = max(w13144)
   
    ## t test for if the width diff between 2 runs is 0, pvalue is index 1 widthresults[1], p low means you have no sign dev from 0 
    widthresults = stats.ttest_rel(w13549, w13144)
        
    widthsdiff = np.subtract(w13144, w13549)
    widthdiffmean = (widthsdiff.mean()) 
    widthdiffstd = np.std(widthsdiff)
        
    ## t test for if the left edge diff between 2 runs is 0, pvalue is index 1 edgeresults[1]
    edgeresults = stats.ttest_rel(e13549, e13144)
    
    leftdiff = np.subtract(e13144, e13549)
    leftdiffmean = (leftdiff.mean())
    leftdiffstd = np.std(leftdiff)

    # flag if the width difference between two runs is statistically different than 0 
    
    if widthresults[1] <0.1: 
        flag = True 
    else: 
        flag = False 

    return widthresults[1], widthdiffmean, widthdiffstd, edgeresults[1], leftdiffmean, leftdiffstd, flag, minw9, maxw9, minw4, maxw4
    
def problemAmplifiers(issuesinfo): #record any amplifiers with downticks (either run) or significant deviation statistics  
    filename = "/home/r/rejnicho/adc_investigation/adcbinning/focalplane/flaggedAmplifiers2.csv"
    if os.path.isfile(filename) == False: 

        columns = ["Amplifier", "start bin", "p value width diff", "downticks 13144", "downticks 13549"] 
        
        with open(filename, 'w') as csvfile:
            csvwriter = csv.writer(csvfile)  
            csvwriter.writerow(columns)  
            csvwriter.writerow(issuesinfo) 
    else: 
        with open(filename, 'a', newline='') as csvfile:  
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(issuesinfo)
            
def recordEdgeProblem(specs): #record any amplifiers which the method cannot make edges for 
    filename = "/home/r/rejnicho/adc_investigation/adcbinning/focalplane/makeedgeproblems.csv"
    if os.path.isfile(filename) == False: 
        columns = ['Amplifier', 'run'] 

        with open(filename, 'w') as csvfile:  
            csvwriter = csv.writer(csvfile)  
            csvwriter.writerow(columns)  
            csvwriter.writerow(specs) 
    else: 
        with open(filename, 'a', newline='') as csvfile:  
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(specs)

def getEdges(sensorname, edgedict13549, edgedict13144, raft):
    # iterate through all amps on the sensor
    for key in edgedict13549:  
        counts13549 = edgedict13549[key][0]
        bins13549 = edgedict13549[key][1]
        cs13549 = edgedict13549[key][2]
            
        counts13144 = edgedict13144[key][0]
        bins13144 = edgedict13144[key][1]
        cs13144 = edgedict13144[key][2]

        ampname = sensorname +'_' + key

        # determine where to start and end binning for edge method 
        start, end = determineStartEnd(counts13549, bins13549, counts13144, bins13144) 

        # do the edge method 
        e13549, w13549 = makeedges2(counts13549, bins13549, cs13549, 1, start, end, ampname, 13549) 
        e13144, w13144 = makeedges2(counts13144, bins13144, cs13144, 1, start, end, ampname, 13144) 
        
        if len(e13549) == 0: #continue skips the rest of the amp functions  
            probleminfo = [ampname, 13549]
            recordEdgeProblem(probleminfo) #record the fact that you had an issue with the amplifier 
            continue 
            
        if len(e13144) == 0: 
            probleminfo = [ampname, 13144]
            recordEdgeProblem(probleminfo) #record the fact that you had an issue with the amplifier 
            continue 


        # record and get the inl info 
        avednl4, stddnl4, maxdnl4, rmsdnl4 = getdnl(e13144, ampname)
        avinl4, stdinl4, maxinl4, rmsinl4 = getinl(e13144, ampname)
    
        avednl9, stddnl9, maxdnl9, rmsdnl9 = getdnl(e13549, ampname)
        avinl9, stdinl9, maxinl9, rmsinl9 = getinl(e13549, ampname)

        # look for statistical deviations between the two runs 
        pwidth, widthdiffmean, widthdiffstd, pedge, leftdiffmean, leftdiffstd, pflag, minw9, maxw9, minw4, maxw4 = getPvalues(e13549, w13549, e13144, w13144) 

        # look for downticks in the distributions
        downflag4 = determineDowntick(counts13144, bins13144)
        downflag9 = determineDowntick(counts13549, bins13549)  
        
        if pflag or downflag4 or downflag9 == True: #if any flag is true, we record the amplifier to be examined later 
            issuesinfo = [ampname, start, pwidth, downflag4, downflag9] 
            problemAmplifiers(issuesinfo)

        # write all the general specifications for recording on the general log 
        specs = [ampname, start, pwidth, pedge, 
                 avednl4, stddnl4, maxdnl4, rmsdnl4, avinl4, stdinl4, maxinl4, rmsinl4,
                 avednl9, stddnl9, maxdnl9, rmsdnl9, avinl9, stdinl9, maxinl9, rmsinl9, 
                 widthdiffmean, widthdiffstd, leftdiffmean, leftdiffstd,
                 minw9, maxw9, minw4, maxw4, e13549[-1], e13144[-1]] 
    
        makeGeneralLn(specs, raft) 

def convertDetectorNumber(numb): ## this function takes the detector number and spits out the raft/sensor name RXXSXX format
    repo_path = "/repo/ir2"
    butler = Butler(repo_path, collections=['LSSTCam/photodiode','LSSTCam/raw/all'], instrument='LSSTCam')
    registry = butler.registry
    recordClasses = butler.registry.queryDimensionRecords('detector', where="instrument='LSSTCam'")
    det_raft_pairs = sorted([(rc.id, rc.full_name) for rc in recordClasses])
    sensorname = det_raft_pairs[numb][1]
    return sensorname 

def readAllRaftFiles(raft, raftlist):
    # range of the sensors on each raft 
    sensors = np.arange(0, 190, 9)
    index = raftlist.index(raft)
    fileleft = sensors[index]
    fileright = sensors[index+1]
    
    detectornumbers = np.arange(fileleft, fileright)
    for detectornumb in detectornumbers: 
        # verifies that amplifier is running analysis 
        print(detectornumb) 
        sensorname = convertDetectorNumber(detectornumb)
        file13144 = f'/sdf/data/rubin/user/rejnicho/edgedicts/dicts13144/sensor{detectornumb}.pkl'
        file13549 = f'/sdf/data/rubin/user/rejnicho/edgedicts/dicts13549/sensor{detectornumb}.pkl'
        
        # write something that if sensor file doesn't exist (yet), skip and continue onwards 
        if os.path.isfile(file13144):
        
            with open(file13144, 'rb') as f:
                edgedict13144 = pkl.load(f) 
        
            with open(file13549, 'rb') as f:
                edgedict13549 = pkl.load(f) 

            getEdges(sensorname, edgedict13549, edgedict13144, raft)
        else: 
            continue 

        # verifies that amplifier is done analysis 
        print(sensorname, "analysis completed")

def runAllRafts(): 
    raftlist = ["R01", "R02", "R03", "R10", "R11", "R12", "R13", "R14", "R20", "R21", "R22", "R23", "R24", 
                "R30", "R31", "R32", "R33", "R34", "R41", "R42", "R43"]
    for raft in raftlist:
        readAllRaftFiles(raft, raftlist)


