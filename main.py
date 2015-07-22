import os
import datetime

import numpy as np
import scipy as sp
import scipy.signal

import matplotlib.pyplot as plt
import matplotlib.dates
import matplotlib.style

#---------------------------------------------------------------------------------------------------
class bmEntry:
    def __init__(self, time, bristolType, duration, note):
        self._time = time
        self._bristolType = bristolType
        self._duration = duration
        self._note = note
        return    
#---------------------------------------------------------------------------------------------------
def parseFile(fn):
    fid = open(fn,'r')
    l = fid.readline().rstrip()
    
    # TODO turn this into regex

    # First line should be entries
    if l != '[ENTRIES]':
        raise('bad file')
        
    while True:
        l = fid.readline().rstrip()
        if len(l) > 0: break
    
    entries = []
    
    # Read until you have text
    done = False
    while not done:        
        # Read entry lines
        time = l
        bType = fid.readline().rstrip()
        duration = fid.readline().rstrip()
        note = fid.readline().rstrip()  # Optional
        
        # Parse date
        time = time[6:23]
        pm = True if time[16] == 'p' else False
        dTime = datetime.datetime(int(time[0:4]), int(time[5:7]), int(time[8:10]), int(time[11:13])+12 if pm and time[11:13] != '12' else int(time[11:13]), int(time[14:16]))
        # Parse bType
        bType = int(bType[6])
        # Parse duration
        # TODO
        
        # Create entry
        entry = bmEntry(dTime, bType, 0, note)
        entries.append(entry)
        
        while True:
            l = fid.readline()
            if len(l) == 0: done = True; break
            if l[0:4] == 'Time': break
                    
    return entries

#---------------------------------------------------------------------------------------------------
def convolve(dates, bss, time, timeWindow, kernelFunctor):
    N = dates.shape[0]
    r = np.zeros(N)
    for k in range(N):
        thisTime = dates[k]
        # Select range
        idx = np.where(np.logical_and(time>(thisTime-timeWindow), time <= thisTime))
        r[k] = kernelFunctor(bss[idx])
    return r
#---------------------------------------------------------------------------------------------------
def convolve2d(bss, time, timeWindow, kernelFunctor):
    N = bss.shape[0]
    r = []
    for k in range(N):
        thisTime = time[k]
        # Select range
        idx = np.where(np.logical_and(time>(thisTime-timeWindow), time <= thisTime))
        r.append(kernelFunctor(bss[idx]))
    return np.array(r)    
#---------------------------------------------------------------------------------------------------
def everyday(start, stop):
    r = []
    d = datetime.timedelta(days=1)
    k = start
    while (k<=stop):
        r.append(k)
        k = k + d
    return np.array(r)
        
#---------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    fn = r'C:\Users\idg101\Desktop\bm\BowelMove 20150717_174633.txt'
    
    matplotlib.style.use(r'https://raw.githubusercontent.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/master/styles/matplotlibrc')
    
    entries = parseFile(fn)
    
    N = len(entries)

    bss = []
    time = []
    for k in entries:
        bss.append(k._bristolType)
        time.append(k._time)
    bss = np.array(bss)
    time = np.array(time)
        
    #h = np.histogram(bss, bins=7, range=(1,8))
    #plt.pie(h[0], labels=['1', '2', '3', '4', '5', '6', '7'])
    #plt.show()e
    
    g = lambda x: np.histogram(x, bins=7, range=(1,7))[0]
    
    dateRange = everyday(time[-1], time[0])

    
    #h = convolve2d(bss, dateRange, datetime.timedelta(days=14), g)
    #plt.imshow(h); plt.show()
          
    g = lambda x: np.max(x)-np.min(x)
    delta1 = datetime.timedelta(days=1)
    delta3 = datetime.timedelta(days=3)
    delta7 = datetime.timedelta(days=7)
    delta14 = datetime.timedelta(days=14)
    delta28 = datetime.timedelta(days=28)
    
    ma1day = convolve(dateRange, bss, time, delta1, np.mean)  
    ma3days = convolve(dateRange, bss, time, delta3, np.mean)        
    ma7days = convolve(dateRange, bss, time, delta7, np.mean)        
    ma14days = convolve(dateRange, bss, time, delta14, np.mean)        
    ma28days = convolve(dateRange, bss, time, delta28, np.mean)        
    plt.plot(dateRange, ma3days); plt.hold(True)
    plt.plot(dateRange, ma14days); 
    plt.plot(dateRange, ma28days); 
    plt.ylabel('Bristol Stool Scale')
    plt.xlabel('Date')
    plt.ylim(0,8); plt.hold(False)
    plt.legend(('3 days', '14 days', '28 days'))
    plt.title('Bristol Stool Scale Moving Average')
    plt.savefig('bss - moving average.png')
    
    plt.clf(); plt.subplot(311)
    plt.plot(dateRange, ma3days, label='3 days'); plt.ylim(0,8); plt.ylabel('Bristol Stool Scale'); plt.legend(shadow=True)
    plt.subplot(312)
    plt.plot(dateRange, ma14days, label = '14 days'); plt.ylim(0,8); plt.ylabel('Bristol Stool Scale');  plt.legend(shadow=True)
    plt.subplot(313)
    plt.plot(dateRange, ma28days, label = '28 days'); plt.ylim(0,8); plt.ylabel('Bristol Stool Scale');  plt.legend(shadow=True)
    plt.xlabel('Date')
    plt.ylim(0,8);
    plt.suptitle('Bristol Stool Scale Moving Average')
    plt.savefig('bss - moving average - subplot.png')

    plt.clf()
    numMovements3 = convolve(dateRange, bss, time, delta3, lambda x: np.shape(x)[0]) / 3
    numMovements14 = convolve(dateRange, bss, time, delta14, lambda x: np.shape(x)[0]) / 14
    numMovements28 = convolve(dateRange, bss, time, delta28, lambda x: np.shape(x)[0]) / 28
    plt.subplot(311)
    plt.plot(dateRange, numMovements3, label='3 days'); plt.ylim(0,8);  plt.legend(shadow=True)
    plt.subplot(312)
    plt.plot(dateRange, numMovements14, label = '14 days'); plt.ylim(0,8);  plt.legend(shadow=True)
    plt.subplot(313)
    plt.plot(dateRange, numMovements28, label = '28 days'); plt.ylim(0,8);  plt.legend(shadow=True)
    plt.xlabel('Date')
    plt.ylim(0,8);
    plt.suptitle('Mean Movements Per Day')
    plt.savefig('mean movements per day - moving average - subplot.png')    
    
    plt.clf()
    numMovements1 = convolve(dateRange, bss, time, delta1, lambda x: np.shape(x)[0]) / 1
    plt.subplot(311)
    plt.plot(dateRange, numMovements1, label='Daily'); plt.ylim(0,8);  plt.legend(shadow=True)
    plt.subplot(312)
    plt.plot(dateRange, numMovements14, label = '14 days'); plt.ylim(0,8);  plt.legend(shadow=True)
    plt.subplot(313)
    plt.plot(dateRange, numMovements28, label = '28 days'); plt.ylim(0,8);  plt.legend(shadow=True)
    plt.xlabel('Date')
    plt.ylim(0,8);
    plt.suptitle('Mean Movements Per Day')
    plt.savefig('mean movements per day 2 - moving average - subplot.png')    
    
    plt.clf()
    h = np.histogram(bss,bins=7, range=(1,7))
    bssPdf = h[0] / np.sum(h[0])
    plt.plot(np.arange(1,8), bssPdf, marker='o')
    plt.title('PDF of Bristol Stool Scale')
    plt.xlabel('Bristol Stool Scale'); plt.xlim(0,8)
    plt.ylabel('p'); plt.ylim(0,1)
    plt.savefig('bss pdf.png')  
    
    plt.clf()
    plt.hist(bss, bins=7)
    plt.title('Histogram of Bristol Stool Scale')
    plt.xlabel('Bristol Stool Scale');
    plt.ylabel('Counts'); 
    plt.savefig('bss histogram.png')     
    
    plt.clf()
    badBss = np.abs(bss - 4)>=2
    badBssHistPdf = np.histogram(badBss, bins=2)[0]
    basBssHistPdf = badBssHistPdf / np.sum(badBssHistPdf)
    plt.pie(basBssHistPdf, labels = ('3-5', '1-2, 6-7'), autopct='%1.1f%%')
    plt.title('Pie Chart of Good/Bad Bristol Stool Scale')
    plt.savefig('bss pie chart.png')     
    
    plt.clf()
    ma1day[np.where(np.isnan(ma1day))] = 4  # data imputation
    ma3days[np.where(np.isnan(ma3days))] = 4  # data imputation
    plt.subplot(311)
    plt.acorr(ma1day, normed=True, maxlags=90, label='1 day'); plt.legend()
    plt.subplot(312)
    plt.acorr(ma14days, normed=True, maxlags=90, label='14 days');plt.legend()
    plt.subplot(313)
    plt.acorr(ma28days, normed=True, maxlags=90, label='28 days');plt.legend()
    plt.xlabel('Days')    
    plt.suptitle('Bristol Stool Scale AutoCorr')
    plt.savefig('bss acorr.png')   
    
    plt.clf()
    plt.subplot(311)
    plt.acorr(numMovements1, normed=True, maxlags=90,  label='1 day'); plt.legend()
    plt.subplot(312)
    plt.acorr(numMovements14, normed=True, maxlags=90,  label='14 day');plt.legend()
    plt.subplot(313)
    plt.acorr(numMovements28, normed=True, maxlags=90, label='28 days'); plt.legend()
    plt.xlabel('Days')    
    plt.suptitle('Num Movements AutoCorr')
    plt.savefig('num movements acorr.png')      
    
    plt.clf()   
    plt.hist2d(numMovements1, ma1day, cmap=matplotlib.cm.gist_heat, normed=True, bins=(5,7), range=((0,5),(1,7)))
    plt.xlabel('Num Movements'); plt.ylabel('Bristol Stool Scale'); plt.colorbar()
    plt.title('1 day moving average')
    plt.savefig('density 1 day.png')      
    
    plt.clf()   
    plt.hist2d(numMovements14, ma14days, cmap=matplotlib.cm.gist_heat, normed=True, bins=(5,7), range=((0,5),(1,7)))
    plt.xlabel('Num Movements'); plt.ylabel('Bristol Stool Scale'); plt.colorbar()
    plt.title('14 day moving average')
    plt.savefig('density 14 days.png')   
    
    plt.clf()   
    plt.hist2d(numMovements28, ma28days, cmap=matplotlib.cm.gist_heat, normed=True, bins=(5,7), range=((0,5),(1,7)))
    plt.xlabel('Num Movements'); plt.ylabel('Bristol Stool Scale'); plt.colorbar()
    plt.title('28 day moving average')
    plt.savefig('density 28 days.png')       
    
    print('done')
