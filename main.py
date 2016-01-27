import os
import datetime

import numpy as np
import scipy as sp
import scipy.signal

import matplotlib.pyplot as plt
import matplotlib.dates
import matplotlib.style

#---------------------------------------------------------------------------------------------------
# http://stackoverflow.com/questions/5734438/how-to-create-a-month-iterator
def month_year_iter( start_month, start_year, end_month, end_year ):
    ym_start= 12*start_year + start_month - 1
    ym_end= 12*end_year + end_month
    for ym in range( ym_start, ym_end ):
        y, m = divmod( ym, 12 )
        yield y, m+1

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
def pctBetterTest(dates, bss, period1, period2):
    assert(len(dates == len(bss)))
    d1_idx = np.where(np.logical_and(dates>=period1[0], dates <= period1[1]))
    d2_idx = np.where(np.logical_and(dates>=period2[0], dates <= period2[1]))
    s1 = bss[d1_idx]
    s2 = bss[d2_idx]

    pctGoodPeriod1 = (np.sum(np.logical_and(s1>=3, s1<=5))/len(s1))
    pctGoodPeriod2 = (np.sum(np.logical_and(s2>=3, s2<=5))/len(s2))
    return ((pctGoodPeriod2-pctGoodPeriod1) / pctGoodPeriod1)*100

#---------------------------------------------------------------------------------------------------
def histogram(dates, bss, period1, numBins=0):
    assert(len(dates == len(bss)))
    d1_idx = np.where(np.logical_and(dates>=period1[0], dates <= period1[1]))
    s1 = bss[d1_idx]

    h = np.histogram(s1, bins=7, range=(1,7))

    return h[0]

#---------------------------------------------------------------------------------------------------
def ksTest(dates, bss, period1, period2):
    assert(len(dates == len(bss)))
    d1_idx = np.where(np.logical_and(dates>=period1[0], dates <= period1[1]))
    d2_idx = np.where(np.logical_and(dates>=period2[0], dates <= period2[1]))
    s1 = bss[d1_idx]
    s2 = bss[d2_idx]
    
    D, p = sp.stats.ks_2samp(s1,s2)
    
    return p
    
#---------------------------------------------------------------------------------------------------
def convolve(dates, bss, time, timeWindow, kernelFunctor):
    N = dates.shape[0]
    r = np.zeros(N)
    for k in range(N):
        thisTime = dates[k]
        # Select range
        idx = np.where(np.logical_and(time>=(thisTime-timeWindow), time <= thisTime))
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
    while (k<=stop+d):
        r.append(k)
        k = k + d
    return np.array(r)

#---------------------------------------------------------------------------------------------------
# http://stackoverflow.com/questions/736043/checking-if-a-string-can-be-converted-to-float-in-python
def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False
#---------------------------------------------------------------------------------------------------

def parseHealthSpreadsheet():
    fn = r'C:\Users\idg101\Desktop\bm\Heath Record - Sheet1.csv'
    
    import csv
    dailyScore = []
    dayOfWeek = np.zeros(7)
    dayOfWeekSum = np.zeros(7)
    date = []
    with open(fn) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            dd = row['Date'].split('/')
            d = datetime.datetime(int(dd[2]), int(dd[0]), int(dd[1]))
            date.append(d)
            ds = row['Daily Score']
            if isfloat(ds):
                dailyScore.append(float(ds))
                dayOfWeek[d.weekday()] += float(ds)
                dayOfWeekSum[d.weekday()] += 1
            else:
                dailyScore.append(np.nan)
    
    dailyScore = np.array(dailyScore)
    
    # Boxplot
    fig, ax = plt.subplots(1)    
    mat = []
    for months in range(1,13):
        r = []
        for k in range(len(dailyScore)):
            if date[k].month==months:
                if np.isfinite(dailyScore[k]):
                    r.append(dailyScore[k])
        mat.append(r)

    plt.boxplot(mat, showmeans=True, whis=[10, 90]); 
    plt.ylabel('HQI'); plt.ylim(0,5)
    plt.xlabel('Month')
    plt.title('Health Quality Index by Month\nWhiskers - 90th Percentile')
    plt.xticks(range(1,13), ('Jan 2015', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'), rotation=15)
    plt.tight_layout()
    plt.savefig('hqi - boxplot.png')        
    
    plt.clf()
    plt.plot(date, dailyScore, label = 'HQI'); plt.ylim(0,5);  plt.legend(shadow=True)
    plt.xlabel('Date')
    plt.ylim(0,5);
    plt.suptitle('Health Quality Index')
    plt.savefig('hqi.png')      
    
    plt.clf()
    plt.bar(np.arange(0,7),(dayOfWeek/dayOfWeekSum))
    plt.ylim(0,4)
    plt.savefig('hqi day of week.png')    
    return
    
        
#---------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    fn = r'C:\Users\idg101\Desktop\bm\BowelMove 20160127_165002.txt'    
    
    matplotlib.style.use(r'https://raw.githubusercontent.com/isaacgerg/matplotlibrc/master/matplotlibrc.txt')
    
    parseHealthSpreadsheet()
    
    entries = parseFile(fn)
    
    N = len(entries)

    bss = []
    time = []
    for k in entries:
        bss.append(k._bristolType)
        time.append(k._time)
    bss = np.array(bss)
    time = np.array(time)
           
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
    delta42 = datetime.timedelta(days=42)
    
    ma1day = convolve(dateRange, bss, time, delta1, np.mean)  
    ma3days = convolve(dateRange, bss, time, delta3, np.mean)        
    min3days = convolve(dateRange, bss, time, delta3, lambda x: np.min(x) if len(x)>0 else np.nan)     
    max3days = convolve(dateRange, bss, time, delta3, lambda x: np.max(x) if len(x)>0 else np.nan)     
    pctGood3days = convolve(dateRange, bss, time, delta3,lambda x: (np.sum(np.logical_and(x>=3, x<=5))/len(x))*100)            
    #ma7days = convolve(dateRange, bss, time, delta7, lambda x: sp.stats.mode(x)[0][0])        
    ma7days = convolve(dateRange, bss, time, delta7,np.mean)        
    pctGood7days = convolve(dateRange, bss, time, delta7,lambda x: (np.sum(np.logical_and(x>=3, x<=5))/len(x))*100)    
    pctGood28days = convolve(dateRange, bss, time, delta28,lambda x: (np.sum(np.logical_and(x>=3, x<=5))/len(x))*100)            
    sd7days = convolve(dateRange, bss, time, delta7, np.std)    
    ma14days = convolve(dateRange, bss, time, delta14, np.mean)        
    ma28days = convolve(dateRange, bss, time, delta28, np.mean)        
    
    if False:
        plt.plot(dateRange, ma3days); plt.hold(True)
        plt.plot(dateRange, ma7days); 
        plt.plot(dateRange, ma28days); 
        plt.ylabel('Bristol Stool Scale')
        plt.xlabel('Date')
        plt.ylim(0,8); plt.hold(False)
        plt.legend(('3 days', '7 days', '28 days'))
        plt.title('Bristol Stool Scale Moving Average')
        plt.savefig('bss - moving average.png')
    
    # Moving ks test
    today = time[0] # rename var "today"
    lastDay = time[-1]    
    pvalues = []
    times = []
    means = []
    numDays = (today-lastDay - 2*delta28).days
    pctBetter = []    
    for k in range(numDays):
        startDate1 = lastDay + datetime.timedelta(days=int(k))
        stopDate1 = startDate1 + delta28
        startDate2 = stopDate1
        stopDate2  = startDate2 + delta28
        times.append(startDate2)
        
        idx1 = np.logical_and(time>=startDate1, time<=stopDate1)
        idx2 = np.logical_and(time>=startDate2, time<=stopDate2)
        d = np.mean(bss[idx2]) - np.mean(bss[idx1])
        if d<0:
            d = 64
        else:
            d = 255-64
        means.append(d)
                
        pctBetter.append(pctBetterTest(time, bss, (startDate1,stopDate1), (startDate2, stopDate2)))
        pvalues.append(ksTest(time, bss, (startDate1,stopDate1), (startDate2, stopDate2)))
    fig, ax = plt.subplots(1)
    cmap = matplotlib.cm.seismic_r
    for k in range(len(times)):
        c = cmap(int(means[k]))
        ax.semilogy(times[k], pvalues[k], marker='.', markeredgewidth=1, markeredgecolor='black', markersize = 20, color=c)
    #ax.semilogy(times, pvalues); 
    
    
    plt.ylabel('p-value'); plt.xlabel('Split Date'); plt.title('KS Testing\np-value of Before-After Period of 4 Weeks');
    fig.autofmt_xdate()
    plt.savefig('bss - p-value of Before-After Period of 4 Weeks.png')
    
    # Pct better scores week 1 compared week 2
    fig, ax = plt.subplots(1)
    cmap = matplotlib.cm.seismic_r
    means = np.array(means)
    for k in range(len(times)):
        c = cmap(int(means[k]))
        ax.plot(times[k], pctBetter[k], marker='.', markeredgewidth=1, markeredgecolor='black', markersize = 20, color=c)
    plt.ylabel('Percentage Impvoement'); plt.xlabel('Split Date'); plt.title('Percentage Improvement bss=[3,5]\nBefore-After Period of 4 Weeks\nColor indicates BSS change delta');
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig('pct difference - before-after Period of 4 Weeks.png')    
    
    # Weekly histogram
    newestDay = time[0] 
    lastDay = time[-1]        
    times = []
    numWeeks = int((newestDay - lastDay - delta7).days/7)
    hgram = []
    for k in range(numWeeks):
        startDate1 = lastDay + datetime.timedelta(days=int(7*k))
        stopDate1 = startDate1 + delta7
        hgram.append(histogram(time, bss, (startDate1,stopDate1)))
    hgram = np.array(hgram)    
    fig, ax = plt.subplots(1)
    
    plt.imshow(hgram, interpolation='nearest', cmap=matplotlib.cm.gist_heat, extent=[1,7,0, numWeeks], origin='upper'); plt.colorbar(); 
    plt.ylabel('Weeks in Past'); plt.xlabel('Bristol Stool Score'); plt.title('BSS Histogram - Weekly');
    plt.savefig('bss - histogram weekly.png')    
    
    # Plot all points
    fig, ax = plt.subplots(1)
    cmap = [matplotlib.cm.RdBu(0), matplotlib.cm.RdBu(32), matplotlib.cm.RdBu(32*2), (0.5,0.5,0.5,1.0), matplotlib.cm.RdBu(32*5), matplotlib.cm.RdBu(32*6), matplotlib.cm.RdBu(32*7)]
    for k in range(len(bss)):
        color = cmap[bss[k]-1]
        plt.plot(time[k], time[k].hour + (time[k].minute/60.0), '.', marker = '|', markeredgewidth = 5, color=color, markersize=10)
    plt.ylim(0,24); plt.ylabel('Hour of Day'); plt.xlabel('Date')
    plt.title('BSS\nRed - 1, Gray - 4, Blue - 7')
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig('bss - type vs time.png')
    
    # Boxplot
    months = []
    years = []
    scores = []
    labels = []
    for k in month_year_iter(time[-1].month, time[-1].year, time[0].month, time[0].year):
        month = k[1]
        year = k[0]        
        labels.append('%d, %d'%(month, year))
        months.append(month)
        years.append(year)
        m = []
        for k in range(len(time)):
            if time[k].month == month and time[k].year == year:
                m.append(bss[k])
        scores.append(m)
                        
    fig, ax = plt.subplots(1)            
    plt.boxplot(scores, showmeans=True, whis=[10, 90]); 
    plt.ylabel('BSS'); plt.ylim(0,8)
    plt.xlabel('Month')
    plt.title('Bristol Stool Scale by Month\nWhiskers - 90th Percentile')
    plt.xticks(range(1,len(months)+1), labels, rotation=15)
    plt.tight_layout()
    plt.savefig('bss - boxplot.png')    
    
    
    plt.clf(); plt.subplot(311)
    plt.plot(dateRange, ma3days, label='3 day MA'); plt.ylim(1,7); plt.ylabel('Bristol Stool Scale'); plt.legend(shadow=True)
    plt.plot(dateRange, 3*np.ones(len(ma3days)), color='black'); plt.plot(dateRange, 5*np.ones(len(ma3days)), color='black')
    #plt.hold(True); plt.plot(dateRange, min3days); plt.plot(dateRange, max3days); plt.hold(False)
    plt.subplot(312)
    plt.plot(dateRange, ma7days, label = '7 day MA'); plt.ylim(1,7); plt.ylabel('Bristol Stool Scale');  plt.legend(shadow=True)
    plt.plot(dateRange, 3*np.ones(len(ma3days)), color='black'); plt.plot(dateRange, 5*np.ones(len(ma3days)), color='black')
    plt.subplot(313)
    plt.plot(dateRange, ma28days, label = '28 day MA'); plt.ylim(1,7); plt.ylabel('Bristol Stool Scale');  plt.legend(shadow=True)
    plt.plot(dateRange, 3*np.ones(len(ma3days)), color='black'); plt.plot(dateRange, 5*np.ones(len(ma3days)), color='black')
    plt.xlabel('Date'); plt.ylim(1,7);
    plt.savefig('bss - moving average - subplot.png')
    
    plt.clf(); 
    plt.subplot(311) 
    plt.plot(dateRange, pctGood3days, label = '3 day window'); plt.ylabel('Perctage 3-5');  plt.legend(shadow=True, loc=4)    
    plt.xlabel('Date'); plt.ylim(0,100);    
    plt.subplot(312) 
    plt.plot(dateRange, pctGood7days, label = '7 day window'); plt.ylabel('Perctage 3-5');  plt.legend(shadow=True, loc = 4)
    plt.xlabel('Date'); plt.ylim(0,100);    
    plt.subplot(313) 
    plt.plot(dateRange, pctGood28days, label = '28 day window'); plt.ylabel('Perctage 3-5');  plt.legend(shadow=True, loc = 4)
    plt.xlabel('Date'); plt.ylim(0,100);        
    plt.suptitle('Bristol Stool Scale Moving Average')
    plt.savefig('bss - pct good - subplot.png')
    
    # Percentage of BSS 3-5 scores in last day
    pvalues = []
    mvalues = []
    times = []
    days = (today-lastDay - delta7).days
    histogramBss3day = np.ones((7,1))
    sd = []
    for k in np.arange(-days, 0, dtype=np.int):
        startDate = today + datetime.timedelta(days=int(k)) - delta7
        stopDate = today + datetime.timedelta(days=int(k))
        times.append(stopDate)
        #assert(len(dates == len(bss)))
        d1_idx = np.where(np.logical_and(time>=startDate, time<=stopDate))
        s1 = bss[d1_idx]
        l = np.sum(np.logical_and(s1>=3, s1<=5))/len(s1)   
        sd.append(s1.std())
        mvalues.append(s1.mean())
        pvalues.append(l*100)       
        h = np.histogram(s1,bins=7, range=(1,7))
        histogramBss3day = np.append(histogramBss3day, h[0].reshape((7,1)), axis=1)
    plt.clf()
    mvalues = np.array(mvalues)
    sd = np.array(sd)
    fig, ax = plt.subplots(1)
    ax2 = ax.twinx()
    ax2.grid(False)
    ax2.plot(times,mvalues, 'g'); ax2.set_ylabel('BSS Score'); ax2.set_ylim(0,7)
    ax.plot(times, pvalues); ax.set_ylabel('Percentage'); plt.xlabel('Date'); plt.title('Percentage Good Score, 7 day MA');
    ax2.plot(times, sd, 'r')
    fig.autofmt_xdate(); ax.set_ylim(0,100)
    plt.savefig('bss - Percentage good.png')        
    

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

    if True:
        plt.clf()
        badBss = np.abs(bss - 4)>=2
        badBssHistPdf = np.histogram(badBss, bins=2)[0]
        basBssHistPdf = badBssHistPdf / np.sum(badBssHistPdf)
        plt.pie(basBssHistPdf, labels = ('3-5', '1-2, 6-7'), autopct='%1.1f%%')
        plt.title('Pie Chart of Good/Bad Bristol Stool Scale')
        plt.savefig('bss pie chart.png')     
       
    
    print('done')
