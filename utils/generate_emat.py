import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert
import random
from scipy.signal import kaiserord, lfilter, firwin
import requests
def area(y0,y1,x2,y2,y3,y4):
    '''calculate area
    center : x2,y2
    '''
    x1=x2-1
    x3=x2+1
    if((y2-y0)/2>(y2-y1)):
        y1=y0
        x1=x1-1
    if((y2-y4)/2>(y2-y3)):
        y3=y4
        x3=x3+1
    return 2*min(y1,y3)

def dataloader(ind_,signal,ekg,s1_label):
    s1_label_= np.array(s1_label[ind_,:],dtype=np.float)
    signal_=np.array(signal[ind_,:],dtype=np.float)
    ekg_=np.array(ekg[ind_,:],dtype=np.float)
    return signal_,ekg_,s1_label_

def detect(signal,ekg,s1_label,fs = 500.0):  
    plot_f=False 
    #hyperparameter    
    duration = 10.0
    
    unit=1000/fs
    samples = int(fs*duration)
    t = np.arange(samples) / fs      
    if(plot_f):
        fig = plt.figure(figsize=(12, 6))

    #normalize
    analytic_signal = hilbert(signal)   
    envelope=abs(analytic_signal)/max(abs(analytic_signal))
    SE=-1*(envelope**2)*np.log(envelope**2) 

    #plot HR
    nyq_rate = 500 / 2.0
    width = 5.0/nyq_rate
    ripple_db = 10.0
    N, beta = kaiserord(ripple_db, width)
    cutoff_hz = 10.0

    # Use firwin with a Kaiser window to create a lowpass FIR filter.
    taps = firwin(N, cutoff_hz/nyq_rate, window=('kaiser', beta))
    filtered_x = lfilter(taps, 1.0, SE)
    if(plot_f):
        ax = fig.add_subplot(5,1,2)
        ax.plot(t, filtered_x)
    ck_peak=np.zeros(filtered_x.shape[0])
    sum_=0
    count=0
    plot_=True
    for i in range(2,filtered_x.shape[0]-2):
        if(filtered_x[i]>filtered_x[i-1] and filtered_x[i]>filtered_x[i+1] and filtered_x[i]>0.1):
            ck_peak[i]=area(filtered_x[i-2],filtered_x[i-1],i,filtered_x[i],filtered_x[i+1],filtered_x[i+2])
            sum_+=ck_peak[i]
            count+=1
    ck_peak_sort=np.sort(ck_peak, axis=None)
    ck_peak_th=sum_/count*0.5#ck_peak_sort[-1000]
    peak_max=0
    peak_idx=0
    #peak_split=0.05
    peak_split=0.05
    for i in range(1,filtered_x.shape[0]-1):        
        if(filtered_x[i]>filtered_x[i-1] and filtered_x[i]>filtered_x[i+1] and ck_peak[i]>ck_peak_th):
            if(filtered_x[i]>peak_max):
                ck_peak[peak_idx]=0
                peak_max=filtered_x[i]
                peak_idx=i
            else:
                ck_peak[i]=0
        if(filtered_x[i]<peak_split):
            peak_max=0
            peak_idx=0 
    peak_position=np.zeros(5000)        
    for i in range(1,filtered_x.shape[0]-1):
        if(filtered_x[i]>filtered_x[i-1] and filtered_x[i]>filtered_x[i+1] and ck_peak[i]>ck_peak_th and plot_):
            if(plot_f):
                ax.scatter(t[i],filtered_x[i], s=50)
            peak_position[i]=1
            #plot_=False
        if(filtered_x[i]<peak_split):
            plot_=True
    #print('peak_position',np.sum(peak_position))

    #plot S1 label
    t=np.arange(5000)/500.0
    if(plot_f):
        ax = fig.add_subplot(5,1,3)  
        ax.plot(t,ekg)
    if(plot_f):
        for i in range(s1_label.shape[1]):
            if(s1_label[2][i]==1):
                ax.plot(t[i+4:i+4+2],ekg[i+4:i+4+2],color='red')

    end=0
    ck=0
    plot_s1 = False
    plot_on_label_s1=False
    if(plot_f):
        ax = fig.add_subplot(5,1,4)
    pred_s1_label=np.zeros(5000)

    s3s4ck=0
    s3s4count=0
    local_max_peak=0
    local_max_peak_position=0
    if(plot_f):
        ax.plot(t,signal,color='blue')
    time_count=0
    for i in range(s1_label.shape[1]-1):
        if(s1_label[2][i]-s1_label[2][i+1]==1):
            time_count+=1
    n=0
    emat_array=[]
    for i in range(s1_label.shape[1]-1):
        if(s1_label[2][i]==1):
            if(ck_peak[i]>ck_peak_th and ck_peak[i]>local_max_peak):
                local_max_peak=ck_peak[i]
                local_max_peak_position=i
                plot_on_label_s1 = True
        if(s1_label[2][i]-s1_label[2][i+1]==1 and i>end):
            plot_s1 = True
            ck=i
            
        if(s1_label[2][i+1]-s1_label[2][i]==1):
            q_onset=i
            #print('*q_onset',q_onset)
        if(i>ck+150 and plot_s1):
            plot_s1 = False
            if(plot_f):
                ax.plot(t[ck+4:ck+4+82],signal[ck+4:ck+4+82],color='red')
            pred_s1_label[ck+4:ck+4+82]=1
        
        
        if((ck_peak[i]>ck_peak_th and plot_s1) or (plot_on_label_s1 and i-local_max_peak_position>150)) :
            #ax.plot(t[i+4:i+4+2],signal[i+4:i+4+2],color='red')
            if(ck_peak[i]>local_max_peak):
                n+=1
                #print(n)
                #print('emat',i,q_onset,i-q_onset)
                if(i-q_onset>0):
                    emat_array.append((i-q_onset)*unit)
                start=i+4-1
                end_plot=i+4+1
                while(filtered_x[start]>0.05):
                    start-=1
                if(end_plot<4999):
                    while(filtered_x[end_plot]>0.05 and end_plot<4999):
                        end_plot+=1
                if(plot_f):
                    ax.plot(t[start:end_plot],signal[start:end_plot],color='red')
                pred_s1_label[start:end_plot]=1
                plot_s1=False
                plot_on_label_s1=False
                local_max_peak=0
                local_max_peak_position=0
                end=i+150
            else:
                n+=1
                #print(n)
                #print('*emat',local_max_peak_position,q_onset,local_max_peak_position-q_onset)
                if(local_max_peak_position-q_onset>0):
                    emat_array.append((local_max_peak_position-q_onset)*unit)
                start=local_max_peak_position+4-1
                end_plot=local_max_peak_position+4+1
                while(filtered_x[start]>0.05):
                    start-=1
                if(end_plot<4999):
                    while(filtered_x[end_plot]>0.05 and end_plot<4999):
                        end_plot+=1
                if(plot_f):
                    ax.plot(t[start:end_plot],signal[start:end_plot],color='red')
                pred_s1_label[start:end_plot]=1
                plot_s1=False
                plot_on_label_s1=False
                end=local_max_peak_position+150
                local_max_peak=0
                local_max_peak_position=0
    emat_array=np.array(emat_array,dtype=np.int)
    #print(np.sum(emat_array)/emat_array.shape[0])
    if(plot_f):
        plt.savefig("emat_plot")
    return np.sum(emat_array)/emat_array.shape[0]
   
def cal_emat(hs_set,ekg_set,fs=500.0):
    ekg_set=np.array(ekg_set,dtype=np.float)
    payload = {"raw": ekg_set.tolist()}
    res = requests.post('http://gpu4.miplab.org:8899/PQRSTSegmentation', json=payload)
    s1_label_set = np.array(res.json()['label'])

    emat_set=[]
    size=hs_set.shape[0]
    for i in range(size):
        signal,ekg,s1_label=dataloader(i,hs_set,ekg_set,s1_label_set)
        emat_set.append(detect(signal,ekg,s1_label,fs))
    return emat_set # list (#signal)
if __name__ == '__main__':   
    hs_set = np.load('audicor_hs_s3s4_10_200_45.npy',allow_pickle=True)[:5,5:]# (#signal,fs*10)
    ekg_set = np.load('audicor_ekg_s3s4.npy',allow_pickle=True)[:5,5:]# (#signal,fs*10)
    #emat_set=np.load('audicor_emat.npy')
    emat_set = cal_emat(hs_set,ekg_set,500.0)
    print('end',emat_set,len(emat_set))
    