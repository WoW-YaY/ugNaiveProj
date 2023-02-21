import numpy as np
from scipy import signal as signal

data_No = './data/a4-5'

# Splitting data with 50% overlap
# output: 6 sheets with 6*128
def spl(data):
    ax = np.zeros((6,128))
    for i in range(6):
        ax[i,:] = data[0,i*64:i*64+128]
    ay = np.zeros((6,128))
    for i in range(6):
        ay[i,:] = data[1,i*64:i*64+128]
    az = np.zeros((6,128))
    for i in range(6):
        az[i,:] = data[2,i*64:i*64+128]
    gx = np.zeros((6,128))
    for i in range(6):
        gx[i,:] = data[3,i*64:i*64+128]
    gy = np.zeros((6,128))
    for i in range(6):
        gy[i,:] = data[4,i*64:i*64+128]
    gz = np.zeros((6,128))
    for i in range(6):
        gz[i,:] = data[5,i*64:i*64+128]
    return ax,ay,az,gx,gy,gz

# filtering
def filtering(data):
    rows = data.shape[0]
# median filtering
    for i in range(rows):
        data[i,:] = signal.medfilt(data[i,:],3)
# butterworth filtering
    b, a = signal.butter(3, 0.8, 'lowpass')
    for i in range(rows):
        data[i,:] = signal.filtfilt(b,a,data[i,:])
    return data


def generate_feature(data):
    f = np.zeros((6,1))
    f[:,0] = np.mean(data,axis=1)
    return f

df = np.loadtxt(data_No+'.csv',skiprows=1,usecols = (1,2,3,4,5,6),delimiter = ',')
filtered = filtering(df.T)
ax,ay,az,gx,gy,gz = spl(filtered)

fax = generate_feature(ax)
fay = generate_feature(ay)
faz = generate_feature(az)
fgx = generate_feature(gx)
fgy = generate_feature(gy)
fgz = generate_feature(gz)
F = np.hstack((fax,fay,faz,fgx,fgy,fgz))


np.savetxt(data_No+'features.csv',F,delimiter = ',')

