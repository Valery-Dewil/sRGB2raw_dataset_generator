import iio
import numpy as np
import matplotlib.pyplot as plt
import os

plt.show()
plt.ion()

##
c0  = [] #first channel
c1  = [] #second channel
c2  = [] #third channel
c3  = [] #fourth channel
all_chan = [] #all the channels

iso=12800

for scene in range(1,11):
    print(scene)
    for i in range(50):
        try:        
            a = iio.read("scene"+str(scene)+"/iso"+str(iso)+"/frame"+str(i)+".tiff".format(i))
        except:        
            a = iio.read("scene"+str(scene)+"/iso"+str(iso)+"/{:03d}.tiff".format(i))
        c0.append(a[:,:,0].flatten())
        c1.append(a[:,:,1].flatten())
        c2.append(a[:,:,2].flattetn())
        c3.append(a[:,:,3].flatten())
        all_chan.append(a.flatten())

## Compute histograms
histo0, bins0 = np.histogram(c0,4095-240)
histo1, bins1 = np.histogram(c1,4095-240)
histo2, bins2 = np.histogram(c2,4095-240)
histo3, bins3 = np.histogram(c3,4095-240)

histo, bins = np.histogram(all_chan,4095)

np.save("histo0.npy", histo0)
np.save("histo1.npy", histo1)
np.save("histo2.npy", histo2)
np.save("histo3.npy", histo3)
np.save("histo_iso3200.npy", histo)
np.save("histo_iso12800.npy", histo)

np.save("bins0.npy", bins0)
np.save("bins1.npy", bins1)
np.save("bins2.npy", bins2)
np.save("bins3.npy", bins3)
np.save("bins_iso3200.npy", bins)
np.save("bins_iso12800.npy", bins)


## Plot histograms
histo0 = np.load("histo0.npy")
histo1 = np.load("histo1.npy")
histo2 = np.load("histo2.npy")
histo3 = np.load("histo3.npy")
histo = np.load("histo_iso3200.npy")
histo = np.load("histo_iso12800.npy")

bins0 = np.load("bins0.npy")
bins1 = np.load("bins1.npy")
bins2 = np.load("bins2.npy")
bins3 = np.load("bins3.npy")
bins = np.load("bins_iso3200.npy")
bins = np.load("bins_iso12800.npy")

plt.figure(1)
plt.clf()
plt.subplot(2,2,1)
plt.bar(bins0[:-1], histo0, width = 0.3)
plt.subplot(2,2,2)
plt.bar(bins1[:-1], histo1, width = 0.3)
plt.subplot(2,2,3)
plt.bar(bins2[:-1], histo2, width = 0.3)
plt.subplot(2,2,4)
plt.bar(bins3[:-1], histo3, width = 0.3)


plt.figure(2)
plt.clf()
plt.bar(bins[:-1], histo, width =1)



## Find percentiles

total = np.sum(histo)
lbda = 100 / total
histo = histo * lbda
cumul = np.cumsum(histo)

centered_bins = (bins[1:] + bins[:-1])  / 2


def find_percentile(p, cumul, centered_bins):
    k = 0 
    percentile = cumul[0]
    while percentile < p:
        k = k+1
        percentile = cumul[k]
    #linearisation
    a = cumul[k] - cumul[k-1]
    b = cumul[k] - a*k
    le_k = (p-b) / a
    
    alpha = centered_bins[k] - centered_bins[k-1]
    beta = centered_bins[k] - alpha * k
        
    return alpha*le_k+beta
    
print(find_percentile(99,cumul, centered_bins))
print(find_percentile(98,cumul, centered_bins))
print(find_percentile(2,cumul, centered_bins))
print(find_percentile(1,cumul, centered_bins))
