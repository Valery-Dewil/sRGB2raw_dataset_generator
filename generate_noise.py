import iio
import numpy as np


## TRAIN ISO3200
for seq in range(240):
    print("train", seq)
    for i in range(0, 498+3, 3):
        #read gt
        gt = iio.read("train/gt_iso3200/{:03d}/{:08d}.tiff".format(seq, i))
        n, m, c = gt.shape
        
        #add noise to the GT
        noisy = gt + np.sqrt( (8.0034*gt-2043.51144).clip(0, np.inf) )*np.random.randn(n,m,c)
        
        #store the output
        iio.write("train/noisy_iso3200/{:03d}/{:08d}.tiff".format(seq, i), noisy)

## VALIDATION ISO3200
for seq in range(30):
    print("validation", seq)
    for i in range(0, 498+3, 3):
        #read gt
        gt = iio.read("validation/gt_iso3200/{:03d}/{:08d}.tiff".format(seq, i))
        n, m, c = gt.shape

        #add noise to the GT
        noisy = gt + np.sqrt( (8.0034*gt-2043.51144).clip(0, np.inf) )*np.random.randn(n,m,c)
        
        #store the output
        iio.write("validation/noisy_iso3200/{:03d}/{:08d}.tiff".format(seq, i), noisy)

