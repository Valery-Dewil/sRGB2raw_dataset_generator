import argparse
import iio
import numpy as np
from scipy.stats import poisson, truncnorm
import torch
import torch.distributions as tdist



## Parse arguments
parser = argparse.ArgumentParser(description="Compute the noise and clean raw data from the RGB")
parser.add_argument("--sigma", type=float, default=0.015, help='std of the truncated gaussian')
parser.add_argument("--close", type=str  , default="center")
parser.add_argument("--p"    , type=int  , default=1)
parser.add_argument("--first", type=int  , default=1)
parser.add_argument("--last" , type=int  , default=1)
parser.add_argument("--step", type=int  , default=3)
parser.add_argument("--BL"  , type=int  , default=240)
parser.add_argument("--WL"  , type=int  , default=4095)
parser.add_argument("--input", type=str  , default="")
parser.add_argument("--output", type=str  , default="")

args = parser.parse_args()



## Functions
def alea(a, b):
    return np.random.rand()*(b-a) + a

def trunc_gauss(a, b, mu, sigma):
    """
    a and b are the borders
    mu is the mean
    sigma is the std
    """
    alpha = (a - mu) / sigma
    beta  = (b - mu) / sigma
    return truncnorm(a=alpha, b=beta, loc=mu, scale=sigma).rvs()

def random_ccm(distrib, close='center', sigma=1):
  """
  Generates random RGB -> Camera color correction matrices.
  
  distrib is either 'uniform' (for the training) or 'truncated_gaussian' for testing
  close is either 'left' or 'center' or 'right', it represent whether the mean of the truncated gaussian is close to the left border, center (same mean as the uniform used for training) or close to the right. Ex: during training we have X ~ U(µ, µ-l, µ-l). 'left' will lead to a mean close to (µ-l), 'center' exactly equal to µ and 'right' close to (µ+l).
  sigma: not used for the Uniform distrib, for the truncated distrib, this is its std.
  """
  ### Takes a random convex combination of XYZ -> Camera CCMs.
  ##xyz2cams = [[[1.0234, -0.2969, -0.2266],
  ##             [-0.5625, 1.6328, -0.0469],
  ##             [-0.0703, 0.2188, 0.6406]]
  ##             ,
  ##            [[0.4913, -0.0541, -0.0202],
  ##             [-0.613, 1.3513, 0.2906],
  ##             [-0.1564, 0.2151, 0.7183]],

  ##            [[0.838, -0.263, -0.0639],
  ##             [-0.2887, 1.0725, 0.2496],
  ##             [-0.0627, 0.1427, 0.5438]],

  ##            [[0.6596, -0.2079, -0.0562],
  ##             [-0.4782, 1.3016, 0.1933],
  ##             [-0.097, 0.1581, 0.5181]]]
  #if distrib=='uniform':
  #    xyz2cam = torch.FloatTensor([[alea(0.4913, 1.0234), alea(-0.2969, -0.0541), alea(-0.2266, -0.0202)],
  #                                  [alea(-0.613, -0.2887), alea(1.0725, 1.6328), alea(-0.0469, 0.2906)],
  #                                  [alea(-0.1564, -0.0627), alea(0.1427, 0.2188), alea(0.5181, 0.7183)]])
  #elif distrib=='truncated_gaussian':
  #   if close=='left':
  #      xyz2cam = torch.FloatTensor([[trunc_gauss(0.4913, 1.0234, 0.55, sigma), trunc_gauss(-0.2969, -0.0541, -0.24, sigma), trunc_gauss(-0.2266, -0.0202, -0.18, sigma)],
  #                                  [trunc_gauss(-0.613, -0.2887, -0.55, sigma), trunc_gauss(1.0725, 1.6328, 1.15, sigma), trunc_gauss(-0.0469, 0.2906, 0.05, sigma)],
  #                                  [trunc_gauss(-0.1564, -0.0627, -0.12, sigma), trunc_gauss(0.1427, 0.2188, 0.155, sigma), trunc_gauss(0.5181, 0.7183, 0.56, sigma)]])
  #   elif close=='center':
  #      xyz2cam = torch.FloatTensor([[trunc_gauss(0.4913, 1.0234, 0.75735, sigma), trunc_gauss(-0.2969, -0.0541, -0.17205, sigma), trunc_gauss(-0.2266, -0.0202, -0.1234, sigma)],
  #                                  [trunc_gauss(-0.613, -0.2887, -0.45085, sigma), trunc_gauss(1.0725, 1.6328, 1.35265, sigma), trunc_gauss(-0.0469, 0.2906, 0.12185, sigma)],
  #                                  [trunc_gauss(-0.1564, -0.0627, -0.10955, sigma), trunc_gauss(0.1427, 0.2188, 0.18075, sigma), trunc_gauss(0.5181, 0.7183, 0.6182, sigma)]])
  #   elif close=='right':
  #      xyz2cam = torch.FloatTensor([[trunc_gauss(0.4913, 1.0234, 0.95, sigma), trunc_gauss(-0.2969, -0.0541, -0.1, sigma), trunc_gauss(-0.2266, -0.0202, -0.05, sigma)],
  #                                  [trunc_gauss(-0.613, -0.2887, -0.35, sigma), trunc_gauss(1.0725, 1.6328, 1.5, sigma), trunc_gauss(-0.0469, 0.2906, 0.22, sigma)],
  #                                  [trunc_gauss(-0.1564, -0.0627, -0.09, sigma), trunc_gauss(0.1427, 0.2188, 0.2, sigma), trunc_gauss(0.5181, 0.7183, 0.65, sigma)]])
  #      
  ##num_ccms = len(xyz2cams)
  ##xyz2cams = torch.FloatTensor(xyz2cams)
  ##weights  = torch.FloatTensor(num_ccms, 1, 1).uniform_(1e-8, 1e8)
  ##weights_sum = torch.sum(weights, dim=0)
  ##xyz2cam = torch.sum(xyz2cams * weights, dim=0) / weights_sum

  ## Multiplies with RGB -> XYZ to get RGB -> Camera CCM.
  #rgb2xyz = torch.FloatTensor([[0.4124564, 0.3575761, 0.1804375],
  #                             [0.2126729, 0.7151522, 0.0721750],
  #                             [0.0193339, 0.1191920, 0.9503041]])
  #rgb2cam = torch.mm(xyz2cam, rgb2xyz)

  ## Normalizes each row.
  #rgb2cam = rgb2cam / torch.sum(rgb2cam, dim=-1, keepdim=True)
  rgb2cam = torch.tensor([[0.6218, 0.2241, 0.1541], [0.1064, 0.6482, 0.2454], [0.084, 0.2383, 0.6777]]).cuda()
  return rgb2cam


def random_gains(n, red_gain, blue_gain):
  """Generates random gains for brightening and white balance."""
  # RGB gain represents brightening.
  #n        = tdist.Normal(loc=torch.tensor([0.8]), scale=torch.tensor([0.1])) 
  #rgb_gain = 1.0 / n.sample()
        
  # RGB gain represents brightening.
  n = torch.tensor([trunc_gauss(0.5, 1.1, 0.6, sigma)]) 
  # Red and blue gains represent white balance.
  red_gain  =  torch.tensor([trunc_gauss(1.9, 2.4, 1.98, sigma)])
  blue_gain =  torch.tensor([trunc_gauss(1.5, 1.9, 1.55, sigma)])

  rgb_gain = 1.0 / n

  return rgb_gain, red_gain, blue_gain


def inverse_smoothstep(image):
  """Approximately inverts a global tone mapping curve."""
  image = image.permute(1, 2, 0) # Permute the image tensor to HxWxC format from CxHxW format
  image = torch.clamp(image, min=0.0, max=1.0)
  out   = 0.5 - torch.sin(torch.asin(1.0 - 2.0 * image) / 3.0) 
  out   = out.permute(2, 0, 1) # Re-Permute the tensor back to CxHxW format
  return out


def gamma_expansion(image):
  """Converts from gamma to linear space."""
  # Clamps to prevent numerical instability of gradients near zero.
  image = image.permute(1, 2, 0) # Permute the image tensor to HxWxC format from CxHxW format
  out   = torch.clamp(image, min=1e-8) ** 2.2
  out   = out.permute(2, 0, 1) # Re-Permute the tensor back to CxHxW format
  return out


def apply_ccm(image, ccm):
  """Applies a color correction matrix."""
  image = image.permute(1, 2, 0) # Permute the image tensor to HxWxC format from CxHxW format
  shape = image.size()
  image = torch.reshape(image, [-1, 3])
  image = torch.tensordot(image, ccm, dims=[[-1], [-1]])
  out   = torch.reshape(image, shape)
  out   = out.permute(2, 0, 1) # Re-Permute the tensor back to CxHxW format
  return out


def safe_invert_gains(image, rgb_gain, red_gain, blue_gain):
  """Inverts gains while safely handling saturated pixels."""
  image = image.permute(1, 2, 0) # Permute the image tensor to HxWxC format from CxHxW format
  gains = torch.stack((1.0 / red_gain, torch.tensor([1.0]), 1.0 / blue_gain)) / rgb_gain
  gains = gains.squeeze()
  gains = gains[None, None, :]
  gains = gains.cuda()
  # Prevents dimming of saturated pixels by smoothly masking gains near white.
  gray  = torch.mean(image, dim=-1, keepdim=True)
  #inflection = 0.9
  inflection = 0.9999999999 # I remove the part which causes discontinuities
  mask  = (torch.clamp(gray - inflection, min=0.0) / (1.0 - inflection)) ** 2.0
  mask = mask.cuda()


  #Saturation appears when gains greater than 1 
 


  safe_gains = torch.max(mask + (1.0 - mask) * gains, gains)
  out   = image * safe_gains
  out   = out.permute(2, 0, 1) # Re-Permute the tensor back to CxHxW format
  return out


def mosaic(image):
  image = image.permute(1, 2, 0) # Permute the image tensor to HxWxC format from CxHxW format
  shape = image.size()
  chan0 = image[0::2, 0::2, 1]
  chan1 = image[0::2, 1::2, 2]
  chan2 = image[1::2, 0::2, 0]
  chan3 = image[1::2, 1::2, 1]
  out  = torch.stack((chan0, chan1, chan2, chan3), dim=-1).cuda()
  out  = torch.reshape(out, (shape[0] // 2, shape[1] // 2, 4))
  out  = out.permute(2, 0, 1) # Re-Permute the tensor back to CxHxW format
  return out


def unprocess(stack):
    """Unprocesses an image from sRGB to realistic raw data.
    stack is a stack of shape 3x5, H, W"""
    
    # Randomly creates image metadata.
    rgb2cam = random_ccm()
    cam2rgb = torch.inverse(rgb2cam)
    rgb_gain, red_gain, blue_gain = random_gains()

    _, H, W = stack.shape
    result = torch.zeros(20, H//2, W//2)
    
    for n in range(5):
        # Approximately inverts global tone mapping.
        image  = inverse_smoothstep(stack[3*n:3*(n+1),:,:])
        # Inverts gamma compression.
        image = gamma_expansion(image)
        # Inverts color correction.
        image = apply_ccm(image, rgb2cam)
        # Approximately inverts white balance and brightening.
        image = safe_invert_gains(image, rgb_gain, red_gain, blue_gain)
        # Clips saturated pixels.
        image = torch.clamp(image, min=0.0, max=1.0)
        # Applies a Bayer mosaic.
        image = mosaic(image)

        result[4*n:4*(n+1),:,:] = image.clone()
    
    return result



def rgb2raw(rgb_batch):
    B, C, H, W = rgb_batch.shape
    raw_batch = torch.zeros(B, 20, H//2, W//2)
    for b in range(B):
        raw_batch[b,:,:,:] = unprocess(rgb_batch[b,:,:,:])

    return raw_batch

def single_image_rgb2raw(img, rgb_gain, red_gain, blue_gain):
    # Randomly creates image metadata.
    rgb2cam = torch.tensor([[0.6218, 0.2241, 0.1541], [0.1064, 0.6482, 0.2454], [0.084, 0.2383, 0.6777]]).cuda()

    img = img.transpose(2,0,1) / 255
    img = torch.tensor(img).cuda()
    
    # Approximately inverts global tone mapping.
    image  = inverse_smoothstep(img)
    # Inverts gamma compression.
    image = gamma_expansion(image)
    # Inverts color correction.
    image = apply_ccm(image, rgb2cam)
    # Approximately inverts white balance and brightening.
    image = safe_invert_gains(image, rgb_gain, red_gain, blue_gain)
    # Clips saturated pixels.
    image = torch.clamp(image, min=0.0, max=1.0)
    # Applies a Bayer mosaic.
    image = mosaic(image)  
    
    return image.permute(1,2,0).cpu().numpy()




def add_Poisson_Gaussian_noise(img):
    a = 13.486051 #sigma shot noise iso 6400
    b = 130.818508 #sigma read noise iso 6400
    img = torch.tensor(img).permute(2,0,1).unsqueeze(0).cuda()
    _, _, H, W = img.shape
    poisson_noisy_img = torch.poisson((img-240)/a)*a
    gaussian_noise = np.sqrt(b)*torch.randn(H, W)
    gaussian_noise = gaussian_noise.cuda()
    img = torch.clamp(240 + poisson_noisy_img + gaussian_noise, 0, 4095)
    img = img.squeeze().permute(1,2,0).cpu().numpy()
    return img



## Script
sigma = 0.1
for seq in range(240):
    n = torch.FloatTensor([trunc_gauss(0, 1.0, 0.8, 0.1)]) #truncated Gaussian to prevent from saturation
    ## Red and blue gains represent white balance.
    red_gain  =  torch.FloatTensor(1).uniform_(1.90, 2.4)
    blue_gain =  torch.FloatTensor(1).uniform_(1.5, 1.9)
    rgb_gain = 1.0 / n
    
    print(seq)

    for i in range(args.first, args.last, args.step):

        img = iio.read(args.input%(seq, i))
        unprocess_img = single_image_rgb2raw(img, rgb_gain, red_gain, blue_gain)
        unprocess_img = unprocess_img* (args.WL-args.BL) + args.BL 
        unprocess_img = np.clip(unprocess_img, 0,2634) #first pass: clipping to the 99-percentile 
        unprocess_img = (4075-268)*(unprocess_img-248.5) / (2628-248.5) + 268 #second pass: affine mapping to CRVD

        iio.write(args.output%(seq, i), unprocess_img)

#for seq in range(30):
#    n = torch.FloatTensor([trunc_gauss(0, 1.0, 0.8, 0.1)]) #truncated Gaussian to prevent from saturation
#    ## Red and blue gains represent white balance.
#    red_gain  =  torch.FloatTensor(1).uniform_(1.90, 2.4)
#    blue_gain =  torch.FloatTensor(1).uniform_(1.5, 1.9)
#    rgb_gain = 1.0 / n
#    
#    print(seq)
#
#    for i in range(args.first, args.last, args.step):
#
#        img = iio.read(args.input%(seq, i))
#        unprocess_img = single_image_rgb2raw(img, rgb_gain, red_gain, blue_gain)
#        unprocess_img = unprocess_img* (args.WL-args.BL) + args.BL
#        unprocess_img = np.clip(unprocess_img, 0,2634) #first pass: clipping to the 99-percentile 
#        unprocess_img = (4075-268)*(unprocess_img-248.5) / (2628-248.5) + 268 #second pass: affine mapping to CRVD
#
#        iio.write(args.output%(seq, i), unprocess_img)

