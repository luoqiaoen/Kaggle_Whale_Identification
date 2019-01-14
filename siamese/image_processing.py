from os.path import isfile
from PIL import Image as pil_image
from tqdm import tqdm
from pandas import read_csv
import pickle
import numpy as np
from imagehash import phash
from math import sqrt
import matplotlib.pyplot as plt
import random
import shutil
import pandas as pd
import sys
import keras
from keras import backend as K
from keras.preprocessing.image import img_to_array,array_to_img
from scipy.ndimage import affine_transform
import glob

def expand_path(p):
    if isfile('../large_dataset/whale_files/selected_train/' + p): return '../large_dataset/whale_files/selected_train/' + p
    if isfile('../large_dataset/whale_files/test/' + p): return '../large_dataset/whale_files/test/' + p
    return p

def match(h1,h2):
    for p1 in h2ps[h1]:
        for p2 in h2ps[h2]:
            i1 =  pil_image.open(expand_path(p1))
            i2 =  pil_image.open(expand_path(p2))
            if i1.mode != i2.mode or i1.size != i2.size: return False
            a1 = np.array(i1)
            a1 = a1 - a1.mean()
            a1 = a1/sqrt((a1**2).mean())
            a2 = np.array(i2)
            a2 = a2 - a2.mean()
            a2 = a2/sqrt((a2**2).mean())
            a  = ((a1 - a2)**2).mean()
            if a > 0.1: return False
    return True

def show_whale(imgs, per_row=2):
    n         = len(imgs)
    rows      = (n + per_row - 1)//per_row
    cols      = min(per_row, n)
    fig, axes = plt.subplots(rows,cols, figsize=(24//per_row*cols,24//per_row*rows))
    for ax in axes.flatten(): ax.axis('off')
    for i,(img,ax) in enumerate(zip(imgs, axes.flatten())): ax.imshow(img.convert('RGB'))
    plt.savefig('show_whale')

# For each images id, select the prefered image
def prefer(ps):
    if len(ps) == 1: return ps[0]
    best_p = ps[0]
    best_s = p2size[best_p]
    for i in range(1, len(ps)):
        p = ps[i]
        s = p2size[p]
        if s[0]*s[1] > best_s[0]*best_s[1]: # Select the image with highest resolution
            best_p = p
            best_s = s
    return best_p


tagged = dict([(p,w) for _,p,w,_,_,_,_ in read_csv('../large_dataset/whale_files/train_final.csv').to_records()]) #training
submit = [p for _,p,_,_,_,_,_ in read_csv('../large_dataset/whale_files/test_final.csv').to_records()] #testing
join   = list(tagged.keys()) + submit #total data
if isfile('../large_dataset/whale_files/p2size.pickle'):
    with open('../large_dataset/whale_files/p2size.pickle', 'rb') as f:
        p2size = pickle.load(f)
else:
    p2size = {} #size of image
    for p in tqdm(join):
        size      = pil_image.open(expand_path(p)).size
        p2size[p] = size
        pickle.dump(p2size, open( "../large_dataset/whale_files/p2size.pickle", "wb" ) )

if isfile('../large_dataset/whale_files/p2h.pickle'):
    with open('../large_dataset/whale_files/p2h.pickle', 'rb') as f:
        p2h = pickle.load(f)
else:
    # Compute phash for each image in the training and test set.
    p2h = {}
    for p in tqdm(join):
        img    = pil_image.open(expand_path(p))
        h      = phash(img)
        p2h[p] = h
    # Find all images associated with a given phash value.
    h2ps = {}
    for p,h in p2h.items():
        if h not in h2ps: h2ps[h] = []
        if p not in h2ps[h]: h2ps[h].append(p)
    for h, ps in h2ps.items():
        while len(ps) > 1:
            print('Images:', ps)
            break

    # Find all distinct phash values
    hs = list(h2ps.keys())

    # If the images are close enough, associate the two phash values (this is the slow part: n^2 algorithm)
    h2h = {}
    for i,h1 in enumerate(tqdm(hs)):
        for h2 in hs[:i]:
            if h1-h2 <= 6 and match(h1, h2):
                s1 = str(h1)
                s2 = str(h2)
                if s1 < s2: s1,s2 = s2,s1
                h2h[s1] = s2

    # Group together images with equivalent phash, and replace by string format of phash (faster and more readable)
    for p,h in p2h.items():
        h = str(h)
        if h in h2h: h = h2h[h]
        p2h[p] = h
    pickle.dump(p2h, open( "../large_dataset/whale_files/p2h.pickle", "wb" ) )


if isfile('../large_dataset/whale_files/h2ps.pickle'):
    with open('../large_dataset/whale_files/h2ps.pickle', 'rb') as f:
        h2ps = pickle.load(f)
else:
    # For each image id, determine the list of pictures
    h2ps = {}
    for p,h in p2h.items():
        if h not in h2ps: h2ps[h] = []
        if p not in h2ps[h]: h2ps[h].append(p)
    pickle.dump(h2ps, open( "../large_dataset/whale_files/h2ps.pickle", "wb" ) )

h2p = {}
for h,ps in h2ps.items(): h2p[h] = prefer(ps)
pickle.dump(h2p, open( "../large_dataset/whale_files/h2p.pickle", "wb" ) )
# Find all the whales associated with new_whale
h2ws = {} #hash to whales
new_whale = 'new_whale'
for p,w in tagged.items():
    if w != new_whale: # Use only identified whales
        h = p2h[p]
        if h not in h2ws: h2ws[h] = []
        if w not in h2ws[h]: h2ws[h].append(w)
for h,ws in h2ws.items():
    if len(ws) > 1:
        known[h] = sorted(ws)
print("number of pics other than new whale: ", len(h2ws))
w2hs = {} #whale to hash
for h,ws in h2ws.items():
    if len(ws) == 1:# Use only unambiguous pictures
        w = ws[0]
        if w not in w2hs: w2hs[w] = []
        if h not in w2hs[w]: w2hs[w].append(h)
for w,hs in w2hs.items():
    if len(hs) > 1:
        w2hs[w] = sorted(hs)
pickle.dump(w2hs, open( "../large_dataset/whale_files/w2hs.pickle", "wb" ) )
#pd.DataFrame.from_dict(data=w2hs, orient='index').to_csv('w2hs.csv', header=False)
print("number of whale ID other than new whale: ", len(w2hs))

# Find the list of training images, keep only whales with at least two images.
train = [] # A list of training image ids
for hs in w2hs.values():
    if len(hs) > 1:
        train += hs
random.shuffle(train)
train_set = set(train)
w2hs_min2 = {} # Associate the image ids from train to each whale id.
for w,hs in w2hs.items():
    for h in hs:
        if h in train_set:
            if w not in w2hs_min2: w2hs_min2[w] = []
            if h not in w2hs_min2[w]: w2hs_min2[w].append(h)
for w,ts in w2hs_min2.items(): w2hs_min2[w] = np.array(ts)

pickle.dump(train, open( "../large_dataset/whale_files/train.pickle", "wb" ) )
pickle.dump(w2hs_min2, open( "../large_dataset/whale_files/w2hs_min2.pickle", "wb" ) )
#pd.DataFrame.from_dict(data=w2hs_min2, orient='index').to_csv('w2hs_min2.csv', header=False)
print("number of unique whales ID with at least two images: ", len(w2hs_min2))

if len(w2hs_min2) == len(w2hs):
    print('Data looks good')
else:
    print("The data set still needs working")


if isfile('../large_dataset/whale_files/bounding-box.pickle'):
    with open('../large_dataset/whale_files/bounding-box.pickle', 'rb') as f:
        p2bb = pickle.load(f)
else:
    path ='../large_dataset/whale_files/'
    filenames = glob.glob(path + "/*.csv")
    p2bb = {}
    for filename in filenames:
        p2bb.update(dict([(p,(xi,yi,xm,ym)) for _,p,_,xi,yi,xm,ym in read_csv(filename).to_records()]))
        pickle.dump(p2bb, open( "../large_dataset/whale_files/bounding-box.pickle", "wb" ) )
#pd.DataFrame.from_dict(data=p2bb, orient='index').to_csv('p2bb.csv', header=False)

img_shape    = (384,384,1) # The image shape used by the model
anisotropy   = 2.15 # The horizontal compression ratio
crop_margin  = 0.05 # The margin added around the bounding box to compensate for bounding box inaccuracy
rotate       = {}
def read_raw_image(p):
    img = pil_image.open(expand_path(p))
    if p in rotate: img = img.rotate(180)
    return img

def build_transform(rotation, shear, height_zoom, width_zoom, height_shift, width_shift):
    """
    Build a transformation matrix with the specified characteristics.
    """
    rotation        = np.deg2rad(rotation)
    shear           = np.deg2rad(shear)
    rotation_matrix = np.array([[np.cos(rotation), np.sin(rotation), 0], [-np.sin(rotation), np.cos(rotation), 0], [0, 0, 1]])
    shift_matrix    = np.array([[1, 0, height_shift], [0, 1, width_shift], [0, 0, 1]])
    shear_matrix    = np.array([[1, np.sin(shear), 0], [0, np.cos(shear), 0], [0, 0, 1]])
    zoom_matrix     = np.array([[1.0/height_zoom, 0, 0], [0, 1.0/width_zoom, 0], [0, 0, 1]])
    shift_matrix    = np.array([[1, 0, -height_shift], [0, 1, -width_shift], [0, 0, 1]])
    return np.dot(np.dot(rotation_matrix, shear_matrix), np.dot(zoom_matrix, shift_matrix))

def read_cropped_image(p, augment):
    """
    @param p : the name of the picture to read
    @param augment: True/False if data augmentation should be performed
    @return a numpy array with the transformed image
    """
    # If an image id was given, convert to filename
    if p in h2p: p = h2p[p]
    size_x,size_y = p2size[p]

    # Determine the region of the original image we want to capture based on the bounding box.
    x0,y0,x1,y1   = p2bb[p]
    if p in rotate: x0, y0, x1, y1 = size_x - x1, size_y - y1, size_x - x0, size_y - y0
    dx            = x1 - x0
    dy            = y1 - y0
    x0           -= dx*crop_margin
    x1           += dx*crop_margin + 1
    y0           -= dy*crop_margin
    y1           += dy*crop_margin + 1
    if (x0 < 0     ): x0 = 0
    if (x1 > size_x): x1 = size_x
    if (y0 < 0     ): y0 = 0
    if (y1 > size_y): y1 = size_y
    dx            = x1 - x0
    dy            = y1 - y0
    if dx > dy*anisotropy:
        dy  = 0.5*(dx/anisotropy - dy)
        y0 -= dy
        y1 += dy
    else:
        dx  = 0.5*(dy*anisotropy - dx)
        x0 -= dx
        x1 += dx

    # Generate the transformation matrix
    trans = np.array([[1, 0, -0.5*img_shape[0]], [0, 1, -0.5*img_shape[1]], [0, 0, 1]])
    trans = np.dot(np.array([[(y1 - y0)/img_shape[0], 0, 0], [0, (x1 - x0)/img_shape[1], 0], [0, 0, 1]]), trans)
    if augment:
        trans = np.dot(build_transform(
            random.uniform(-5, 5),
            random.uniform(-5, 5),
            random.uniform(0.8, 1.0),
            random.uniform(0.8, 1.0),
            random.uniform(-0.05*(y1 - y0), 0.05*(y1 - y0)),
            random.uniform(-0.05*(x1 - x0), 0.05*(x1 - x0))
            ), trans)
    trans = np.dot(np.array([[1, 0, 0.5*(y1 + y0)], [0, 1, 0.5*(x1 + x0)], [0, 0, 1]]), trans)

    # Read the image, transform to black and white and comvert to numpy array
    img   = read_raw_image(p).convert('L')
    img   = img_to_array(img)

    # Apply affine transformation
    matrix = trans[:2,:2]
    offset = trans[:2,2]
    img    = img.reshape(img.shape[:-1])
    img    = affine_transform(img, matrix, offset, output_shape=img_shape[:-1], order=1, mode='constant', cval=np.average(img))
    img    = img.reshape(img_shape)

    # Normalize to zero mean and unit variance
    img  -= np.mean(img, keepdims=True)
    img  /= np.std(img, keepdims=True) + K.epsilon()
    return img

def read_for_training(p):
    """
    Read and preprocess an image with data augmentation (random transform).
    """
    return read_cropped_image(p, True)

def read_for_validation(p):
    """
    Read and preprocess an image without data augmentation (use for testing).
    """
    return read_cropped_image(p, False)

#p = list(tagged.keys())[192]
#imgs = [
#    read_raw_image(p),
#    array_to_img(read_for_validation(p)),
#    array_to_img(read_for_training(p))
#]
#show_whale(imgs, per_row=3)
