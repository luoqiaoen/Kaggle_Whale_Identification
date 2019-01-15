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
import keras

def expand_path(p):
    if isfile('../large_dataset/whale_files/train/' + p): return '../large_dataset/whale_files/train/' + p
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


tagged = dict([(p,w) for _,p,w in read_csv('../large_dataset/whale_files/train.csv').to_records()]) #training
submit = [p for _,p,_ in read_csv('../large_dataset/whale_files/sample_submission.csv').to_records()] #testing
join   = list(tagged.keys()) + submit #total data
p2size = {} #size of image
for p in tqdm(join):
    size      = pil_image.open(expand_path(p)).size
    p2size[p] = size

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
    # Notice how 25460 images use only 20913 distinct image ids.

for h, ps in h2ps.items():
    while len(ps) > 1:
        print('Images:', ps)
        break
    #fb3879dc7 and 195f7ca52 both in test and train, the other two are actually not the same

h2p = {}
for h,ps in h2ps.items(): h2p[h] = prefer(ps)

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
print("number of whale ID other than new whale: ", len(w2hs))
#pd.DataFrame.from_dict(data=w2hs, orient='index').to_csv('w2hs.csv', header=False)

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

#pd.DataFrame.from_dict(data=w2hs_min2, orient='index').to_csv('w2hs_min2.csv', header=False)
print("number of unique whales ID with at least two images: ", len(w2hs_min2))
