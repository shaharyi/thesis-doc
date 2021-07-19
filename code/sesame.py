# -*- coding: utf-8 -*-
"""
SESAME.PY
3D reconstruction from cross-sections

Requirements:
EPD academic version (for numpy, scipy, mayavi, sklearn) with Python 2.7
http://www.enthought.com/repo/.hidden_epd_installers
http://www.enthought.com/products/edudownload.php

PyGame 1.9
http://pygame.org/download.shtml

OpenCV 2.4
http://opencv.willowgarage.com/wiki/InstallGuide
install, then copy
c:/opencv/build/python/2.7/cv2.pyd
to
D:/Python27/Lib/site-packages
and
add the DLLs' path to your PATH:
C:/opencv/build/x86/vc10/bin

if VC++2010 not installed, install VC++ 2010 x64 Runtime:
http://www.microsoft.com/en-us/download/details.aspx?id=14632
or x86 for 32bit:
http://www.microsoft.com/en-us/download/details.aspx?id=5555

pydicom
http://code.google.com/p/pydicom/downloads/list
use installer or
extract source and copy folder "dicom" to
D:/Python27/Lib/site-packages

psutil
http://code.google.com/p/psutil/
for monitoring available RAM

Written by
Shahar Yifrah
May, 2012
"""

import time
import sys
import os
import copy
import multiprocessing
import warnings
import cPickle
import cProfile
import heapq
import operator
import logging
from copy import deepcopy
from pprint import pformat
from pdb import set_trace, pm
from math import log, sqrt, atan
from collections import OrderedDict, deque
from itertools import *
from logging import debug

import dicom
import pygame
import pygame.gfxdraw
from pygame.locals import *
from numpy import *
import numpy
from numpy.linalg import *
from scipy import interpolate, signal, ndimage, io, misc
from scipy.ndimage import interpolation
from scipy.ndimage.morphology import distance_transform_cdt,distance_transform_edt
from scipy.ndimage.filters import generic_gradient_magnitude, prewitt, laplace
from scipy.spatial import KDTree, distance
import scipy.linalg
import psutil

import cv2
import cv2.cv as cv
from mayavi import mlab

#from sklearn.cluster.dbscan_sparse import DBSCAN
#from dbscan_sparse import DBSCAN

#from sklearn import metrics

HELP		    = 'Help ESCape Autoload XYZ rt/lt=slices Lockdown Save ENTER=3D' \
                      'Del Clr Rbf-2d Normals Gradients XYZ up/dn=scribbles ' \
                      'Worst randOm p/PgUp/PgDn=grades Matlabload ' \
                      'k=toggle-grades i=toggle-rbf t=calc_rbf_grades'

AUTO_EVENTS_WHERE   = 0
AUTO_EVENTS_RANDOM  = 0
AUTO_LOCKDOWN       = False
AUTO_EVENTS_LOAD    = False
AUTO_EVENTS_INIT_RANDOM = False #(AUTO_EVENTS_RANDOM>0)
RANDOM_USE_CENTROID = False
RANDOM_USE_BOX_CENTER = False
RANDOM_ON_AXIS = False
SAVE_RBF        = True
PRECALC_GRADIENTS = True
SHOW_QUIVER     = False

#'sphere' #'ninja_twisted' #'letterS' #'shape_letterC' #'shape_horseshoe' #
DATA_FUNC = '' #'shape_blimps'
DATA_SIZE = 128

EXTRACT_SLICE_MULTICORE = False
SUPPORT_RANDOM_ALIGNED  = False

FONT_SIZE = 28

LOAD_SCRIBBLE       = 0
LOAD_SCRIBBLE2      = 0
#   '' #'s3_ventricle2.npy' #'s3_rbf_256.npy' #'bamba4.npy' #'ninja128_worm.npy'
NUMPY_DATA_FILENAME = '' #'Cerebellum_box_gt.npy' #rightvent_box.npy' #'s3_ventricle2_iso_gt.npy'
GROUND_TRUTH_DATA   = ''
#cache only applies to self-produced data file, not Matlab input for example
CACHE_DATA_FILENAME = 'cached_data.npy'
#MATLAB_DATA_FILENAME= '../data_synth/bamba.mat'
#MATLAB_DATA_FILENAME= '../data_synth/Cube_s.mat'
MATLAB_DATA_FILENAME = ''
MHD_DATA_FILENAME   = ''
#IMAGE_FILENAME     = '../data/Head1/MR000057.dcm'
##DATA                = '../SE000002'
##DATA                = '../17'
DATA                = 'D:/Shahar/thesis/s2/s2'           #'../s3'
REF_MARKS_MATFILE   = 'Cerebellum_box.mat' #'RightVent_box.mat'  #'Ventricle2_box.mat'
RBF_2D_REDUCTION    = True
IMAGE               = 'MRbrain.%i'
#DATA               = '../ct_data'
#IMAGE              = 'CThead.%i'
#DATA               = '../data'
#IMAGE              = 'MR0000%02i.dcm'
#IMAGE_FILENAME     = '../mri_data/MRbrain.1'
IMAGE_FILENAME      = 'MRbrain.50'
VIEW_FACTOR         = 4
WHITE_RGB           = (255,255,255)
MATLAB_SHRINK_FACTOR= 2
DOWNSAMPLE          = 1 # .5 #1 #1.5
RESAMPLE_XY         = 1 #DOWNSAMPLE*2.133
RESAMPLE_Z          = DOWNSAMPLE
RESAMPLE_ZXY        = (RESAMPLE_Z, RESAMPLE_XY, RESAMPLE_XY)
RESAMPLE            = False
ELIM_TOO_CLOSE      = True
GRADE_BY_NN_SIGNATURE = False

ROUND_PRECISION     = 8

PATCH_LEN           = 4
PATCH_SIZE          = PATCH_LEN*ones(3,int)
SIGNATURE_CUTOFF    = 0.0

GRADE_THRESHOLD     = 200
#pixels
MIN_SLICE_MASS      = 64

MAX_PCA_PTS = 1024

GRAY    = 0xF0F0F0
WHITE   = 0xFFFFFF
BLACK   = 0x000000
RED     = 0xFF0000
GREEN   = 0x00FF00
BLUE    = 0x0000FF
CYAN    = 0x00FFFF
MAGENTA = 0xFF00FF

NORM_MAX_LEN        = 40
GRAD_COLOR          = WHITE #0x7F7FFF
GRAD_LOC_COLOR      = CYAN #BLUE
GRAD_LOC_RADIUS     = 3

DEBUG               = 1

RBF_NORMAL_RADIUS   = 3
TOO_CLOSE_DISTANCE  = RBF_NORMAL_RADIUS
RBF_GRID_SIZE       = 32
RBF_FULL            = 1
RBF_BBOX_ONLY       = 0
RBF_ALPHA           = 67
RBF_INSIDE_COLOR    = BLUE
RBF_SHOULDER        = PATCH_LEN * array([1,1,1])
RBF_GRADIENT_THRESHOLD = 30
RBF_THRESHOLD       = .3
RBF_THRESHOLD_SURF  = .1
DATA_BOUNDARY_THRESH= .8
MAX_RBF_POINTS      = 1000
RBF_REDUCTION_ACCURACY = .01

GRADES_ALPHA        = 67
REF_ALPHA           = 67
REF_MARKS_AXIS      = 0

SAMPLE_INTERVAL     = 2
CONTOUR_THRESH      = 0.0 #0.01
INSIDE_BRIGHTNESS   = 255 #255 is the normal

GRADIENT_WEIGHT     = 0.5

#Noise Robust Gradient Operators
#http://www.holoborodko.com/pavel/image-processing/edge-detection/
#smoothing operator, from:
SMOOTH = [ 1,  4,  6,  4,  1]
#derivative operator
DERIV  = [-1, -4, -5,  0,  5,  4,  1]
#mul by 1/512 to normalize
GRAD_2D_X = outer(DERIV,SMOOTH)
GRAD_2D_Y = outer(SMOOTH,DERIV)
GRAD_OP_NORMALIZER = 512
GRAD_3D_NORMALIZER = 512*4

MB = 1024*1024

#vectors for scanning point neighbors
NEIGHBORS_STEPS_CLOCKWISE = array([(-1,-1)] + 2*[(0,1)]+2*[(1,0)]+2*[(0,-1)]+2*[(-1,0)])
NEIGHBORS_OFFSETS_CLOCKWISE = cumsum(NEIGHBORS_STEPS_CLOCKWISE, 0)


DBSCAN_MIN_SAMPLES  = 10  
DBSCAN_EPS          = 1.5 # allow only "+" neigbors, not diag 

USE_HALF_REF_ONLY = False
TOUCH_ENDING_SLICES = False
START_POS = False
DICOM_NORMALIZE = True
PNG_DATA_DIRNAME = ''
INIT_AXIS_2_KEY = K_y
RESAMPLE_GT = RESAMPLE
SPACING = [1,1,1]

CONFIG    = 'S3_CROP'

if CONFIG=='KNOT':
    NUMPY_DATA_FILENAME = 'trefoil_knot.npy'

if CONFIG=='HEPATIC':
    NUMPY_DATA_FILENAME = 'FJ2415_BP5820_FMA14339_Left hepatic vein_Filled.npy'
    
if CONFIG=='L1':
    NUMPY_DATA_FILENAME = 'FJ3157_BP8948_FMA13072_First lumbar vertebra_Filled.npy'
    
if CONFIG=='CALCANEUS':
    NUMPY_DATA_FILENAME = 'FJ3360_BP9040_FMA24497_Right calcaneus_Filled.npy'

if CONFIG=='L1':
    NUMPY_DATA_FILENAME = 'FJ3157_BP8948_FMA13072_First lumbar vertebra_Filled.npy'
    
if CONFIG=='TALUS':
    NUMPY_DATA_FILENAME = 'FJ3385_BP8033_FMA24482_Right talus_Filled.npy'
    INIT_AXIS_2_KEY     = K_z

if CONFIG=='HUM_GT':
    NUMPY_DATA_FILENAME = 'hum_dia.npy'
    GROUND_TRUTH_DATA   = 'hum_dia.npy'

if CONFIG=='HUM_CROP_PAD':
    NUMPY_DATA_FILENAME = 'hum_dia_crop_pad.npy'
    GROUND_TRUTH_DATA   = 'hum_dia_gt_crop_pad.npy'
    INIT_AXIS_2_KEY     = K_z
    
if CONFIG=='HUM_CROP':
    DATA = '../hum_dia'
    REF_MARKS_MATFILE   = 'Hum_crop'
    DICOM_NORMALIZE     = False
    GROUND_TRUTH_DATA   = 'hum_dia.npy'
    
if CONFIG=='ILIAC':
    PNG_DATA_DIRNAME = '../iliac'
    PRECALC_GRADIENTS   = False
    
if CONFIG=='SAVE_ILIAC':
    DATA = '../iliac'
    RESAMPLE            = True
    RESAMPLE_XY         = 512/250.0
    RESAMPLE_Z          = 1.0
    RESAMPLE_ZXY        = (RESAMPLE_Z, RESAMPLE_XY, RESAMPLE_XY)
    DICOM_NORMALIZE     = False
    
if CONFIG=='PROMISE12':
    NUMPY_DATA_FILENAME = 'Case21.npy'
    GROUND_TRUTH_DATA   = 'Case21_segmentation.npy'

if CONFIG.startswith('PROMISE12'):
    RESAMPLE            = True
    RESAMPLE_XY         = 3.6
    RESAMPLE_Z          = .625
    RESAMPLE_ZXY        = (RESAMPLE_Z, RESAMPLE_XY, RESAMPLE_XY)
    
if CONFIG=='PROMISE12_GT':
    NUMPY_DATA_FILENAME = 'Case21_segmentation.npy'
    
if CONFIG=='TRIPOD':
    NUMPY_DATA_FILENAME = 'tripod.npy'

if CONFIG=='SAVE_CB_BOX':
    USE_HALF_REF_ONLY   = True
    NUMPY_DATA_FILENAME = 'Cerebellum_box.npy'
    REF_MARKS_MATFILE   = 'Cerebellum_box.mat'
    TOUCH_ENDING_SLICES = True
    SAVE_RBF            = True
    GRADIENT_WEIGHT     = 0.0
    
if CONFIG=='CB_BOX_GT':
    NUMPY_DATA_FILENAME = 'Cerebellum_box_gt.npy'
    
if CONFIG=='CB_BOX':
    NUMPY_DATA_FILENAME = 'Cerebellum_box.npy'
    REF_MARKS_MATFILE   = 'Cerebellum_box.mat'
    GROUND_TRUTH_DATA   = 'Cerebellum_box_gt.npy'

if CONFIG=='CB_BOX_GT_ROT':
    NUMPY_DATA_FILENAME = 'Cerebellum_box_gt_rot.npy'
    
if CONFIG=='CB_BOX_ROT':
    NUMPY_DATA_FILENAME = 'Cerebellum_box_rot.npy'
    GROUND_TRUTH_DATA   = 'Cerebellum_box_gt_rot.npy'
    
if CONFIG=='BAMBA4':
    NUMPY_DATA_FILENAME = 'bamba4.npy'
    DATA                = ''
    
if CONFIG=='S2_BOX_GT':
    NUMPY_DATA_FILENAME = 'RightVent_box_gt.npy'
    
if CONFIG=='SAVE_S2_BOX':
    NUMPY_DATA_FILENAME = 'RightVent_box.npy'
    REF_MARKS_MATFILE   = 'RightVent_box.mat'
    TOUCH_ENDING_SLICES = True
    SAVE_RBF            = True
    GRADIENT_WEIGHT     = 0.0

if CONFIG in ['S3', 'SAVE_S3']:
    DATA_FUNC           = ''
    DATA                = '../s3'
    CACHE_DATA_FILENAME = 'cached_data.npy'
    REF_MARKS_MATFILE   = 'Ventricle2.mat'
    VIEW_FACTOR         = 1
    GRADIENT_WEIGHT     = 0.0 #1.0 means use only grayscale gradient
    RESAMPLE            = True #False
    ###
    DICOM_NORMALIZE     = False 
    RESAMPLE_XY         = 2
    RESAMPLE_Z          = 1
    RESAMPLE_ZXY        = (RESAMPLE_Z, RESAMPLE_XY, RESAMPLE_XY)
    
if CONFIG == 'SAVE_S3':
    DOWNSAMPLE          = 2 #4 #1.5
     
if CONFIG == 'S3_BOX_GT':
    DATA = '../s3'
    NUMPY_DATA_FILENAME = 's3_ventricle2_iso_gt.npy'
    START_POS = [98/2+1, 113/2, 117/2]

if CONFIG in ['S3_BOX', 'SAVE_S3_BOX', 'S3_CROP']:
    START_POS           = [98/2+1, 113/2, 117/2]
    DATA                = ''
    GROUND_TRUTH_DATA   = 's3_ventricle2_iso_gt.npy'
    NUMPY_DATA_FILENAME = 's3_ventricle2.npy'
    CACHE_DATA_FILENAME = 'cached_data.npy'
    REF_MARKS_MATFILE   = 'Ventricle2_box.mat'
    VIEW_FACTOR         = 4
    GRADIENT_WEIGHT     = 0.0
    RESAMPLE            = True
    RESAMPLE_GT         = False
    RESAMPLE_XY         = DOWNSAMPLE
    RESAMPLE_Z          = .5#DOWNSAMPLE*240.0/512
    RESAMPLE_ZXY        = (RESAMPLE_Z, RESAMPLE_XY, RESAMPLE_XY)

if CONFIG=='S3_CROP':
    START_POS           = [49/2+1, 113/2, 117/2] #[98/2+1, 113/2, 117/2]
    DATA                = '../s3_crop'
    GROUND_TRUTH_DATA   = 's3_ventricle2_gt.npy'
    REF_MARKS_MATFILE   = 'Ventricle2_box.mat'
    DICOM_NORMALIZE     = False
    NUMPY_DATA_FILENAME = ''
    GRADIENT_WEIGHT     = 0.5
    RESAMPLE            = False
    SPACING             = [2.133, 1, 1]
    
if CONFIG == 'SAVE_S3_BOX':
    NUMPY_DATA_FILENAME = 's3_ventricle2.npy'
    DOWNSAMPLE          = 1
    USE_HALF_REF_ONLY   = False
    
if CONFIG in ['SAVE_S3', 'SAVE_S3_BOX']:
    TOUCH_ENDING_SLICES = False
    SAVE_RBF            = True
    RESAMPLE_XY         = DOWNSAMPLE
    RESAMPLE_Z          = .5#DOWNSAMPLE*240.0/512
    RESAMPLE_ZXY        = (RESAMPLE_Z, RESAMPLE_XY, RESAMPLE_XY)

    
##################
#Roberts' Cross 2D
##################
R0=[[1,0],[0,-1]]
R1=[[0,1],[-1,0]]
#((ndimage.convolve(a,R0)**2 + ndimage.convolve(a,R1)**2)**.5).round().astype(int)
#(ndimage.convolve(a,R0)**2 + ndimage.convolve(a,R1)**2)

########################
#Sobel 3D with smoothing
########################
#adapted from
#http://en.wikipedia.org/wiki/Sobel_operator#Extension_to_other_dimensions
#
#smoothing operator
vs = [ 1,  4,  6,  4,  1]
#derivative operator
vd = [-1, -4, -5,  0,  5,  4,  1]

def Hx(x,y,z):
    return vd[x]*vs[y]*vs[z]
def Hy(x,y,z):
    return vd[y]*vs[x]*vs[z]
def Hz(x,y,z):
    return vd[z]*vs[x]*vs[y]

def GetZ():
    f=[]
    for z in range(len(vd)):
        f.append([])
        for y in range(len(vs)):
            f[z].append([])
            for x in range(len(vs)):
                f[z][y].append(Hz(x,y,z))
    return f
        
def GetX():
    f=[]
    for z in range(len(vs)):
        f.append([])
        for y in range(len(vs)):
            f[z].append([])
            for x in range(len(vd)):
                f[z][y].append(Hx(x,y,z))
    return f

def GetY():
    f=[]
    for z in range(len(vs)):
        f.append([])
        for y in range(len(vd)):
            f[z].append([])
            for x in range(len(vs)):
                f[z][y].append(Hy(x,y,z))
    return f
####################
GRAD_3D_Y = array(GetY())
GRAD_3D_X = array(GetX())
GRAD_3D_Z = array(GetZ())

def synth_data(N,centers,radii):
    x,y,z = mgrid[0:N,0:N,0:N]
    cube = zeros((N,N,N),bool)
    for i,c in enumerate(centers):
        sphere = (x-c[0])**2 + (y-c[1])**2 +(z-c[2])**2 <= radii[i]**2
        cube = cube + sphere
    return 255*cube.astype(uint8)

def shape_blimps(N, n=3):
    centers = (N * array([
        [.5, .5, .5],
        [.3, .3, .5],
        [.7, .3, .5],
        [.3, .7, .5],
                      ])).round().astype(int)
    radii   = N * array([.35, .2, .2, .2])
    return synth_data(N,centers[:n+1],radii[:n+1])

def tube(N,c1,c2,r):
    cube = zeros((N,N,N))
    c1,c2 = map(array,[c1,c2])
    dist = N*norm(c2-c1)
    step = (c2-c1)/dist
    c = c1
    origr  = r
    for i in range(int(dist)):
        fluc = .5 if random.rand()<.5 else -.5
        fluc /= N
        if True and abs((r+fluc)/origr-1)<.35 and random.rand()<.5:
            r += fluc
        cube += sphere(N,c,r)
        c += step
    return cube,r

def xarc(N,c,R,r):
    c=array(c)
    cube = zeros((N,N,N),dtype=uint8)
    for rad in linspace(-pi/2,0,num=20):
        x,y,z = R*cos(rad), R*sin(rad), 0
        cube += sphere(N,c+(x,y,z),r)
    return cube

def yarc(N,c,R,r):
    c=array(c)
    cube = zeros((N,N,N),dtype=uint8)
    for rad in linspace(-pi/2,0,num=20):
        x,y,z = 0,R*cos(rad), R*sin(rad)
        cube += sphere(N,c+(x,y,z),r)
    return cube

def ninja(N,margin=.23,radius=.1,arc=.15):
    m=margin
    r=radius
    a=arc
    cube = zeros((N,N,N),uint8)
    c, r = tube(N,(m,m,m),(1-m-a,m,m),r)
    cube += c
    cube += xarc(N,(1-m-a,m+a,m),a,r)
    c, r = tube(N,(1-m,m+a,m),(1-m,1-m-a,m),r)
    cube += c
    cube += yarc(N,(1-m,1-m-a,m+a),a,r)
    c, r = tube(N,(1-m,1-m,m+a),(1-m,1-m,1-m),r)
    cube += c
    return 255*cube.astype(uint8)

def ninja_twisted(N,margin=.23):
    m=margin
    cube = ninja(N,margin=m)
    cube = interpolation.rotate(cube,angle=30,axes=(0,1),reshape=False)
    c = zeros((N,N,N),uint8)
    c[:,:,N*m/2:] = cube[:,:,:-N*m/2]
    cube = c
    cube = interpolation.rotate(cube,angle=20,axes=(0,2),reshape=False)
    return 255*cube.astype(uint8)

def shape_bamba(N):
    centers = (N * array([[.35,.25,.35],[.5,.5,.5],[.35,.75,.35]]
                         )).round().astype(int)
    radii   = N * array([.1875, .25, .1875])
    return synth_data(N,centers,radii)

def moveX(N,cube,s):
    cube[:-s*N,:,:] = cube[s*N:,:,:]
    cube[-s*N:,:,:] = 0
    return cube
    
def shape_letterC(N):
    r=.14
    cube = torus(N,(.5,.5,.5),.26,r)
    xlen = cube.shape[0]
    cube[:.5*N,:,:] = 0
    cube += sphere(N,(.5,.24,.5),r)    
    cube += sphere(N,(.5,.76,.5),r)
    cube[cube>0] = 255
    return moveX(N,cube,.2)

def shape_letterC_humble(N):
    cube = torus(N,(.5,.5,.2),.26,.12)
    xlen = cube.shape[0]
    cube[:.5*N,:,:] = 0
    cube += sphere(N,(.5,.24,.2),.12)    
    cube += sphere(N,(.5,.76,.2),.12)
    return cube

def sphere(N,c=(.5,.5,.5),r=.4):
    x,y,z = mgrid[0:N,0:N,0:N]
    r = N*r
    c = N*array(c)
    cube = (x-c[0])**2 + (y-c[1])**2 +(z-c[2])**2 <= r**2
    return 255*cube.astype(uint8)

def torus(N,c,R,r,xstretch=.8):
    x,y,z = mgrid[0:N,0:N,0:N]
    R,r = N*R, N*r
    c = N*array(c)
    cube = (R-sqrt((xstretch*(x-c[0]))**2+(y-c[1])**2))**2 + (z-c[2])**2 <= r**2
    return 255*cube.astype(uint8)

def letterS(N):
    R, r = .175, .12
    cube = torus(N,(.5,.5,.5),R,r)
    start = round((.5+r)*N)
    end   = round((.5+R+r)*N)
    cube[:.5*N,:start,:] = cube[:.5*N,end-start:end,:]
    cube[:.5*N,start:,:] = 0
        
    start = round((.5-r)*N)
    end   = round((.5-R-r)*N)
    cube[.5*N:,start:,:] = cube[.5*N:,end:end+(N-start),:]
    cube[.5*N:,:start,:] = 0
    ##cube = interpolation.shift(cube,(0,N*.15,0))

    cube += sphere(N, (.5, .5-2*R, .5), r)
    cube += sphere(N, (.5, .5+2*R, .5), r)
    cube[cube>0]=1
    
    cube = interpolation.rotate(cube,angle=30,axes=(0,2),reshape=False)    
    cube = interpolation.rotate(cube,angle=60,axes=(1,2),reshape=False)    
    return 255*cube.astype(uint8)

def shape_torus(N):
    cube = torus(N,(.5,.5,.5),.26,.14,.82)
    cube = interpolation.rotate(cube,angle=45,axes=(0,2),reshape=False)
    cube = interpolation.rotate(cube,angle=45,axes=(0,1),reshape=False)
    return cube

def shape_horseshoe(N):
    cube = torus(N,(.5,.5,.3),.26,.12)
    xlen = cube.shape[0]
    cube[:xlen/2,:,:] = 0
    cube[.2*N:xlen/2,:,:] = cube[xlen/2,:,:]
    cube += sphere(N,(.2,.24,.3),.12)
    cube += sphere(N,(.2,.76,.3),.12)
    cube[cube>0] = 1
##    cube = interpolation.rotate(cube,angle=30,axes=(0,2),reshape=False)
##    cube[:-10,:,:] = cube[10:,:,:]
##    cube[-10:,:,:] = 0
    return 255*cube.astype(uint8)

def shape_horseshoe1(N):
    centers = (N * array([[.30, .25, .30],                      
                          [.45, .38, .45],
                          [.50, .50, .50],
                          [.45, .62, .45],                      
                          [.30,.75,.30],
                          ])).round().astype(int)
    radii   = N * array([.15, .16, .17, .16, .15])
    return synth_data(N,centers,radii)

def shape_torus_rot(N,c=(.5,.5,.4),R=.25,r=.12,angle=45,axes=(0,2),stretch=.84):
    x,y,z = mgrid[0:N,0:N,0:N]
    R,r = N*R, N*r
    c = N*array(c)
    cube = (R-sqrt((stretch*(x-c[0]))**2+(y-c[1])**2))**2 + (z-c[2])**2 <= r**2
    cube = 255*cube.astype(uint8)
    cube = interpolation.rotate(cube,angle,axes,reshape=False)
    return cube

def shape_s2(N):
    centers = (N * array([
       [70,200,40],
       [45,140,50],
       [90,85,60],
       [130,140,70],
       [170,190,80],
       [205,140,90],
##       [220,140,90],
       [200,80,100],
       ])/256).round().astype(int)
    radii   = N * .15* ones(len(centers))
    return synth_data(N,centers,radii)

def shape_s(N):
    centers = (N * array([
                          #[.70, .12, .70],
                          [.80, .18, .80],
                          [.84, .30, .84],
                          [.78, .40, .78],                      
                          [.70, .46, .70],                      
                          [.60, .54, .60],
                          [.52, .58, .52],
                          [.46, .70, .46],
                          [.52, .82, .52],
                          #[.62, .88, .62],
                          ])).round().astype(int)
    radii   = N * array([.1,.1,.11,.1,.11,.11,.11,.1,.1,.1])
    return synth_data(N,centers,radii)
    
"""
DBSCAN: Density-Based Spatial Clustering of Applications with Noise
Shahar: I derive neighborhood with sparse "locality based" method
  instead of dense "pairwise_distances" matrix.
"""
# Author: Robert Layton <robertlayton@gmail.com>
#
# License: BSD

import warnings
import numpy as np

from sklearn.base import BaseEstimator
from sklearn.metrics import pairwise_distances
from sklearn.utils import check_random_state
from scipy.spatial import distance
def dbscan(X, eps=0.5, min_samples=5, metric='euclidean',
           random_state=None):
    """Perform DBSCAN clustering from vector array or distance matrix.

    Parameters
    ----------
    X: array [n_samples, n_samples] or [n_samples, n_features]
        Array of distances between samples, or a feature array.
        The array is treated as a feature array unless the metric is given as
        'precomputed'.
    eps: float, optional
        The maximum distance between two samples for them to be considered
        as in the same neighborhood.
    min_samples: int, optional
        The number of samples in a neighborhood for a point to be considered
        as a core point.
    metric: string, or callable
        The metric to use when calculating distance between instances in a
        feature array. If metric is a string or callable, it must be one of
        the options allowed by metrics.pairwise.calculate_distance for its
        metric parameter.
        If metric is "precomputed", X is assumed to be a distance matrix and
        must be square.
    random_state: numpy.RandomState, optional
        The generator used to initialize the centers. Defaults to numpy.random.

    Returns
    -------
    core_samples: array [n_core_samples]
        Indices of core samples.

    labels : array [n_samples]
        Cluster labels for each point.  Noisy samples are given the label -1.

    Notes
    -----
    See examples/plot_dbscan.py for an example.

    References
    ----------
    Ester, M., H. P. Kriegel, J. Sander, and X. Xu, "A Density-Based
    Algorithm for Discovering Clusters in Large Spatial Databases with Noise".
    In: Proceedings of the 2nd International Conference on Knowledge Discovery
    and Data Mining, Portland, OR, AAAI Press, pp. 226–231. 1996
    """
    X = np.asarray(X)
    n = X.shape[0]
    # If index order not given, create random order.
    random_state = check_random_state(random_state)
    index_order = np.arange(n)
    random_state.shuffle(index_order)

################
##    D2 = pairwise_distances(X, metric=metric)

    offsets=[]
    for i in range(-1,2):
        for j in range(-1,2):
            for k in range(-1,2):
                if distance.euclidean((0,0,0),(i,j,k)) < eps:
                    offsets.append((i,j,k))
    offsets = np.array(offsets)
    
    coords = [(tuple(v),i) for i,v in enumerate(X)]
    coord = dict(coords)
    Xset = frozenset([tuple(x) for x in X])
    N=X.shape[0]    
    neighborhoods=[]
    for i in range(N):
        neig = [tuple(p) for p in X[i]+offsets]
        hood = [coord[p] for p in neig if p in Xset]
        hood.sort()
        neighborhoods.append(np.array(hood))
        
    # Calculate neighborhood for all samples. This leaves the original point
    # in, which needs to be considered later (i.e. point i is the
    # neighborhood of point i. While True, its useless information)
##    neighborhoods2 = [np.where(x > inveps)[0] for x in S2]
#################
 
    # Initially, all samples are noise.
    labels = -np.ones(n)
    # A list of all core samples found.
    core_samples = []
    # label_num is the label given to the new cluster
    label_num = 0
    # Look at all samples and determine if they are core.
    # If they are then build a new cluster from them.
    for index in index_order:
        if labels[index] != -1 or len(neighborhoods[index]) < min_samples:
            # This point is already classified, or not enough for a core point.
            continue
        core_samples.append(index)
        labels[index] = label_num
        # candidates for new core samples in the cluster.
        candidates = [index]
        while len(candidates) > 0:
            new_candidates = []
            # A candidate is a core point in the current cluster that has
            # not yet been used to expand the current cluster.
            for c in candidates:
                noise = np.where(labels[neighborhoods[c]] == -1)[0]
                noise = neighborhoods[c][noise]
                labels[noise] = label_num
                for neighbor in noise:
                    # check if its a core point as well
                    if len(neighborhoods[neighbor]) >= min_samples:
                        # is new core point
                        new_candidates.append(neighbor)
                        core_samples.append(neighbor)
            # Update candidates for next round of cluster expansion.
            candidates = new_candidates
        # Current cluster finished.
        # Next core point found will start a new cluster.
        label_num += 1
    return core_samples, labels


class DBSCAN(BaseEstimator):
    """Perform DBSCAN clustering from vector array or distance matrix.

    DBSCAN - Density-Based Spatial Clustering of Applications with Noise.
    Finds core samples of high density and expands clusters from them.
    Good for data which contains clusters of similar density.

    Parameters
    ----------
    eps : float, optional
        The maximum distance between two samples for them to be considered
        as in the same neighborhood.
    min_samples : int, optional
        The number of samples in a neighborhood for a point to be considered
        as a core point.
    metric : string, or callable
        The metric to use when calculating distance between instances in a
        feature array. If metric is a string or callable, it must be one of
        the options allowed by metrics.pairwise.calculate_distance for its
        metric parameter.
        If metric is "precomputed", X is assumed to be a distance matrix and
        must be square.
    random_state : numpy.RandomState, optional
        The generator used to initialize the centers. Defaults to numpy.random.

    Attributes
    ----------
    'core_sample_indices_' : array, shape = [n_core_samples]
        Indices of core samples.

    'components_' : array, shape = [n_core_samples, n_features]
        Copy of each core sample found by training.

    'labels_' : array, shape = [n_samples]
        Cluster labels for each point in the dataset given to fit().
        Noisy samples are given the label -1.

    Notes
    -----
    See examples/plot_dbscan.py for an example.

    References
    ----------
    Ester, M., H. P. Kriegel, J. Sander, and X. Xu, "A Density-Based
    Algorithm for Discovering Clusters in Large Spatial Databases with Noise".
    In: Proceedings of the 2nd International Conference on Knowledge Discovery
    and Data Mining, Portland, OR, AAAI Press, pp. 226–231. 1996
    """

    def __init__(self, eps=0.5, min_samples=5, metric='euclidean',
            random_state=None):
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.random_state = check_random_state(random_state)

    def fit(self, X, **params):
        """Perform DBSCAN clustering from vector array or distance matrix.

        Parameters
        ----------
        X: array [n_samples, n_samples] or [n_samples, n_features]
            Array of distances between samples, or a feature array.
            The array is treated as a feature array unless the metric is
            given as 'precomputed'.
        params: dict
            Overwrite keywords from __init__.
        """
        if params:
            warnings.warn('Passing parameters to fit methods is '
                        'depreciated', stacklevel=2)
            self.set_params(**params)
        self.core_sample_indices_, self.labels_ = dbscan(X,
                                                         **self.get_params())
        self.components_ = X[self.core_sample_indices_].copy()
        return self

#end of DBSCAN
#################################################################

last_message = 'Greetings'
def display_box(screen, message=None, pos=(25,20)):
    "debug a message in a box in the middle of the screen"
    global last_message
    if message==None:
        message = last_message
    assert(message is not None and message!='')
    fontobject = pygame.font.Font(None,FONT_SIZE)
    text_surf = fontobject.render(message, 1, (255,255,255))
    xs,ys = text_surf.get_size()
    if pos==None:
        pos = ((screen.get_width()-xs)/2, (screen.get_height()-ys)/2)
    x,y = pos    
    #clear area for text
    prev_text_surf = fontobject.render(last_message, 1, (255,255,255))
    last_message = message
    xsp,ysp = prev_text_surf.get_size()
    pygame.draw.rect(screen, (0,0,0), (x,y, max(xsp,xs),ysp), 0)
    #blit text
    screen.blit(text_surf,(x,y))
    #blit box
##    pygame.draw.rect(screen, (255,255,255),
##                   (x-2,y-2, xs+4,ys+4), 1)
    pygame.display.flip()
  
def wait_event(event=KEYDOWN,show=None):
    pygame.event.clear()
    while True:
        e = pygame.event.wait()
        if e.type == pygame.QUIT or \
            (e.type == KEYDOWN and e.key == K_ESCAPE):
            return pygame.QUIT
        if e.type == event:
            break
    return 0

def convolve3d_pt(P, K, (x,y,z)):
    """Single image pt convolution
    not optimized - use separable convolution to optimize relevant filters
    """
    K=K.T
    xsize,  ysize,  zsize  = P.shape
    kX, kY, kZ = K.shape
    kCenterX, kCenterY, kCenterZ = array(K.shape) / 2
    s=0
    for m in range(kX):
        mm = kX - 1 - m      # row index of flipped kernel
        for n in range(kY):  # kernel columns
            nn = kY - 1 - n  # column index of flipped kernel
            for d in range(kZ):
                dd = kZ - 1 - d
                #index of input signal, used for checking boundary
                ii = x + (m - kCenterX)
                jj = y + (n - kCenterY)
                hh = z + (d - kCenterZ)
                #ignore input samples which are out of bound
                if  ii>=0 and ii<xsize and \
                    jj>=0 and jj<ysize and \
                    hh>=0 and hh<zsize:
                    s += P[ii,jj,hh] * K[mm,nn,dd]
    return s

def convolve_pt(P, K, (x,y)):
    """Single image pt convolution
    not optimized - use separable convolution to optimize relevant filters
    """
    K=K.T
    rows,  cols  = P.shape
    kRows, kCols = K.shape
    kCenterY, kCenterX = array(K.shape) / 2
    s=0
    for m in xrange(kRows):
        mm = kRows - 1 - m      # row index of flipped kernel
        for n in xrange(kCols):  # kernel columns
            nn = kCols - 1 - n  # column index of flipped kernel
            #index of input signal, used for checking boundary
            ii = y + (m - kCenterY)
            jj = x + (n - kCenterX)
            #ignore input samples which are out of bound
            if ii>=0 and ii<rows and jj>=0 and jj<cols:
                s += P[ii,jj] * K[mm,nn]
    return s

def bilinear(plane, (x1,y1), topLeft, step):
    topLeftX, topLeftY = topLeft
    topRight = (topLeftY,       topLeftX+step)
    botLeft  = (topLeftY+step,  topLeftX)
    botRight = (topLeftY+step,  topLeftX+step)
    ulC = plane[topLeft]
    urC = plane[topRight]
    blC = plane[botLeft]
    brC = plane[botRight]
    step = float(step)
    rightFactorX = (x1-topLeftX)/step
    topC = ulC * (1-rightFactorX) + urC * rightFactorX
    botC = blC * (1-rightFactorX) + brC * rightFactorX
    botFactorY = (y1-topLeftY)/step
    finalC = topC * (1-botFactorY) + botC * botFactorY
##    if ulC or urC or blC or brC:
##        pdb.set_trace()
    return int(round(finalC))

def histeq(im,nbr_bins=256):
   #get image histogram
   imhist,bins = numpy.histogram(im.flatten(),nbr_bins,density=True)
   cdf = imhist.cumsum() #cumulative distribution function
   cdf = 255 * cdf / cdf[-1] #normalize
   #use linear interpolation of cdf to find new pixel values
   im2 = numpy.interp(im.flatten(),bins[:-1],cdf)
   return im2.reshape(im.shape).astype('uint32')

def read_data(filename):
    "return data"
    if filename.endswith('.dcm'):
        dcm = dicom.ReadFile(filename)
        a = dcm.pixel_array
    elif filename.endswith('.1'):
        data = array([numpy.fromfile(filename, dtype='>u2')])
        data.shape = (1, 256, 256)
        #data = data.T
        a=data[0]
    return data

def read_mhd_data(filename):
    """
    Input: filename - without extension
    Purpose: read "mhd" to decide on shape, then read raw
    Return: array as int8 or int16
    """
    with open(filename+'.mhd','rt') as f:
        for line in f:
            if line.startswith('DimSize'):
                the_shape = [int(x) for x in line.split()[-3:]]
            if line.startswith('ElementType'):
                if line.endswith('MET_CHAR'):
                    the_type = int16
                elif line.endswith('MET_SHORT'):
                    the_type = int8
                else:
                    raise Exception('Unknown type: ' + line)            
    d = fromfile(filename+'.raw',the_type)
    return d.reshape(*the_shape)      

def read_png_data(dirname):
    dirlist = sorted(os.listdir(dirname))
    paths = [os.path.join(dirname,p) for p in dirlist]
    aa = [misc.imread(p) for p in paths]
    return array(aa)

def read_matlab_data(filename):
    data = io.loadmat(filename)['Cube']
    f = MATLAB_SHRINK_FACTOR
    data = data[::f,::f,::f]
    data = 255*data
    return data

def read_data_dicom(dirname):
    "Read folder files and return data as 3d array"
    #dirlist = sorted(os.listdir(dirname),key=lambda x: int(x.split('.')[1]))
    dirlist = sorted(os.listdir(dirname))
    half = len(dirlist)/2
##    dirlist = dirlist[half-5:half+5]
    paths = [os.path.join(dirname,p) for p in dirlist]
    debug(pformat(paths))
    #a = array([numpy.fromfile(p,dtype='>u2') for p in paths])
    aa = [dicom.ReadFile(p,force=True).pixel_array for p in paths]
    #sometimes the last image has different size, so drop bad imgs
    the_shape = aa[0].shape
    #filter out different size images
    aa = array([a for a in aa if a.shape==the_shape])
    ##aa = array([histeq(a) for a in aa])
    if DICOM_NORMALIZE:
        debug('* Normalizing DICOMs')
        #normalize all images
        amin,amax = aa.min(), aa.max()
    ##    aa = (aa - amin)/(amax-amin) * 255
        aa -= amin
        aa *= 255.0
        aa /= (amax-amin)
        aa = aa.round().astype('uint8')
    debug('Read {} images'.format(len(aa)))
        #a=histeq(a)[0]
        #a *= 255.0 / a.max()
        ##a=a.astype('ubyte')
        #a=a.astype('uint32')
        #a += (a<<8) + (a<<16) + (255<<24)
        ##a += 256*a + 256*256*a + 256*256*256*255
    return aa

def resample(data,(x,y,z)):
    M=eye(3)*(x,y,z)
    #M=array([[x,0,0],[0,y,0],[0,0,z]])
    newshape = (array(data.shape) / (x,y,z)).round().astype(int)
    debug('Resampling with %s',(x,y,z))
    bb=ndimage.affine_transform(data, M, prefilter=False, order=0,
                                output_shape=newshape)
    bb.reshape(newshape)
    return bb
    
def byte2rgba(plane, (red,green,blue)=(0xff,0xff,0xff), alpha=255):
    a = plane.astype('uint32')
    #if CONFIG in ['HUM_CROP_PAD']:        a = a/2**4
    a = a / (a.max()/255.0)
    a = a.round().astype('uint32')
    a = (red&a) + ((green&a)<<8) + ((blue&a)<<16) + (alpha<<24)
    ret = pygame.image.frombuffer(a.tostring(),a.shape[::-1],'RGBA')
    return ret.convert()

def calc_gradients(plane, pts):
    "Return 2d gradients of the pts"
    gradients = empty((len(pts),2), int)
    for i,pt in enumerate(pts):
        if any(pt):
            g = array(gradient(plane, pt))
            if any(g):
                gradients[i] = g
                #ng = NORM_MAX_LEN * (g/norm(g))
            else:
                debug('any(g) is False')
        else:
            debug('any(pt) is False')
    return gradients

def show_gradients(plane,pts,surf):
    grads = calc_gradients(plane, pts)
    show_normals(pts, grads, surf)
    
def show_normals(pts, normals, surf):
    "util to show vector field"
    for pt,normal in zip(pts,normals):
        pygame.draw.circle(surf,GRAD_LOC_COLOR,pt,GRAD_LOC_RADIUS)
        pygame.draw.line(surf,GRAD_COLOR,pt,pt+normal,2)

def show_contour_normals(pts,segments,surf):    
    normals = calc_all_contours_normals_2d(pts, segments)
    show_normals(concatenate(pts.tolist()), normals, surf)


def calc_combined_normals(pts,gradient_field,normals):
    "Combine gradient and contour-normals"
    ret = empty_like(pts)
    gradients = gradient_field[pts.T.tolist()]
    pts = [tuple(p) for p in pts]
    normlist = array([normals[p] for p in pts])
    i=0
    for normal, gradient in izip(normlist, gradients):
        if norm(gradient) > 0:            
            angle = arccos(dot(gradient,normal)/(norm(gradient)*norm(normal)))        
            if angle<pi/2:            
##            if any(normal==0):
##                #The normal is on axis-oriented plane
##                missing_axis = where(normal==0)[0][0]
##                factor = (norm(delete(normal,missing_axis)) /
##                          norm(delete(gradient,missing_axis)))
##                normal[missing_axis] = gradient[missing_axis] * factor
##            else:
                ng = gradient/norm(gradient)
                nn = normal/norm(normal)
                nnormal = GRADIENT_WEIGHT*ng + (1-GRADIENT_WEIGHT)*nn
                normal = nnormal * norm(normal)
        ret[i] = normal
        i += 1
    return ret

def calc_contour_normals_2d_OLD(pts):
    """calc 2d normals of contour pts
    assume the pts are clockwise around the object
    (row,col)
    """
    normals = empty((len(pts),2),int)
    for i in xrange(len(pts)):
        p1,p2,p3 = [array(pts[x]) for x in (i-2, i, (i+2)%len(pts))]
        a = p2 - p1
        b = p3 - p2
        angles = [arctan2(x[1],x[0]) for x in (a,b)]
        mid_angle = sum(angles)/2-pi/2
        x1,y1,x2,y2,x3,y3 = concatenate((p1,p2,p3))
        #turn sign
        z = x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2)
        if  sign(z)== 1 and angles[0]>angles[1] or \
            sign(z)==-1 and angles[0]<angles[1]:
            mid_angle = mid_angle-pi
        cx,cy = cos(mid_angle), sin(mid_angle)
        c = NORM_MAX_LEN*array([cx,cy])
        normals[i] = c.round().astype(int)
##    print("pts:")
##    pprint(pts)
##    print("normals:")
##    pprint(normals)
    return normals

def calc_contour_normals_2d(pts,seg):
    """calc 2d normals of contour pts
    assume the pts are clockwise around the object
    (row,col)
    """
    pset = set([tuple(x) for x in pts])
    normals = empty((len(pts),2),int)
    j=0
    for i in xrange(len(seg)):
        p1,p2,p3 = [array(seg[x]) for x in (i-2, i, (i+2)%len(seg))]
        if tuple(p2) in pset:
            a = p2 - p1
            b = p3 - p2
            angles = [arctan2(x[1],x[0]) for x in (a,b)]
            mid_angle = sum(angles)/2-pi/2
            x1,y1,x2,y2,x3,y3 = concatenate((p1,p2,p3))
            #turn sign
            z = x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2)
            if  sign(z)== 1 and angles[0]>angles[1] or \
                sign(z)==-1 and angles[0]<angles[1]:
                mid_angle = mid_angle-pi
            cx,cy = cos(mid_angle), sin(mid_angle)
            c = NORM_MAX_LEN*array([cx,cy])
            normals[j] = c.round().astype(int)
            j+=1
##    print("pts:")
##    pprint(pts)
##    print("normals:")
##    pprint(normals)
    return normals

def calc_all_contours_normals_2d(pts,segments):
    """
    assume that each segment is around the target clockwise
    """
    normals = []
    for i in xrange(len(pts)):
        seg_normals = calc_contour_normals_2d(pts[i], segments[i])
        normals.extend(seg_normals)
    ret = array(normals)
    return ret

def calc_contour_normals(marks):
    """2d slice is embedded in 3d
    returns 3D
    """
    nmarks = len(marks)
    normals = {}
    for i,(triplet,trans,pshape,axis,iPlane,segments,pts) in enumerate(marks):
        normals2d = calc_all_contours_normals_2d(pts,segments)
        pts = concatenate(pts.tolist()) 
        #fill in for the missing dimension of the slice
        normals2d_yx = roll(normals2d,shift=1,axis=1)
        if trans: #oblique
            (A,t),axes = trans
            #pts_yx = roll(pts, shift=1, axis=1)
            normals3d = [(dot(A,p+n) - dot(A,p)).round().astype(int)
                         for p,n in zip(pts,normals2d)]
        else:     #orthographic
            val = 0
            if TOUCH_ENDING_SLICES and i in (0,nmarks-1):
                #normals2d_yx = zeros(normals2d_yx.shape)
                val = -20 if i==0 else 20
            #insert value (usually 0) for the missing axis
            normals3d = insert(normals2d_yx, axis, values=val, axis=1)
            normals3d = normals3d.tolist()
        pts3d = pts_2d_to_3d(pts, trans, axis, iPlane)
        normals.update(zip([tuple(p) for p in pts3d],normals3d))
    return normals
        
def show_slice(plane,rbf_plane=None,grades_plane=None,ref_plane=None,marks_plane=None):
    global screen, img, obliq_slice
    img_size = VIEW_FACTOR * array(plane.shape,dtype='uint32')
    img_size = img_size[::-1] #was y,x
    screen = pygame.display.set_mode(img_size)
    img = byte2rgba(plane)
    img = pygame.transform.smoothscale(img, img_size)
    #s32 = pygame.image.tostring(img,'RGBA')
    #s8 = s32[::4]
    #img8 = pygame.image.frombuffer(s8,img_size,'P')
    #img8.set_palette([(c,c,c) for c in range(256)])
    #img = pygame.image.frombuffer(a.tostring(),a.shape,'P')
    #img.set_palette([(c,c,c) for c in range(256)])
    screen.blit(img,(0,0))
    if ref_plane != None:
        ref_surface = byte2rgba(ref_plane, (0,0xff,0))
        ref_surface = pygame.transform.smoothscale(ref_surface, img_size)
##        ref_surface.set_alpha(REF_ALPHA) #0 is transparent, 255 is opaque
        screen.blit(ref_surface, (0,0), special_flags=BLEND_ADD)
    if rbf_plane != None:
        rbf_byte = zeros(rbf_plane.shape,uint8)
        rbf_byte[rbf_plane>0] = 0xFF        
        rbf_surface = byte2rgba(rbf_byte, (0xff,0,0))
        rbf_surface = pygame.transform.smoothscale(rbf_surface, img_size)
##        rbf_surface.set_alpha(RBF_ALPHA) #0 is transparent, 255 is opaque
        screen.blit(rbf_surface, (0,0), special_flags=BLEND_ADD)
    if grades_plane != None:
        nonzero = count_nonzero(grades_plane)
        plane_grade = 0 if nonzero==0 else float(grades_plane.sum()) / nonzero
        debug("Grade: {}".format(plane_grade))
        grades_surface = byte2rgba(grades_plane, (0,0xff,0))
        grades_surface = pygame.transform.smoothscale(grades_surface, img_size)
##        grades_surface.set_alpha(GRADES_ALPHA) #0 is transparent, 255 is opaque
        screen.blit(grades_surface, (0,0), special_flags=BLEND_ADD)
    if marks_plane != None:
        marks_surface = byte2rgba(marks_plane, (0xff,0xff,0xff))
        marks_surface = pygame.transform.smoothscale(marks_surface, img_size)
##        marks_surface.set_alpha(REF_ALPHA) #0 is transparent, 255 is opaque
        screen.blit(marks_surface, (0,0), special_flags=BLEND_ADD)
    if not obliq_slice:
        display_box(screen, 'Axis: '+ 'XYZ'[axis], pos=(25,20))
    else:
        display_box(screen)    # repeat last msg
    pygame.display.flip()
    arr = pygame.surfarray.array2d(img)
    arr = arr.T
    #make RGB 0x030303 -> 0x000003
    arr = arr & 255 
    return arr

def gradient(plane, pt):
    Gx = convolve_pt(plane, GRAD_2D_X, pt) / GRAD_OP_NORMALIZER
    Gy = convolve_pt(plane, GRAD_2D_Y, pt) / GRAD_OP_NORMALIZER
##    G = sqrt(Gx**2+ Gy**2)
##    angle = atan(float(Gy)/float(Gx))
    return Gx,Gy

def gradient3d(data, pt):
    Gx = convolve3d_pt(data, GRAD_3D_X, pt) / GRAD_3D_NORMALIZER
    Gy = convolve3d_pt(data, GRAD_3D_Y, pt) / GRAD_3D_NORMALIZER
    Gz = convolve3d_pt(data, GRAD_3D_Z, pt) / GRAD_3D_NORMALIZER
##    G = sqrt(Gx**2+ Gy**2)
##    angle = atan(float(Gy)/float(Gx))
    return Gx,Gy,Gz

def convolve_ndim(img, kernel, normalizer):
    return ndimage.filters.convolve(img, kernel) / normalizer

def calc_gradient3d(box_data):
    "Disribute job, 1 core per dimension"
    debug('* Calc gradientds3d of {0} .. '.format(box_data.shape))
    img = box_data.astype(int32)
    #leave one core out, to be responsive
    ncores = multiprocessing.cpu_count() - 1
    pool = multiprocessing.Pool(ncores)
    results = [pool.apply_async(convolve_ndim,(img, kernel, GRAD_3D_NORMALIZER))
               for kernel in (GRAD_3D_Z, GRAD_3D_Y, GRAD_3D_X)]
    pool.close()
    pool.join()
    ret = [r.get() for r in results]
    #make into single cube of triplets
    ret = transpose(ret, axes=(1,2,3,0))
    debug('* .. done')
    return ret

def trilinear(data, (x,y,z), topleft, step):
    "http://en.wikipedia.org/wiki/Trilinear_interpolation#Method"
    x0, y0, z0 = topleft
    p1 = topleft + step*ones(3)
    tshape = array(data.shape)
    over = p1>=tshape
    p1[over] = tshape[over]-1
    x1, y1, z1 = p1

    #d = (pt - pt0)/(pt1-pt0)
    xd = 0 if x1==x0 else (x - x0)/(x1 - x0)
    yd = 0 if y1==y0 else (y - y0)/(y1 - y0)
    zd = 0 if z1==z0 else (z - z0)/(z1 - z0)

    c00 = data[x0, y0, z0] * (1 - xd) + data[x1, y0, z0] * xd 
    c10 = data[x0, y1, z0] * (1 - xd) + data[x1, y1, z0] * xd 
    c01 = data[x0, y0, z1] * (1 - xd) + data[x1, y0, z1] * xd 
    c11 = data[x0, y1, z1] * (1 - xd) + data[x1, y1, z1] * xd
    
    c0 = c00*(1 - yd) + c10*yd
    c1 = c01*(1 - yd) + c11*yd

    c = c0*(1 - zd) + c1 * zd    
    return c.astype(data.dtype)

def Rbf__myinit__(self, *args, **kwargs):
    """
    patch for scipy.interpolate.rbf.Rbf to prevent re-calc rbf in every task
    only the last line is actually different
    """
    self.xi = asarray([asarray(a, dtype=float_).flatten()
                       for a in args[:-1]])
    self.N = self.xi.shape[-1]
    self.di = asarray(args[-1]).flatten()

    if not all([x.size==self.di.size for x in self.xi]):
        raise ValueError("All arrays must be equal length.")

    self.norm = kwargs.pop('norm', self._euclidean_norm)
    r = self._call_norm(self.xi, self.xi)
    self.epsilon = kwargs.pop('epsilon', r.mean())
    self.smooth = kwargs.pop('smooth', 0.0)

    self.function = kwargs.pop('function', 'multiquadric')

    # attach anything left in kwargs to self
    #  for use by any user-callable function or
    #  to save on the object returned.
    for item, value in kwargs.items():
        setattr(self, item, value)

    #was self.A but it's not used elsewhere in Rbf class so just A is more clear
    A = self._init_function(r) - eye(self.N)*self.smooth
    if not hasattr(self,'nodes'):
        self.nodes = linalg.solve(A, self.di)
    
def compute_3d_rbf_reduction(pts, grads):
    "return indices of valuable pts for RBF"
    N = len(pts)
    idx = range(0,N,10)
    old_len = 0
    orig = pts.T.tolist()
    accuracy = 1
    min_accuracy = 1
    debug('starting 3D rbf reduction, total: %d pts', N)
    while accuracy>RBF_REDUCTION_ACCURACY and len(idx)>old_len:
        debug('computing 3D rbf with %d pts of %d', len(idx), N)
        rbf = compute_3d_rbf_func(pts[idx], grads[idx])
        if rbf==None:
            return None,None
        #we expect zero on the level-set
        err = abs(rbf(*orig))
        accuracy = sum(err>RBF_THRESHOLD) / float(N)
        debug('accuracy is %f', accuracy)
        if accuracy < min_accuracy:
            min_accuracy = accuracy
            best_idx = list(idx)
            best_rbf = deepcopy(rbf)
        if accuracy>RBF_REDUCTION_ACCURACY:
            isort = argsort(err)[::-1]
            i=0
            add_idx = 0
            old_len = len(idx)
            while i<len(isort) and add_idx<N/10:
                last=i
                #take only middle of cluster
                i+=1
                while i<len(isort) and abs(isort[i] - isort[last]) == 1:
                    i+=1
                ii = (last+i)/2
                if abs(isort[ii] - idx).min() > 1:
                    idx.append(isort[ii])
                    add_idx += 1
    return best_rbf, best_idx
    
def compute_3d_rbf_func(pts, grads):
    "return rbf function"
    radius = RBF_NORMAL_RADIUS
    pos_pts = [tuple((p-g*(radius/norm(g))).round().astype(int))
               for (p,g) in zip(pts,grads)]
    neg_pts = [tuple((p+g*(radius/norm(g))).round().astype(int))
               for (p,g) in zip(pts,grads)]
    pts = pos_pts + neg_pts
    sx, sy, sz = zip(*pts)
    sw = len(pos_pts)*[1] + len(neg_pts)*[-1]
    #'cubic': r**3
    #'thin_plate': r**2 * log(r)
    warnings.filterwarnings('default')
    try:
        rbf = interpolate.Rbf(sx,sy,sz,sw, function='thin_plate')
    except LinAlgError as detail:
        debug('* LinAlgError in interpolate.Rbf(): ' + str(detail))
        return None
    return rbf

def compute_3d_rbf_surface(rbf, data_shape, pts):
    """
    pts is for starting points on the surface, so every connected-component is reached.
    return ones except rbf boundary.
    """
    offsets=[]
    for i in range(-1,2):
        for j in range(-1,2):
            for k in range(-1,2):
                offsets.append((i,j,k))
    offsets = array(offsets)
    visited = set()
    stack = [p.round().astype(int) for p in pts]    
    #stack.append(pts[0].round().astype(int))
    rbf_data = ones(data_shape)
    rbf_data_shape = array(rbf_data.shape)
    debug('* Starting RBF eval on zero-set walk...')
    while stack:
        pt = stack.pop()
        neig = pt+offsets
        legit = [tuple(p) for p in neig if all(p<rbf_data_shape) and all(p>=0)]
        zset = [p for p in legit if p not in visited]
        if zset:
            zz = zip(*zset)
            rbf_data[zz] = rbf(*zz)
            zset = [p for p in zset if abs(rbf_data[p]) < RBF_THRESHOLD]
            stack.extend(zset)
            visited.update(zset)
    debug('* ...done')    
    return rbf_data, array(list(visited))

def compute_3d_rbf_slice( rbf_state, i1,i2, step, x1,x2, y1,y2, z1,z2 ):    
    XI, YI, ZI = mgrid[x1:x2:step, y1:y2:step, z1:z2:step]
    sx,sy,sz,sw = 4*[[0]]
    #patch to prevent re-calc rbf in every task
    interpolate.Rbf.__init__ = Rbf__myinit__
    rbf = interpolate.Rbf(sx,sy,sz,sw, **rbf_state)
    r = rbf(XI[i1:i2,:,:], YI[i1:i2,:,:], ZI[i1:i2,:,:])
    return r
    
def compute_3d_rbf_volume(rbf, data_shape, npts):
    "Returns value-field of the function, as 3d array"
##    warnings.filterwarnings('error')
    length = max(data_shape)
    step = 1 if RBF_FULL else length/RBF_GRID_SIZE
        
    x1, x2 = 0, data_shape[0]
    y1, y2 = 0, data_shape[1]
    z1, z2 = 0, data_shape[2]
    xti = arange(x1, x2, step).round().astype(int)
    yti = arange(y1, y2, step).round().astype(int)
    zti = arange(z1, z2, step).round().astype(int)
    XI, YI, ZI = mgrid[x1:x2:step, y1:y2:step, z1:z2:step]

##    WI = rbf(XI,YI,ZI)
    ncores = multiprocessing.cpu_count() - 1
    pool = multiprocessing.Pool(ncores)
    rbf_state = rbf.__dict__
    #remove bound methods as they are unpicklable (for multiprocessing)
    rbf_state.pop('norm')
    rbf_state.pop('_function')
    results = []

    avail = psutil.virtual_memory().available 
    debug('* npts=%d avail=%d', npts, avail)
    #leave one part alone
    mem_per_core = avail*(ncores-1)/ncores / ncores
    total_mem_needed = 2 * 8 * 3*prod(XI.shape)*2*npts
    xlen = XI.shape[0]
    mem_needed_per_x = total_mem_needed / xlen    
    x_per_slice = mem_per_core / mem_needed_per_x
    debug('x_per_slice=%d', x_per_slice)
    i, i1, i2 = 0, 0, 0
    while i2<xlen:
        i1 = i*x_per_slice
        i2 = min(xlen, i1+x_per_slice)
        r = pool.apply_async(compute_3d_rbf_slice,(rbf_state,i1,i2,step,x1,x2,y1,y2,z1,z2))
##        r = pool.apply(compute_3d_rbf_slice,(rbf_state,i1,i2,step,x1,x2,y1,y2,z1,z2))
        results += [r]        
        if (i+1)%ncores == 0:
            [r.wait() for r in results[-ncores:]]
        i+=1
        
    pool.close()
    debug('* now join()... avail=%d', psutil.virtual_memory().available)
    pool.join()
    debug('All process: '+str(all([r.successful() for r in results])))
    WI = concatenate([r.get() for r in results]).reshape(XI.shape)
    
##    WI = concatenate(results).reshape(XI.shape)
    
    if RBF_FULL:
        ret = WI.reshape(data_shape)
    else:
        debug('fill-in with linear interpolation, step=%d' % step)
        ret = linear_fill_in(data_shape, xti,yti,zti,WI)
    return ret
    
def linear_fill_in(data_shape,xti,yti,zti,WI):
    ret = zeros(data_shape, float)    
    for i,x in enumerate(xti):
        debug('{0}, {1}'.format(i,x))
        for j,y in enumerate(yti):
            for k,z in enumerate(zti):                
                ret[x,y,z] = WI[i,j,k]
                if j>0 and i>0 and k>0:
                    corner = (x-step, y-step, z-step)
                    for x1 in range(corner[0]+1,x):
                        for y1 in range(corner[1]+1,y):
                            for z1 in range(corner[2]+1,z):
                                c1 = trilinear(ret, (x1,y1,z1), corner, step)
                                ret[x1,y1,z1] = c1
    return ret

def upscale_rbf(data, newshape):
    x,y,z = data.shape
    XI,YI,ZI = mgrid[0:2:1,0:y:1,0:z:1]
    sx,sy,sz = [ravel(x) for x in (XI,YI,ZI)]
    sw = ravel(data[:2,:,:])
    rbf = interpolate.Rbf(sx,sy,sz,sw, function='thin_plate')

def compute_ground_truth(data_shape, refmarks = None):
    gt = -1*ones(data_shape)
    if refmarks==None:
        refmarks = io.loadmat(REF_MARKS_MATFILE).values()[0]
    for iPlane,segments in enumerate(refmarks):
        for seg in segments:
            if seg.size > 0:
                pts = seg.round().astype(int)
                pts = roll(pts,1,1)
                pts = enforce_clockwise(pts)
                gt[iPlane,:,:] = compute_2d_rbf_standalone(pts, data_shape[1:][::-1])
    gt[gt<0] = 0
    gt[gt>0] = 1
    return gt

def compute_2d_rbf_standalone(pts, plane_shape):
    grads = calc_contour_normals_2d(pts,pts)    
    rbf = compute_2d_rbf_func(pts, None, grads)
    ymax, xmax = plane_shape
    XI, YI = mgrid[0:xmax, 0:ymax]
    ZI = rbf(XI,YI)
    return ZI

def compute_2d_rbf_reduction(pts, plane):
    """
    return sorted indexes of important points
    """
    N = len(pts)
    #start with min 3 points, max tenth of the points
    step = min(max(1,N/3), 10)
    idx = range(0,N,step)
    best_idx = idx
    old_len = 0
    orig = pts.T.tolist()
    accuracy = 1
    min_accuracy = 1
    grads = calc_contour_normals_2d(pts,pts)
    debug('starting 2D rbf reduction, total: %d pts', N)
    while accuracy > RBF_REDUCTION_ACCURACY and  len(idx)>old_len:
        #show_normals(pts[idx],grads[idx],screen)
        debug('computing 2D rbf with %d pts of %d', len(idx), N)
        try:
            rbf = compute_2d_rbf_func(pts[idx], plane, grads[idx])
        except LinAlgError as detail:
            debug('* LinAlgError in interpolate.Rbf(): ' + str(detail))
            step -=1
            if step>0:
                idx = range(0,N,step)
                old_len=0                
                continue
            else:
                return sorted(best_idx)
        #we expect zero on the level-set
        err = abs(rbf(*orig))
        #accuracy = max(err)
        accuracy = sum(err>RBF_THRESHOLD) / float(N)
        debug('accuracy is %f', accuracy)
        if accuracy < min_accuracy:
            min_accuracy = accuracy
            best_idx = list(idx)        
        if accuracy > RBF_REDUCTION_ACCURACY:
            isort = argsort(err)[::-1]
            i=0
            add_idx = 0
            old_len = len(idx)
            while i<len(isort) and add_idx<N/10:
                last=i
                #take only middle of cluster
                i+=1
                while i<len(isort) and abs(isort[i] - isort[last]) == 1:
                    i+=1
                ii = (last+i)/2
                if abs(isort[ii] - idx).min() > 2:
                    idx.append(isort[ii])
                    add_idx += 1
    return sorted(best_idx)

def compute_2d_rbf_func(pts, plane=None, grads = None):
    "Returns value-field of the function as 2d array"
    sx, sy = zip(*pts)
    xmin,xmax = min(sx), max(sx)
    ymin,ymax = min(sy), max(sy)
    if grads==None:
        grads  = array([gradient(plane,p) for p in pts])
    #ngrads = array([g/norm(g) for g in grads])
    radius = RBF_NORMAL_RADIUS
    #inside body
    pos_pts = [tuple((p-g*(radius/norm(g))).round().astype(int))
               for (p,g) in zip(pts,grads)]
    #outside body
    neg_pts = [tuple((p+g*(radius/norm(g))).round().astype(int))
               for (p,g) in zip(pts,grads)]
    pts = pos_pts + neg_pts
    sx, sy = zip(*pts)
    sz = len(pos_pts)*[1] + len(neg_pts)*[-1]
    #'cubic': r**3
    #'thin_plate': r**2 * log(r)
    rbf = interpolate.Rbf(sx,sy,sz, function='thin_plate')
    return rbf

def compute_2d_rbf(rbf, plane_shape):
    length = max(plane_shape)
    step = 1 if RBF_FULL else length/RBF_GRID_SIZE
    if RBF_BBOX_ONLY:
        step = 1 if RBF_FULL else (20+max(xmax-xmin,ymax-ymin))/RBF_GRID_SIZE
        x1, x2 = xmin-10, xmax+10
        y1, y2 = ymin-10, ymax+10
        #xti = linspace(xmin, xmax, RBF_GRID_SIZE)
        #yti = linspace(ymin, ymax, RBF_GRID_SIZE)
    else:
        x1, x2 = 0, plane_shape[1]
        y1, y2 = 0, plane_shape[0]
    XI, YI = mgrid[x1:x2:step, y1:y2:step]
    ZI = rbf(XI,YI)
##    pygame.display.set_caption('hover for RBF... any key to continue')
##    print('hover for RBF... hit a key to continue')
##    wait_event(show=ZI.swapaxes(0,1))
    """
    flatZI = [a for b in ZI for a in b]
    zmin, zmax = min(flatZI), max(flatZI)
    zrange = zmax-zmin
    normZI = [[(a-zmin)/zrange*2-1 for a in b] for b in ZI]
    """
    return ZI

def compute_2d_rbf_img(ZI):
    ret = zeros(ZI.shape, dtype=int)
    x1, x2 = 0, ZI.shape[0]
    y1, y2 = 0, ZI.shape[1]
    length = max(ZI.shape)
    step = 1 if RBF_FULL else length/RBF_GRID_SIZE
    xti = arange(x1, x2, step)
    yti = arange(y1, y2, step)
    xti = xti.round().astype(int)[:-1]
    yti = yti.round().astype(int)[:-1]
    #clamp ZI values to [-1,1]
    normZI = clip(ZI,-1,1)
##    radius = (xmax-xmin)/len(xti)/2
    #halfstep = step/2
    for j,y in enumerate(yti):
        for i,x in enumerate(xti):
            v = normZI[i,j]
            if abs(v)<CONTOUR_THRESH:
                c = RBF_CONTOUR_COLOR
            else:
                b = abs(int(v*INSIDE_BRIGHTNESS))
                #green inside, purple outside
                c = b+(b<<16)  if v<0 else  b<<8
##            pygame.draw.circle(ret, c, (x,y), radius)
            if v>0:
                #r = pygame.Rect(x-halfstep, y-halfstep, step,step)
                #ret.fill(WHITE, rect=r)
                ret[y,x] = RBF_INSIDE_COLOR
            #pxarray[x,y] = c #gray2rgb(gray)
            if not RBF_FULL and j>0 and i>0:
                topLeftX, topLeftY = x-step,y-step
                for x1 in range(topLeftX+1,x):
                    for y1 in range(topLeftY+1,y):
                        c1 = bilinear(ret,(x1,y1),(topLeftX,topLeftY), step)
                        ret[y1,x1] = c1
    return ret

def draw_scribble(segments):
    for seg in segments:
        x1,y1 = seg[0]
        for (x,y) in seg:
            pygame.gfxdraw.line(screen, x1,y1,x,y, WHITE_RGB)
            x1,y1 = x, y
    pygame.display.flip()


def draw_pts(pts, surf):
    for (x,y) in pts:
        pygame.gfxdraw.circle(surf,x,y,1,WHITE_RGB)
    pygame.display.flip()

def get_scribble():
    "return list of pt-lists (segments)"
    segments = []
    done = False
    while not done:
        #for event in pygame.event.get():
        event = pygame.event.wait()
        if event.type == MOUSEMOTION:
            if event.buttons[0]:
                x2, y2 = pygame.mouse.get_pos()
                segments[-1].append((x2,y2))
                pygame.gfxdraw.line(screen, x1,y1,x2,y2, WHITE_RGB)
                x1,y1 = x2,y2
        elif event.type == MOUSEBUTTONDOWN:
            if event.button == 1:
                x1,y1 = event.pos
                segments.append([(x1,y1)])
        elif event.type == KEYDOWN:
            done = True
        pygame.display.flip()
    return segments

def lockdown(pts, plane):
    """return Greedy Snakes points
    """
    mask = array2cv(plane.astype('uint8'))
    a,b,c = [.1],[.1],[.1]
    pts = [tuple(p) for p in pts]
    #only supports IPL_DEPTH_8U
    pts = cv2.cv.SnakeImage(mask,pts,a,b,c,(19,19),
                        (cv.CV_TERMCRIT_ITER | cv.CV_TERMCRIT_EPS, 100, 0.001))
    return array(pts)

def show_scribble_rbf(pts, plane):
    global screen, img
    rbf_data = compute_2d_rbf(pts, plane)
    rbf_plane = compute_2d_rbf_img(rbf_data)
##    rbf_plane = zeros(plane.shape)
    #rbf_surface = pygame.surfarray.make_surface(rbf_plane)    
    rbf_surface = byte2rgba(rbf_plane)
##    img_size = VIEW_FACTOR * array(rbf_plane.shape)[::-1]
##    rbf_surface = pygame.transform.smoothscale(rbf_surface, img_size)    
    pygame.image.save(rbf_surface, 'rbf.JPG')
    rbf_surface.set_alpha(RBF_ALPHA) #0 is transparent, 255 is opaque
    show_gradients(plane, pts, rbf_surface)
    screen.blit(img, (0,0)) #,special_flags=BLEND_ADD)
    draw_pts(pts, screen)
    pygame.display.set_caption('After-snake points, <SPACE> to continue')
##    wait_event()
    screen.blit(rbf_surface, (0,0))
    pygame.display.flip()
    pygame.display.set_caption('RBF with Normals, <SPACE> to continue')
    debug('Showing RBF with Normals, <SPACE> to continue')
    
#2 useful func:
#cv2.cv.SnakeImage
#cv2.grabCut, which outputs back to the given mask

#Numpy and OpenCV have different orderings for dimensions, hence the tuple reversals here.

def cv2array(im):
  depth2dtype = {
        cv.IPL_DEPTH_8U: 'uint8',
        cv.IPL_DEPTH_8S: 'int8',
        cv.IPL_DEPTH_16U: 'uint16',
        cv.IPL_DEPTH_16S: 'int16',
        cv.IPL_DEPTH_32S: 'int32',
        cv.IPL_DEPTH_32F: 'float32',
        cv.IPL_DEPTH_64F: 'float64',
    }

  arrdtype=im.depth
  a = numpy.fromstring(
         im.tostring(),
         dtype=depth2dtype[im.depth],
         count=im.width*im.height*im.nChannels)
  a.shape = (im.height,im.width,im.nChannels)
  return a

def array2cv(a):
  dtype2depth = {
        'uint8':   cv.IPL_DEPTH_8U,
        'int8':    cv.IPL_DEPTH_8S,
        'uint16':  cv.IPL_DEPTH_16U,
        'int16':   cv.IPL_DEPTH_16S,
        'int32':   cv.IPL_DEPTH_32S,
        'float32': cv.IPL_DEPTH_32F,
        'float64': cv.IPL_DEPTH_64F,
    }
  try:
    nChannels = a.shape[2]
  except:
    nChannels = 1
  cv_im = cv.CreateImageHeader((a.shape[1],a.shape[0]),
          dtype2depth[str(a.dtype)],
          nChannels)
  cv.SetData(cv_im, a.tostring(),
             a.dtype.itemsize*nChannels*a.shape[1])
  return cv_im

def update_cut_plane(cut_plane, triplet=None, axis=None, iPlane=None):
    if cut_plane==None:
        return
    if triplet==None:
        X,Y,Z = data.shape - ones(3)
        triplet = empty((3,3))
        if axis==0:
            triplet[0,:] = iPlane,0,0
            triplet[1,:] = iPlane,Y,0
            triplet[2,:] = iPlane,0,Z
        elif axis==1:
            triplet[0,:] = 0,iPlane,0
            triplet[1,:] = X,iPlane,0
            triplet[2,:] = 0,iPlane,Z
        elif axis==2:
            triplet[0,:] = 0,0,iPlane
            triplet[1,:] = X,0,iPlane
            triplet[2,:] = 0,Y,iPlane
    triplet *= array(SPACING)
##    mlab.plot3d(*triplet.T)
##    mlab.points3d(*triplet.T)
    a,b,c,d = calc_plane_def(triplet)
    n = array([a,b,c]) #normal
    n = n / norm(n)    #normalize to size 1
    implicit_plane = cut_plane.filters[0]
    implicit_plane.widget.origin = triplet[0]
    implicit_plane.widget.normal = n  

cut_plane = None
def show_data(data, pts, rbf_data, normals, grades, ref_data, triplet=None):
    """
    @param triplet is for the current yet-unmarked "best next" plane
    """
    global cut_plane
    if mlab.gcf():  #get current figure
        mlab.clf()  #clear it
    else:
        mlab.figure(bgcolor=(0, 0, 0), size=(900,1000))

    #show friendly to the orientation of X-axis in 2D image 
    scene = mlab.get_engine().scenes[0]
    scene.scene.camera.position = [565, -144, 252]
    scene.scene.camera.focal_point = [49, 56, 86]
    scene.scene.camera.view_angle = 30.0
    scene.scene.camera.view_up = [-0.355, -0.934, -0.027]
##    scene.scene.camera.clipping_range = [315.61130981057175, 910.08620958969675]
    scene.scene.camera.compute_view_plane_normal()
##    scene.scene.render()
    
    if ref_data!=None:
        ref_int = zeros(ref_data.shape,int32)
        ref_int[ref_data>0] = 0xFF
        src_ref = mlab.pipeline.scalar_field(ref_int)
        src_ref.spacing = SPACING
        iso_ref = mlab.pipeline.iso_surface(src_ref)
        iso_ref.actor.property.opacity = 0.1
        
    if rbf_data!=None:
        rbf_int = 255 * rbf_data.astype(int32)
        src_rbf = mlab.pipeline.scalar_field(rbf_int)
        src_rbf.spacing = SPACING
        for _triplet,_,_,_axis,_iPlane,_,_ in marks:
            cp = mlab.pipeline.cut_plane(src_rbf)
            surf = mlab.pipeline.surface(cp)
            surf.enable_contours = True
            surf.contour.auto_contours = False
            surf.actor.property.line_width = 4
            cp.children[0].scalar_lut_manager.lut_mode = 'summer'
            implicit_plane = cp.filters[0]
            implicit_plane.widget.enabled = False
            update_cut_plane(cp,_triplet, _axis, _iPlane)
        if triplet!=None:
            cut_plane = mlab.pipeline.cut_plane(src_rbf)
            surf = mlab.pipeline.surface(cut_plane)
            surf.enable_contours = True
            surf.contour.auto_contours = False
            surf.actor.property.line_width = 4
            cut_plane.children[0].scalar_lut_manager.lut_mode = 'Blues'
            implicit_plane = cut_plane.filters[0]
            implicit_plane.widget.enabled = False
            update_cut_plane(cut_plane, triplet)
        
    if grades==None:
        if rbf_data!=None:
##          mlab.contour3d(voi_raw, contours=1, transparent=True)
            iso_rbf = mlab.pipeline.iso_surface(src_rbf)
            iso_rbf.actor.property.opacity = 0.3
            iso_rbf.actor.property.backface_culling = True
##          mlab.pipeline.volume(voi) #, vmin=0, vmax=0.8)
    else:
        src_grades = mlab.pipeline.scalar_field(grades)
        src_grades.spacing = SPACING
        iso_grades = mlab.pipeline.iso_surface(src_grades)
        iso_grades.actor.property.backface_culling = True
        iso_grades.contour.number_of_contours = 10
        iso_grades.actor.property.opacity = 0.3
        src_grades.children[0].scalar_lut_manager.lut_mode = 'Reds'
##        src_grades.children[0].scalar_lut_manager.reverse_lut = True
##        iso_rbf.actor.actor.visibility = False
        if triplet!=None:
            xx,yy,zz = triplet.transpose()
            cluster_centers = mlab.points3d(xx,yy,zz,
                                            scale_factor=4,color=(.1,.1,1))
    """
    src_raw = mlab.pipeline.scalar_field(data)
    voi_raw = mlab.pipeline.extract_grid(src_raw)
    voi_rbf = mlab.pipeline.extract_grid(src_rbf)
    pts = array(pts)
    xmax, ymax, zmax = pts.max(0) + RBF_SHOLDER
    xmin, ymin, zmin = pts.min(0) - RBF_SHOLDER
    voi_raw.set(x_min=xmin,x_max=xmax,y_min=ymin,y_max=ymax,z_min=zmin,z_max=zmax)
    voi_rbf.set(x_min=xmin,x_max=xmax,y_min=ymin,y_max=ymax,z_min=zmin,z_max=zmax)
    """
    # Our data is not equally spaced in all directions:
##    src.spacing = [1, 1, 1.5]
##    src.update_image_data = True
    
##    mlab.pipeline.iso_surface(voi_raw)

    #pts_grads = g[pts.T.tolist()]
    if pts!=None and SHOW_QUIVER:
        x,y,z = pts.T
        u,v,w = normals.T
##    u,v,w = gradients[x,y,z].T
        mlab.quiver3d(x,y,z,u,v,w)
##    mlab.view(-125, 54, 326, (145.5, 138, 66.5))
##    mlab.roll(-175)

##    mlab.show()

def calc_3d_gradients(data, pts):
    pos = zeros((len(pts),3),int32)
    mag = zeros((len(pts),3),int32)
    for i,pt in enumerate(pts):
        pos[i] = pt
        mag[i] = gradient3d(data, pt)
        debug("{} : {}".format(pt,mag[i]))
    return pos,mag

def dice(data, axis, i, like_shape=None):
    """like_shape means you want to embed the plane in larger plane of zeros
    in the position bbox
    """
    if data==None:
        return None
    if axis==0:
        plane = data[i,:,:]
    elif axis==1:
        plane = data[:,i,:]
    elif axis==2:
        plane = data[:,:,i]
    else:
        raise Exception('BUG!')
    ret = plane
    return ret

def histogram(data, (x,y,z),(w,h,d), cutoff = 0):
    """returns sorted OrderedDict - creates 0 for non-occuring colors
    cutoff of 0.1 will throw 10% (outliers)
    """
    hist={}
    for i in xrange(x,x+w):
        for j in xrange(y,y+h):
            for k in xrange(z,z+d):
                c=data[i,j,k]
                hist[data[i,j,k]] = hist.get(c,0) + 1
    hist = OrderedDict(sorted(hist.items(), key=lambda t: t[0]))
    num = w*h*d
    limit = (1-cutoff) * num
    while num > limit:
        minC,minN = hist.popitem(last=False)
        num -= hist[minC]
    return hist

def hist_median(dic):
    "Given a hist, return its median"
    total = sum(dic.values())
    num=0
    for k,v in dic.items():
        num += v
        if num>= total/2:
            return k

def hist_med_quick(data, (x,y,z),(w,h,d), sample=0):
    """Given data, calc histogram and return its median.
    sample=2 means take half etc.
    """
    if sample:
        data_sample = numpy.random.choice(data[x:x+w,y:y+h,z:z+d], size=w*h*d/sample, replace=True)
    else:
        data_sample = data[x:x+w,y:y+h,z:z+d].ravel()
    if data_sample.size == 0:
        return 0
    hist = bincount(data_sample)
    half = data_sample.size/2
    num = 0
    col = 0
    while num < half:
        num += hist[col]
        col += 1
    return col

def grade_patch(signature, pt, data, gradients):
    "returns a grade: 0..255, high is similar"
    src_in_med, src_out_med = signature
    dst_in_med, dst_out_med = get_signature(data,pt,gradients)
    src_in_out = float(abs(src_in_med-src_out_med))
    if src_in_out == 0:
        src_in_out = 1
    in_in  = abs(src_in_med - dst_in_med) /src_in_out    
    in_out = abs(src_in_med - dst_out_med)/src_in_out
    g = 1-max(in_in , 1-in_out)
    #debug('in_in={0} 1-in_out={1} g={2}'.format(in_in, 1-in_out, g))
    return uint8(max(g, 0)*255)

def get_signature(data, pt, gradients, size=PATCH_SIZE):
    """get signature of the patch 
    input: surface and patch-middle point
    output: tuple of 2 numbers
    """
    half_size = size/2
    gx, gy, gz = gradients[tuple(pt)]
    x = 0 if gx<0 else half_size[0]
    y = 0 if gy<0 else half_size[1]
    z = 0 if gz<0 else half_size[2]
    pt = array(pt)
    corner = array((x,y,z))
    inside_topleft = pt - corner
    outside_topleft= pt - (half_size-corner)
    hist_med_inside  = hist_med_quick(data, inside_topleft,  half_size)
    hist_med_outside = hist_med_quick(data, outside_topleft, half_size)
    return (hist_med_inside, hist_med_outside)
##    hist_inside  = histogram(data, inside_topleft,  half_size, SIGNATURE_CUTOFF)
##    hist_outside = histogram(data, outside_topleft, half_size, SIGNATURE_CUTOFF)
##    return [hist_median(h) for h in [hist_inside, hist_outside]]

def get_bounding_box(pts, dshape, shoulder = RBF_SHOULDER):
    pts = array(pts)
    x2,y2,z2 = [int(min(s,b)+1) for s,b in zip(dshape,  pts.max(0) + shoulder)]
    x1,y1,z1 = [int(max(s,b))   for s,b in zip([0,0,0], pts.min(0) - shoulder)]
    return x1,x2,y1,y2,z1,z2

def grade_pts_ground_truth(data, total_pts, gradients, pts):
    """grade a list of pts
    return low-confidence list and grades-dict"""
    grades = {}
    low = []
    for pt in pts:
        in_med, out_med = get_signature(data,pt,gradients)
        g = in_med - out_med
        grades[pt] = uint8(max(g, 0))
        if grades[pt] < GRADE_THRESHOLD: #so low confidence
            low.append(pt)
    return grades, low

def grade_pts_nn_patch(data, total_pts, gradients, signatures, pts):
    """grade a list of pts
    return low-confidence list and grades-dict"""
    low = []
    grades = {}
    tree = KDTree(total_pts, leafsize=total_pts.shape[0]+1)
    for pt in pts:
        # find 1 nearest points within 1.1 times the real NN
        distances, ndx = tree.query([pt], eps=.1, k=1)
        nearest_mark_sgnture = signatures[ndx[0]]
        grades[pt] = grade_patch(nearest_mark_sgnture, pt, data, gradients)
        if grades[pt] < GRADE_THRESHOLD: #so low confidence
            low.append(pt)
    return grades, low

def extract_surface(rbf_data,starting_pt):
    """return list of pts that make the surface"""
    offsets=[]
    for i in range(-1,2):
        for j in range(-1,2):
            for k in range(-1,2):
                offsets.append((i,j,k))
    offsets = array(offsets)
    
    #tree = KDTree(pts, leafsize=pts.shape[0]+1)
    visited = set()
    stack = []
##    q.append(tuple(pts[0].tolist()))
    stack.append(starting_pt.round().astype(int))
    rbf_data_shape = array(rbf_data.shape)
    debug('* Starting zero-set walk...')
    while stack:
        pt = stack.pop()
        neig = pt+offsets
        legit = [tuple(p) for p in neig if all(p<rbf_data_shape) and all(p>=0)]
        zset = [p for p in legit if p not in visited and \
                abs(rbf_data[p]) < RBF_THRESHOLD]
        stack.extend(zset)
        visited.update(zset)
    debug('* ...done')        
    isosurf = list(visited)
    return isosurf

def calc_gradient_polar(grad):
    set_trace()
    X,Y,Z,_ = shape(grad)
    grad = grad.reshape(X*Y*Z,3)
    polar = [(calc_angle(g), norm(g).round().astype(int)) for g in grad]
    polar = array(polar).reshape(X,Y,Z,2)
    
def grade_rbf_by_gradient(data, rbf_data, rbf_surf, pts, data_grad):
    """
    return grades cube(zeros where non-rbf), low-grade pts list, avg-grade
    """
    #detect sharp edges (transitions)
    #another option: gaussian_laplace(..,1) for pre-smooth
    #but then response has zero-crossing on edge
    c = laplace(data.astype(int))
    c = (c-c.mean())/c.std()
    c = abs(c)
    c[c < 3] = 0
    c = distance_transform_edt(c==0)
    rbf_surf_idx = where(rbf_surf>0)
    rbf_grades = c[rbf_surf_idx]
##    rbf_grades = rbf_grades.max() - rbf_grades
    grades = zeros(data.shape, uint8)
    grades[rbf_surf_idx] = rbf_grades
    low = where(rbf_grades > rbf_grades.mean())
    low = transpose(rbf_surf_idx)[low]
    return grades, low, average(rbf_grades)
    
def grade_rbf_patchwise(data, rbf_data, pts, gradients):
    """Return grades plane for visualization, and low grades pts
    Walk from user-mark pt to other RBF 0 level-set, by a front saved
    in a stack
    @todo try use deque
    """
    isosurf = extract_surface(rbf_data,pts[0])
    debug('* Grading voxels (multitaksing) ...')
    signatures = [get_signature(data, pt, gradients) for pt in pts]    
    ncores = multiprocessing.cpu_count() - 1
    pool = multiprocessing.Pool(ncores)
    niso = len(isosurf)
    pts_per_core = niso/ncores
    results = []
    for i in range(ncores):
        i1 = i * pts_per_core
        i2 = niso if i==ncores-1 else (i+1)*pts_per_core
        if GRADE_BY_NN_SIGNATURE:
            r = pool.apply_async(grade_pts_nn_patch,
                                 (data, pts, gradients, signatures, isosurf[i1:i2]))
        else:        
            r = pool.apply_async(grade_pts_ground_truth,
                                 (data, pts, gradients, isosurf[i1:i2]))
        results += [r]
    pool.close()
    pool.join()
    debug('* ... done')
    debug([r.successful() for r in results])
    #numpy sum won't work for lists
    low =           __builtins__.sum([r.get()[1] for r in results], [])
    grades_items =  __builtins__.sum([r.get()[0].items() for r in results], [])
    grades_dict = dict(grades_items)
    avg_grade = mean(grades_dict.values())
    # for visualization only:
    grades = zeros(data.shape, uint8)
    for pt,g in grades_dict.items():
        grades[pt] = g
    return grades, low, avg_grade

def grade_rbf_raster(data, rbf_data, pts, gradients):
    """return grades as 3D array
    Distribute work by giving several slices per core.
    """
    signatures = [get_signature(data, pt, gradients) for pt in pts]
    #leave 1 core for the user to surf meanwhile..
    ncores = multiprocessing.cpu_count() - 1
    pool = multiprocessing.Pool(ncores)
    grades = zeros(data.shape, uint8)
    results = []
    xsize = data.shape[0]
    x_per_core = xsize/ncores
    for i in range(ncores):
        x1 = i * x_per_core
        x2 = xsize if i==ncores-1 else  (i+1) * x_per_core
##        r = pool.apply(grade_planes,(data,rbf_data,pts,gradients,signatures,x1,x2))
        r = pool.apply_async(grade_planes,(data,rbf_data,pts,gradients,signatures,x1,x2))
        results += [r]
    pool.close()
    pool.join()
    debug([r.successful() for r in results])
    grades = vstack([r.get() for r in results])
    return grades

def grade_planes(data,rbf_data,pts,gradients,signatures,x1,x2):
    "Atomic grading-job for a core, process slices x1..x2"
    xsize, ysize, zsize = (x2-x1,) + data.shape[1:]
    grades = zeros((xsize,ysize,zsize), uint8)
    tree = KDTree(pts, leafsize=pts.shape[0]+1)
    for x in xrange(xsize):
        for y in xrange(ysize):
            for z in xrange(zsize):
                pt = (x1+x, y, z)
                if abs(rbf_data[pt]) > .1:
                    grades[x,y,z] = 0 #mark non-participating
                else:
                    #gradient = gradient3d(rbf_data, pt)
                    #debug("{},{}".format(rbf_data[pt],norm(gradient)))
                    # find 1 nearest points within 1.1 times the real NN
                    distances, ndx = tree.query([pt], eps=.1, k=1)
                    nearest_mark_sgnture = signatures[ndx[0]]
                    grades[x,y,z] = grade_patch(nearest_mark_sgnture, pt, data, gradients)
                    debug('%d,%d,%d,%g',x,y,z,grades[x,y,z])
    return grades

def calc_grade_for_axis(grades, axis):
    axis_grades = []
    for i in xrange(grades.shape[axis]):
        grades_plane = dice(grades,axis,i)
        nonzero = count_nonzero(grades_plane)
        grade = sys.maxint if nonzero==0 else float(grades_plane.sum())/nonzero
        axis_grades.append((axis, i, grade))
    return axis_grades

def calc_grade_list(grades):
    plane_grades = []
    for axis in (0,1,2):
        plane_grades.append( calc_grade_for_axis(grades, axis) )    
    ret = reduce(operator.add, plane_grades)
    #sort by grade
    ret = sorted(ret,key = lambda x: x[2])
    return ret

def pca_plane_basis(pts):
    """fit plane using PCA
    return (centroid,pca1,pca2)"""
    if len(pts)>MAX_PCA_PTS:
        #pts = random.choice(pts, size=1, replace=False)
        idx = random.randint(len(pts), size=MAX_PCA_PTS)
        pts = pts[idx]
    avail = psutil.virtual_memory().available
    if len(pts)**2*8 > avail:
        debug('Not enough memory for PCA!')
        return None
    P = array(pts)
    m = mean(P,0)
    M=P-m
    M= M.T
    U,s,Vt = scipy.linalg.svd(M)
    #a basis plus normal (all normalized to have magnitude 1)
    b1,b2,n = U[:,0], U[:,1], U[:,2]
    #scale up so the base is wider, less numeric errors
    #TODO: instead of using the singular values,
    # project all points onto b1 then set b1 to be the largest projection.
    # Projection of vector "a" over unit-sized b: (a.b)b
    # see Java code Ex3 in CG_2013: RayTracer.maxProjections 
    b1 *= sqrt(s[0])
    b2 *= sqrt(s[1])
    return [0+m, b1+m, b2+m]

def is_slice_too_big(pshape,data_shape):
    return (prod(pshape) > (2*max(data_shape))**2)

def random_slice(data_shape, pts=[], use_centroid = RANDOM_USE_CENTROID):
##    centers = empty((3,3), float)
    shape_too_big = True
    while shape_too_big:
        a = random.rand(3,3)
        triplet = a*array(data_shape)
        if pts!=[] and use_centroid:
            centers[2] = mean(pts,0) #marks centroid
            triplet[2] = centers[2]
        elif RANDOM_USE_BOX_CENTER:
            triplet[2] = .5*array(data_shape)
        trans, pshape = calc_trans(triplet, data_shape)
        shape_too_big = is_slice_too_big(pshape, data_shape)
        if shape_too_big:
            debug('* too big a slice, re-trying '+ str(pshape))    
    return triplet, trans, pshape

def where_to_slice_next(low_grade_pts, pts, data_shape):
    """pts - the marks of the user
    return triplet:
    centers of 3 largest clusters of low-confidence
    if not enough clusters, triplet for PCA fitted plane
    """
    pts = array(low_grade_pts)
##    D = distance.squareform(distance.pdist(pts))
##    db = DBSCAN().fit(D, eps=1.5, min_samples=10, metric='precomputed')
    debug('* num of low-grade: {0}'.format(len(pts)))
    save('uncertainty',pts)
    db = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES).fit(pts)
    core_samples = db.core_sample_indices_
    labels = db.labels_

    #Number of clusters in labels
    label_set = set(labels)
    n_clusters = len(label_set)
    #label -1 is the noise "cluster"
    if -1 in label_set: n_clusters -= 1
    debug('n_clusters={0}'.format(n_clusters))

    centers = empty((3,3), float)
    #take only (up to 3) largest clusters
    sizes = [sum(labels == k) for k in label_set if k!=-1]
    largest_labels = argsort(sizes)[::-1]
    i=0
    ki=0
    while i<3 and ki<len(largest_labels):
        k = largest_labels[ki]
        ##class_members = [index[0] for index in np.argwhere(labels == k)]
        cluster_size = sizes[k]        
        debug('size #{0} = {1}'.format(k,cluster_size))
        next_size = sizes[largest_labels[ki+1]]  if ki+1<len(largest_labels) else 0
        cluster_core_samples = [index for index in core_samples
                                if labels[index] == k]
        cluster_pts = pts[cluster_core_samples]
        if i<2 and cluster_size/2 >= next_size:
            debug('next_size={0}'.format(next_size))
            debug('cluster is >2*next, so use its PCA basis')
            centers[i:] = pca_plane_basis(cluster_pts)[:3-i]
            restart_i=i
            i=3
        else:            
            centers[i] = cluster_pts.mean(axis=0)
            restart_i=i
            too_close=False
            for j in range(i):
                if norm(centers[i]-centers[j])<DBSCAN_EPS:
                    too_close=True
            if too_close:
                debug('too close centers, try next cluster')
            else:
                i+=1
        if i==3:
            triplet = centers
            trans, pshape = calc_trans(triplet, data_shape)
            debug('slice shape: {0}'.format(pshape))
            if is_slice_too_big(pshape, data_shape):
                debug('too big slice, move to next cluster')
                i=restart_i
            elif MIN_SLICE_MASS:
                plane_gt = extract_slice(gt_data, trans, pshape)
                mass = sum(plane_gt>127)
                debug('mass=%d',mass);
                if mass<MIN_SLICE_MASS:
                    debug('mass too small, move to next cluster')
                    i=restart_i
        ki+=1
        
    if i<3:
        centers[i:] = pca_plane_basis(pts)[:3-i]
        triplet = centers
        trans, pshape = calc_trans(triplet, data_shape)
        debug('slice shape: {0}'.format(pshape))
        if is_slice_too_big(pshape, data_shape):
            debug('slice too big after all, return random slice (use marks-centroid)')
            triplet, trans, pshape = random_slice(data_shape,pts,use_centroid=True)
    plane = extract_slice(data, trans, pshape)
    debug(pformat(triplet))
    return triplet, trans, pshape, plane

def intersect_plane_with_box(tshape, plane_def):
    """For each 'unknown' dimension, let the knowns be the other dimensions.
    The known ones define the edges of the box.
    Then find the unknown by the plane-equation.
    But this gives also out-of-box points, so filter them out.
    """
    normal,d = plane_def[:3],plane_def[3]
    pts = empty((3*2*2,3),float)
    i=0
    for unknown in (0,1,2):
        knowns = [x for x in (0,1,2) if x!=unknown]
        for known1_val in (0,tshape[knowns[0]]):
            for known2_val in (0,tshape[knowns[1]]):
                known_coefs = delete(normal,unknown)
                unknown_coef = normal[unknown]
                known_vals = (known1_val, known2_val)
                pts[i][knowns] = known_vals
                pts[i][unknown] = -(dot(known_coefs,known_vals)+d) / float(unknown_coef)
                i+=1
    #return only those within box
    # '*' is logical_and
    return pts[all(pts>=0,axis=1) * all(pts<=tshape,axis=1)]

def trilinear_interpolate(data, pt):
    "http://en.wikipedia.org/wiki/Trilinear_interpolation#Method"
    x,y,z = pt

    p0 = floor(pt)
    x0, y0, z0 = p0
    p1 = ceil(pt)
    tshape = array(data.shape)
    over = p1>=tshape
    p1[over] = tshape[over]-1
    x1, y1, z1 = p1

    #d = (pt - pt0)/(pt1-pt0)
    xd = 0 if x1==x0 else (x - x0)/(x1 - x0)
    yd = 0 if y1==y0 else (y - y0)/(y1 - y0)
    zd = 0 if z1==z0 else (z - z0)/(z1 - z0)

    c00 = data[x0, y0, z0] * (1 - xd) + data[x1, y0, z0] * xd 
    c10 = data[x0, y1, z0] * (1 - xd) + data[x1, y1, z0] * xd 
    c01 = data[x0, y0, z1] * (1 - xd) + data[x1, y0, z1] * xd 
    c11 = data[x0, y1, z1] * (1 - xd) + data[x1, y1, z1] * xd
    
    c0 = c00*(1 - yd) + c10*yd
    c1 = c01*(1 - yd) + c11*yd

    c = c0*(1 - zd) + c1 * zd    
    return c.astype(data.dtype)

def calc_plane_def((p1,p2,p3)):
    "return plane params (a,b,c,d)"
    normal = cross(p2-p1,p3-p1)
    a,b,c = normal.astype(float)
    d = -dot(normal, p1)
    return (a,b,c,d)

def calc_corners_xy((a,b,c,d), brect):
    x,y = brect[0,:2] #xmin, ymin
    z = -(a*x+b*y+d)/c    
    topleft = array([x,y,z])
    x = brect[1,0] #xmax
    z = -(a*x+b*y+d)/c
    topright = array([x,y,z])
    x,y = brect[0,0], brect[1,1]  #xmin, ymax
    z = -(a*x+b*y+d)/c
    botleft = array([x,y,z])
    return topleft, topright, botleft

def calc_corners(axes, abcd, brect):
    ax1,ax2 = axes
    ax3 = 3-sum(axes)
    dims = [1 if a in axes else 0 for a in (0,1,2)]
    dims = array(dims+[1])
    coeff3 = dot(1-dims, abcd)
    #last 1 for later extracting missing coord: -dot(abcd,pt)/coeff3
    pt = array([0,0,0,1], float)
    
    pt[[ax1,ax2,ax3]] = brect[0,ax1], brect[0,ax2], 0
    pt[ax3] = -dot(abcd,pt) / coeff3
    topleft = array(pt[:3])
    
    pt[[ax1,ax3]] = brect[1,ax1], 0
    pt[ax3] = -dot(abcd,pt) / coeff3
    topright = array(pt[:3])
    
    pt[[ax1,ax2,ax3]] = brect[0,ax1], brect[1,ax2], 0
    pt[ax3] = -dot(abcd,pt) / coeff3
    botleft = array(pt[:3])
    return topleft, topright, botleft

def calc_trans_for_corners(topleft, topright, botleft):    
    w = norm(topright - topleft)
    h = norm(botleft - topleft)
    t = topleft  #the translate
    A0 = (topright - t)/w  #first  column
    A1 = (botleft - t)/h   #second column
    A = array((A0,A1)).T
    return (A,t), (int(h),int(w))

def calc_trans(triplet, dshape):
    """return trans 2d->3d and the slice shape"""
    abcd = calc_plane_def(triplet)
    pts = intersect_plane_with_box(dshape, abcd)
    brect = [(pts[:,d].min(),pts[:,d].max()) for d in (0,1,2)]
    brect = array(brect).T

    min_area = -1
    for axes in ((0,1),(1,2),(0,2)):
        topleft, topright, botleft = calc_corners(axes, abcd, brect)
        trans, (h,w) = calc_trans_for_corners(topleft,topright,botleft)
        debug('area=%d',h*w)
        if h*w<min_area or min_area==-1:
            min_area = h*w
            ret = (trans,axes), (h,w)
    return ret
    
def extract_slice_part(data,trans,w,ystart,yend):
    box_max = array(data.shape)
    ysize = yend - ystart
    plane_part = zeros((ysize,w), data.dtype)
    (A,t),axes = trans
    for yi in range(ysize):
        y = ystart + yi
        for x in range(w):
            pt3d = dot(A,(x,y)) + t
            if all(pt3d >= 0) and all(pt3d < box_max):
                plane_part[yi,x] = trilinear_interpolate(data,pt3d)
    return plane_part

def extract_slice(data, trans, pshape):
    """optional params meaning - see dice() above
    Algorithm:
    1. find intersections of the plane with the data box edges
    2. for these pts, find axis-oriented b-box
    3. find the trans (A,T) from R2 to R3, like X' = AX + T
       use (0,0), (0,h), (w,0) which are easy to calculate
    4. use the trans and interpolate every value in the 2D (w,h) image
    """
    if data==None:
        return None
    
    h,w = pshape

    ncores = multiprocessing.cpu_count() - 1
    pool = multiprocessing.Pool(ncores)
    results = []
    y_per_core = h/ncores
    for i in range(ncores):
        y1 = i * y_per_core
        y2 = h if i==ncores-1 else  (i+1) * y_per_core
        if EXTRACT_SLICE_MULTICORE:
            r = pool.apply_async(extract_slice_part,(data,trans,w,y1,y2))
####        r = pool.apply(extract_slice_part,(data,trans,w,y1,y2))
        else:
            r = extract_slice_part(data,trans,w,y1,y2)        
        results += [r]        
    pool.close()
    pool.join()    
##    debug([r.successful() for r in results])
    if EXTRACT_SLICE_MULTICORE:    
        plane = vstack([r.get() for r in results])
    else:
        plane = vstack(results)
    return plane

def enforce_clockwise(pts):
    """If most of the points take right-turn, do nothing.
    Otherwise, reverse the list in-place"""
    total = 0
    for i in xrange(len(pts)):
        p1,p2,p3 = [array(pts[x]) for x in (i-1, i, (i+1)%len(pts))]
        x1,y1,x2,y2,x3,y3 = concatenate((p1,p2,p3))
        #turn sign
        z = x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2)
        total += sign(z)
    if total<0:
        pts = pts[::-1] 
    return pts
    
def load_ref_marks():
    debug('* Loading ref marks..')
    global marks, imark
    imark = 0
    pts3d = []
    refmarks = io.loadmat(REF_MARKS_MATFILE).values()[0]
    for iPlane,segments in enumerate(refmarks):
        if USE_HALF_REF_ONLY and iPlane%2==1:
            continue
        for pts in segments:
            if pts.size > 0:
                if RESAMPLE:
                    pts = pts / RESAMPLE_XY
                pts = pts.round().astype(int)
                pts = enforce_clockwise(pts)
                mark = (None, None, pshape, REF_MARKS_AXIS, iPlane/RESAMPLE_Z, [pts], array([pts]))
                marks.insert(imark, mark)
                debug("added new: %d" % imark)
                imark += 1
    ##            cPickle.dump(mark, open('mark{0}.pkl'.format(imark),'wb'))
                p2d = roll(pts,1,1) #swap to (row,col)                            
                #insert missing axis value, iPlane
                p3d = insert(p2d, REF_MARKS_AXIS, iPlane, 1)
                pts3d.append(p3d)
    total_pts = concatenate(pts3d)
    if RESAMPLE:
        total_pts /= RESAMPLE_ZXY
    total_pts = total_pts.round().astype(int)
    ref_byte = zeros(data.shape, uint8)
    for p in total_pts:
        ref_byte[tuple(p)] = 0xFF
    debug('* ..done')
    return ref_byte

def change_view_factor(marks, old_factor, new_factor):
    """@TODO changing up and down loses accuracy - use only original marks
    This way this function is not needed
    """
    for im,mark in enumerate(marks):
        (triplet, trans, pshape, axis, i, segments, pts) = mark
        pts = pts * new_factor / old_factor
        seg = array(segments) * new_factor / old_factor
        segments = [s for s in seg]
        marks[im] = (triplet, trans, pshape, axis, i, segments, pts)
    return marks

def find_too_close(total_pts, new_pts):
    if len(total_pts)==0 or len(new_pts)==0:
        return []
    total_tree = KDTree(total_pts, leafsize=4)
    new_tree =   KDTree(new_pts,   leafsize=4)
    near = new_tree.query_ball_tree(total_tree, r=TOO_CLOSE_DISTANCE)
    bb = [x!=[] for x in near]
    return nonzero(bb)[0]

def eliminate_too_close(total_pts, new_pts):
    if len(total_pts)==0 or len(new_pts)==0:
        return total_pts + new_pts
    total_tree = KDTree(total_pts, leafsize=4)
    new_tree =   KDTree(new_pts,   leafsize=4)
    near = total_tree.query_ball_tree(new_tree, r=TOO_CLOSE_DISTANCE)
    bb = [x==[] for x in near]
    tt = array(total_pts)
    tt = tt[nonzero(bb)]
    ret = tt.tolist() + new_pts
    return ret
  
def sample_segment_by_angle(pts):
    pts = pts[::3]
    angles = zeros(len(pts))
    for i,p in enumerate(pts):
        prev, succ = pts[(i-1)%len(pts)], pts[(i+1)%len(pts)]
        a, b = p-prev, p-succ
        cosine = dot(a,b)/(norm(a)*norm(b))
        angles[i] = arccos(round(cosine,ROUND_PRECISION))

    seglen = len(pts)/SAMPLE_INTERVAL  if len(pts)>3*SAMPLE_INTERVAL else len(pts)
    maxangle = sort(angles)[seglen-1]    
    seg = [p for i,p in enumerate(pts) if angles[i]<=maxangle]
    interval = len(seg)/seglen
    seg = seg[::interval]
##    #use stable sort to preserve order
##    best = argsort(angles,kind='mergesort')[:seglen]
##    seg = pts[best]
    return seg
        
def sample_segment(pts, plane):
    global auto
    if not auto:
        pts = pts[::VIEW_FACTOR]
    idx = compute_2d_rbf_reduction(pts, plane)
##    idx = range(0,len(pts),5)
    seg = pts[idx] 
    #for cyclic intervals (around a body) -
    # verify that last point isn't too close to start point
    if len(seg)>1:
        start = array(seg[0])
        if norm(start-seg[-1]) < .5*norm(seg[1]-start):
            seg=seg[:-1]
    return seg

def pts_2d_to_3d(pts, trans, axis, iPlane):
    stp = array(pts) / VIEW_FACTOR
    if trans:
        (A,t),axes = trans
        p3d = array([dot(A,x)+t for x in stp])    
    else:
        stp = roll(stp,shift=1,axis=1) #swap to (row,col)
        #insert missing axis value, iPlane
        p3d = insert(stp, axis, iPlane, 1)                              
    return p3d

def find_couple(plane, seq, start=(0,0)):
    """find couple that the 1st has >2 neighbors
    so it is not antenna but meaningful body edge"""
    h,w = plane.shape
    y,x = start
    while y<h:        
        if plane[y,x-1]==seq[0] and plane[y,x]==seq[1]:
            #[:-1] to exclude repeating upper-left
            locs = (y,x) + NEIGHBORS_OFFSETS_CLOCKWISE[:-1]  
            vals = plane[locs.T.tolist()]
            if sum(vals==1) > 2:
                break
        x+=1
        if x==w:
            y+=1
            x=1 #not from zero (looking for a couple)
    return (y,x) if y<h else None

def auto_mark_blob(plane, pt):
    """find [0,1] but not [0,1,0]
    and set pt to pos-of-1
    """
    seg = []
    seg.append(pt)
    plane[pt] = 2
    target=array([0,1], uint8).tostring()
    while pt!=None:
        locs = pt + NEIGHBORS_OFFSETS_CLOCKWISE
        vals = plane[locs.T.tolist()]
        vstr = vals.tostring()
        targetpos = vstr.find(target)  #/vals.itemsize
        pos = targetpos
        #skip antenna
        while targetpos!=-1 and vstr[(pos+2)%len(vstr)]=='\x00':
            targetpos = vstr[pos+1:].find(target)  #/vals.itemsize
            pos = pos+1 + targetpos
        if targetpos==-1:
            pt=None
        else:
            pt = tuple(locs[pos+1])
            seg.append(pt)
            plane[pt] = 2
    return seg

def write_plane(plane):
    with open('a.txt','wt') as f:
        for l in plane:
            f.write(str(l).replace('\n','')[1:-1])
            f.write('\n')
            
def read_plane():
    s=[]
    with open('a.txt','rt') as f:
        for l in f:
            a=fromstring(l,dtype=int,sep=' ')
            s.append(a)
    return array(s)
            
def auto_mark(orig_plane):
    """get plane of 0/1, change boundary to 2 in process
    return new_segments
    as if human marked each closed shape around mass clockwise
    return only segments with min. 3 pts
    """
    segs = []
    pt = (0,0)  
    plane = (orig_plane > 127).astype(uint8)
    pt = find_couple(plane, (0,1), pt)
    while pt!=None:
        seg = auto_mark_blob(plane, pt)
        if len(seg)>3:
            #roll since we worked locally here in form (y,x) 
            seg = roll(seg,shift=1,axis=1)
            seg = VIEW_FACTOR * array(seg)            
            segs.append(seg.tolist())
        pt = find_couple(plane, (0,1), pt)
    return segs

def del_mark(marks,imark,marks_data):
    new_segments = []
    if imark>=0:
        (triplet, trans, pshape, axis, iPlane, segments, pts) = marks.pop(imark)
        for seg_pts in pts:
            p3d = pts_2d_to_3d(seg_pts, trans, axis, iPlane)
            for p in p3d:
                marks_data[tuple(p)] = 0                    
        debug("Deleted {0}".format(imark))
        imark = -1
    return new_segments, imark, marks_data

def auto_events(num_events,smart=True):
    #first obtain 2 marks
    if AUTO_EVENTS_LOAD:
        e = pygame.event.Event(KEYDOWN,key=K_a)
        pygame.event.post(e)
    else:
        if AUTO_EVENTS_INIT_RANDOM:
            if RANDOM_ON_AXIS:
                e = pygame.event.Event(KEYDOWN,key=K_b)
            else:
                e = pygame.event.Event(KEYDOWN,key=K_o)
            pygame.event.post(e)
        e = pygame.event.Event(KEYDOWN,key=cCE)
        pygame.event.post(e)
        e = pygame.event.Event(KEYDOWN,key=K_n)
        pygame.event.post(e)
        e = pygame.event.Event(KEYDOWN,key=K_s)
        pygame.event.post(e)
        if AUTO_EVENTS_INIT_RANDOM:
            if RANDOM_ON_AXIS:
                e = pygame.event.Event(KEYDOWN,key=K_b)
            else:
                e = pygame.event.Event(KEYDOWN,key=K_o)
        else:
            e = pygame.event.Event(KEYDOWN,key=INIT_AXIS_2_KEY)
        pygame.event.post(e)
        e = pygame.event.Event(KEYDOWN,key=K_SPACE)
        pygame.event.post(e)
        e = pygame.event.Event(KEYDOWN,key=K_n)
        pygame.event.post(e)
        e = pygame.event.Event(KEYDOWN,key=K_s)
        pygame.event.post(e)
    e = pygame.event.Event(KEYDOWN,key=K_RETURN)
    pygame.event.post(e)
    for i in range(num_events):
        if smart:
            #rbf grades
            e = pygame.event.Event(KEYDOWN, key=K_t)
            pygame.event.post(e)
            e = pygame.event.Event(KEYDOWN, key=K_w)
        else:
            if RANDOM_ON_AXIS:
                e = pygame.event.Event(KEYDOWN, key=K_b)
            else:            
                e = pygame.event.Event(KEYDOWN, key=K_o)
        pygame.event.post(e)        
        e = pygame.event.Event(KEYDOWN,key=K_SPACE)
        pygame.event.post(e)
        e = pygame.event.Event(KEYDOWN,key=K_n)
        pygame.event.post(e)
        e = pygame.event.Event(KEYDOWN, key=K_s)
        pygame.event.post(e)
        e = pygame.event.Event(KEYDOWN, key=K_RETURN)
        pygame.event.post(e)
        e = pygame.event.Event(KEYDOWN, key=K_f)
        pygame.event.post(e)
        
def reverse_enumerate(l):
    return izip(xrange(len(l)-1, -1, -1), reversed(l))

def random_sample(cube, n = 1000,thresh=1):
    pts = where(cube>=thresh)
    pts = vstack(pts).T
    idx = random.randint(len(pts),size=n)
    return pts[idx]

def earth_mover_dist(hist1, hist2):
    e = 0
    emd = 0
    for i in range(len(hist1)):
        e = hist1[i] + e - hist2[i]
        emd += abs(e)
    return emd

def D2(cube,thresh):
    N = cube.shape[0]
    pts1 = random_sample(cube, N**3/2, thresh)
    pts2 = random_sample(cube, N**3/2, thresh)
    d = sqrt(sum((pts1-pts2)**2,axis=1))
    h,bins = numpy.histogram(d,bins=N, range=(0,N), density=True)    
    return h

def load_data():
    global RESAMPLE
    debug('* Data source:')
    if DATA_FUNC:
        debug(DATA_FUNC)
        data = eval(DATA_FUNC)(DATA_SIZE)
    elif os.path.exists(NUMPY_DATA_FILENAME):
        debug(NUMPY_DATA_FILENAME)
        data = load(NUMPY_DATA_FILENAME)
    elif os.path.exists(MHD_DATA_FILENAME):
        debug(MHD_DATA_FILENAME)
        data = read_mhd_data(MHD_DATA_FILENAME)    
    elif os.path.exists(MATLAB_DATA_FILENAME):
        debug(MATLAB_DATA_FILENAME)
        data = read_matlab_data(MATLAB_DATA_FILENAME)
    elif os.path.exists(PNG_DATA_DIRNAME):
        debug(PNG_DATA_DIRNAME)
        data = read_png_data(PNG_DATA_DIRNAME)
    elif os.path.exists(CACHE_DATA_FILENAME):
        debug(CACHE_DATA_FILENAME)
        data = load(CACHE_DATA_FILENAME)
        RESAMPLE = False
    else:
        debug(DATA)
        data = read_data_dicom(DATA)
    if RESAMPLE:
        data = resample(data, RESAMPLE_ZXY)
    with open(CACHE_DATA_FILENAME,'wb') as f:
        save(f,data)
    debug("* Data TYPE: {0}, SHAPE: {1}".format(data.dtype, data.shape))
    return data

def calc_distance():
    """see Metrics section in:
    http://promise12.grand-challenge.org/Details
    """
    debug('* calc distances...')
    rbf_data_DT = distance_transform_cdt(1-rbf_surf, metric='taxicab')
    grades = zeros(data.shape)
    dist = []
    m = multiply(rbf_surf, surfaceDT)    
    dist.append( m[m>0] )
    m = multiply(data_boundary, rbf_data_DT)
    dist.append( m[m>0] )
    avg = [average(d) for d in dist] if len(dist)>0 else []
    ptl = [percentile(d,q=95,overwrite_input=True) for d in dist] if len(dist)>0 else []
    debug('*\n* distances:')
    for i,v in enumerate(avg+ptl):
        debug('metric_%d = %g',i,v)
    if avg:
        debug('* max avg = %g', max(avg))
    if ptl:
        debug('* max 95th percentile = %d', max(ptl))
    sum_rbf_data = float(sum(rbf_data > 0))
    sum_gt_data =  float(sum(gt_data  > 0))
    dice_coeff = 2 * sum(logical_and(rbf_data>0, gt_data>0)) /  \
                (sum_rbf_data + sum_gt_data)
    debug('* dice coeff = %g', dice_coeff)
    rel = abs( sum_rbf_data / sum_gt_data - 1 )
    debug('* relative vol diff = %g', rel)
    debug('*')
    display_box(screen, 'DC = %.2f' % dice_coeff)

def render3d(grades=None,triplet=None):
    debug('* Rendering...')
    show_data(data, total_pts, rbf_data, comb_normals,
              grades, ref_data, triplet)
    debug('* Done')

def calc_rbf_surf(rbf_data):
    rbf_surf = generic_gradient_magnitude(rbf_data, prewitt)
    #normalize to about 0..1
    rbf_surf = rbf_surf / 9.0
    rbf_surf[rbf_surf <  DATA_BOUNDARY_THRESH] = 0
    rbf_surf[rbf_surf >= DATA_BOUNDARY_THRESH] = 1
    return rbf_surf.astype(uint8)
    
if __name__=='__main__':
    ts = time.strftime("%Y%m%d_%H%M%S")
    if not os.path.isdir('Logs'):
        os.mkdir('Logs')           
    logger = logging.getLogger('')
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    fh = logging.FileHandler('Logs/'+ts+'.log',mode='w')
    fh.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s %(message)s',datefmt='%H:%M:%S')
    fh.setFormatter(formatter)
    # add the handlers to logger
    logger.addHandler(ch)
    logger.addHandler(fh)    

    debug('* Started logging')
##    warnings.filterwarnings('error')
    os.environ['SDL_VIDEO_WINDOW_POS'] = '985,30'
    pygame.quit() #clean out anything running
    pygame.display.init()
##    pygame.key.set_repeat(1,50) #(delay_ms, interval_ms)
    pygame.font.init()
    data = load_data()
    is_show_rbf, is_show_grades = True, False
    axis = 0
    obliq_slice = False    
    ld = len(data)
    pts = (ld*array([[.5,.5,.5]])).astype(int)
    #pos = (ld/2*ones(3)).astype(uint)
    pos = array(data.shape)/2
    if DATA=='../s3' and MATLAB_DATA_FILENAME==None:
        pos = [ld/3,ld/2,ld/2.5]
    if START_POS:
        pos = START_POS
    iPlane = pos[axis]
    debug('iPlane=%d',iPlane)
    plane = dice(data,axis, iPlane)
    view_plane = show_slice(plane)
    new_segments = []
    done = False
    marks = []  #list of (axis, iPlane, segments)
    imark = -1
    triplet = None
    low_grades = None
    triplet, trans, pshape = None, None, None
    marks_data, marks_plane = zeros(data.shape,uint8), None
    rbf_data, rbf_plane  = None, None
    ref_data, ref_plane = None,None
    grades, grades_plane = None, None
    pts = None
    if PRECALC_GRADIENTS:
        data_gradients = calc_gradient3d(data)
##    g = empty(data.shape+(3,))    
##    g[...,0], g[...,1], g[...,2] = numpy.gradient(data)
##    data_gradients = g
##    del(g)

    gt_data = data
    if GROUND_TRUTH_DATA:
        gt_data = load(GROUND_TRUTH_DATA) 
        if RESAMPLE_GT:
            gt_data = resample(gt_data, RESAMPLE_ZXY)
    plane_gt = dice(gt_data,axis, iPlane)
    debug('* extract gt edges')
    data_boundary = generic_gradient_magnitude(gt_data,prewitt)
    #normalize to about 0..1
    data_boundary = data_boundary / 9.0
    #data_boundary[data_boundary >= DATA_BOUNDARY_THRESH] = 1
    data_boundary[data_boundary <  DATA_BOUNDARY_THRESH] = 0
    data_boundary = (data_boundary>0)
    save('data_boundary', data_boundary.astype(uint8))
##    histGT = D2(data_boundary, 1-RBF_THRESHOLD)
##    #data_boundary = any(data_gradients,3)
##    sum_data_boundary = sum(data_boundary)

    total_pts, comb_normals, ref_data, triplet = None, None, None, None
    
    debug('* calc gt innerDT')
    innerDT = distance_transform_cdt(gt_data,metric='taxicab')
    gt_data_inv = gt_data.max() - gt_data
    debug('* calc gt outerDT')
    outerDT = distance_transform_cdt(gt_data_inv,metric='taxicab')
    surfaceDT = innerDT + outerDT - 1
    
    iteration=0
    pygame.display.set_caption('basics: Help ESCape <Arrows> SPACE Save XYZ ENTER=3D t=gradeRBF WhereNext')
    if AUTO_EVENTS_WHERE: 
        auto_events(AUTO_EVENTS_WHERE, smart=True)
    if AUTO_EVENTS_RANDOM:
        auto_events(AUTO_EVENTS_RANDOM, smart=False)
    render3d()        
    while not done:
        #for event in pygame.event.get():
        event = pygame.event.wait()
        if event.type == MOUSEMOTION:
            if event.buttons[0]:
                x2, y2 = pygame.mouse.get_pos()
                new_segments[-1].append((x2,y2))
                pygame.gfxdraw.line(screen, x1,y1,x2,y2, WHITE_RGB)
                x1,y1 = x2,y2
        elif event.type == MOUSEBUTTONDOWN:
            if event.button == 1:
                x1,y1 = event.pos
                new_segments.append([(x1,y1)])
        elif event.type == KEYDOWN:
            char = chr(event.key&0xff).lower()
            debug(pygame.key.name(event.key) + ' pressed')
            if event.key == K_F1 or event.key == K_h:
                #display_box(screen, HELP)
                debug(HELP)
            elif event.key == K_ESCAPE:
                done = True
            elif event.key == K_f:
                fh.flush()
            elif not obliq_slice and event.key in (K_RIGHT,K_LEFT,K_b) or char in "xyz":
                obliq_slice = False
                trans = None
                triplet = None
                new_segments = []
                imark = -1
                if char in 'xyz':
                    pos[axis] = iPlane
                    axis = 'xyz'.index(char)
                    iPlane = pos[axis]
                    plane = dice(data,axis,iPlane)
                    plane_gt = dice(gt_data,axis,iPlane) 
                    rbf_plane, grades_plane, ref_plane, marks_plane = \
                        [dice(d,axis,iPlane,plane.shape) \
                         for d in [rbf_data, grades, ref_data, marks_data]]
                    view_plane = show_slice(plane,
                                            rbf_plane if is_show_rbf else None,
                                            grades_plane if is_show_grades else None,
                                            ref_plane, marks_plane)
                elif event.key in (K_RIGHT, K_LEFT, K_b):
                    if event.key in (K_RIGHT,K_LEFT):
                        step = {K_RIGHT:1, K_LEFT:-1}[event.key]
                        iPlane = (iPlane+step) % data.shape[axis]
                    elif SUPPORT_RANDOM_ALIGNED:
                        iPlane = random.randint(data.shape[axis])
                    else:
                        debug('SUPPORT_RANDOM_ALIGNED is false!')
                    plane = dice(data,axis,iPlane)
                    plane_gt = dice(gt_data,axis, iPlane)                    
                    rbf_plane, grades_plane, ref_plane, marks_plane = \
                        [dice(d,axis,iPlane,plane.shape) \
                         for d in [rbf_data, grades, ref_data, marks_data]]
                    view_plane = show_slice(plane,
                                            rbf_plane if is_show_rbf else None,
                                            grades_plane if is_show_grades else None,
                                            ref_plane, marks_plane)
                    debug(iPlane)
                update_cut_plane(cut_plane, triplet, axis, iPlane)
            elif event.key in (K_UP, K_DOWN):
                if marks:
                    step = {K_DOWN:1, K_UP:-1}[event.key]
                    imark = (imark + step) % len(marks)
                    triplet,trans,pshape,axis,iPlane,new_segments,pts = marks[imark]
                    debug("{0}: {1}".format(imark, iPlane))
                    if not trans:
                        plane = dice(data,axis,iPlane)
                        plane_gt = dice(gt_data,axis, iPlane)                        
                        rbf_plane, grades_plane, ref_plane, marks_plane = \
                            [dice(d,axis,iPlane,plane.shape) \
                             for d in [rbf_data, grades, ref_data, marks_data]]
                    else:
                        debug('* extract_slice() pshape={}'.format(pshape))
                        plane = extract_slice(data, trans, pshape)
                        plane_gt = extract_slice(gt_data, trans, pshape)
                        debug('* [extract_slice()]')
                        rbf_plane, grades_plane, ref_plane, marks_plane = \
                            [extract_slice(d,trans,pshape)
                             for d in [rbf_data, grades, ref_data, marks_data]]
                        debug('* show_slice()')
                    view_plane = show_slice(plane,
                                            rbf_plane if is_show_rbf else None,
                                            grades_plane if is_show_grades else None,
                                            ref_plane, marks_plane)
                    debug('* done')                       
                    draw_scribble(new_segments)
            elif event.key == K_l and pts!=None:
                #for each segment
                for j in xrange(len(pts)):
                    pts[j] = lockdown(pts[j], view_plane)
                screen.blit(img, (0,0))
                cat_pts = concatenate(pts.tolist())
                show_gradients(view_plane, cat_pts , screen)            
            elif event.key == K_s and pts!=None and pts!=[]:
                #save current input
                #pprint(pts)
                mark = (triplet, trans, pshape, axis, iPlane, segments, pts)
                for seg_pts in pts:
                    p3d = pts_2d_to_3d(seg_pts, trans, axis, iPlane)
                    for p in p3d:
                        marks_data[tuple(p)] = 0xFF
                if imark >= 0:
                    marks[imark] = mark
                    debug("saved current: %d" % imark)
                else:
                    imark = len(marks)
                    marks.insert(imark, mark)
                    debug("added new: %d" % imark)
                cPickle.dump(mark, open('mark{0}.pkl'.format(imark),'wb'))
            elif event.key == K_SPACE:
                auto= (len(new_segments)==0)
                if auto:
                    debug('* No user marks, do auto_mark()')
                    new_segments = auto_mark(plane_gt)
                segments = [array(s) for s in new_segments]
                if not RBF_2D_REDUCTION:
                    pts = array(segments)
                else:
                    pts = array([sample_segment(seg, view_plane) for seg in segments])
                if auto and AUTO_LOCKDOWN:
                    debug('* auto lockdown')
                    for j in xrange(len(pts)):
                        if len(pts[j])>2:
                            pts[j] = lockdown(pts[j], view_plane)
##                debug(pformat(pts))
                screen.blit(img, (0,0))                
##                cat_pts = concatenate(pts.tolist())   if len(pts)>0 else []
##                show_gradients(view_plane,cat_pts,screen)
                show_contour_normals(pts,segments,screen)
                display_box(screen)    #repeat last message
            elif event.key == K_r and cat_pts!=None:
                show_scribble_rbf(cat_pts, view_plane)
            elif event.key == K_g and cat_pts!=None:
                show_gradients(view_plane, cat_pts, screen)                
            elif event.key == K_n and pts!=None:
                screen.blit(img, (0,0))            
                if len(pts)>0:
                    show_contour_normals(pts,segments,screen)
                display_box(screen)    #repeat last message
            elif event.key == K_w and low_grades!=None or event.key == K_o:
                display_box(screen, 'Computing next best slice...')
                new_segments = []
                imark = -1
                obliq_slice = True
                if event.key == K_w:
                    debug('* where_to_slice_next()')
                    triplet,trans,pshape,plane = where_to_slice_next(
                        low_grades, pts, data.shape)
                elif event.key == K_o:
                    debug('* random_slice()')
                    triplet,trans,pshape = random_slice(data.shape, pts)
                    debug(pformat(triplet))
                    plane = extract_slice(data, shape)
                display_box(screen, 'Algo done, extracting the slice...')
                plane_gt = extract_slice(gt_data, trans, pshape)
                debug('* slice shape: {0}'.format(pshape))
                if sum(plane_gt>127)<MIN_SLICE_MASS:
                    debug('mass too small, skipping slice');
                else:
                    debug('* [extract_slice()]')
                    rbf_plane, grades_plane, ref_plane, marks_plane = \
                        [extract_slice(d,trans,pshape)
                         for d in [rbf_data, grades, ref_data, marks_data]]
                    debug('* show_slice()')
    ##                is_show_rbf = False
    ##                is_show_grades = False
                    view_plane = show_slice(plane,
                                            rbf_plane if is_show_rbf else None,
                                            grades_plane if is_show_grades else None,
                                            ref_plane)
                    if len(marks)>=2:
                        render3d(grades, triplet)
                display_box(screen, 'Best next slice ready')
            elif event.key in (K_p,K_PAGEUP,K_PAGEDOWN) and grades!=None:
                trans = None
                imark = -1
                if event.key == K_p:
                    igrade = 0
                else:
                    step = {K_PAGEDOWN:1, K_PAGEUP:-1}[event.key]
                    igrade = (igrade + step) % len(plane_grades)                    
                new_segments = []
                axis, iPlane, grade = plane_grades[igrade]
                plane = dice(data,axis,iPlane)
                plane_gt = dice(gt_data,axis,iPlane)
                rbf_plane, grades_plane, ref_plane, marks_plane = \
                    [dice(d,axis,iPlane,plane.shape) \
                     for d in [rbf_data, grades, ref_data, marks_data]]
                view_plane = show_slice(plane,
                                        rbf_plane if is_show_rbf else None,
                                        grades_plane if is_show_grades else None,
                                        ref_plane, marks_plane)
            elif event.key == K_k:
                is_show_grades = not is_show_grades
                view_plane = show_slice(plane,
                                        rbf_plane if is_show_rbf else None,
                                        grades_plane if is_show_grades else None,
                                        ref_plane, marks_plane)
            elif event.key == K_i:
                is_show_rbf = not is_show_rbf
                view_plane = show_slice(plane,
                                        rbf_plane if is_show_rbf else None,
                                        grades_plane if is_show_grades else None,
                                        ref_plane, marks_plane)
            elif event.key == K_c:
                new_segments = []
                view_plane = show_slice(plane,
                                        rbf_plane if is_show_rbf else None,
                                        grades_plane if is_show_grades else None,
                                        ref_plane, marks_plane)
            elif event.key == K_d:
                new_segments, imark, marks_data = del_mark(marks,imark,marks_data)
            elif event.key == K_q:
                set_trace()
            elif event.key in [K_KP_PLUS,K_KP_MINUS]:
                step = {K_KP_PLUS:1, K_KP_MINUS:-1}[event.key]
                marks = change_view_factor(marks, VIEW_FACTOR, VIEW_FACTOR+step)
                VIEW_FACTOR += step
                view_plane = show_slice(plane,
                                        rbf_plane if is_show_rbf else None,
                                        grades_plane if is_show_grades else None,
                                        ref_plane, marks_plane)
                debug('new VIEW_FACTOR: {0}'.format(VIEW_FACTOR))
            elif event.key == K_a:
                marks = []
                imark=0
                file_exists=True
                while file_exists:
                    file_path='mark{}.pkl'.format(imark)
                    file_exists= os.path.exists(file_path)
                    if file_exists:
                        mark = cPickle.load(open(file_path,'rb'))
                        marks.insert(imark, mark)
                        debug('loaded ' + file_path)
                        imark+=1
            elif event.key == K_m:
                ref_data = load_ref_marks()
                save('ref_data',ref_data)
            elif event.key == K_t:
                display_box(screen, 'Computing RBF grades...')
                debug('* Calc rbf grades..')
                grades, low_grades, avg_grade = grade_rbf_by_gradient( \
                    data, rbf_data, rbf_surf, total_pts, data_gradients)
                debug('* avg grade: {0:g}'.format(avg_grade))
                plane_grades = calc_grade_list(grades)
                igrade = 0
                debug('* ..done')
                render3d(grades)
                debug("Done.")
                display_box(screen, 'RBF grades ready')
            elif (event.key == K_RETURN or event.key == K_KP_ENTER):
                display_box(screen, 'Computing 3D RBF...')
                debug('Calculating 3D..')
                pts3d = []
                ndx = []
                for imark,(triplet,trans,pshape,axis,iPlane,seg,pts) in enumerate(marks):
                    new_pts = []
                    for p2d in pts:
                        p3d = pts_2d_to_3d(p2d, trans, axis, iPlane)
                        if ELIM_TOO_CLOSE:
                            elim = find_too_close(pts3d, p3d.tolist())
                            debug('* mark#{0}: eliminating {1} pts of {2}'
                                  .format(imark, len(elim), len(p3d)))
                            p2d = delete(p2d, elim, axis=0)
                            p3d = delete(p3d, elim, axis=0)
                        if len(p2d)>0:
                            pts3d.extend(p3d)
                            new_pts.append(p2d)
                    if len(new_pts)>0:
                        marks[imark] = (triplet,trans,pshape,axis,iPlane,seg,array(new_pts))
                    else:
                        new_segments, imark, marks_data = del_mark(marks,imark,marks_data)
                total_pts = array(pts3d)
                debug("* Calc contour normals")                
                normals = calc_contour_normals(marks)
                debug("* Calc combined normals")                    
                comb_normals = calc_combined_normals(total_pts, data_gradients, normals)
                rbf, idx = compute_3d_rbf_reduction(total_pts, comb_normals)
                if rbf==None:
                    new_segments, imark, marks_data = del_mark(marks,imark,marks_data)
                    debug('* rbf failed, last mark deleted!')
                else:
                    #rbf = compute_3d_rbf_func(total_pts[idx], comb_normals[idx])
                    debug('* Calc rbf_volume ..')
                    rbf_data = compute_3d_rbf_volume(rbf, data.shape, len(idx))
                    rbf_data[rbf_data<=0] = 0
                    rbf_data[rbf_data> 0] = 1
                    rbf_data = rbf_data.astype(int8)
                    rbf_surf = calc_rbf_surf(rbf_data)
##                    debug('* Calc rbf_surface ..')
##                    rbf_data, rbf_surf = compute_3d_rbf_surface(rbf, data.shape, total_pts[idx])
                    debug('* done')                    
                    calc_distance()
                    render3d()
                    if SAVE_RBF:
                        save('rbf' + str(iteration), rbf_data)
                        iteration+=1
                    debug("Done.")
        pygame.display.flip()
    mlab.close(all=True) 
    mlab.get_engine().stop()
    pygame.quit()
    
