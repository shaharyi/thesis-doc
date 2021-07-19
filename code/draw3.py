"""
DRAW3.PY
2D reconstruction from cross-sections

Requirements:
Python 2.6 for multiprocessing
PyGame, NumPy, SciPy

Written by
Shahar Yifrah
august, 2011
"""

import time
import sys
import os
import copy
import multiprocessing
import warnings
import cPickle
from pprint import pprint
from pdb import set_trace,pm
from math import log, sqrt, atan
from bisect import bisect
from itertools import count, izip
from collections import OrderedDict

import pygame
from pygame.locals import *
import numpy as np
from scipy import interpolate, signal, ndimage

IMAGE_FILENAME      = 'kidney1.gif' #'brain_t2_003.jpg' #

GRAY    = 0xF0F0F0
WHITE   = 0xFFFFFF
BLACK   = 0x000000
RED     = 0xFF0000
GREEN   = 0x00FF00
BLUE    = 0x0000FF
CYAN    = 0x00FFFF
MAGENTA = 0xFF00FF

BLOB_COLOR      = 0x808000
SEG_IN_COLOR  = 0x00FF00
SEG_OUT_COLOR = 0xFF0000
CHULL_COLOR   = 0xFFFF00
LINE_BG_COLOR   = BLOB_COLOR^GREEN
LINE_FG_COLOR   = GREEN
#BORDER_COLOR    = BLUE

SOBEL_X = [[-1,-2,-1],
           [ 0, 0, 0],
           [ 1, 2, 1]]
SOBEL_Y = [[-1, 0, 1],
           [-2, 0, 2],
           [-1, 0, 1]]

#Noise Robust Gradient Operators
#http://www.holoborodko.com/pavel/image-processing/edge-detection/
#mul by 1/32:
GRAD32_X = [[-1,-2,-1],
            [-2,-4,-2],
            [ 0, 0, 0],
            [ 2, 4, 2],
            [ 1, 2, 1]]
GRAD32_Y = [[-1,-2, 0, 2, 1],
            [-2,-4, 0, 4, 2],
            [-1,-2, 0, 2, 1]]

GRAD512_XS=[ 1,  4,  6,  4,  1]         #smoothing operator
GRAD512_YS=[-1, -4, -5,  0,  5,  4,  1] #derivative operator
#mul by 1/512 to normalize
GRAD512_X = np.outer(GRAD512_YS, GRAD512_XS)
GRAD512_Y = np.outer(GRAD512_XS, GRAD512_YS)

#LINECOLOR = 0x6030A8
RBF_ALPHA           = 67
RBF_GRID_SIZE       = 200
RBF_SAMPLE_INTERVAL = 0
RBF_NORMAL_RADIUS   = 3
#position centers along image gradient normal
#0 means position along the segment near the in/out crossing
RBF_NORMALS         = 1  
RBF_CONTOUR_COLOR   = 0x7F
RBF_INSIDE_COLOR    = GRAY
RBF_FULL            = 1
BBOX_ONLY           = 0
NORM_MAX_LEN        = 40
CONTOUR_THRESH      = 0.0 #0.01
INSIDE_BRIGHTNESS   = 255 #255 is the normal
GRAD_COLOR          = WHITE #0x7F7FFF
GRAD_LOC_COLOR      = CYAN #BLUE
GRAD_LOC_RADIUS     = 3
GRAD_OP_X           = GRAD512_X
GRAD_OP_Y           = GRAD512_Y
GRAD_OP_XS          = GRAD512_XS
GRAD_OP_YS          = GRAD512_YS
GRAD_OP_NORMALIZER  = 512
LOAD_SEGMENTS       = 1
SAVE_SEGMENTS       = 1
RBF                 = 1
EPSILON             = 1e-2
USE_GRID            = 0
GRID_STEP           = 20
PATCH_LEN           = 16
PATCH_HALF_LEN      = PATCH_LEN/2
PATCH_SIZE          = (PATCH_LEN,PATCH_LEN)
PATCH_HALF_SIZE     = (PATCH_LEN/2,PATCH_LEN/2)
WALK_CONTOUR_STEP   = PATCH_LEN
SIGNATURE_CUTOFF    = 0.0
CONTOUR_BREATH      = 1
SHOW_GRADIENTS      = 0
BREATH_STEP         = PATCH_LEN/2
BREATH_NUM          = 5


def printlog(s):
    pass

def line((x0, y0), (x1, y1), color=0xFF08FF, xor=0, get_pts=0, draw=-1, extra=0):
    """Bresenham's line algorithm from Wikipedia
    if extra>0 continue in both ends upto border, with color=extra
    """
    if draw==-1: draw = not get_pts
    pts = []
    dx = abs(x1-x0)
    dy = abs(y1-y0) 
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx-dy
    c = color
    in_extra = False
    x = x0
    y = y0
    while True:        
        if draw:
            if xor: c = pxarray[x][y] ^ color
            pxarray[x][y] = c
        if get_pts:
            pts.append((x,y,pxarray[x][y]))
        if x == x1 and y == y1:
            if not extra:
                break
            elif get_pts:
                break
            else:
                in_extra = True
                color = extra
        if extra and (x in [0,WIDTH-1] or y in [0,HEIGHT-1]):
            break
        e2 = 2*err
        if e2 > -dy:
            err = err - dy
            x = x + sx
        if e2 <  dx:
            err = err + dx
            y = y + sy
            
    if extra and not get_pts:
        x  =  x0; y  =  y0
        sx = -sx; sy = -sy
        err = dx-dy
        e2 = 2*err
        if e2 > -dy:
            err = err - dy
            x = x + sx
        if e2 <  dx:
            err = err + dx
            y = y + sy        
        in_extra = True
        color = extra
        ray=[]
        while True:
            if get_pts: 
                ray.append((x,y,pxarray[x][y]))
            else:
                if xor: c = pxarray[x][y] ^ color
                pxarray[x][y] = c
            if (x in [0,WIDTH-1] or y in [0,HEIGHT-1]):
                break
            e2 = 2*err
            if e2 > -dy:
                err = err - dy
                x = x + sx
            if e2 <  dx:
                err = err + dx
                y = y + sy
        ray.reverse()
        ray.extend(pts)
        pts = ray
    return pts
    
def grid_lines():
    lines = []
    for x in range(GRID_STEP,WIDTH,GRID_STEP):
        pts = line((x,0), (x,HEIGHT-1),color=LINE_BG_COLOR,xor=1,draw=1,get_pts=1)
        lines.append(pts)
    for y in range(GRID_STEP,HEIGHT,GRID_STEP):
        pts = line((0,y), (WIDTH-1,y),color=LINE_BG_COLOR,xor=1,draw=1,get_pts=1)
        lines.append(pts)
    pygame.display.flip()
    return lines

def collect_segments(pts, vals):
    lsegments = []
    i=0
    segval = 0
    while i<len(vals):
        if segval!=vals[i] or i==len(vals)-1: #new segment start
            if segval!=0:
                p0,p1 = pts[last][0:2], pts[i-1][0:2]
                s0,s1 = min(p0,p1), max(p0,p1)
                seglen = distp(p0, p1)
                seg = (mean(p0,p1), tuple(s0), tuple(s1), seglen, segval)
                lsegments.append(seg)
            segval = vals[i]
            last = i
        i += 1        
    return lsegments

def draw_segments(pts):
    draw_on = False
    last_line = None
    vals = [-1] * len(pts)
    dx,dy,dc = [a-b for a,b in zip(pts[-1],pts[0])]
    if abs(dx)>abs(dy):
        if dx<0:
            dx=-dx
            pts.reverse()
        vec = [x for x,y,c in pts]
    else:
        if dy<0:
            dy=-dy
            pts.reverse()
        vec = [y for x,y,c in pts]
    while True:
        e = pygame.event.wait()
        if e.type == pygame.QUIT:
            break
        if e.type == KEYDOWN:   # and e.key == K_ESCAPE:
            break
        if e.type == MOUSEBUTTONDOWN:
            draw_on = True
            p = e.pos[0] if dx>dy else e.pos[1]
            startloc = min(bisect(vec, p), len(pts)-1)
            start = pts[startloc][0:2]
        if e.type == MOUSEBUTTONUP:
            draw_on = False
            last_line = None
            p = e.pos[0] if dx>dy else e.pos[1]
            endloc = min(bisect(vec, p), len(pts)-1)
            end = pts[endloc][0:2]
            seg_pts = line(start, end,color=LINE_FG_COLOR,xor=0,get_pts=0)#,extra=BORDER_COLOR)
            sloc,eloc = min(startloc,endloc),max(startloc,endloc)
            vals[sloc:eloc] = [1] * (eloc-sloc)
        if e.type == MOUSEMOTION:
            if draw_on:
                if last_line:
                    line(last_line[0], last_line[1],color=LINE_BG_COLOR,xor=1)
                p = e.pos[0] if dx>dy else e.pos[1]
                loc = min(bisect(vec, p), len(pts)-1)
                end = pts[loc][0:2]
                line(start,end,color=LINE_FG_COLOR,xor=0)
                last_line = (start,end)
        pygame.display.flip()
    return collect_segments(pts,vals)
    
def draw_lines():
    lines = []
    segments = []
    while True:
        e = pygame.event.wait()
        if e.type == pygame.QUIT or e.type == KEYDOWN:
            break
        if e.type == MOUSEBUTTONDOWN:
            line, segs = draw_line(e.pos)
            lines.append(line)
            segments.append(segs)
    return lines, segments

def draw_line(start_pos):
    last_line = None
    pts = []
    segments = []
    while True:
        e = pygame.event.wait()
        if e.type == MOUSEBUTTONUP:
            last_line = None
            pts = line(start_pos, e.pos,color=LINE_BG_COLOR,xor=1,get_pts=1)#,extra=BORDER_COLOR)
            #lines.append(pts)
            segments = draw_segments(pts)
            break
        if e.type == MOUSEMOTION:
            if last_line:
                line(last_line[0], last_line[1],color=LINE_BG_COLOR,xor=1)#,extra=BORDER_COLOR)
            line(start_pos, e.pos,color=LINE_BG_COLOR,xor=1)#,extra=BORDER_COLOR)
            last_line = (start_pos,e.pos)
        pygame.display.flip()
    return pts, segments

def redraw_segments(segments):
    """redraw segments, id is color-coded
    @return segments dict, key is id"""
    dseg = {}
    i=0
    for lsegments in segments:
        for seg in lsegments:
            pmean,p0,p1,l,v = seg
            i+=1
            c = i+SEG_IN_COLOR if v==1 else i+SEG_OUT_COLOR
            line(p0,p1,color=c,xor=0)
            dseg[i] = seg
    return dseg

def check_escape():
    e = pygame.event.poll()
    ret = e.type == KEYDOWN and e.key == K_ESCAPE
    if ret:
        pygame.event.post(e)
    return ret

def wait_event(event=KEYDOWN,show=None):
    global pxarray, signat
    pygame.event.clear()
    while True:
        e = pygame.event.wait()
        if e.type == pygame.QUIT:
            break
        if e.type == event:
            break
        if e.type == MOUSEBUTTONDOWN:
            pt = e.pos
            print(pt)
            if show is not None:
                if show=='grad':
                    show_gradients(img, [pt])
                elif show=='grade':
                    if e.button==1 and globals().has_key('signat'):
##                        pygame.draw.rect(screen,GREEN,(topleft,PATCH_SIZE),1)
##                        pygame.display.flip()
                        grade_patch(signat, pt, pxarray)
                    if e.button==3:
                        signat = get_signature(pxarray,pt)
        if e.type == MOUSEMOTION and type(show)==type(''):
            if show is not None:
                try:
                    print('{0}:{1}'.format(pt,show[e.pos]))
                except:
                    pass
    return e

def avg(seq):
    return sum(seq)/len(seq)

def distp(p,q):
##    return sqrt(sqr(sub(p,q)))
    px,py=p
    qx,qy=q
    return sqrt((px-qx)**2 + (py-qy)**2)

def dist((x,y),(u,v)):
    return sqrt((x-u)**2 + (y-v)**2)

def rgb2int(r,g,b):
    return r<<16 | g<<8 | b

def rgb2gray(c):
    "From Matlab doc of rgb2gray"
    R,G,B = (c & 0xff0000)>>16, (c & 0x00ff00)>>8, c & 0x0000ff
    return int(0.2989 * R + 0.5870 * G + 0.1140 * B)

def gray2rgb(g):
    return g<<16 | g<<8 | g

#Some linear algebra
####################
def dot_product(u,v):
   return sum([a*b for (a,b) in zip(u,v)])

def dot(a, b):
    return sum([a[i]*b[i] for i in range(len(a))])
def sub(a,b):
    return [a[i]-b[i] for i in range(len(a))]
def add(a,b):
    return [a[i]+b[i] for i in range(len(a))]
def mul(v,t):
    return [t*a for a in v]
def div(v,t):
    return [a/t for a in v]
def norm(v):
    return sqrt(dot(v,v))
def norm2(x,y):
    return sqrt(x**2+y**2)
def roundv(v):
    return [int(round(a)) for a in v]
####################

def line_pt(l, q):
    """Compute nearest point on a line l to some point q"""
    "t = (q-p0)v / vv"
    p0,p1 = l
    v = sub(p1,p0)
    t = dot(sub(q,p0),v) / float(dot(v,v))
    pt = add(p0,mul(v,t))
    r = [int(x) for x in pt]
    return r if r == pt else pt

def mean(p,q):
    return tuple(div(add(p, q), 2))

def meani(p,q,n,i):
	return tuple(add(p,mul(div(sub(q,p),n),i)))
    
def preprocess(segments):
    """Split segment in half if surrounded by inverse segments
    This way it is naturally handled
    """
    ret = []
    for lsegments in segments:
        retl = []
        sv = 0
        for i in range(len(lsegments)):
            seg = lsegments[i]
            prev_sv = sv
            next_sv = 0 if i+1>=len(lsegments) else lsegments[i+1][4]
            mid,p0,p1,sl,sv = seg
            if next_sv==prev_sv and sv==-next_sv:
                m = mean(p0,p1)
                new1 = (mean(p0,m), p0, m, sl/2, sv)
                new2 = (mean(m,p1), m, p1, sl/2, sv)
                retl.extend([new1, new2])
            else:
                retl.append(seg)
        ret.append(retl)
    return ret

def sample(segments, interval):
    samples = []
    for lsegments in segments:
        for s in lsegments:
            mid,p0,p1,sl,sv = s
            times = sl/interval
            seg_samples = [meani(p0,p1,times,i) for i in range(int(times))]
            samples.extend([(p[0],p[1],sv) for p in seg_samples])
            samples.append((p1[0],p1[1],sv));
    sx, sy, sz = zip(*samples)
    return sx,sy,sz

def bilinear(ret_pxarray, (x1,y1), topLeft, step):
    topLeftX, topLeftY = topLeft
    topRight = (topLeftX+step, topLeftY)
    botLeft  = (topLeftX,      topLeftY+step)
    botRight = (topLeftX+step, topLeftY+step)
    ulC = ret_pxarray[topLeft]
    urC = ret_pxarray[topRight]
    blC = ret_pxarray[botLeft]
    brC = ret_pxarray[botRight]
    step = float(step)
    rightFactorX = (x1-topLeftX)/step
    topC = ulC * (1-rightFactorX) + urC * rightFactorX
    botC = blC * (1-rightFactorX) + brC * rightFactorX
    botFactorY = (y1-topLeftY)/step
    finalC = topC * (1-botFactorY) + botC * botFactorY
##    if ulC or urC or blC or brC:
##        pdb.set_trace()
    return int(round(finalC))

##############
#
#http://en.wikipedia.org/wiki/Union_find
#

class UFNode:
    """label is a numeric value
    no make_set() since it's included here
    """
    def __init__ (self, label):
        self.label = label
        self.parent = self
        self.rank   = 0
    def __str__(self):
        return self.label

def uf_union(x, y):
    xRoot = uf_find(x)
    yRoot = uf_find(y)
    if xRoot.rank > yRoot.rank:
        yRoot.parent = xRoot
    elif xRoot.rank < yRoot.rank:
        xRoot.parent = yRoot
    elif xRoot != yRoot: # Unless x and y are already in same set, merge them
        yRoot.parent = xRoot
        xRoot.rank = xRoot.rank + 1
    return x.parent

def uf_find(x):
    if x.parent == x:
        return x
    else:
        x.parent = uf_find(x.parent)
        return x.parent
#    
##############

def extract_blobs(px):
     
    """en.wikipedia.org/wiki/Blob_extraction
    input:  PixelArray with "0" for background
    output: dict with blobs labels 1,8,... (where no background)
            areas dict: { 1: 439,  8: 458 }
    """
    linked = {}
    M,N = px.surface.get_size()

    labels = {}
    next_label = 1
    
    #First pass
    for row in range(M):
        for col in range(N):
            pos = row,col
            if px[pos] != 0:
                #connected elements with the current element's value
                neighbors = {}
                if row > 0:
                    north = row-1, col
                    if px[north] == px[pos]:
                        neighbors[north] = px[north]
                if col > 0:
                    west = row, col-1
                    if px[west] == px[pos]:
                        neighbors[west] = px[west]
                if neighbors == {}:
                    new_node = UFNode(next_label)
                    linked[next_label] = new_node
                    labels[pos] = new_node
                    next_label += 1
                else:
                    #Find the smallest label                   
                    L = [labels[x] for x in neighbors.keys()]
                    seq = [x.label for x in L]
                    minvalue, minindex = min(zip(seq, range(len(L))))
                    labels[pos] = L[minindex]
                    for labelx in L:
                        for labely in L:
                            u = uf_union(linked[labelx.label], labely)
                        linked[labelx.label] = u
    #Second pass
    areas = {}
    for row in range(M):
        for col in range(N):
            pos = row,col
            if px[pos] != 0:
                labels[pos] = uf_find(labels[pos])
                px[pos] = labels[pos].label
                L = labels[pos].label
                areas[L] = areas.get(L,0) + 1
    return labels, areas

def compute_rbf((Gx,Gy), segments, use_dist_trans=True):
    ret_surface = screen.convert(8)
    ret_surface.fill(0)
    ret_pxarray = pygame.PixelArray (ret_surface)
    #only take p0,p1
    pts = [p for l in segments for s in l for p in s[1:3]]
    #sign per point
    sz = [v for l in segments for s in l for v in 2*[s[4]]]
    sx,sy = zip(*pts)
    xmin,xmax = min(sx), max(sx)
    ymin,ymax = min(sy), max(sy)
    if RBF_SAMPLE_INTERVAL:
        sx,sy,sz = sample(segments, RBF_SAMPLE_INTERVAL)
    elif RBF_NORMALS:
        pos_pts = [p for l in segments for s in l if s[4]==1 for p in s[1:3]]
        grads = [(Gx[p],Gy[p]) for p in pos_pts]
        ng = [div(g,norm(g)) for g in grads]
        radius = RBF_NORMAL_RADIUS
        neg_pts = [roundv(add(p,mul(ng,radius/norm(ng))))
                   for (p,ng) in zip(pos_pts,ng)]
        pts = pos_pts + neg_pts
        sx,sy = zip(*pts)
        sz = len(pos_pts)*[1] + len(neg_pts)*[-1]
    else:
        out_pts = [l[0][2] for l in segments if l[0][4]==-1]
        out_sgn = len(out_pts)*[-1]
        inside_pts = [p for l in segments for s in l[1:-1] for p in s[1:3]]
        inside_sgn = [v for l in segments for s in l[1:-1] for v in 2*[s[4]]]
        out2_pts = [l[-1][1] for l in segments if l[-1][4]==-1]
        out2_sgn = len(out2_pts)*[-1]
        pts = out_pts + inside_pts + out2_pts
        sz  = out_sgn + inside_sgn + out2_sgn
        sx,sy = zip(*pts)
    if not DEBUG:
        pool = multiprocessing.Pool(multiprocessing.cpu_count())

    #'cubic': r**3
    #'thin_plate': r**2 * log(r)
    rbf = interpolate.Rbf(sx,sy,sz, function='thin_plate')
    length = max(ret_surface.get_width(), ret_surface.get_height())
    step = 1 if RBF_FULL else length/RBF_GRID_SIZE
    if BBOX_ONLY:
##        xti = np.linspace(xmin, xmax, RBF_GRID_SIZE)
##        yti = np.linspace(ymin, ymax, RBF_GRID_SIZE)
        xti = np.arange(xmin, xmax, step)
        yti = np.arange(ymin, ymax, step)
    else:
        xti = np.arange(0, ret_surface.get_width(),  step)
        yti = np.arange(0, ret_surface.get_height(), step)
    xti = [int(round(x)) for x in xti[:-1]]
    yti = [int(round(y)) for y in yti[:-1]]
    XI, YI = np.meshgrid(xti, yti)
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
    #clamp ZI values to [-1,1]
    normZI = [[a if abs(a)<=1 else cmp(a,0) for a in b] for b in ZI]
    radius = (xmax-xmin)/len(xti)/2
    halfstep = step/2
    for j,y in enumerate(yti):
        for i,x in enumerate(xti):
            v = normZI[j][i]
            if abs(v)<CONTOUR_THRESH:
                c = RBF_CONTOUR_COLOR
            else:
                b = abs(int(v*INSIDE_BRIGHTNESS))
                #green inside, purple outside
                c = b+(b<<16)  if v<0 else  b<<8
            #pygame.draw.circle(ret_surface, c, (x,y), radius)
            if v>0:
                #r = pygame.Rect(x-halfstep, y-halfstep, step,step)
                #ret_surface.fill(WHITE, rect=r)
                ret_pxarray[x,y] = RBF_INSIDE_COLOR
            #pxarray[x,y] = c #gray2rgb(gray)
            if j>0 and i>0:
                topLeftX, topLeftY = x-step,y-step
                for x1 in range(topLeftX+1,x):
                    for y1 in range(topLeftY+1,y):
                        c1 = bilinear(ret_pxarray,(x1,y1),(topLeftX,topLeftY), step)
                        ret_pxarray[x1,y1] = c1
    return ret_surface

def neighbors4((x,y), clip=0):
    pts = ((x+1,y), (x-1,y), (x,y+1), (x,y-1))
    if clip:
        pts = tuple([p for p in pts if not cliprect.collidepoint(*p)])
    return pts

def convolve(P, K, (x,y)):
    """Single image pt convolution
    not optimized - use separable convolution to optimize relevant filters
    """
    cols, rows = screen.get_size()
    kCols = len(K)
    kRows = len(K[0])
    kCenterX = kCols / 2;
    kCenterY = kRows / 2;
    s=0
    for m in range(kRows):
        mm = kRows - 1 - m      # row index of flipped kernel
        for n in range(kCols):  # kernel columns
            nn = kCols - 1 - n  # column index of flipped kernel
            #index of input signal, used for checking boundary
            ii = y + (m - kCenterY)
            jj = x + (n - kCenterX)
            #ignore input samples which are out of bound
            if ii>=0 and ii<rows and jj>=0 and jj<cols:
                s += P[jj][ii] * K[nn][mm]
    return s

def img_gradient(surface, rowK,colK,normalizer):
    "returns 2D numpy.array"
    if surface.get_bitsize()!=32:
        surface = surface.convert(32)
    arr = pygame.surfarray.pixels2d(surface)
    #separable filter 2d convolution
    Gx = signal.sepfir2d(arr,rowK,colK) / normalizer
    Gy = signal.sepfir2d(arr,colK,rowK) / normalizer
    return Gx,Gy
    
def gradient(px, pt):
    Gx = convolve(px, GRAD_OP_X, pt)
    Gy = convolve(px, GRAD_OP_Y, pt)
##    G = sqrt(Gx**2+ Gy**2)
##    angle = atan(float(Gy)/float(Gx))
    return Gx,Gy

def gradients(im, pts):
    px = pygame.PixelArray(im)
    ret = []
    for pt in pts:
        gx, gy = gradient(px,pt)
        ret.append((gx,gy))
    del px
    return ret

def walk_grad(px, (x,y)):
    """return max gradient location, over the env that is 
    not already colored as contour"""
    no_max_yet = True
    for i in range(x-1,x+2):
        for j in range(y-1,y+2):
            if px[i][j] != RBF_CONTOUR_COLOR:
                gx,gy = gradient(px,(i,j))
                G = norm2(gx,gy)
                if no_max_yet or maxG<G:
                    maxG = G
                    mi,mj = i,j
                    no_max_yet = False
    return (-1,-1) if no_max_yet else (mi,mj) 

def aprx_dist(p,q):
    return abs(p[0]-q[0]) + abs(p[1]-q[1])

def earth_mover_dist(hist1,hist2):
    x=0;
    for i in hist1.keys():
        x = x + abs(hist2[i]-hist1[i]);
        if hist2[i] > hist1[i]:
            hist2[i+1] = hist2[i+1]+(hist2[i]-hist1[i]);
            hist2[i] = hist1[i];
        else:
            hist1[i+1] = hist1[i+1]+(hist1[i]-hist2[i]);
            hist1[i] = hist2[i];
    return x

def get_loc_behind_normal(px, pt, dist):
    """return the color a bit "behind" the pt,
       with respect to the "gray image" normal there"""
    G = normalized_gradient(px, pt)
    offset = mul(G,dist)
    loc = sub(pt,offset) 
    return roundv(loc)

def histogram(px, (x,y),(w,h), cutoff = 0):
    """returns sorted OrderedDict - creates 0 for non-occuring colors
    cutoff .1 will throw 10% (outliers)
    """
    d={}
    for i in range(x,x+w):
        for j in range(y,y+h):
            c=px[i,j]
            d[px[i,j]] = d.get(c,0) + 1
    num = w*h
    d = OrderedDict(sorted(d.items(), key=lambda t: t[0]))
    while num > (1-cutoff)*(w*h):
        minC,minN = d.popitem(last=False)
        num -= d[minC]
    return d

def zero_bg(px, fg_colors):
    for i in range(PATCH_LEN):
        for j in range(PATCH_LEN):
            px[i,j] = 1 if px[i,j] in fg_colors else 0

def biggest_blob(px):
    labels, areas = extract_blobs(px)
    if len(labels)==0:
        return 0
    max_label,max_area = max(areas.items(), key=lambda x: x[1])    
    for i in range(PATCH_LEN):
        for j in range(PATCH_LEN):
            if px[i,j] != max_label:
                px[i,j] = 0
    return max_area

def max_contig_border(px):
    top=[]; right=[]; bottom=[]; left=[]
    for i in range(PATCH_LEN):
        top.append(px[i,0])
        right.append(px[PATCH_LEN-1,i])
        bottom.append(px[PATCH_LEN-1-i,PATCH_LEN-1])
        left.append(px[0,PATCH_LEN-1-i])
    border = top+right+bottom+left    
    count=0
    ret=0
    for c in border:
        if ret<count: ret=count
        count = count+1 if c!=0 else 0
    return ret

def grade_patch_1blob(fg_colors, patch):
    "returns a grade 0..1"
    patch_copy = patch.copy()
    patch_px = pygame.PixelArray(patch_copy)
    zero_bg(patch_px, fg_colors)
    area = biggest_blob(patch_px)
    #pprint(patch_px)
    max_border = max_contig_border(patch_px)
    border_percent = float(max_border) / (4*PATCH_LEN)
    area_percent = float(area) / (PATCH_LEN**2)
    #best percent is 0.35
    border_grade = 1-abs(min(border_percent,.70) - 0.35)/.35
    #best percent is 0.35
    area_grade   = 1-abs(min(area_percent,.70)   - 0.35)/.35
    grade = min(border_grade,area_grade)
    print('border: {0} area: {1} grade:{2}'.format(
        border_percent,area_percent,grade))
    return grade

def chi_sq_dist(hist1, hist2):
    sum1,sum2 = [sum(h.values()) for h in hist1,hist2]
    ratio = float(sum1)/sum2
    ret = 0
    keys = set(hist1.keys() + hist2.keys())
    for k in keys:
        p,q= hist1.get(k,0), hist2.get(k,0)
        q *= ratio
        ret += (p-q)**2 / (p+q)
    return 1 - ret/2.0 / sum1

def grade_patch_chi_sq(signature, pt, px):
    "returns a grade 0..1"
    dest = get_signature(px,pt)
    in_in  = chi_sq_dist(signature[0], dest[0])
    in_out = chi_sq_dist(signature[0], dest[1])
    print('in_in={0} in_out={1}'.format(in_in, in_out))
    return min(in_in, 1-in_out)

def weighted_avg(dic):
    tot=0
    num=0
    for k,v in dic.items():
        tot += k*v
        num += v
    return tot/num

def hist_median(dic):
    total = sum(dic.values())
    num=0
    for k,v in dic.items():
        num += v
        if num>= total/2:
            return k

def grade_patch(signature, pt, px):
    "returns a grade 0..1"    
    sign_in_med, sign_out_med = signature
    dest_in_med, dest_out_med = get_signature(px,pt)
    sign_in_out = float(abs(sign_in_med-sign_out_med))
    in_in  = abs(sign_in_med - dest_in_med) /sign_in_out    
    in_out = abs(sign_in_med - dest_out_med)/sign_in_out
    g = 1-max(in_in , 1-in_out)
    #print('in_in={0} 1-in_out={1} g={2}'.format(in_in, 1-in_out, g))
    return g

def get_signature(px,pt, size=PATCH_SIZE):
    """get histogram of the patch 
    input: surface and patch-middle point
    output: hostogram of colors
    """
    half_size = div(size,2)
    gx, gy = gradient(px, pt)
    x = 0 if gx<0 else half_size[0]
    y = 0 if gy<0 else half_size[1]
    inside_topleft = sub(pt, (x,y))
    outside_topleft= sub(pt, sub(half_size,(x,y)))
    hist_inside  = histogram(px, inside_topleft,  half_size, SIGNATURE_CUTOFF)
    hist_outside = histogram(px, outside_topleft, half_size, SIGNATURE_CUTOFF)    
    return [hist_median(h) for h in [hist_inside, hist_outside]]

def get_signature_1blob(surf,pt):
    """get colors of histogram of the patch quarter that is surely inside the blob
    input: surface and patch-middle point
    output: set of foreground colors
    TODO: histogram is not needed, just the colors are
    """
    px = pygame.PixelArray(surf)
    gx, gy = gradient(px, pt)
    x = 0 if gx<0 else PATCH_LEN
    y = 0 if gy<0 else PATCH_LEN
    topleft = sub(pt, (x,y))
##    pygame.draw.rect(screen,RED,(topleft,PATCH_SIZE),1)
##    pygame.display.flip()
    hist = histogram(px, topleft, PATCH_SIZE)
    num = PATCH_LEN**2
    while num > SIGNATURE_THRESH*PATCH_LEN**2:
        minC,minN = min(hist.items(), key=lambda x: x[1])
        num -= hist[minC]
        del hist[minC]
    fg_colors = set(hist.keys())
    botright = add(topleft,PATCH_SIZE)
    pprint(px[topleft[0]:botright[0], topleft[1]:botright[1]])
    print('hist={0}'.format(hist))
    return fg_colors

def get_patch(px,pt):
    left,top = sub(pt,PATCH_HALF_SIZE)
    rite,bot = add(pt,PATCH_HALF_SIZE)
    return px[left:rite,top:bot]
    

def reached_some_mark(pt,ptSet):
    for p in ptSet:
        if aprx_dist(pt,p)<=4:
            return p
    return None

def normalized_gradient(px, pt):
    gx,gy = gradient(px,pt)
    mag = norm2(gx,gy)
    return div((gx,gy),mag)

def breath(signature, pt_anchor, px, px_contour):
    grades = [] #list of (pt,grade)
    ngx, ngy = normalized_gradient(px_contour, pt_anchor)
    breath_delta = mul((ngx,ngy), BREATH_STEP)
    size = sub(img.get_size(), PATCH_SIZE)
    allowed_rect = pygame.Rect(PATCH_HALF_SIZE, size)
    for i in range(BREATH_NUM):
        for f in add,sub:
            p = f(pt_anchor, mul(breath_delta,i))
            pt = roundv(p)
            if allowed_rect.collidepoint(*pt):
                g = pt, grade_patch(signature, pt, px)
                grades.append(g)
    return grades

def walk_contour(surface, interest_pts):
    """walk from some pt to detected another.
    until processing upto the start.
    TODO: switch to next signature at midway to next mark
    """
    px_contour = pygame.PixelArray(surface)
    px = pygame.PixelArray(img)
    pts = set(interest_pts)
    pt_start = pts.pop()
    mark = pt_start
    signature = get_signature(px,mark,PATCH_SIZE)
    done = False
    grades=[]
    while not done:
        pt = mark
        i=0
        next_mark = reached_some_mark(pt, pts) 
        while not next_mark:    #TODO  and pt!=(-1,-1):        
            pt = walk_grad(px_contour, pt)
            px_contour[pt] = RBF_CONTOUR_COLOR
            if i % WALK_CONTOUR_STEP == 0:
                breath_grades = breath(signature, pt, px, px_contour)
                grades.extend(breath_grades)
                #g = pt, grade_patch(signature, pt, px)
                #grades.append(g)
                #print('{0} {1} {2}'.format(i,*g))
            next_mark = reached_some_mark(pt, pts) 
            i+=1
        pts.add(mark)
        mark = next_mark
        pts.remove(mark)
        signature = get_signature(px,mark,PATCH_SIZE)
        done = (mark == pt_start)
            
    del px
    del px_contour
    for pt,grade in iter(grades):
        #print('{0} {1}'.format(pt,grade))
        pygame.draw.circle(surface,WHITE,pt,max(2,int(grade*10)),1)

def show_contour(surface, pt_start):
    px = pygame.PixelArray(surface)
    pt = pt_start
    i=0
    #first walk away a bit and then start testing if we're back near start
    while (i<10 or aprx_dist(pt,pt_start)>2) and pt!=(-1,-1):
        pt = walk_grad(px, pt)
        px[pt] = RBF_CONTOUR_COLOR
        i+=1
    del px

def show_gradients(surf, (Gx,Gy), pts):
    for pt in pts:
        g=(Gx[pt],Gy[pt])
        gx,gy = mul(div(g,norm(g)),NORM_MAX_LEN)
        end = pt[0]+gx, pt[1]+gy
        pygame.draw.circle(surf,GRAD_LOC_COLOR,pt,GRAD_LOC_RADIUS)
        pygame.draw.line(surf,GRAD_COLOR,pt,end,2)

def show_gradients_old(im, pts):
    px = pygame.PixelArray(im)
    for pt in pts:
        g = gradient(px, pt)
        gx,gy = mul(div(g,norm(g)),NORM_MAX_LEN)
        end = pt[0]+gx, pt[1]+gy
        pygame.draw.circle(im,GRAD_LOC_COLOR,pt,GRAD_LOC_RADIUS)
        pygame.draw.line(screen,GRAD_COLOR,pt,end,2)
    del px
    pygame.display.flip()

def main():
    global screen, pxarray, cliprect, img
    #warnings.simplefilter("error")    #"always"
    try:
        img = pygame.image.load(IMAGE_FILENAME)
        #screen is a Surface
        ##screen = pygame.display.set_mode((WIDTH,HEIGHT))
        img_size = img.get_size()
        screen = pygame.display.set_mode(img_size)
        img = img.convert(8)
        screen.blit(img,(0,0))
        pygame.display.flip()

##        pxarray = pygame.PixelArray (img)
##        pygame.display.set_caption('r-click=sign, click=grade')
##        print("r-click=sign, click=grade")
##        wait_event(show='grade')
##        pxarray = None

        gradient_field = img_gradient(img, GRAD_OP_XS,GRAD_OP_YS,GRAD_OP_NORMALIZER)        
        pxarray = pygame.PixelArray (screen)
        
        pygame.display.set_caption('Pls Draw lines, <SPACE> to go on')
        print('Pls Draw lines, another <SPACE> to go on')
        if USE_GRID:
            new_lines, new_segments = grid_lines()
        elif LOAD_SEGMENTS:
            with open('draw3.pickle') as f:
                new_lines, new_segments = cPickle.load(f)
                print('Loaded segments:')
                pprint(new_segments)
        else:
            new_lines, new_segments = draw_lines()

        lines, segments = [], []
        pygame.display.set_caption('RBF with Normals, <SPACE> to continue')
        print('Showing RBF with Normals, <SPACE> to continue')
        while new_lines:
            lines += new_lines
            segments += new_segments
            if SAVE_SEGMENTS:
                with open('draw3.pickle','w') as f:
                    cPickle.dump((lines, segments),f)
            screen.fill(0)
            rbf_surface = compute_rbf(gradient_field, segments, use_dist_trans=True)
            rbf_surface.set_alpha(RBF_ALPHA) #0 is transparent, 255 is opaque
            interest_pts=[p for l in segments for s in l if s[4]==1 for p in s[1:3]]
            if CONTOUR_BREATH:
                walk_contour(rbf_surface, interest_pts)
            if SHOW_GRADIENTS:
                show_gradients(rbf_surface, gradient_field, interest_pts)
            pxarray = None
            screen.blit(img,(0,0)) #,special_flags=BLEND_ADD)
            screen.blit(rbf_surface, (0,0))
            pxarray = pygame.PixelArray (screen)            
            seg_dict = redraw_segments(segments)
            pxarray = None
            pygame.display.flip()
            pygame.image.save(screen, 'rbf.png')            
            pxarray = pygame.PixelArray (screen)            
            new_lines, new_segments = draw_lines()
            pygame.display.flip()
            
        pygame.image.save(screen, 'screen_capture.JPG')        
        px = pygame.PixelArray(rbf_surface)
        print("hit any key to quit..")
        wait_event(show='grade')

        pygame.quit()
    except:
        pygame.quit()
        raise

if __name__=='__main__':
    DEBUG=1
    if DEBUG:
        print 'Debug mode'
        import pdb
    else:
        print 'Production mode'
        print 'Multiprocessing in on'
##        try:
##            import psyco
##            psyco.full()
##            print 'Psyco is on'
##        except:
##            print 'No Psyco for you'

    os.chdir(sys.path[0])
    main()
    
