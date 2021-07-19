"""
DRAW.PY
2D reconstruction from cross-sections with and w/o distance transform

Written by
Shahar Yifrah
March, 2011
"""

import pygame, random, time, pprint, os
from math import log, sqrt
from pygame.locals import *

DEBUG=0
if DEBUG:
    print 'Debug mode'
    import pdb
else:
    print 'Production mode'
    try:
        import psyco
        psyco.full()
    except:
        print 'No Psyco for you'


"""First line drawn must have blob in it"""

def printlog(s):
    pass

def integrate_dist(a,b,c, x):
    """Integrate[1/Sqrt[a + b*x + c*x^2], x]"""
    """matlab:
        log((b/2 + c*x)/c^(1/2) + (c*x^2 + b*x + a)^(1/2))/c^(1/2)"""
    a = float(a)
    b = float(b)
    c = float(c)
    x = float(x)
    ret = log(b + 2*c*x + 2*sqrt(c)*sqrt(a + b*x + c*x**2))/ sqrt(c)    
    return ret

def integrate_dist_grow(a,b,c, t):
    """Integrate[x/Sqrt[a + b*x + c*x^2], x]"""
    """matlab:
        (c*x^2 + b*x + a)^(1/2)/c -
        (b*log((b/2 + c*x)/c^(1/2) +
            (c*x^2 + b*x + a)^(1/2)))/(2*c^(3/2))"""
    ret =  sqrt(c*t**2 + b*t + a)/c -  \
                (b*log((b/2 + c*t)/sqrt(c) + \
                sqrt(c*t**2 + b*t + a)))/(2*sqrt(c)**3)
    return ret

def integrate_dist_decay(a,b,c, t):
    """Integrate[(1 - x)/Sqrt[a + b*x + c*x^2], x]"""
    """matlab:
        log((b/2 + c*x)/c^(1/2) +
            (c*x^2 + b*x + a)^(1/2))/c^(1/2) -
        (c*x^2 + b*x + a)^(1/2)/c +
        (b*log((b/2 + c*x)/c^(1/2) +
           (c*x^2 + b*x + a)^(1/2)))/(2*c^(3/2))    """
    ret =  log((b/2 + c*t)/sqrt(c) +
                    sqrt(c*t**2 + b*t + a))/sqrt(c) - \
                sqrt(c*t**2 + b*t + a)/c +     \
                (b*log((b/2 + c*t)/sqrt(c) +     \
                       sqrt(c*t**2 + b*t + a)))/(2*sqrt(c)**3)
    return ret

def flood_fill((x, y), border, value):
    "Flood fill on a region of non-BORDER_COLOR pixels."
    if not cliprect.collidepoint(x,y) or pxarray[x][y] == border:
        return
    edge = [(x, y)]
    pxarray[x][y] = value
    while edge:
        newedge = []
        for (x, y) in edge:
            for (s, t) in ((x+1, y), (x-1, y), (x, y+1), (x, y-1)):
                if cliprect.collidepoint(s,t) and pxarray[s][t] not in [border, value]:
                    pxarray[s][t] = value
                    #pdb.set_trace();
                    newedge.append((s, t))
        edge = newedge
        pygame.display.flip()

    
def line((x0, y0), (x1, y1), color=0xFF08FF, xor=0, get_pts=False):
    "Bresenham's line algorithm from Wikipedia"
    pts = []
    dx = abs(x1-x0)
    dy = abs(y1-y0) 
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx-dy
    c = color
    while True:
        if get_pts: 
            pts.append((x0,y0,pxarray[x0][y0]))
        else:
            if xor: c = pxarray[x0][y0] ^ color
            pxarray[x0][y0] = c
        if x0 == x1 and y0 == y1:
            break
        e2 = 2*err
        if e2 > -dy:
            err = err - dy
            x0 = x0 + sx
        if e2 <  dx:
            err = err + dx
            y0 = y0 + sy 
    return pts
    
def roundline(srf, color, start, end, radius=1):
    dx = end[0]-start[0]
    dy = end[1]-start[1]
    distance = max(abs(dx), abs(dy))
    for i in range(distance):
        x = int( start[0]+float(i)/distance*dx)
        y = int( start[1]+float(i)/distance*dy)
        pygame.draw.circle(srf, color, (x, y), radius)

def draw_blob():
    draw_on = False
    try:
        while True:
            e = pygame.event.wait()
            if e.type == pygame.QUIT:
                raise StopIteration
            if e.type == KEYDOWN and e.key == K_ESCAPE:
                raise StopIteration
            if e.type == pygame.MOUSEBUTTONDOWN:
                start = e.pos
                pygame.draw.circle(screen, BLOB_COLOR, e.pos, radius)
                draw_on = True
            if e.type == pygame.MOUSEBUTTONUP:
                draw_on = False
                roundline(screen, BLOB_COLOR, start, e.pos,  radius)
            if e.type == pygame.MOUSEMOTION:
                if draw_on:
                    pygame.draw.circle(screen, BLOB_COLOR, e.pos, radius)
                    roundline(screen, BLOB_COLOR, e.pos, last_pos,  radius)
                last_pos = e.pos
            pygame.display.flip()

    except StopIteration:
        pass

def draw_lines():
    last_line = None
    draw_on = False
    linecolor = 0x6030A8;
    lines = []
    try:
        while True:
            e = pygame.event.wait()
            if e.type == pygame.QUIT:
                raise StopIteration
            if e.type == KEYDOWN and e.key == K_ESCAPE:
                raise StopIteration
            if e.type == pygame.MOUSEBUTTONDOWN:
                draw_on = True
                start = e.pos
            if e.type == pygame.MOUSEBUTTONUP:
                draw_on = False
                last_line = None
                pts = line(start, e.pos,color=linecolor,xor=1,get_pts=True)
                lines.append(pts)
            if e.type == pygame.MOUSEMOTION:
                if draw_on:
                    if last_line:
                        line(last_line[0], last_line[1],color=linecolor,xor=1)
                    line(start, e.pos,color=linecolor,xor=1)
                    last_line = (start,e.pos)
            pygame.display.flip()
    except StopIteration:
        pass
    return lines

def count_run(l,start):
    i=start+1
    while i<len(l) and l[i]==l[start]:
        i+=1
    return i-start

def extract_segments(lines):
    fg_pt = min(lines[0], key=lambda x: x[2]);
    fg = fg_pt[2];
    segments = []
    for l in lines:
        lsegments = []
        colors = [x[2] for x in l]
        i = 0
        while i<len(colors):
            c = colors[i]
            segcount = count_run(colors,i)
            segval = 1 if c==fg else -1
            p0 = l[i][0:2]
            p1 = l[i+segcount-1][0:2]
            mean = div(add(p0, p1), 2)
            seglen = distp(p0, p1)
            lsegments.append((tuple(mean), tuple(p0), tuple(p1), seglen, segval))
            i = i + segcount
        segments.append(lsegments)
    print 'Segments:'
    pprint.pprint(segments)
    return segments
                
def redraw_lines(lines):
    "First line drawn must have blob in it"
    NO_MORE_FG = -2
    NO_FG_YET = -1
    fg_pt = max(lines[0], key=lambda x: x[2])
    fg = fg_pt[2]
    dt_lines=[]
    for l in lines:
        dt_line = []
        dt_lines.append(dt_line)
        colors = [x[2] for x in l]
        try:
            fg_start = colors.index(fg)
        except:
            fg_start = NO_MORE_FG
        npts = len(l)
        next_fg = NO_FG_YET
        prev_fg = NO_FG_YET
        for i in range(0,npts):
            (x,y,c) = l[i]
            newc=LINE_FG_COLOR if c==fg else LINE_BG_COLOR 
            pxarray[x,y] = newc
##            pdb.set_trace()
            if c==fg:
                dist_transform = 0
                if prev_fg < i:
                    prev_fg = i
            else:
                if next_fg==NO_FG_YET or (next_fg>=0 and i>next_fg):
##                    pdb.set_trace()
                    prev_fg = next_fg
                    try:
                        next_fg = colors.index(fg,i)
                    except:
                        next_fg = NO_MORE_FG
                        pass
                if prev_fg >= 0:
                    if next_fg >= 0:
                        dist_transform = min(next_fg - i, i-prev_fg)
                    else:
                        dist_transform = i - prev_fg
                elif next_fg>=0:
                    dist_transform = next_fg - i
            dt_line.append((x,y,newc, dist_transform))
    return dt_lines

def check_escape():
    e = pygame.event.poll()
    ret = e.type == KEYDOWN and e.key == K_ESCAPE
    if ret:
        pygame.event.post(e)
    return ret

def wait_escape():
    try:
        while True:
            e = pygame.event.wait()
            if e.type == pygame.QUIT:
                raise StopIteration
            if e.type == KEYDOWN and e.key == K_ESCAPE:
                raise StopIteration
            if e.type == pygame.MOUSEMOTION: #MOUSEBUTTONDOWN:
                print(mvc[e.pos])
    except StopIteration:
        pass

def distp(p,q):
##    return sqrt(sqr(sub(p,q)))
    px,py=p
    qx,qy=q
    return sqrt((px-qx)**2 + (py-qy)**2)

def dist(x,y,u,v):
    return sqrt((x-u)**2 + (y-v)**2)

def rgb2gray(c):
    "From Matlab doc of rgb2gray"
    R,G,B = (c & 0xff0000)>>16, (c & 0x00ff00)>>8, c & 0x0000ff
    return int(0.2989 * R + 0.5870 * G + 0.1140 * B)

def gray2rgb(g):
    return g<<16 | g<<8 | g

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

def line_pt(l, q):
    """Compute nearest point on a line l to some point q"""
    "t = (q-p0)v / vv"
    p0,p1 = l
    v = sub(p1,p0)
    t = dot(sub(q,p0),v) / float(dot(v,v))
    pt = add(p0,mul(v,t))
    r = [int(x) for x in pt]
    return r if r == pt else pt

def compute_pt_exhaust(x,y,lines):
    "return gray level"
    total_d = 0
    v = 0
    for l in lines:
        llen = len(l)
        for (lx,ly,c,dt) in l:
            d = dist(x,y,lx,ly)
            if d == 0:
                continue
            w = d #+ float(dt)/llen #(d*dt) if dt>0 else d
            total_d += 1/w
            delta = 1  if  c==LINE_FG_COLOR  else  -1
            v += delta/w
    v /= total_d
    assert(v>=-1 and v<=1)
    if v>0.1:
        gray = 255
    else:
        gray = (v+1)/2*255
    gray = int(gray);        
    if gray >= 256:
        pdb.set_trace()
    return gray

def compute_pt_heur(x,y,segments):
    v = 0
    total_w = 0
    for lsegments in segments:
        for seg in lsegments:
            m,p0,p1,sl,sv = seg
            if m == (x,y):
                continue
            w = sl / dist(x,y,*m)
            total_w += w
            v += sv*w
    v /= total_w       
    if v<-1 or v>1:
        print '*',
    if abs(v)<0.01:
        gray = 255
    else:
        gray = (v+1)/2*255
    gray = int(gray);        
    if gray >= 256:
        pdb.set_trace()
    return gray
        
def compute_pt_exact(x,y,segments, dt=True):
    v = 0
    total_w = 0
    for lsegments in segments:
        sv = 0
        for i in range(len(lsegments)):
            seg = lsegments[i]
            prev_sv = sv
            next_sv = 0 if i+1>=len(lsegments) else lsegments[i+1][4]
            mean,p0,p1,sl,sv = seg
            if (x,y)==p0 or (x,y)==p1:
                continue
##            pdb.set_trace()
            p0x, p0y = p0
            p1x, p1y = p1
            a = (x-p0x)**2.0 + (y-p0y)**2.0
            b = 2.0*(x-p0x)*(p0x-p1x) + 2.0*(y-p0y)*(p0y-p1y)
            c = (p0x - p1x)**2.0  + (p0y-p1y)**2.0
            try:
                if not dt or sv==1 or (not prev_sv and not next_sv):
                    t1 = integrate_dist(a,b,c,1)                               
                    t0 = integrate_dist(a,b,c,0)
                elif sv==-1:
                    if next_sv==1 and not prev_sv:
                        t1 = integrate_dist_grow(a,b,c,1.0)
                        t0 = integrate_dist_grow(a,b,c,0.0)
    ##                    pdb.set_trace()
                    elif not next_sv and prev_sv==1:
                        t1 = integrate_dist_decay(a,b,c,1.0)
                        t0 = integrate_dist_decay(a,b,c,0.0)
                    elif next_sv==1 and prev_sv==1:
                        tm_decay = integrate_dist_decay(a,b,c,0.5)
                        t0_decay = integrate_dist_decay(a,b,c,0.0)
                        t1 = tm_decay - t0_decay
                        t1_grow  = integrate_dist_grow(a,b,c,1.0)
                        tm_grow  = integrate_dist_grow(a,b,c,0.5)
                        #swapped since next line should've added this
                        t0 = tm_grow - t1_grow
                    t1 *= 2
                    t0 *= 2
                else:
                    pdb.set_trace()
            except:
                continue
            w = t1 - t0
            total_w += w
            v += sv*w
    if total_w!=0:    
        v /= total_w
    else:
        v=0
    if v<-1 or v>1:
        printlog('v outside -1,1: %f' % v)
        v=min(v,1)
        v=max(v,-1)
    return v


def compute_mvc(lines, use_dist_trans=False):
    segments = extract_segments(lines)
    for x in range(0,WIDTH):
        if x & 15 == 0:
            pygame.display.flip()
            if check_escape():
                break
        for y in range(0,HEIGHT):
            v = compute_pt_exact(x,y,segments,dt=use_dist_trans)
            mvc[(x,y)] = v
            if abs(v)<0.01:
                c = WHITE
            else:
                b=int(v*255)
                c = -b+(-b<<16) if v<0 else b<<8
            pxarray[x,y] = c #gray2rgb(gray)
    pygame.display.flip()

def main():
    global screen, pxarray, cliprect, mvc
    screen = pygame.display.set_mode((WIDTH,HEIGHT))
    pygame.display.set_caption('MVC (Draw first line through blob!)')
    pxarray = pygame.PixelArray (screen)
    cliprect = screen.get_clip()
##    width = screen.get_width()
##    height = screen.get_height()
    center = (WIDTH/2, HEIGHT/2)
    mvc = {}
    pygame.draw.circle(screen, GREEN, center, 5)
    draw_blob()
    flood_fill(center, value=BLOB_COLOR, border=color)
    lines = draw_lines()
    screen.fill(0)
    dt_lines = redraw_lines(lines)
    compute_mvc(dt_lines)
    wait_escape()
    
    screen.fill(0)
    dt_lines = redraw_lines(lines)
    compute_mvc(dt_lines, use_dist_trans=True)
    wait_escape()
    
    pygame.quit()

WIDTH,HEIGHT = 320,240
EPSILON = 1e-2
draw_on = False
last_pos = (0, 0)
GREEN = 0x00FF00
GRAY  = 0xF0F0F0
WHITE = 0xFFFFFF
RED   = 0xFF0000
LINE_BG_COLOR = RED
LINE_FG_COLOR = GREEN
BLOB_COLOR = 0x808000
radius = 1

##os.chdir('w:/thesis/code')
if __name__=='__main__':
    main()
    
    
