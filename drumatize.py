import sys
import math
import matplotlib.pyplot as plt
import numpy as np
import struct
import soundfile as sf
from random import random

mode = sys.argv[1] if len(sys.argv) > 1 else 'hardkick'

def ReadStuff():
       
    data, rate = sf.read('./sample_kick.wav')
    print(len(data))
    print(rate)
           
    #WriteStuff(data, rate)
    #PlotStuff(data, rate)
    
    WriteNewStuff(rate)
    
def PlotStuff(data, rate):
    
    x = np.arange(len(data))
    y = [d[0] for d in data]
    
    plt.plot(x, y, 'r-')
    plt.ylabel('shit')
    #plt.show(block=True)    

def WriteStuff(data, rate):
    sf.write('./shit.wav', data, rate, 'PCM_24')


def WriteNewStuff(rate):

    with open("./drum_template","r") as template:
        lines = template.readlines()  
 
    synstr = lines[0]

    # translate GLSL to Python
    sin         = math.sin
    cos         = math.cos
    exp         = math.exp
    PI          = np.pi
    step        = lambda a,x : 0 if x < a else 1
    smoothstep  = lambda a,b,x : 0 if x < a else 1 if x > b else 3*((x-a)/(b-a))**2 - 2 * ((x-a)/(b-a))**3
    doubleslope = lambda x,a,d,s: smoothstep(-.00001,a,x) - (1.-s) * smoothstep(0.,d,x-a);
    clamp       = lambda x,a,b: min(b, max(a,x))
    fract       = lambda x: math.modf(x)[0]
    _sin        = lambda x: math.sin(2. * np.pi * (x % 1.))
    _saw        = lambda x: 2.*fract(x) - 1.
    _tri        = lambda x: 4.*abs(fract(x)-.5)-1
    s_atan      = lambda x: 2./np.pi * math.atan(x)
    s_crzy      = lambda x: clamp(s_atan(x) - 0.1*math.cos(0.9*x*math.exp(x)), -1., 1.)
    mix         = lambda x, y, a: a*y + (1-a)*x


#float _sin(float a) { return sin(2. * PI * mod(a,1.)); }
#float _unisin(float a,float b) { return (.5*_sin(a) + .5*_sin((1.+b)*a)); }
#float _sq(float a) { return sign(2.*fract(a) - 1.); }
#float _squ(float a,float pwm) { return sign(2.*fract(a) - 1. + pwm); }
#float _tri(float a) { return (4.*abs(fract(a)-.5) - 1.); }
#float _saw(float a) { return (2.*fract(a) - 1.); }
#float quant(float a,float div,float invdiv) { return floor(div*a+.5)*invdiv; }
#float quanti(float a,float div) { return floor(div*a+.5)/div; }
#float clip(float a) { return clamp(a,-1.,1.); }
#float theta(float x) { return smoothstep(0., 0.01, x); }
#float freqC1(float note){ return 32.7 * pow(2.,note/12.); }
#float minus1hochN(int n) { return (1. - 2.*float(n % 2)); }

    print()
    print(synstr)

    L = 2 # how to set from within template file?
    for i in range(1,len(lines)):
        print(lines[i])
        exec(lines[i])

    print(L)
#    data = list(map(eval(synstr), i/rate for i in range(L*rate)))

#    temp = [i/rate for i in range(L*rate)]
#    data = [eval(synstr) for t in temp]
    
    data = []
    drumseq = [1, 5, 3, 5, 2, 5, 4, 5];
    drumseq_N = len(drumseq);
    subdiv_N = 4;
    loop_N = 4;
    bpm = 170;
    tick_len = 60/bpm * 1/subdiv_N;
    L = drumseq_N/subdiv_N * loop_N * 60/bpm;

    # AmpComp1 Kick_f1 Kick_f2 Kick_Dec ... see for yourself
    P_init = [1.0, 60,  150, 0.30, 1.0, 1.0, 6000.,  800., 350., 0.010, .10, 1.0, 2.0, 0.05, 0.3, 0.3]
    P_strt = [0.5, 40,  100, 0.01, 0.5, 0.5, 4000.,  600., 200., 0.002, .20, 0.2, 1.0, 0.01, 0.2, 0.2]
    P_stop = [1.0, 100, 350, 0.50, 1.0, 1.0, 9999., 1500., 400., 0.015, .02, 1.5, 2.0, 0.20, 0.4, 0.4]

    PM = []
    for i in range(0, max(drumseq)):
        P = []
        for p in range(0, len(P_init)):
            #P.append(mix(P_init[p], np.random.uniform(P_strt[p], P_stop[p]), .5))
            P.append(mix(P_init[p], np.random.uniform(P_strt[p], P_stop[p]), .8))
        PM.append(P)
        
    print("writing", int(L*rate), " samples")
    for i in range(int(L*rate)):
        T = i/rate
        t = T % tick_len
        drumseq_i = int(T / tick_len) % drumseq_N
        s = 0.

        mode = 'random'

        vel = 1 - drumseq[drumseq_i]/6;

        for p in range(0, len(P_init)): P[p] = PM[drumseq[drumseq_i]-1][p]

        if mode == 'random':
            
            f   = P[1] + P[2] * smoothstep(-P[3], 0., -t)
            env = smoothstep(0.,0.01,t) * smoothstep(-0.1, 0.2, 0.3 - t)
            #kick_body = env * .1*TRISQ(t, f, 100, 1., 1., .1, 16., 10.)
            kick_body = (_tri(t*f) + .7*_sin(t*f*.5)) * env
            kick_click = s_atan(40.*(1.-exp(-1000.*t))*exp(-80.*t) * _sin((1200.-1000.*sin(1000.*t*sin(30.*t)))*t))
            kick_blobb = s_crzy(10.*(1.-exp(-1000.*t))*exp(-30.*t) * _sin((300.-300.*t)*t))
            
            kick = clamp(P[0] * kick_body + P[4] * kick_blobb + P[5] * kick_click, -1.0, 1.0)

            f1 = P[6]
            f2 = P[7]
            f3 = P[8]
            dec12 = P[9]
            dec23 = P[9]
            rel = P[10]
            length = 3*P[10]
            snr = _sin(t * (f3 + (f1-f2)*smoothstep(-dec12,0.,-t)
                             + (f2-f3)*smoothstep(-dec12-dec23,-dec12,-t))) * smoothstep(-rel,-dec12-dec23,-t);
            #noise = 2. * fract(sin(t * 90000.) * 45000.) * doubleslope(t,0.05,0.3,0.3);
            noise = (2. * random()-1) * doubleslope(t,P[13],P[14],P[15])
            snr = s_atan(P[12] * snr + P[11] * noise) * doubleslope(t,0.0,0.25,0.3) * step(t,length);

            s = s_atan(kick + snr);
            
        elif mode == 'hardkick':
            f   = 60. + 150. * smoothstep(-0.3, 0., -t)
            env = smoothstep(0.,0.01,t) * smoothstep(-0.1, 0.2, 0.3 - t)
            kick_body = env * .1*TRISQ(t, f, 100, 1., 1., .1, 16., 10.)
            kick_body += .7 * (smoothstep(0.,0.01,t) * smoothstep(-0.2, 0.2, 0.3 - t)) * _sin(t*f*.5)
            kick_click = 1.5*step(t,0.05) * _sin(t*5000. * _saw(t*1000.))
            kick_click = s_atan(40.*(1.-exp(-1000.*t))*exp(-80.*t) * _sin((1200.-1000.*sin(1000.*t*sin(30.*t)))*t))
            kick_blobb = s_crzy(10.*(1.-exp(-1000.*t))*exp(-30.*t) * _sin((300.-300.*t)*t))
            s = 2.*clamp(kick_body + kick_blobb + kick_click,-1.5,1.5)

        elif mode == 'facekick':
            f   = 50. + 150. * smoothstep(-0.12, 0., -t)
            env = smoothstep(0.,0.015,t) * smoothstep(-0.08, 0., 0.16 - t)
            kick_body = env * TRISQ(t, f, 3, 1., 0.8, 8., 4., 1.)
            kick_click = 0.4 * step(t,0.03) * _sin(t*1100. * _saw(t*800.))
            kick_blobb = (1.-exp(-1000.*t))*exp(-40.*t) * _sin((400.-200.*t)*t * _sin(1.*f*t))
            s = (kick_body + kick_blobb + 0.1*kick_click)

        elif mode == 'snare':
            f1 = 6000.
            f2 = 800.
            f3 = 350.
            dec12 = 0.01
            dec23 = 0.01
            rel = 0.1
            length = 0.3
            snr = _sin(t * (f3 + (f1-f2)*smoothstep(-dec12,0.,-t)
                             + (f2-f3)*smoothstep(-dec12-dec23,-dec12,-t))) * smoothstep(-rel,-dec12-dec23,-t);
            #noise = 2. * fract(sin(t * 90000.) * 45000.) * doubleslope(t,0.05,0.3,0.3);
            noise = (2. * random()-1) * doubleslope(t,0.05,0.3,0.3)
            noise = 0
            s = clamp(1.7 * (2. * snr + noise), -1, 1) * doubleslope(t,0.0,0.25,0.3) * step(t,length);

        elif mode == 'bass1':
            s = 2. * fract(_sin(t * 90.) * 45000.) * doubleslope(t,0.05,0.3,0.3);

        else:
            pass
#            s = eval(synstr)
        
        s *= vel;
        
        data.append(s)
        
    sf.write('./shit.wav', data, rate, 'PCM_24')
    
#    sf.write('noise.wav', np.random.randn(10000, 2), 44100, 'PCM_24')

def TRISQ(t, f, MAXN, MIX, INR, NDECAY, RES, RES_Q):
    ret = 0.
    Ninc = 8 # try this: leaving out harmonics...

    _sin = lambda x: math.sin(2. * np.pi * x)
    
    for N in range(0, MAXN, Ninc):
        mode     = 2.*N + 1.;
        inv_mode = 1./mode
        comp_TRI = (-1 if N % 2 == 1 else 1) * inv_mode*inv_mode;
        comp_SQU = inv_mode;
        filter_N = math.pow(1. + math.pow(N * INR,2.*NDECAY),-.5) + RES * math.exp(-math.pow(N*INR*RES_Q,2.));
        ret += (MIX * comp_TRI + (1.-MIX) * comp_SQU) * filter_N * _sin(mode * f * t);
    
    return ret


if __name__ == '__main__':
    print("haha", math.sin(math.acos(-1)))
    ReadStuff()
