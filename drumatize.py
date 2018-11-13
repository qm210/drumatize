import math
import wave
import matplotlib.pyplot as plt
import numpy as np
import struct
import soundfile as sf

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
    sin = math.sin
    cos = math.cos
    exp = math.exp
    PI = np.pi
    smoothstep = lambda a,b,x : 0 if x < a else 1 if x > b else 3*((x-a)/(b-a))**2 - 2 * ((x-a)/(b-a))**3

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
    for i in range(int(L*rate)):
        t = i/rate
        s = eval(synstr)
        data.append(s)
        
    sf.write('./shit.wav', data, rate, 'PCM_24')
    
#    sf.write('noise.wav', np.random.randn(10000, 2), 44100, 'PCM_24')

if __name__ == '__main__':
    print("haha", math.sin(math.acos(-1)))
    ReadStuff()
