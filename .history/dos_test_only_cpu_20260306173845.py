import os
os.environ["OMP_NUM_THREADS"] = "8" 

import minijclsquant as jcl
import numpy as np
from time import perf_counter
from math import pi,sqrt
import scipy as sci
N=2**(3+15)


S=jcl.lattice_hexagonal(N)

print('Number of atoms : '+str(np.shape(S)[0]))

t=-2.7+0j
m=0.5+0j

start=perf_counter()

H=jcl.H_graphene(S,t,m,0+0j,True,'ELL')
end=perf_counter()

print('Time it takes to build the H : '+str( end-start)+' [s]')

print('Area of the system : '+str(H.Omega)+' [nm**2]')


random_vector=jcl.random_vector_generator(N)


start=perf_counter()
dos_cpu=jcl.kpm_dos(H,int(np.sqrt(N)),random_vector)

dos_cpu[:,1]=dos_cpu[:,1]/(N*H.bounds[1])
end=perf_counter()

print('Time in cpu : '+str(end-start))

print('Integral of dos cpu : '+str(sci.integrate.simpson(dos_cpu[:,1],x=dos_cpu[:,0])))



