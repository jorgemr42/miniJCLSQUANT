import scipy as sci
import numpy as np
from time import perf_counter
import copy
from math import pi,sqrt
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
from skopt import gp_minimize


import numpy as np

from scipy.constants import pi,k,e,hbar

from scipy.optimize import minimize,NonlinearConstraint,BFGS,SR1

from scipy.integrate import trapezoid as trapz
from scipy.integrate import cumulative_trapezoid as cumtrapz

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

from jclsquant.constants import *
import matplotlib
from jclsquant.kernel_and_moments import *



###########################################################
# Define class containing relevant functions / parameters #
# for constrained minimization                            #
###########################################################
class problem:
    """
    Def:
        Problem class build by Aron W.Cummings for the minimization.
    """
    # Class init: read in DOS, calculate needed quantities
    def __init__(self, dos,deltaE,u0,T0):

        k=kb
        data = dos                 # Read in DOS from file
        E    = data[:,0]           # [eV]
        dE   = E[1]-E[0]           # [eV]
        dos  = 2*data[:,1]         # [e-/eV/C-atom]
        kT0  = k*T0                # [eV]
        
        # Index of charge neutrality point
        ntot = trapz(dos,dx=dE)
        ncum = cumtrapz(dos)*dE
        icnp = np.where(ncum<=ntot/2)[0][-1]
        ecnp = E[len(E)//2]
        # print()
        # # print('icnp = '+str(icnp))
        # print('ntot = '+str(ntot)+' e- / C-atom')
        # print('ncnp = ',trapz(dos[0:icnp],dx=dE))
        # print('ecnp = '+str(ecnp)+' eV')
        # print()

        # Save needed variables for use in functions
        self.E    = E - ecnp  # Shift energy so that E=0 is the CNP
        self.dE   = dE
        self.dos  = dos
        self.icnp = icnp

        # Starting energy and density, plus target energy
        self.n0  = self.n_fun([u0,kT0])
        self.e0  = self.e_fun([u0,kT0])

        # print('Initial total energy  : '+str(self.e0))
        # print('Initial total density of electrons  : '+str(self.n0))


        # print('Total energy for thermal distribution : '+str(self.e_fun([u0,kT0])))
        # print('Total density of electrons for thermal distribution : '+str(self.n_fun([u0,kT0])))

        self.etarget = self.e0 + deltaE
        self.ntarget = self.n0

    # Function to calculate energy at a given u and T
    def e_fun(self,x):
        u  = x[0]
        kT = x[1]

        ee = (self.E-u)/(2*kT)
        eh = (self.E+u)/(2*kT)
        fe = 1 / (1 + np.exp(2*ee))
        fh = 1 / (1 + np.exp(2*eh))

        return trapz( ((fe+fh)*self.E*self.dos)[self.icnp:-1] , dx=self.dE)


    # Function to calculate dE/du
    def dedu_fun(self,x):
        u  = x[0]
        kT = x[1]

        ee = (self.E-u)/(2*kT)
        eh = (self.E+u)/(2*kT)
        dfe_du =  1/np.cosh(ee)**2 / (4*kT)
        dfh_du = -1/np.cosh(eh)**2 / (4*kT)

        return trapz( ((dfe_du+dfh_du)*self.E*self.dos)[self.icnp:-1],dx=self.dE)


    # Function to calculate dE/dkT
    def dedkT_fun(self,x):
        u  = x[0]
        kT = x[1]

        ee = (self.E-u)/(2*kT)
        eh = (self.E+u)/(2*kT)
        dfe_dkT = ee / np.cosh(ee)**2 / (2*kT)
        dfh_dkT = eh / np.cosh(eh)**2 / (2*kT)

        return trapz( ((dfe_dkT+dfh_dkT)*self.E*self.dos)[self.icnp:-1],dx=self.dE)


    # Function to calculate density at a given u and T
    def n_fun(self,x):
        u  = x[0]
        kT = x[1]

        ee = (self.E-u)/(2*kT)
        eh = (self.E+u)/(2*kT)
        fe = 1 / (1 + np.exp(2*ee))
        fh = 1 / (1 + np.exp(2*eh))

        return trapz( ((fe-fh)*self.dos)[self.icnp:-1] , dx=self.dE)


    # Function to calculate dn/du
    def dndu_fun(self,x):
        u  = x[0]
        kT = x[1]

        ee = (self.E-u)/(2*kT)
        eh = (self.E+u)/(2*kT)
        dfe_du =  1/np.cosh(ee)**2 / (4*kT)
        dfh_du = -1/np.cosh(eh)**2 / (4*kT)

        return trapz( ((dfe_du-dfh_du)*self.dos)[self.icnp:-1] , dx=self.dE)


    # Function to calculate dn/dkT
    def dndkT_fun(self,x):
        u  = x[0]
        kT = x[1]

        ee = (self.E-u)/(2*kT)
        eh = (self.E+u)/(2*kT)
        dfe_dkT = ee / np.cosh(ee)**2 / (2*kT)
        dfh_dkT = eh / np.cosh(eh)**2 / (2*kT)

        return trapz( ((dfe_dkT-dfh_dkT)*self.dos)[self.icnp:-1] , dx=self.dE)


    # Function to calculate d^2n/du^2
    def d2ndu2_fun(self,x):
        u  = x[0]
        kT = x[1]

        ee = (self.E-u)/(2*kT)
        eh = (self.E+u)/(2*kT)

        d2fe_du2 = np.tanh(ee) / np.cosh(ee)**2 / (2*kT)**2
        d2fh_du2 = np.tanh(eh) / np.cosh(eh)**2 / (2*kT)**2

        return trapz( ((d2fe_du2-d2fh_du2)*self.dos)[self.icnp:-1] , dx=self.dE)


    # Function to calculate d^2n/dkT^2
    def d2ndkT2_fun(self,x):
        u  = x[0]
        kT = x[1]

        ee = (self.E-u)/(2*kT)
        eh = (self.E+u)/(2*kT)

        d2fe_dkT2 = (ee*np.tanh(ee) - 1) * ee / np.cosh(ee)**2 / kT**2
        d2fh_dkT2 = (eh*np.tanh(eh) - 1) * eh / np.cosh(eh)**2 / kT**2

        return trapz( ((d2fe_dkT2-d2fh_dkT2)*self.dos)[self.icnp:-1] , dx=self.dE)


    # Function to calculate d^2n/(du*dkT)
    def d2ndudkT_fun(self,x):
        u  = x[0]
        kT = x[1]

        ee = (self.E-u)/(2*kT)
        eh = (self.E+u)/(2*kT)

        d2fe_dudkT =  (2*ee*np.tanh(ee) - 1) / np.cosh(ee)**2 / (2*kT)**2
        d2fh_dudkT = -(2*ee*np.tanh(eh) - 1) / np.cosh(eh)**2 / (2*kT)**2

        return trapz( ((d2fe_dudkT-d2fh_dudkT)*self.dos)[self.icnp:-1] , dx=self.dE)


    # # Minimization function to find target energy
    def emin_fun(self,x):
        return 0.8*(self.e_fun(x) - self.etarget)**2/(self.etarget)**2+0.2*(self.n_fun(x)-self.ntarget)**2

    # Derivative of minimization function with respect to u
    def demindu_fun(self,x):
        return 2*0.8*(self.e_fun(x) - self.etarget) * self.dedu_fun(x)/(self.etarget)**2+2*0.2*(self.n_fun(x)-self.ntarget)*self.dndu_fun(x)


    # Derivative of minimization function with respect to kT
    def demindkT_fun(self,x):
        return 2*0.8*(self.e_fun(x) - self.etarget) * self.dedkT_fun(x)/(self.etarget)**2+2*0.2*(self.n_fun(x)-self.ntarget)*self.dndkT_fun(x)
    
###########################################################
###########################################################
###########################################################



def minimization(deltaE,dos,initial_guess,niter=1000):    
    """
    Def:
        Minimization function build by Aron W.Cummings and implemented by me to JLSQUANT

    Inputs:
        n_tot: Initial electron density, be carefull as it must the obtained 
                by the dos being normalized to int^inf_Ecnp(dos)=4 (In the case of graphene).
        n_tot: Initial total Energy, be carefull as it must the obtained 
                by the dos being normalized to int^inf_Ecnp(dos)=4 (In the case of graphene).
        dos: Density of states so that dos[:,0] are the energyes and dos[:,1] is the dos WITHOUT normalization of any kind.
        initial_guess: array [mu_0 [eV],T_0 [eV]] of the initial guess for the minimization
        deltaE: Difference of energy between the excited and the thermal distibution both with the same energy.
        niter: Number of iterations by default=1000

    Outputs:
        m_new: Final chemical potential
        T_new: Final temperature in eV
    """
    dos_norm = np.copy(dos)
    dos_norm[:,1] = 2*dos[:,1] / sci.integrate.simpson(dos[:,1],x=dos[:,0],dx=np.abs(dos[0,0]-dos[1,0]))
    k=kb
    u0,T0 = initial_guess

    T0 = T0/kb
    # print('mu old : '+str(u0))
    # print('Temp old : '+str(T0))

    p = problem(dos_norm,deltaE,u0,T0)
    #method = 'trust-constr'
    # method = 'L-BFGS-B'
    method = 'SLSQP'
    #method = 'COBYLA'

    # Use gradients of E and n to step our way towards a good initial guess for the minimizer
    uiter = np.zeros(2*niter+1)
    Titer = np.zeros(2*niter+1)
    uiter[0] = u0
    Titer[0] = T0
    uold  = u0
    Told  = T0

    uguess = uold
    Tguess = Told




    # Set up nonlinear constraint for energy minimization
    # Constraint function
    def constr_eq(x):
        return p.n_fun(x)-p.n0

    def constr_ineq(x):
        return [(p.n0+p.n0/1e10)-p.n_fun(x) , p.n_fun(x)-(p.n0-p.n0/1e10)]

    # Jacobian of constraint
    def constr_jac(x):
        return [p.dndu_fun(x) , p.dndkT_fun(x)]

    # Hessian of constraint
    def constr_hess(x,v):
        return v[0]*np.array([[p.d2ndu2_fun(x)   , p.d2ndudkT_fun(x)] ,
                            [p.d2ndudkT_fun(x) , p.d2ndkT2_fun(x)]])

    # Jacobian of minimization function
    def emin_jac(x):
        return [p.demindu_fun(x),p.demindkT_fun(x)]

    # Set up the constraint -- density must be equal to initial density n0
    # and call the minimizer to get the new chemical potential + temperature
    hyper_repetitions=101
    T_guess_array=np.linspace(Tguess/2,5*Tguess,hyper_repetitions)
    if uguess>=0:
        u_guess_array=np.linspace(uguess/2,5*uguess,hyper_repetitions)
    else:
        u_guess_array=np.linspace(10*uguess,uguess/2,hyper_repetitions)

    results_T=np.zeros(hyper_repetitions)
    results_u=np.zeros(hyper_repetitions)
    results_min=np.zeros(hyper_repetitions)
    for i in range(hyper_repetitions):
        if method == 'trust-constr':
            nl_constr = NonlinearConstraint(p.n_fun, p.n0,p.n0,
                                            jac=constr_jac, hess=constr_hess)
            result = minimize(p.emin_fun,[uguess,k*Tguess],
                            jac=emin_jac, hess=BFGS(),
                            constraints=nl_constr,
                            method=method,
                            options={'verbose':1,'gtol':1e-100,'xtol':1e-100})
        elif method == 'SLSQP':
            nl_constr = { 'type':'eq' , 'fun':constr_eq , 'jac':constr_jac }
            tolerance = abs(deltaE)/1e10
            result = minimize(p.emin_fun,[u_guess_array[i],T_guess_array[i]*k],
                            jac=emin_jac,
                            constraints=nl_constr,
                            method=method,
                            tol=tolerance)
        elif method == 'COBYLA':
            nl_constr = { 'type':'ineq' , 'fun':constr_ineq , 'jac':constr_jac }
            tolerance = abs(deltaE)/1e10
            result = minimize(p.emin_fun,[uguess,k*Tguess],
                            constraints=nl_constr,
                            method=method,
                            tol=tolerance,
                            options={'disp':True})

        elif method == 'L-BFGS-B':
            nl_constr = { 'type':'eq' , 'fun':constr_eq , 'jac':constr_jac }
            tolerance = abs(deltaE)/1e10
            result = minimize(p.emin_fun,[uguess,k*Tguess],
                            jac=emin_jac,
                            constraints=None,
                            method=method,)
        

        results_u[i]= result.x[0]
        results_T[i] = result.x[1]
        results_min[i]=p.emin_fun([results_u[i],results_T[i]])
        
    unew=results_u[np.argmin(results_min)]
    Tnew=results_T[np.argmin(results_min)]/k
    # plt.plot(results_T/k)
    # plt.show()
    # plt.plot(results_min)
    # plt.show()
    # print('mu new : '+str(unew))
    # print('Temp new : '+str(Tnew))
    return unew,Tnew






def minimization2(deltaE,dos,initial_guess,niter=1000):    
    """
    Def:
        Minimization function build by Aron W.Cummings and implemented by me to JLSQUANT

    Inputs:
        n_tot: Initial electron density, be carefull as it must the obtained 
                by the dos being normalized to int^inf_Ecnp(dos)=4 (In the case of graphene).
        n_tot: Initial total Energy, be carefull as it must the obtained 
                by the dos being normalized to int^inf_Ecnp(dos)=4 (In the case of graphene).
        dos: Density of states so that dos[:,0] are the energyes and dos[:,1] is the dos WITHOUT normalization of any kind.
        initial_guess: array [mu_0 [eV],T_0 [eV]] of the initial guess for the minimization
        deltaE: Difference of energy between the excited and the thermal distibution both with the same energy.
        niter: Number of iterations by default=1000

    Outputs:
        m_new: Final chemical potential
        T_new: Final temperature in eV
    """
    dos_norm = np.copy(dos)
    dos_norm[:,1] = 2*dos[:,1] / sci.integrate.simpson(dos[:,1],x=dos[:,0],dx=np.abs(dos[0,0]-dos[1,0]))
    k=kb
    u0,T0 = initial_guess

    T0 = T0/kb
    print('mu old : '+str(u0))
    print('Temp old : '+str(T0))

    p = problem(dos_norm,deltaE,u0,T0)
    #method = 'trust-constr'
    # method = 'L-BFGS-B'
    method = 'SLSQP'
    #method = 'COBYLA'

    # Use gradients of E and n to step our way towards a good initial guess for the minimizer
    uiter = np.zeros(2*niter+1)
    Titer = np.zeros(2*niter+1)
    uiter[0] = u0
    Titer[0] = T0
    uold  = u0
    Told  = T0




    uguess = uold
    Tguess = Told




    # Set up nonlinear constraint for energy minimization
    # Constraint function
    def constr_eq(x):
        return p.n_fun(x)-p.n0

    def constr_ineq(x):
        return [(p.n0+p.n0/1e10)-p.n_fun(x) , p.n_fun(x)-(p.n0-p.n0/1e10)]

    # Jacobian of constraint
    def constr_jac(x):
        return [p.dndu_fun(x) , p.dndkT_fun(x)]

    # Hessian of constraint
    def constr_hess(x,v):
        return v[0]*np.array([[p.d2ndu2_fun(x)   , p.d2ndudkT_fun(x)] ,
                            [p.d2ndudkT_fun(x) , p.d2ndkT2_fun(x)]])

    # Jacobian of minimization function
    def emin_jac(x):
        return [p.demindu_fun(x),p.demindkT_fun(x)]

    # Set up the constraint -- density must be equal to initial density n0
    # and call the minimizer to get the new chemical potential + temperature
    if uguess>=0:
        search_space = [((uguess / 2), (10 * uguess)),(k*Tguess / 2, 10 * Tguess*k)]
    else:
        search_space = [((10*uguess ), ( uguess/2)),(k*Tguess / 2, 10 * Tguess*k)]

    bo_result = gp_minimize(p.emin_fun,                  # Objective function
                            search_space,               # Bounds
                            n_calls=10,                 # Number of evaluations
                            random_state=42,            # Reproducibility
                            n_initial_points=5)   
    print("Bayesian Optimization Result:")
    print(f"  x = {bo_result.x[0]:.4f}, f(x) = {bo_result.fun:.4f}")  
    print(bo_result.x[0])
    print(bo_result.x[1]/k)
    if method == 'trust-constr':
        nl_constr = NonlinearConstraint(p.n_fun, p.n0,p.n0,
                                        jac=constr_jac, hess=constr_hess)
        result = minimize(p.emin_fun,[uguess,k*Tguess],
                        jac=emin_jac, hess=BFGS(),
                        constraints=nl_constr,
                        method=method,
                        options={'verbose':1,'gtol':1e-100,'xtol':1e-100})
    elif method == 'SLSQP':
        nl_constr = { 'type':'eq' , 'fun':constr_eq , 'jac':constr_jac }
        tolerance = abs(deltaE)/1e10
        result = minimize(p.emin_fun,[bo_result.x[0],bo_result.x[1]],
                        jac=emin_jac,
                        constraints=nl_constr,
                        method=method,
                        tol=tolerance)
    elif method == 'COBYLA':
        nl_constr = { 'type':'ineq' , 'fun':constr_ineq , 'jac':constr_jac }
        tolerance = abs(deltaE)/1e10
        result = minimize(p.emin_fun,[uguess,k*Tguess],
                        constraints=nl_constr,
                        method=method,
                        tol=tolerance,
                        options={'disp':True})

    elif method == 'L-BFGS-B':
        nl_constr = { 'type':'eq' , 'fun':constr_eq , 'jac':constr_jac }
        tolerance = abs(deltaE)/1e10
        result = minimize(p.emin_fun,[uguess,k*Tguess],
                        jac=emin_jac,
                        constraints=None,
                        method=method,)
        
    unew=result.x[0]
    Tnew=result.x[1]/k
    print('mu new : '+str(unew))
    print('Temp new : '+str(Tnew))
    return unew,Tnew























































    

def FD_optical_v3(E_vec,mu,T_mod,P_max,omega,W):
    if mu>-((omega/2)+1.76*0.5*W) and mu<((omega/2)+1.76*0.5*W) :
        return (1/(1+np.exp((E_vec-mu)/T_mod)))+(P_max/np.cosh((E_vec-omega/2)/W))-(P_max/np.cosh((E_vec+omega/2)/W))    
    else:
        return (1/(1+np.exp((E_vec-mu)/T_mod)))
    
def FD_optical(E_vec,mu,T_mod,P_max,omega,W):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if mu<-(omega/2+1.76*W):
            result=(1/(1+np.exp((E_vec-mu)/T_mod)))
        elif mu>=-(omega/2+1.76*W) and mu<-(omega/2)+1.76*W:
            result=np.heaviside((mu)-E_vec,0)*(1-P_max/np.cosh((E_vec+omega/2)/W))+np.heaviside(E_vec-mu,0)*(1/(1+np.exp((E_vec-mu)/T_mod)))+np.heaviside((mu+omega)-E_vec,0)*(P_max/np.cosh((E_vec-omega/2)/W))
        elif mu>-(omega/2)+1.76*W and mu<(omega/2)-1.76*W:
            result=(1/(1+np.exp((E_vec-mu)/T_mod)))-P_max/np.cosh((E_vec+omega/2)/W)+P_max/np.cosh((E_vec-omega/2)/W)
        elif mu>=(omega/2)-1.76*W and mu<=(omega/2)+1.76*W:
            # return np.heaviside((mu)-E_vec,0)*(1-P_max/np.cosh((E_vec+omega/2)/W))+np.heaviside(E_vec-mu,0)*(1/(1+np.exp((E_vec-mu)/T_mod)))+np.heaviside((mu+omega)-E_vec,0)*(P_max/np.cosh((E_vec-omega/2)/W))
            result=np.heaviside(mu-E_vec,0)*(1/(1+np.exp((E_vec-mu)/T_mod)))-np.heaviside((mu-omega)-E_vec,0)*(P_max/np.cosh((E_vec+omega/2)/W))+np.heaviside(E_vec-mu,0)*(P_max/np.cosh((E_vec-omega/2)/W))
        elif mu>=(omega/2)+1.76*W:
            result=(1/(1+np.exp((E_vec-mu)/T_mod)))
    result[result>1]=1
    result[result<0]=0
    return result




def FD_T(E_vec,mu,T_mod):
    return (1/(1+np.exp((E_vec-mu)/T_mod)))

def E_tot(dos,mu_0,T_0,P_max,omega_exc,W):
    return sci.integrate.simpson(dos[:,0]*dos[:,1]*(FD_optical(dos[:,0],mu_0,T_0,P_max,omega_exc,W)+FD_optical(dos[:,0],-mu_0,T_0,P_max,omega_exc,W)),x=dos[:,0],dx=np.abs(dos[1,0]-dos[0,0]))
def E_tot_0_v2(dos,mu_0,T_0,P_max,omega_exc,W):
    return sci.integrate.simpson(dos[:,0]*dos[:,1]*(FD_optical_v3(dos[:,0],mu_0,T_0,P_max,omega_exc,W)+FD_optical_v3(dos[:,0],-mu_0,T_0,P_max,omega_exc,W)),x=dos[:,0],dx=np.abs(dos[1,0]-dos[0,0]))

def n_tot_0(dos,mu_0,T_0,P_max,omega_exc,W):
    return sci.integrate.simpson(dos[:,1]*(FD_optical(dos[:,0],mu_0,T_0,P_max,omega_exc,W)-FD_optical_v2(dos[:,0],-mu_0,T_0,P_max,omega_exc,W)),x=dos[:,0],dx=np.abs(dos[1,0]-dos[0,0]))


def E_tot_1(dos,mu_0,T_0):
    return sci.integrate.simpson(dos[:,0]*dos[:,1]*(FD_T(dos[:,0],mu_0,T_0)+FD_T(dos[:,0],-mu_0,T_0)),x=dos[:,0],dx=np.abs(dos[1,0]-dos[0,0]))

def n_tot_1(dos,mu,T):
    return sci.integrate.simpson(dos[:,1]*(FD_T(dos[:,0],mu,T)-FD_T(dos[:,0],-mu,T)),x=dos[:,0],dx=np.abs(dos[1,0]-dos[0,0]))

def n_e_tot(dos,mu_0,T_0,P_max,omega_exc,W):
    """
    Def:
        Gives the electron density and the total energy of an excited distribution with mu_0, T_0  and omega_exc.

    Inputs:
        dos: Density of states so that dos[:,0] are the energyes and dos[:,1] is the dos WITHOUT normalization of any kind.
        mu_0: Chemical potential of the excited distribution in [eV]. 
        T_0: Temperature of the excited distribution in [eV].
        P_max: Height of the excitation i the range [0,1]
        omega_exc: Energy of the excitation in [eV].
        W: Width of the excitation.

    Outputs:
        n_tot: Density of electrons
        E_tot: Total energy 
    """


    # W=2*(dos[1,0]-dos[0,0])
    # print('Broadening of the excitation : '+str(W)+' [eV]')


    dos_norm_2=np.copy(dos)
    dos_norm_2[:,1]=4*dos[:,1]/sci.integrate.simpson(dos[:,1],x=dos[:,0],dx=np.abs(dos[0,0]-dos[1,0]))
    ntot = sci.integrate.simpson(dos_norm_2[:,1],dx=np.abs(dos[1,0]-dos[0,0]))
    ncum = sci.integrate.cumulative_simpson(dos_norm_2[:,1])*np.abs(dos[1,0]-dos[0,0])
    icnp = np.where(ncum<=ntot/2)[0][-1]
    ecnp = dos_norm_2[icnp,0]

    dos_norm_2[:,0]=dos_norm_2[:,0]-ecnp

    # print('Total n : '+str(ntot))
    # print('Charge neutrality point : '+str(ecnp)+' [eV]')



    E_tot_exc=E_tot(dos_norm_2[icnp:-1,:],mu_0,T_0,P_max,omega_exc,W)
    E_tot_thermal=E_tot_1(dos_norm_2[icnp:-1,:],mu_0,T_0)
    return E_tot_exc,E_tot_thermal

def n_e_tot_2(dos,mu_0,T_0,P_max,omega_exc,W):
    """
    Def:
        Gives the electron density and the total energy of an excited distribution with mu_0, T_0  and omega_exc.

    Inputs:
        dos: Density of states so that dos[:,0] are the energyes and dos[:,1] is the dos WITHOUT normalization of any kind.
        mu_0: Chemical potential of the excited distribution in [eV]. 
        T_0: Temperature of the excited distribution in [eV].
        omega_exc: Energy of the excitation in [eV].

    Outputs:
        n_tot: Density of electrons
        E_tot: Total energy 
    """


    # W=2*(dos[1,0]-dos[0,0])
    # print('Broadening of the excitation : '+str(W)+' [eV]')


    dos_norm_2=np.copy(dos)
    dos_norm_2[:,1]=4*dos[:,1]/sci.integrate.simpson(dos[:,1],x=dos[:,0],dx=np.abs(dos[0,0]-dos[1,0]))
    ntot = sci.integrate.simpson(dos_norm_2[:,1],dx=np.abs(dos[1,0]-dos[0,0]))
    ncum = sci.integrate.cumulative_simpson(dos_norm_2[:,1])*np.abs(dos[1,0]-dos[0,0])
    icnp = np.where(ncum<=ntot/2)[0][-1]
    ecnp = dos_norm_2[icnp,0]

    dos_norm_2[:,0]=dos_norm_2[:,0]-ecnp

    # print('Total n : '+str(ntot))
    # print('Charge neutrality point : '+str(ecnp)+' [eV]')



    # n_tot=n_tot_0(dos_norm_2[icnp:-1,:],mu_0,T_0,omega_exc,W)
    E_tot_exc=E_tot_0_v2(dos_norm_2[icnp:-1,:],mu_0,T_0,P_max,omega_exc,W)






    n_tot_2=n_tot_1(dos_norm_2[icnp:-1,:],mu_0,T_0)
    E_tot_thermal=E_tot_1(dos_norm_2[icnp:-1,:],mu_0,T_0)



    # print('Initial total energy  : '+str(E_tot)+'  [eV]')
    # print('Initial total density of electrons  : '+str(n_tot))


    # print('Total energy for thermal distribution : '+str(E_tot_2))
    # print('Total density of electrons for thermal distribution : '+str(n_tot_2))
    return E_tot_exc,E_tot_thermal




def FD_diff(dos,P_max,omega_exc,W,mu_0,T_0,unew,Tnew,t_vec,tau):
    # W=2*(dos[1,0]-dos[0,0])
    FD_1=FD_optical(dos[:,0],mu_0,T_0,P_max,omega_exc,W)
    FD_2=FD_T(dos[:,0],unew,Tnew)
    FD_3=FD_T(dos[:,0],mu_0,T_0)
    delta_t=t_vec[1]-t_vec[0]
    FD_time=np.zeros((len(dos[:,0]),len(t_vec)))
    FD_T_time=np.zeros((len(dos[:,0]),len(t_vec)))
    FD_time[:,0]=FD_1
    for i in range(1,len(t_vec)):
        FD_T_time[:,i]=FD_3
        FD_time[:,i]=(1-delta_t/tau)*FD_time[:,i-1]+delta_t*FD_2/tau
    return FD_time,FD_T_time

def moments_FD_diff(M,dos,P_max,omega_exc,W,mu_0,T_0,unew,Tnew,t_vec,tau):
    # W=2*(dos[1,0]-dos[0,0])
    FD_1=FD_optical(dos[:,0],mu_0,T_0,P_max,omega_exc,W)
    FD_2=FD_T(dos[:,0],unew,Tnew)
    FD_3=FD_T(dos[:,0],mu_0,T_0)
    delta_t=t_vec[1]-t_vec[0]
    
    moments_FD_time=np.zeros((M,len(t_vec)))
    
    moments_FD_time[:,0]=moments_FD(M,FD_1)
    moments_FD_2=moments_FD(M,FD_2)
    moments_FD_3=moments_FD(M,FD_3)

    for i in range(1,len(t_vec)):
        moments_FD_time[:,i]=(1-delta_t/tau)*moments_FD_time[:,i-1]+delta_t*moments_FD_2/tau
    return moments_FD_time,moments_FD_3







viridis = matplotlib.cm.get_cmap('viridis', 100)

def plotting_distribution(dos,P_max,omega_exc,W,mu_0,T_0,unew,Tnew,t_vec,tau):
    
    FD_diff_vec,_=FD_diff(dos,P_max,omega_exc,W,mu_0,T_0,unew,Tnew,t_vec,tau)

    norm = matplotlib.colors.Normalize(vmin=t_vec.min(), vmax=t_vec.max())
    colors = viridis(norm(t_vec)) 
    for i in np.arange(0,len(t_vec),len(t_vec)//10):
        plt.plot(dos[:,0],FD_diff_vec[:,i],color=colors[i],linewidth=2)

    plt.xlim([-3,3])
    
    sm = matplotlib.cm.ScalarMappable(cmap=viridis, norm=norm)
    sm.set_array(t_vec)  # Empty array for ScalarMappable
    plt.colorbar(sm, ax=plt.gca(), label=r'$t$ [fs]')
    plt.xlabel(r'$E_F$ [eV]')
    plt.ylabel(r'Occupation number')
    plt.show()
    