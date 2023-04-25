###################################################################################################
# Test Routines for Signal Extraction
###################################################################################################
from signal_extraction import *
import matplotlib.pyplot as plt 

# Test Signal Extraction 
phi,sig_e,sig_v = 0.9,0.5,0.1 
InitState=0
InitVariance=sig_v**2/(1-(phi**2))
N,T = 1000,50
which_path_filtered = 10 

Observations,States,e,v = SignalSimulation_1D(N,T,InitState,InitVariance,phi,sig_e,sig_v,seed_init=0)
OneStepCondMeans,OneStepVariances,KalmanGains,Likelihood_vec,LogLikelihood = SignalExtractionProblem_1D(InitState,InitVariance,Observations[:,which_path_filtered],phi,sig_e,sig_v)

plt.plot(Observations[:,which_path_filtered],label='observation input')
plt.plot(States[:,which_path_filtered],label='state realized')
plt.plot(OneStepCondMeans,label='filtered conditional mean')
plt.legend() 
plt.show()

