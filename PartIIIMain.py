import numpy as np
import matplotlib.pyplot as plt
import PartIIISimulations as simulations

# Initialization
sigma = np.empty((4,2,2),np.complex)
sigma[0] = [[1,0],[0,1]]
sigma[1] = [[0,1],[1,0]]
sigma[2] = [[0,-1j],[1j,0]]
sigma[3] = [[1,0],[0,-1]]

t1 = 1 # Nearest-neighbour hopping amplitude
t2 = 1/3 # Next-nearest-neighbour hopping amplitude
phi = np.pi/4 # Next-nearest-neighbour hopping phase
delta = (3.67423-1)/3 # On-site energy
# Nearest-Neighbour and Next-Nearest-Neighbour Lattice Vectors
a1 = np.array([0,1])
a2 = np.array([np.sin(np.pi/3),-np.cos(np.pi/3)])
a3 = np.array([-np.sin(np.pi/3),-np.cos(np.pi/3)])
b1 = np.array([2*np.sin(np.pi/3),0])
b2 = np.array([-np.sin(np.pi/3),-np.cos(np.pi/3)-1])
b3 = np.array([-np.sin(np.pi/3),1+np.cos(np.pi/3)])

beta = 5 # Inverse temperature
boundaryconditions = 0 # 0 for PBCPBC and 1 for PBCOBC
nonzerotemp = 1 # 0 for zero temperature and 1 for non-zero temperature
measpoints = 3000
L = 80 # Number of sites in the Rhombically Shaped System equals 1/2*L^2

if boundaryconditions == 0:
    # Determining the Bulk Spectrum of the Haldane Model (i.e., PBC in both directions)
    eigvals = simulations.BulkSpectrumPBCPBC(measpoints,sigma,t1,t2,phi,delta,a1,a2,a3,b1,b2,b3)
    
    # Determining Chemical Potential such that System is at Half-Filling for PBC in both directions
    mu = (np.max(eigvals[:,0,:]) + np.min(eigvals[:,1,:]))/2

if boundaryconditions == 1:
    # Determining the Spectrum for the Haldane Model with PBC in one direction and OBC in the other direction
    height = L#40 # Height of the Cylindrical Shape
    eigvalsobc, kvector = simulations.SpectrumPBCOBC(measpoints,sigma,t1,t2,phi,delta,a1,a2,a3,b1,b2,b3,height)
    
    # Determining Chemical Potential such that System is at Half-Filling for PBC in one direction and OBC in other direction
    mu = (np.max(eigvalsobc[:,np.int(height/2)-1]) + np.min(eigvalsobc[:,np.int(height/2)]))/2

# Generating Haldane Model Hamiltonian in Position Basis for Rhombic System (with number of sites N = LxL/2)
N = np.int(L**2/2)

if boundaryconditions == 0:
    H = simulations.HamiltonianRhombicPBCPBC(t1,t2,phi,delta,L,N)
flux = np.pi/np.sqrt(3) - np.pi/(L*np.sin(np.pi/3))
if boundaryconditions == 1:
    H = simulations.HamiltonianRhombicPBCOBC(t1,t2,phi,delta,L,N,flux)

# Obtaining H at Half-Filling
evalsHrhombic, evecsHrhombic = np.linalg.eigh(H)
Hhalffilling = np.matmul(np.matmul(evecsHrhombic,np.diag(evalsHrhombic - mu)),np.matrix(evecsHrhombic).H)

# Evaluating Tensor Network Link Matrix at Half-Filling
hT = np.zeros((N,N), dtype = np.complex)
if nonzerotemp == 0:
    # Zero T
    hT = np.matmul(np.matmul(evecsHrhombic,np.diag(np.sign(evalsHrhombic - mu))),np.matrix(evecsHrhombic).H)
if nonzerotemp == 1:
    # Non-zero T
    hT = np.matmul(np.matmul(evecsHrhombic,np.diag(np.cosh(beta/2*(evalsHrhombic - mu))/np.sinh(beta/2*(evalsHrhombic - mu)))),np.matrix(evecsHrhombic).H)

# Evaluating the Matrix Element Magnitude as a Function of the Distance between the Associated Sites in the Honeycomb Lattice of the Haldane Model
# Obtaining a Matrix containing Distances between Lattice Sites
if boundaryconditions == 0:
    distarray = simulations.HoneycombDistancesPBCPBC(L,N)
if boundaryconditions == 1:
    distarray = simulations.HoneycombDistancesPBCOBC(L,N)
# Obtaining Matrix Element Magnitude (for H and hT, both for half-filling) as a Function of Distance between Sites
uniquedists, Hweights, hTweights = simulations.HhTMagnitude(distarray,Hhalffilling,hT)

# Implementation of the Truncation-Reconstruction Scheme and Evaluation of several Observables
numLm = 100
Lminterval = 1
eta, Lm, Str, S, R, correlations_tr, correlations = simulations.TruncRecObs(numLm,Lminterval,evalsHrhombic,mu,N,distarray,uniquedists,hT,beta,nonzerotemp)


