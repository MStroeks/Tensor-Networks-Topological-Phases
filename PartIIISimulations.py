import numpy as np
import matplotlib.pyplot as plt

def BulkSpectrumPBCPBC(measpoints,sigma,t1,t2,phi,delta,a1,a2,a3,b1,b2,b3):
    kx = np.arange(measpoints)*2*np.pi/(measpoints-1) - np.pi
    ky = np.arange(15)*np.pi/(15-1)
    ham = np.zeros((2,2,measpoints),np.complex)
    eigvals = np.zeros((15,2,measpoints))
    
    for y in range(15):
        for i in range(measpoints):
            k = np.array([kx[i],ky[y]])
            ham[:,:,i] = 2*t2*np.cos(phi)*(np.cos(np.dot(k,b1))+np.cos(np.dot(k,b2))+np.cos(np.dot(k,b3)))*sigma[0] + t1*(np.cos(np.dot(k,a1))+np.cos(np.dot(k,a2))+np.cos(np.dot(k,a3)))*sigma[1] + t1*(np.sin(np.dot(k,a1))+np.sin(np.dot(k,a2))+np.sin(np.dot(k,a3)))*sigma[2] + (delta-2*t2*np.sin(phi)*(np.sin(np.dot(k,b1))+np.sin(np.dot(k,b2))+np.sin(np.dot(k,b3))))*sigma[3]
            eigvals[y,:,i] = np.linalg.eigvalsh(ham[:,:,i])
        plt.plot(kx, eigvals[y,0,:], 'k')
        plt.plot(kx, eigvals[y,1,:], 'k')
        plt.xlim(-np.pi,np.pi)
        plt.ylim(-3.5,5)
        plt.ylabel('$E(\\mathbf{k})$', fontsize = 23)
        plt.xlabel('$k_{x}$', fontsize = 24)
        plt.xticks(fontsize = 19)
        plt.yticks(fontsize = 19)
    plt.plot(np.arange(20)-10,np.zeros(20), 'k--')
    plt.xticks([-np.pi,-np.pi/2,0,np.pi/2,np.pi],['$-\pi$','$-\pi/2$','$0$','$\pi/2$','$\pi$'])
    
    return eigvals

def SpectrumPBCOBC(measpoints,sigma,t1,t2,phi,delta,a1,a2,a3,b1,b2,b3,height):
    eigvalsobc = np.zeros((measpoints,height))
    kvector = np.zeros((2,measpoints))
    kvector[0,:] = np.arange(measpoints)/measpoints*2*np.pi/np.sqrt(3)-np.pi/np.sqrt(3)
    for m in range(measpoints):
        k = kvector[:,m]
        oddvec = np.array([ t2*(np.exp(1j*(-phi+np.dot(b3,k))) + np.exp(1j*(phi-np.dot(b2,k)))), t1*np.exp(1j*np.dot(a1,k)), delta + t2*(np.exp(1j*(-phi+np.dot(b1,k))) + np.exp(1j*(phi-np.dot(b1,k)))), t1*(np.exp(1j*np.dot(a2,k)) + np.exp(1j*np.dot(a3,k))), t2*(np.exp(1j*(-phi+np.dot(b2,k))) + np.exp(1j*(phi-np.dot(b3,k)))) ], dtype = np.complex)
        evenvec = np.array([ t2*(np.exp(1j*(phi+np.dot(b3,k))) + np.exp(1j*(-phi-np.dot(b2,k)))), t1*(np.exp(-1j*np.dot(a2,k)) + np.exp(-1j*np.dot(a3,k))), -delta + t2*(np.exp(1j*(phi+np.dot(b1,k))) + np.exp(1j*(-phi-np.dot(b1,k)))), t1*np.exp(-1j*np.dot(a1,k)), t2*(np.exp(1j*(phi+np.dot(b2,k))) + np.exp(1j*(-phi-np.dot(b3,k)))) ], dtype = np.complex)
        hamobc = np.zeros((height,height), dtype = np.complex)
        
        for i in range(height):
            if np.remainder(i+1,2) == 1: # Odd
                hamobc[:,i] = (np.concatenate(( np.concatenate((np.zeros(i), oddvec)), np.zeros(height - np.size(oddvec) + 3 - i) )))[2:height+2]
            if np.remainder(i+1,2) == 0: # Even
                hamobc[:,i] = (np.concatenate(( np.concatenate((np.zeros((i-1)), evenvec)), np.zeros(height - np.size(evenvec) + 3 - (i-1)) )))[1:height+1]
        eigvalsobc[m,:] = np.linalg.eigh(hamobc)[0]    
    
    return eigvalsobc, kvector

def HamiltonianRhombicPBCPBC(t1,t2,phi,delta,L,N):
    H = np.zeros((N,N), dtype = np.complex)
    
    for h in range(np.int(L/2)):
        for i in range(L):
            s = i + h*L
            # Filled dot
            if np.remainder(s,2) == 0:
                # On-site term
                H[s,s] += -delta
                # Nearest-neighbour hopping term
                H[s,h*L+np.mod(i+1,L)] += t1
                H[s,h*L+np.mod(i-1,L)] += t1
                H[s,np.mod(s+L,N)+1] += t1
                # Next-nearest-neighbour hopping term
                H[s,np.mod(s-L,N)] += t2*np.exp(-1j*phi)
                H[s,np.mod(h-1,np.int(L/2))*L+np.mod(i-2,L)] += t2*np.exp(+1j*phi)
                H[s,h*L+np.mod(i-2,L)] += t2*np.exp(-1j*phi)
                H[s,h*L+np.mod(i+2,L)] += t2*np.exp(+1j*phi)
                H[s,np.mod(s+L,N)] += t2*np.exp(+1j*phi)
                H[s,(np.mod(h+1,np.int(L/2)))*L+np.mod(i+2,L)] += t2*np.exp(-1j*phi)
            # Empty dot
            if np.remainder(s,2) == 1:
                # On-site term
                H[s,s] += +delta
                # Nearest-neighbour hopping term
                H[s,h*L+np.mod(i+1,L)] += t1
                H[s,h*L+np.mod(i-1,L)] += t1
                H[s,np.mod(s-L,N)-1] += t1
                # Next-nearest-neighbour hopping term
                H[s,np.mod(s-L,N)] += t2*np.exp(+1j*phi)
                H[s,np.mod(h-1,np.int(L/2))*L+np.mod(i-2,L)] += t2*np.exp(-1j*phi)
                H[s,h*L+np.mod(i-2,L)] += t2*np.exp(+1j*phi)
                H[s,h*L+np.mod(i+2,L)] += t2*np.exp(-1j*phi)
                H[s,np.mod(s+L,N)] += t2*np.exp(-1j*phi)
                H[s,(np.mod(h+1,np.int(L/2)))*L+np.mod(i+2,L)] += t2*np.exp(+1j*phi)

    return H

def HamiltonianRhombicPBCOBC(t1,t2,phi,delta,L,N,flux):
    H = np.zeros((N,N), dtype = np.complex)
    
    for h in range(np.int(L/2)):
        for i in range(L):
            s = i + h*L
            # Filled dot
            if np.remainder(s,2) == 0:
                # On-site term
                H[s,s] += -delta
                # Nearest-neighbour hopping term
                H[s,h*L+np.mod(i+1,L)] += t1
                H[s,h*L+np.mod(i-1,L)] += t1*np.exp(-1j*flux/(L/2*2*np.sin(np.pi/3)))
                if s + L + 1 < N:
                    H[s,s+L+1] += t1
                # Next-nearest-neighbour hopping term
                if s - L >= 0:
                    H[s,s-L] += t2*np.exp(-1j*phi)
                if h > 0:    
                    H[s,(h-1)*L+np.mod(i-2,L)] += t2*np.exp(+1j*phi)*np.exp(-1j*flux/(L/2*2*np.sin(np.pi/3)))
                H[s,h*L+np.mod(i-2,L)] += t2*np.exp(-1j*phi)*np.exp(-1j*flux/(L/2*2*np.sin(np.pi/3)))
                H[s,h*L+np.mod(i+2,L)] += t2*np.exp(+1j*phi)*np.exp(+1j*flux/(L/2*2*np.sin(np.pi/3)))
                if s + L < N:
                    H[s,s+L] += t2*np.exp(+1j*phi)
                if h < np.int(L/2) - 1:
                    H[s,(h+1)*L+np.mod(i+2,L)] += t2*np.exp(-1j*phi)*np.exp(+1j*flux/(L/2*2*np.sin(np.pi/3)))
            # Empty dot
            if np.remainder(s,2) == 1:
                # On-site term
                H[s,s] += +delta
                # Nearest-neighbour hopping term
                H[s,h*L+np.mod(i+1,L)] += t1*np.exp(+1j*flux/(L/2*2*np.sin(np.pi/3)))
                H[s,h*L+np.mod(i-1,L)] += t1
                if s - L - 1 >= 0:
                    H[s,s-L-1] += t1
                # Next-nearest-neighbour hopping term
                if s - L >= 0:
                    H[s,s-L] += t2*np.exp(+1j*phi)
                if h > 0:    
                    H[s,(h-1)*L+np.mod(i-2,L)] += t2*np.exp(-1j*phi)*np.exp(-1j*flux/(L/2*2*np.sin(np.pi/3)))
                H[s,h*L+np.mod(i-2,L)] += t2*np.exp(+1j*phi)*np.exp(-1j*flux/(L/2*2*np.sin(np.pi/3)))
                H[s,h*L+np.mod(i+2,L)] += t2*np.exp(-1j*phi)*np.exp(+1j*flux/(L/2*2*np.sin(np.pi/3)))
                if s + L < N:
                    H[s,s+L] += t2*np.exp(-1j*phi)
                if h < np.int(L/2) - 1:
                    H[s,(h+1)*L+np.mod(i+2,L)] += t2*np.exp(+1j*phi)*np.exp(+1j*flux/(L/2*2*np.sin(np.pi/3)))

    return H

def HoneycombDistancesPBCPBC(L,N):
    xdist = np.zeros(N)
    ydist = np.zeros(N)
    cartdistvectors = np.zeros((N,2))
    dist = np.zeros(N)
    distarray = np.zeros((N,N))
    
    for h in range(np.int(L/2)):
        for i in range(L):
            s = i + h*L
            xdist[s] = (i-h)*np.sin(np.pi/3)
            if np.remainder(i,2) == 0:
                ydist[s] = 2*h*np.sin(np.pi/3)
            if np.remainder(i,2) == 1:
                ydist[s] = 2*h*np.sin(np.pi/3) - np.sin(np.pi/3)
            
            if np.mod(ydist[s] - L*np.sin(np.pi/3)/2, L*np.sin(np.pi/3)) < L*np.sin(np.pi/3)/2 and h > 0:
                xdist[s] = -(L/2*np.sin(np.pi/3) + xdist[s])
            xdist[s] = min(np.abs(xdist[s]), L*np.sin(np.pi/3) - np.abs(xdist[s]))
            cartdistvectors[s,0] = xdist[s]
            
            if np.remainder(i,2) == 1:
                if np.mod(ydist[s] - L*np.sin(np.pi/3)/2, L*np.sin(np.pi/3)) > L*np.sin(np.pi/3)/2:
                    ydist[s] = min(np.abs(ydist[s]), L*np.sin(np.pi/3) - np.abs(ydist[s]))
                    cartdistvectors[s,1] = (ydist[s] + np.sin(np.pi/3))*np.cos(np.pi/6) - np.cos(np.pi/3)
                if np.mod(ydist[s] - L*np.sin(np.pi/3)/2, L*np.sin(np.pi/3)) < L*np.sin(np.pi/3)/2:
                    ydist[s] = min(np.abs(ydist[s]), L*np.sin(np.pi/3) - np.abs(ydist[s]))
                    cartdistvectors[s,1] = (ydist[s] + np.sin(np.pi/3))*np.cos(np.pi/6) - 1
            ydist[s] = min(np.abs(ydist[s]), L*np.sin(np.pi/3) - np.abs(ydist[s]))
            if np.remainder(i,2) == 0:
                cartdistvectors[s,1] = ydist[s]*np.cos(np.pi/6)
            dist[s] = (cartdistvectors[s,0]**2 + cartdistvectors[s,1]**2)**(1/2)
    
    for h in range(np.int(L/2)):
        for i in range(L):
            disttemp = np.zeros((np.int(L/2),L))
            s = i + h*L
            distyrolled = np.roll(dist, h*L)
            for l in range(np.int(L/2)):
                disttemp[l,:] = np.reshape(distyrolled[l*L:(l+1)*L], L)
            disttemp = np.roll(disttemp, i, axis = 1)
            
            disttempoddreverse = np.copy(disttemp)
            if np.mod(i,2) == 1:
                for r in range(np.int(L/2)-1):
                    disttemprow = disttemp[np.mod(h-1-r,np.int(L/2)),:]
                    disttempoddreverse[np.mod(h+1+r,np.int(L/2)),:] = np.roll(disttemprow, 2*(np.mod(1+r,np.int(L/2))))
                disttemp = np.copy(disttempoddreverse)
            distf = np.zeros(N)
            for l in range(np.int(L/2)):
                distf[l*L:(l+1)*L] = disttemp[l,:]
            distarray[s,:] = distf
    
    distarray = np.around(distarray, decimals = 2)
    
    return distarray

def HoneycombDistancesPBCOBC(L,N):
    xdist = np.zeros(N)
    ydist = np.zeros(N)
    cartdistvectors = np.zeros((N,2))
    disteven = np.zeros(N)
    distodd = np.zeros(N)
    distarray = np.zeros((N,N))
    
    for k in range(2):
        for h in range(np.int(L/2)):
            for i in range(L):
                s = i + h*L
                xdist[s] = (i-h)*np.sin(np.pi/3)
                if np.remainder(i,2) == 0:
                    ydist[s] = 2*h*np.sin(np.pi/3)
                if np.remainder(i,2) == 1:
                    ydist[s] = 2*h*np.sin(np.pi/3) + (2*k-1)*np.sin(np.pi/3)
                    
                xdist[s] = min(np.abs(xdist[s]), L*np.sin(np.pi/3) - np.abs(xdist[s]))
                cartdistvectors[s,0] = xdist[s]
                
                if k == 0:
                    if np.remainder(i,2) == 1:
                        if h == 0:
                            cartdistvectors[s,1] = (np.abs(ydist[s]) + np.sin(np.pi/3))*np.cos(np.pi/6) - 1
                        else:
                            cartdistvectors[s,1] = (np.abs(ydist[s]) + np.sin(np.pi/3))*np.cos(np.pi/6) - np.cos(np.pi/3)
                    if np.remainder(i,2) == 0:
                        cartdistvectors[s,1] = ydist[s]*np.cos(np.pi/6)
                if k == 1:
                    if np.remainder(i,2) == 1:
                        cartdistvectors[s,1] = (np.abs(ydist[s]) + np.sin(np.pi/3))*np.cos(np.pi/6) - 1
                    if np.remainder(i,2) == 0:
                        cartdistvectors[s,1] = ydist[s]*np.cos(np.pi/6)                    
        if k == 0:
            disteven = (cartdistvectors[:,0]**2 + cartdistvectors[:,1]**2)**(1/2)
        else:
            distodd = (cartdistvectors[:,0]**2 + cartdistvectors[:,1]**2)**(1/2)
    
    disttemparray = np.zeros((N,N))
    for h in range(np.int(L/2)):
        for i in range(L):
            s = i + h*L
            if np.mod(i,2) == 0:
                for l in range(np.int(L/2)):
                    if l < h:
                        disttemparray[s,l*L:(l+1)*L] = np.roll(distodd[(h-l)*L:(h+1-l)*L],-2*(h-l))
                    else:
                        disttemparray[s,l*L:(l+1)*L] = disteven[(l-h)*L:(l+1-h)*L]  
                for l in range(np.int(L/2)):
                    distarray[s,l*L:(l+1)*L] = np.roll(disttemparray[s,l*L:(l+1)*L],i)
            if np.mod(i,2) == 1:
                for l in range(np.int(L/2)):
                    if l < h:
                        disttemparray[s,l*L:(l+1)*L] = np.roll(disteven[(h-l)*L:(h+1-l)*L],-2*(h-l))
                    else:
                        disttemparray[s,l*L:(l+1)*L] = distodd[(l-h)*L:(l+1-h)*L]  
                for l in range(np.int(L/2)):
                    distarray[s,l*L:(l+1)*L] = np.roll(disttemparray[s,l*L:(l+1)*L],i)
    
    distarray = np.around(distarray, decimals = 2)
    
    return distarray

def HhTMagnitude(distarray,Hhalffilling,hT):
    uniquedists = np.unique(distarray)
    numdists = np.size(uniquedists)
    Hweights = np.zeros(numdists)
    hTweights = np.zeros(numdists)
    
    for i in range(numdists):
        indices = np.argwhere(distarray == uniquedists[i])
        numindices = np.size(indices)/2
        
        for j in range(np.int(numindices)):
            a = indices[j,0]
            b = indices[j,1]
            Hweights[i] += np.abs(Hhalffilling[a,b])/numindices
            hTweights[i] += np.abs(hT[a,b])/numindices
    
    return uniquedists, Hweights, hTweights

def TruncRecObs(numLm,Lminterval,evalsHrhombic,mu,N,distarray,uniquedists,hT,beta,nonzerotemp):
    # Implementing the Truncation-Reconstruction Scheme and Calculating Observables of Interest
    eta = np.zeros((2,numLm))
    Lm = np.zeros(numLm)
    Str = np.zeros(numLm)
    S = np.zeros(numLm)
    if nonzerotemp == 0:
        evalshT = np.sign(evalsHrhombic - mu)
    if nonzerotemp == 1:
        evalshT = np.cosh(beta/2*(evalsHrhombic - mu))/np.sinh(beta/2*(evalsHrhombic - mu))
    evalminhT = np.min(evalshT[evalshT > 0])
    evalmaxhT = np.max(evalshT[evalshT < 0])
    
    hT_truncated = np.zeros((N,N),np.complex)
    
    correlations_tr = np.zeros((numLm,N-1), dtype = np.complex)
    correlations = np.zeros((numLm,N-1), dtype = np.complex)
    R = np.zeros(N-1)
    hT_inv = np.linalg.inv(hT)
    for k in range(numLm):
        print(k)
        hT_truncated_r = np.zeros((N,N),np.complex)
        truncationparameter = np.int(Lminterval*k) + 2 # Number of sites after which matrix element equals zero
        if k == 0:
            prevtruncationparameter = 0
        if k != 0:
            prevtruncationparameter = np.int(Lminterval*(k-1)) + 2
        for i in range(truncationparameter - prevtruncationparameter):
            p = i + prevtruncationparameter
            indices = np.argwhere(distarray == uniquedists[p])
            numindices = np.size(indices)/2
            
            for j in range(np.int(numindices)):
                a = indices[j,0]
                b = indices[j,1]
                hT_truncated[a,b] = hT_truncated[a,b] + hT[a,b]
        alleta = np.linalg.eigh(hT_truncated)[0]
        eta[0,k] = np.min(alleta[alleta>0])
        eta[1,k] = np.max(alleta[alleta>0])
        Lm[k] = uniquedists[truncationparameter-1]
        print(eta[:,k])
        # Reconstruction
        evalminhTtruncated = np.real(np.min(alleta[alleta>0]))
        evalmaxhTtruncated = np.real(np.max(alleta[alleta<0]))
        middlediff = (evalminhTtruncated + evalmaxhTtruncated)/2 - (evalminhT + evalmaxhT)/2
        hT_truncated_r = (1 + 10**(-11))*((evalminhT - evalmaxhT)/2)/((evalminhTtruncated - evalmaxhTtruncated)/2)*(hT_truncated - (middlediff + (((evalminhTtruncated - middlediff) + (evalmaxhTtruncated - middlediff))/2))*np.eye(N)) + (((evalminhTtruncated - middlediff) + (evalmaxhTtruncated - middlediff))/2)*np.eye(N)
        alleta_r = (1 + 10**(-11))*((evalminhT - evalmaxhT)/2)/((evalminhTtruncated - evalmaxhTtruncated)/2)*(alleta - middlediff - (((evalminhTtruncated - middlediff) + (evalmaxhTtruncated - middlediff))/2)) + (((evalminhTtruncated - middlediff) + (evalmaxhTtruncated - middlediff))/2)
        beta_t_r_epsilonk_t_r = 2*np.arctanh(1/alleta_r)
        pk_t_r = 1/(np.exp(beta_t_r_epsilonk_t_r) + 1)
        Str[k] = -np.sum(pk_t_r*np.log(pk_t_r) + (1 - pk_t_r)*np.log(1 - pk_t_r))
        print(Str[k])
        pk = 1/(np.exp(beta*(evalsHrhombic-mu)) + 1)
        S[k] = -np.sum(pk*np.log(pk) + (1 - pk)*np.log(1-pk))
        print(S[k])
        
        correlationstemp_tr = np.zeros(N, dtype = np.complex)
        correlationstemp = np.zeros(N, dtype = np.complex)
        Rtemp = np.zeros(N)
        hT_truncated_r_inv = np.linalg.inv(hT_truncated_r)
        l = 0
        correlationstemp_tr = np.abs((hT_truncated_r_inv)[l,:]/2)
        correlationstemp = np.abs((hT_inv)[l,:]/2)
        Rtemp = distarray[l,:]
        permutation = np.argsort(Rtemp)
        R = (Rtemp[permutation])[1:]
        correlations_tr[k,:] = (correlationstemp_tr[permutation])[1:]
        correlations[k,:] = (np.ravel(correlationstemp)[permutation])[1:]

    return eta, Lm, Str, S, R, correlations_tr, correlations


