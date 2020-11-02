import numpy as np
from numpy import sqrt as Sqrt
from numpy import pi as Pi
from numpy import exp as Exp
from datetime import datetime
from biophysical_gaussian_process.prior_distribution import mean_cov_model as mean_cov_model
##############################################################################
#############################DIVISON, LIKELIHOOD AND POSTERIOR###############
##############################################################################
def division_forward(m,C,sdx2,sdg2):
    """Given p(z_t) find division at t. Return mean and cov"""
    F = np.zeros((4,4))
    np.fill_diagonal(F,[1,1/2,1,1])
    f = np.array([-np.log(2),0,0,0]).reshape(4,1)
    D = np.zeros((4,4))
    np.fill_diagonal(D,[sdx2,sdg2,0,0])
    return F@m+f, D+F@C@F.T
def division_backward(m,C,sdx2,sdg2):
    """Given p(z_t) find division at t. Return mean and cov"""
    F = np.zeros((4,4))
    np.fill_diagonal(F,[1,2,1,1])
    f = np.array([np.log(2),0,0,0]).reshape(4,1)
    D = np.zeros((4,4))
    np.fill_diagonal(D,[sdx2,sdg2,0,0])
    return F@m+f, D+F@C@F.T
def reverse_mean_covariance(m,C):
    """From the backward process reverse mean and covariance as if growth rate and q were positive"""
    minus = lambda x: -x
    m[2,0],m[3,0],C[0,2],C[0,3],C[1,2],C[1,3]=list(map(minus,[m[2,0],m[3,0],C[0,2],C[0,3],C[1,2],C[1,3]]))
    return m,C
def gaussinan_multiplication(m1,C1,m2,C2):
    """Multiply two gaussian distributions"""
    C = np.linalg.inv(np.linalg.inv(C1)+np.linalg.inv(C2))
    m = C@np.linalg.inv(C1)@m1+C@np.linalg.inv(C2)@m2
    return m,C
def posterior(y,m,C,sx2,sg2):
    """Given y_t and the p(z_t) prior, find p(z_t|y_t). Return mean and cov"""
    D = np.zeros((2,2))
    D[0,0]=sx2;D[1,1]=sg2
    Si = np.linalg.inv(C[:2,:2]+D)
    K = C[:2,:]
    y = y-m[:2,:]
    return m+K.T@Si@y, C-K.T@Si@K
def prediction_forward(Y,m,C,ml,gl,sl2,mq,gq,sq2,b,sx2,sg2,sdx2,sdg2):
    """Y=[x,g,time,cell_id] must be connected in genealogy, time ordered and
    maximun 1 division apart i.e. mother daugther relationship """
    mean=[]; covariance = []
    XG = Y[:,:2].astype('float64');T=Y[:,2:3].astype('float64');ID=Y[:,3:4]
    T = T-T[0,0]
#    start = datetime.now()
    for i in range(T.shape[0]):
        #Initial condition
        if i==0:
            nm=m;nC=C
        # Cell division
        if i!=0 and ID[i,0]!=ID[i-1,0]:
#            sta1 = datetime.now()
            nm,nC = division_forward(nm,nC,sdx2,sdg2)
#            print("div",datetime.now()-sta1)
        # Measure
        y = XG[i:i+1,:].T
        # COMPUTE POSTERIOR
#        sta1 = datetime.now()
        nm,nC= posterior(y,nm,nC,sx2,sg2)
        mean.append(nm)
        covariance.append(nC)
#        print('posterior',datetime.now()-sta1)
        # NEXT TIME POINT PRIOR NO DIVISION
        if i<T.shape[0]-1:
            dt = T[i+1,0]-T[i,0]
            #sta1=datetime.now()
            nm,nC = mean_cov_model(nm,nC,dt,ml,gl,sl2,mq,gq,sq2,b)
#            print('update',datetime.now()-sta1)
#    print('total time',datetime.now()-start)
    return mean,covariance
def prediction_backward(Y,m,C,ml,gl,sl2,mq,gq,sq2,b,sx2,sg2,sdx2,sdg2):
    """Y=[x,g,time,cell_id] must be connected in genealogy, time ordered and
    maximun 1 division apart i.e. mother daugther relationship """
    mean=[]; covariance = []
    Y = Y[::-1] # reverse time order
    XG = Y[:,:2].astype('float64');T=Y[:,2:3].astype('float64');ID=Y[:,3:4]
    T = abs(T-T[0,0])
#    start = datetime.now()
    for i in range(T.shape[0]):
        #Initial condition
        if i==0:
            nm=m;nC=C
        # Cell division
        if i!=0 and ID[i,0]!=ID[i-1,0]:
#            sta1 = datetime.now()
            nm,nC = division_backward(nm,nC,sdx2,sdg2)
#            print("div",datetime.now()-sta1)
        # Measure
        y = XG[i:i+1,:].T
        # COMPUTE POSTERIOR
#        sta1 = datetime.now()
        nm,nC= posterior(y,nm,nC,sx2,sg2)
        rnm,rnC = reverse_mean_covariance(nm,nC)
        mean.append(rnm)
        covariance.append(rnC)
#        print('posterior',datetime.now()-sta1)
        # NEXT TIME POINT PRIOR NO DIVISION
        if i<T.shape[0]-1:
            dt = T[i+1,0]-T[i,0]
           # sta1=datetime.now()
            nm,nC = mean_cov_model(nm,nC,dt,-ml,gl,sl2,-mq,gq,sq2,-b)
#            print('update',datetime.now()-sta1)
#    print('total time',datetime.now()-start)
    return mean,covariance
def prediction(Y,m0,C0,mT,CT,ml,gl,sl2,mq,gq,sq2,b,sx2,sg2,sdx2,sdg2):
    """The initial condition for the forward (m0,C0) and backward (mT,CT) algorithm are clearly different"""
    mean=[]; error=[]
    fm,fC = prediction_forward(Y,mT,CT,ml,gl,sl2,mq,gq,sq2,b,sx2,sg2,sdx2,sdg2)
    bm,bC = prediction_backward(Y,m0,C0,ml,gl,sl2,mq,gq,sq2,b,sx2,sg2,sdx2,sdg2)
    for k in range(len(fm)):
        m,C= gaussinan_multiplication(fm[k],fC[k],bm[-1-k],bC[-1-k])
        mean.append(m);error.append(np.sqrt(np.diag(C)))
    return np.hstack(mean),np.vstack(error).T