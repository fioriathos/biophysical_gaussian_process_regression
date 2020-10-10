import numpy as np
import mpmath
from numpy import sqrt as Sqrt
from numpy import pi as Pi
#from numpy import exp as Exp
from mpmath import erfi as Erfi
from mpmath import log as Log
from numpy import exp as Exp
from scipy.integrate import dblquad,quad
#from numpy import log as Log
# m=4 dim initial state vector, C=4x4 initial state matrix
# ml,mq,gl,gq,sl2,sq2 Ou process variables
# z_=(x_t,g_t,l_t,q_t)
def zerotauint__(a,b,c,t):
    ret = (Exp(-b**2/(4.*a) + c)*Sqrt(Pi)*(-Erfi(b/(2.*Sqrt(a))) + Erfi((b + 2*a*t)/(2.*Sqrt(a)))))/(2.*Sqrt(a))
    return ret
def zerotauint_(a,b,c,t):
    return quad(lambda x: np.exp(a*x**2+b*x+c),0,t)[0]
def onetauint__(a,b,c,t):
    return (Exp(-b**2/(4.*a) + c)*(2*Sqrt(a)*Exp(b**2/(4.*a))*(-1 + Exp(t*(b + a*t))) + b*Sqrt(Pi)*Erfi(b/(2.*Sqrt(a))) -\
           b*Sqrt(Pi)*Erfi((b + 2*a*t)/(2.*Sqrt(a)))))/(4.*a**1.5)

def onetauint_(a,b,c,t):
    return quad(lambda x: x*np.exp(a*x**2+b*x+c),0,t)[0]
def twotauint__(a,b,c,t):
    return (Exp(-b**2/(4.*a) + c)*(2*Sqrt(a)*Exp(b**2/(4.*a))*(b - b*Exp(t*(b + a*t)) + 2*a*Exp(t*(b + a*t))*t) +\
           (2*a - b**2)*Sqrt(Pi)*Erfi(b/(2.*Sqrt(a))) + (-2*a + b**2)*Sqrt(Pi)*Erfi((b + 2*a*t)/(2.*Sqrt(a)))))/(8.*a**2.5)
def twotauint_(a,b,c,t):
    return quad(lambda x: x**2*np.exp(a*x**2+b*x+c),0,t)[0]

def mean_cov_model(m,C,t,ml,gl,sl2,mq,gq,sq2,b):
    """Given p(z0)=n(m,C) find p(z1) with no cell division"""
    nC = np.zeros((4,4))
    nm = np.zeros((4,1))
    bt = b*t
    zerotauint=lambda x,y,z: zerotauint_(x,y,z,t)
    onetauint=lambda x,y,z: onetauint_(x,y,z,t)
    twotauint=lambda x,y,z: twotauint_(x,y,z,t)
    eglt = np.exp(-gl*t)
    egqt = np.exp(-gq*t)
    bx=m[0,0]; bg=m[1,0]; bl=m[2,0]; bq=m[3,0]
    Cxx=C[0,0];Cxg=C[0,1];Cxl=C[0,2];Cxq=C[0,3];Cgg=C[1,1];Cgl=C[1,2];Cgq=C[1,3];Cll=C[2,2];Clq=C[2,3];Cqq=C[3,3]
#Mean
    nm[0,0] = bx+ml*t+(bl-ml)*(1-np.exp(-gl*t))/gl
    nm[2,0] = ml+(bl-ml)*np.exp(-gl*t)
    nm[3,0] = mq+(bq-mq)*np.exp(-gq*t)
    #nm[1,0] = bg/Exp(b*t)+Clq*onetauint(Cll/2.,b+bl+Cxl-gq,bx+Cxx/2.-b*t)+mq*zerotauint(Cll/2.,b+bl+Cxl,bx+Cxx/2.-b*t) +\
    #   (bq+Cxq-mq)*zerotauint(Cll/2.,b+bl+Cxl-gq,bx+Cxx/2.-b*t)
    nm[1,0] = bg*Exp(-b*t) +quad(lambda tau:(Exp(bx + Cxx/2. + (b + bl + Cxl - gq)*tau + (Cll*tau**2)/2.)*(bq + Cxq - mq + Exp(gq*tau)*mq + Clq*tau))/Exp(b*t),0,t)[0]
#Cov
    vx = sl2/(2*gl**3)*(2*gl*t-3+4*eglt-eglt**2)
    cxl = sl2/(2*gl**2)*(1-eglt)**2
    cxq = 0; clq = 0
    vl = sl2/(2*gl)*(1-eglt**2)
    vq = sq2/(2*gq)*(1-egqt**2)
    nC[0,0] = Cll*(1-eglt)**2/gl**2+2*Cxl*(1-eglt)/gl+Cxx+vx
    nC[0,2] = cxl+Cll*eglt*(1-eglt)/gl+Cxl*eglt
    nC[0,3] = Clq*(1-eglt)*egqt/gl+Cxq*egqt
    nC[2,2] = Cll*eglt**2 + vl
    nC[2,3] =  Clq*eglt*egqt
    nC[3,3] = vq + Cqq*egqt**2
#    nC[1,2] = (bg*bl)/Exp((b + gl)*t) + Cgl/Exp((b + gl)*t) + (bg*ml)/Exp(b*t) - (bg*ml)/Exp((b + gl)*t) + \
#       Cll*mq*onetauint(Cll/2.,b + bl + Cxl,bx + Cxx/2. - b*t - gl*t) + Clq*ml*onetauint(Cll/2.,b + bl + Cxl - gq,bx + Cxx/2. - b*t) + \
#       (bq*Cll + bl*Clq + Clq*Cxl + Cll*Cxq - Clq*ml - Cll*mq)*onetauint(Cll/2.,b + bl + Cxl - gq,bx + Cxx/2. - b*t - gl*t) + \
#       Cll*Clq*twotauint(Cll/2.,b + bl + Cxl - gq,bx + Cxx/2. - b*t - gl*t) + ml*mq*zerotauint(Cll/2.,b + bl + Cxl,bx + Cxx/2. - b*t) + \
#       (bl*mq + Cxl*mq - ml*mq)*zerotauint(Cll/2.,b + bl + Cxl,bx + Cxx/2. - b*t - gl*t) + \
#       (bq*ml + Cxq*ml - ml*mq)*zerotauint(Cll/2.,b + bl + Cxl - gq,bx + Cxx/2. - b*t) + \
#       (bl*bq + Clq + bq*Cxl + bl*Cxq + Cxl*Cxq - bq*ml - Cxq*ml - bl*mq - Cxl*mq + ml*mq)*\
#        zerotauint(Cll/2.,b + bl + Cxl - gq,bx + Cxx/2. - b*t - gl*t) - nm[1,0]*nm[2,0]
    nC[1,2] =  (bg*bl + Cgl)/Exp((b + gl)*t) + (bg*ml)/Exp(b*t) - (bg*ml)/Exp((b + gl)*t) +\
              quad(lambda tau: Exp(bx + Cxx/2. - b*t - gl*t + (b + bl + Cxl - gq)*tau + (Cll*tau**2)/2.)*\
               ((bq + Cxq + (-1 + Exp(gq*tau))*mq)*(Cxl + (-1 + Exp(gl*t))*ml + Cll*tau) + bl*(bq + Cxq - mq + Exp(gq*tau)*mq + Clq*tau) +\
                Clq*(1 + Cxl*tau + (-1 + Exp(gl*t))*ml*tau + Cll*tau**2)),0,t)[0]-nm[1,0]*nm[2,0]


#    nC[0,1] = (bg*bx)/Exp(b*t) + Cxg/Exp(b*t) + (bg*bl)/(Exp(b*t)*gl) + Cgl/(Exp(b*t)*gl) - (bg*bl)/(Exp((b + gl)*t)*gl) - \
#       Cgl/(Exp((b + gl)*t)*gl) - (bg*ml)/(Exp(b*t)*gl) + (bg*ml)/(Exp((b + gl)*t)*gl) + (bg*ml*t)/Exp(b*t) + \
#       (Cxl*mq + (Cll*mq)/gl)*onetauint(Cll/2.,b + bl + Cxl,bx + Cxx/2. - b*t) - \
#       (Cll*mq*onetauint(Cll/2.,b + bl + Cxl,bx + Cxx/2. - b*t - gl*t))/gl + \
#       (bx*Clq + bq*Cxl + Cxl*Cxq + Clq*Cxx + (bq*Cll)/gl + (bl*Clq)/gl + (Clq*Cxl)/gl + (Cll*Cxq)/gl - (Clq*ml)/gl - Cxl*mq - \
#          (Cll*mq)/gl + Clq*ml*t)*onetauint(Cll/2.,b + bl + Cxl - gq,bx + Cxx/2. - b*t) + \
#       (-((bq*Cll)/gl) - (bl*Clq)/gl - (Clq*Cxl)/gl - (Cll*Cxq)/gl + (Clq*ml)/gl + (Cll*mq)/gl)*\
#        onetauint(Cll/2.,b + bl + Cxl - gq,bx + Cxx/2. - b*t - gl*t) + \
#       (Clq*Cxl + (Cll*Clq)/gl)*twotauint(Cll/2.,b + bl + Cxl - gq,bx + Cxx/2. - b*t) - \
#       (Cll*Clq*twotauint(Cll/2.,b + bl + Cxl - gq,bx + Cxx/2. - b*t - gl*t))/gl + \
#       (bx*mq + Cxx*mq + (bl*mq)/gl + (Cxl*mq)/gl - (ml*mq)/gl + ml*mq*t)*zerotauint(Cll/2.,b + bl + Cxl,bx + Cxx/2. - b*t) + \
#       (-((bl*mq)/gl) - (Cxl*mq)/gl + (ml*mq)/gl)*zerotauint(Cll/2.,b + bl + Cxl,bx + Cxx/2. - b*t - gl*t) + \
#       (bq*bx + Cxq + bx*Cxq + bq*Cxx + Cxq*Cxx + (bl*bq)/gl + Clq/gl + (bq*Cxl)/gl + (bl*Cxq)/gl + (Cxl*Cxq)/gl - (bq*ml)/gl - \
#          (Cxq*ml)/gl - bx*mq - Cxx*mq - (bl*mq)/gl - (Cxl*mq)/gl + (ml*mq)/gl + bq*ml*t + Cxq*ml*t - ml*mq*t)*\
#        zerotauint(Cll/2.,b + bl + Cxl - gq,bx + Cxx/2. - b*t) + \
#       (-((bl*bq)/gl) - Clq/gl - (bq*Cxl)/gl - (bl*Cxq)/gl - (Cxl*Cxq)/gl + (bq*ml)/gl + (Cxq*ml)/gl + (bl*mq)/gl + (Cxl*mq)/gl - \
#          (ml*mq)/gl)*zerotauint(Cll/2.,b + bl + Cxl - gq,bx + Cxx/2. - b*t - gl*t)- nm[1,0]*nm[0,0]
    nC[0,1] = ((bg*bl + Cgl)/Exp(b*t) - (bg*bl + Cgl)/Exp((b + gl)*t) + ((bg*bx + Cxg)*gl)/Exp(b*t) - (bg*ml)/Exp(b*t) + (bg*ml)/Exp((b + gl)*t) + \
         (bg*gl*ml*t)/Exp(b*t) + quad(lambda tau: Exp(bx + Cxx/2. - b*t - gl*t + (b + bl + Cxl - gq)*tau + (Cll*tau**2)/2.)*\
          (-(bq*Cxl) - Cxl*Cxq + bq*Cxl*Exp(gl*t) + Cxl*Cxq*Exp(gl*t) + bq*bx*Exp(gl*t)*gl + Cxq*Exp(gl*t)*gl + bx*Cxq*Exp(gl*t)*gl + \
            bq*Cxx*Exp(gl*t)*gl + Cxq*Cxx*Exp(gl*t)*gl + bq*ml + Cxq*ml - bq*Exp(gl*t)*ml - Cxq*Exp(gl*t)*ml + Cxl*mq - Cxl*Exp(gl*t)*mq - \
            Cxl*Exp(gq*tau)*mq + Cxl*Exp(gl*t + gq*tau)*mq - bx*Exp(gl*t)*gl*mq - Cxx*Exp(gl*t)*gl*mq + bx*Exp(gl*t + gq*tau)*gl*mq + \
            Cxx*Exp(gl*t + gq*tau)*gl*mq - ml*mq + Exp(gl*t)*ml*mq + Exp(gq*tau)*ml*mq - Exp(gl*t + gq*tau)*ml*mq + bq*Exp(gl*t)*gl*ml*t + \
            Cxq*Exp(gl*t)*gl*ml*t - Exp(gl*t)*gl*ml*mq*t + Exp(gl*t + gq*tau)*gl*ml*mq*t - bq*Cll*tau - Cll*Cxq*tau + \
            bq*Cll*Exp(gl*t)*tau + Cll*Cxq*Exp(gl*t)*tau + bq*Cxl*Exp(gl*t)*gl*tau + Cxl*Cxq*Exp(gl*t)*gl*tau + Cll*mq*tau - \
            Cll*Exp(gl*t)*mq*tau - Cll*Exp(gq*tau)*mq*tau + Cll*Exp(gl*t + gq*tau)*mq*tau - Cxl*Exp(gl*t)*gl*mq*tau + \
            Cxl*Exp(gl*t + gq*tau)*gl*mq*tau + bl*(-1 + Exp(gl*t))*(bq + Cxq - mq + Exp(gq*tau)*mq + Clq*tau) + \
            Clq*(-1 - Cxl*tau + ml*tau - Cll*tau**2 + Exp(gl*t)*\
                (1 + bx*gl*tau + Cxx*gl*tau - ml*tau + gl*ml*t*tau + Cll*tau**2 + Cxl*tau*(1 + gl*tau)))),0,t)[0])/gl-nm[1,0]*nm[0,0]

#    nC[1,3] = (bg*bq)/Exp((b + gq)*t) + Cgq/Exp((b + gq)*t) + (bg*mq)/Exp(b*t) - (bg*mq)/Exp((b + gq)*t) + \
#       Clq*mq*onetauint(Cll/2.,b + bl + Cxl,bx + Cxx/2. - b*t - gq*t) + Clq*mq*onetauint(Cll/2.,b + bl + Cxl - gq,bx + Cxx/2. - b*t) + \
#       (2*bq*Clq + 2*Clq*Cxq - 2*Clq*mq)*onetauint(Cll/2.,b + bl + Cxl - gq,bx + Cxx/2. - b*t - gq*t) + \
#       Clq**2*twotauint(Cll/2.,b + bl + Cxl - gq,bx + Cxx/2. - b*t - gq*t) + mq**2*zerotauint(Cll/2.,b + bl + Cxl,bx + Cxx/2. - b*t) + \
#       (bq*mq + Cxq*mq - mq**2)*zerotauint(Cll/2.,b + bl + Cxl,bx + Cxx/2. - b*t - gq*t) + \
#       (bq*mq + Cxq*mq - mq**2)*zerotauint(Cll/2.,b + bl + Cxl - gq,bx + Cxx/2. - b*t) - \
#       (sq2*zerotauint(Cll/2.,b + bl + Cxl - gq,-bt + bx + Cxx/2. - gq*t))/(2.*gq) + \
#       (bq**2 + Cqq + 2*bq*Cxq + Cxq**2 - 2*bq*mq - 2*Cxq*mq + mq**2)*zerotauint(Cll/2.,b + bl + Cxl - gq,bx + Cxx/2. - b*t - gq*t) + \
#       (sq2*zerotauint(Cll/2.,b + bl + Cxl + gq,-bt + bx + Cxx/2. - gq*t))/(2.*gq)- nm[1,0]*nm[3,0]
    nC[1,3] = quad(lambda tau:(Exp(bx - b*t + b*tau + bl*tau + (Cxx + tau*(2*Cxl + Cll*tau))/2.)*(Exp(gq*(-t + tau)) - Exp(-(gq*(t + tau))))*sq2)/(2.*gq),0,t)[0]+\
              (bg*bq + Cgq)/Exp((b + gq)*t) + (bg*mq)/Exp(b*t) - (bg*mq)/Exp((b + gq)*t) + quad(lambda tau:\
               Exp(bx + Cxx/2. - b*t - gq*t + b*tau + bl*tau + Cxl*tau - gq*tau + (Cll*tau**2)/2.)*\
              (bq**2 + Cqq + (Cxq + (-1 + Exp(gq*t))*mq + Clq*tau)*(Cxq + (-1 + Exp(gq*tau))*mq + Clq*tau) +\
               bq*(2*Cxq + (-2 + Exp(gq*t) + Exp(gq*tau))*mq + 2*Clq*tau)),0,t)[0]-nm[1,0]*nm[3,0]
    def vargt_z0(t1,t2):
        return sq2/(2*gq)*(np.exp(-np.abs(t1-t2)*gq)-np.exp(-gq*(t1+t2)))*Exp(2*bx+2*Cxx-2*b*t+(b+bl+2*Cxl)*(t1+t2)+(Cll*(t1+t2)**2)/2.)
    vgtz0 = dblquad(vargt_z0, 0, t, lambda x: 0, lambda x: t)[0]

    def mean_gtz0_part1(tau1):
        return  Exp(bx + (Cxx + tau1*(2*bl + 2*Cxl + Cll*tau1))/2.)*\
        (Exp(b*(-2*t + tau1))*mq*(bg + Cxg + Cgl*tau1) - Exp(-2*b*t + b*tau1 - gq*tau1)*mq*(bg + Cxg + Cgl*tau1) + \
          Exp(-2*b*t + b*tau1 - gq*tau1)*(Cgq + bg*(bq + Cxq + Clq*tau1) + (Cxg + Cgl*tau1)*(bq + Cxq + Clq*tau1)))
    def mean_gtz0_part2(tau1,tau2):
        return Exp(2*bx + 2*Cxx - gq*(tau1 + tau2) + b*(-2*t + tau1 + tau2) + ((tau1 + tau2)*(2*bl + 4*Cxl + Cll*(tau1 + tau2)))/2.)*\
        (bq**2 + Cqq + (2*Cxq + (-1 + Exp(gq*tau1))*mq + Clq*(tau1 + tau2))*(2*Cxq + (-1 + Exp(gq*tau2))*mq + Clq*(tau1 + tau2)) + \
          bq*(4*Cxq + (-2 + Exp(gq*tau1) + Exp(gq*tau2))*mq + 2*Clq*(tau1 + tau2)))
    m2gtz0 = (Cgg+bg**2)/Exp(2*b*t) +2*quad(mean_gtz0_part1, 0, t)[0] +\
              dblquad(mean_gtz0_part2, 0, t, lambda x: 0, lambda x: t)[0]
    nC[1,1] = vgtz0 +m2gtz0- nm[1,0]**2 
    #print('vg',vgtz0)
    #print('m2g',m2gtz0)
    #print('n2',nm[1,0]**2)
    nC[1,0]=nC[0,1];nC[2,0]=nC[0,2];nC[2,1]=nC[1,2];nC[3,0]=nC[0,3];nC[3,1]=nC[1,3];nC[3,2]=nC[2,3]
    return nm,nC
def division(m,C,sdx2,sdg2):
    """Given p(z_t) find division at t. Return mean and cov"""
    F = np.zeros((4,4))
    np.fill_diagonal(F,[1,1/2,1,1])
    f = np.array([-np.log(2),0,0,0]).reshape(4,1)
    D = np.zeros((4,4))
    np.fill_diagonal(D,[sdx2,sdg2,0,0])
    return F@m+f, D+F@C@F.T
def log_likelihood(y,m,C,sx2,sg2):
    """Given y_t=(x_t,g_t), p(z_t) and errors find p(y_t). Return log lik """
    D = np.zeros((2,2))
    D[0,0]=sx2;D[1,1]=sg2
    M = np.linalg.inv(C[:2,:2]+D)
    y = y-m[:2,:]
    return -1/2*y.T@M@y-1/2*np.log((2*Pi)**4*np.linalg.det(M))
def posterior(y,m,C,sx2,sg2):
    """Given y_t and the p(z_t) prior, find p(z_t|y_t). Return mean and cov"""
    D = np.zeros((2,2))
    D[0,0]=sx2;D[1,1]=sg2
    M = np.linalg.inv(C[:2,:2]+D)
    K = C[:2,:]
    return m+K.T@M@(y-m[:2,:]), C-K.T@M@K




############################################################################
############################################################################
########################Multi measures cov##################################
############################################################################
############################################################################
def K0_K(Y,m,C,ml,gl,sl2,mq,gq,sq2,b,Dx=None,Dg=None):
    """Y=[x,g,t] vector in once cell"""


















