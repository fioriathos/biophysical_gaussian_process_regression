import numpy as np
from datetime import datetime
from multiprocessing import  Pool, Process
from pathos.multiprocessing import Pool as  path_pool
from numba import jit
import mpmath
from numpy import sqrt as Sqrt
from numpy import pi as Pi
from numpy import exp as Exp
from scipy.integrate import dblquad,quad
def mean_x(t,bx,bg,bl,bq,Cxx,Cxg,Cxl,Cxq,Cgg,Cgl,Cgq,Cll,Clq,Cqq,ml,gl,sl2,mq,gq,sq2,b):
    return bx+ml*t+(bl-ml)*(1-np.exp(-gl*t))/gl
def d_mean_x(t,bx,bg,bl,bq,Cxx,Cxg,Cxl,Cxq,Cgg,Cgl,Cgq,Cll,Clq,Cqq,ml,gl,sl2,mq,gq,sq2,b,key):
    """key identify which gradient component to compute"""
    db_=None; dsdg2_=None; dsdx2_=None; dsg2_=None; dsx2_=None;
    dsq2_=None; dgq_=None; dmq_=None; dsl2_=None; dml_=None; dgl_=None;
    if key=='ml':
        dml_ = t-(1-np.exp(-gl*t))/gl
    if key=='gl':
        dgl_ = (bl-ml)*(t*Exp(-gl*t)*gl-(1-np.exp(-gl*t)))/gl**2
    if key=='sl2':
        dsl2_ = 0
    if key=='mq':
        dmq_ = 0
    if key=='gq':
        dgq_ = 0
    if key=='sq2':
        dsq2_ = 0
    if key=='sx2':
        dsx2_ = 0
    if key == 'sg2':
        dsg2_ = 0
    if key=='sdx2':
        dsdx2_ = 0
    if key == 'sdg2':
        dsdg2_ = 0
    if key=='b':
        db_ = 0
    return {'ml':dml_,'gl':dgl_,'sl2':dsl2_,'mq':dmq_,'gq':dgq_,'sq2':dsq2_\
		,'sx2':dsx2_,'sg2':dsg2_,'b':db_,'sdx2':dsdx2_,'sdg2':dsdg2_}
def mean_g(t,bx,bg,bl,bq,Cxx,Cxg,Cxl,Cxq,Cgg,Cgl,Cgq,Cll,Clq,Cqq,ml,gl,sl2,mq,gq,sq2,b):
    return bg*Exp(-b*t) +quad(lambda tau:(Exp(bx + Cxx/2. + (b + bl + Cxl - gq)*tau + (Cll*tau**2)/2.)*(bq + Cxq - mq + Exp(gq*tau)*mq + Clq*tau))/Exp(b*t),0,t)[0]
def d_mean_g(t,bx,bg,bl,bq,Cxx,Cxg,Cxl,Cxq,Cgg,Cgl,Cgq,Cll,Clq,Cqq,ml,gl,sl2,mq,gq,sq2,b,key):
    db_=None; dsdg2_=None; dsdx2_=None; dsg2_=None; dsx2_=None;
    dsq2_=None; dgq_=None; dmq_=None; dsl2_=None; dml_=None; dgl_=None;
    if key=='ml':
        dml_ = 0
    if key=='gl':
        dgl_ = 0
    if key=='sl2':
        dsl2_ = 0
    if key=='mq':
        dmq_ = quad(lambda tau:Exp(bx + Cxx/2. - b*t + (b + bl + Cxl - gq)*tau + (Cll*tau**2)/2.)*(-1 + Exp(gq*tau)) ,0,t)[0]
    if key=='gq':
        dgq_ = quad(lambda tau:-(Exp(bx + Cxx/2. - b*t + (b + bl + Cxl - gq)*tau + (Cll*tau**2)/2.)*tau*(bq + Cxq - mq + Clq*tau)) ,0,t)[0]
    if key=='sq2':
        dsq2_ = 0
    if key=='sdx2':
        dsdx2_ = 0
    if key == 'sdg2':
        dsdg2_ = 0
    if key=='sx2':
        dsx2_ = 0
    if key == 'sg2':
        dsg2_ = 0
    if key=='b':
        db_ =  -((bg*t)/Exp(b*t)) +quad(lambda tau: Exp(bx + Cxx/2. + (b + bl + Cxl - gq)*tau + (Cll*tau**2)/2.)*\
            (-((bq*t)/Exp(b*t)) - (Cxq*t)/Exp(b*t) + (mq*t)/Exp(b*t) - Exp(-(b*t) + gq*tau)*mq*t +\
            (bq/Exp(b*t) + Cxq/Exp(b*t) - mq/Exp(b*t) + Exp(-(b*t) + gq*tau)*mq - (Clq*t)/Exp(b*t))*tau+(Clq*tau**2)/Exp(b*t)),0,t)[0]
    return {'ml':dml_,'gl':dgl_,'sl2':dsl2_,'mq':dmq_,'gq':dgq_,'sq2':dsq2_\
		,'sx2':dsx2_,'sg2':dsg2_,'b':db_,'sdx2':dsdx2_,'sdg2':dsdg2_}
def mean_l(t,bx,bg,bl,bq,Cxx,Cxg,Cxl,Cxq,Cgg,Cgl,Cgq,Cll,Clq,Cqq,ml,gl,sl2,mq,gq,sq2,b):
    return ml+(bl-ml)*np.exp(-gl*t)
def mean_q(t,bx,bg,bl,bq,Cxx,Cxg,Cxl,Cxq,Cgg,Cgl,Cgq,Cll,Clq,Cqq,ml,gl,sl2,mq,gq,sq2,b):
    return mq+(bq-mq)*np.exp(-gq*t)
def cov_xx(t,bx,bg,bl,bq,Cxx,Cxg,Cxl,Cxq,Cgg,Cgl,Cgq,Cll,Clq,Cqq,ml,gl,sl2,mq,gq,sq2,b):
    vx = sl2/(2*gl**3)*(2*gl*t-3+4*Exp(-gl*t)-Exp(-gl*t)**2)
    return Cll*(1-Exp(-gl*t))**2/gl**2+2*Cxl*(1-Exp(-gl*t))/gl+Cxx+vx
def d_cov_xx(t,bx,bg,bl,bq,Cxx,Cxg,Cxl,Cxq,Cgg,Cgl,Cgq,Cll,Clq,Cqq,ml,gl,sl2,mq,gq,sq2,b,key):
    db_=None; dsdg2_=None; dsdx2_=None; dsg2_=None; dsx2_=None;
    dsq2_=None; dgq_=None; dmq_=None; dsl2_=None; dml_=None; dgl_=None;
    if key=='ml':
        dml_ = 0
    if key=='gl':
        dgl_ =2*Exp(-gl*t)*t*Cll*(1-Exp(-gl*t))/gl**2-2*Cll*(1-Exp(-gl*t))**2/gl**3+\
                2*Cxl*(t*Exp(-gl*t))/gl-2*Cxl*(1-Exp(-gl*t))/gl**2+\
                -3*sl2/(2*gl**4)*(2*gl*t-3+4*Exp(-gl*t)-Exp(-gl*t)**2)+\
                sl2/(2*gl**3)*(2*t-t*4*Exp(-gl*t)+2*t*Exp(-gl*t)**2)
    if key=='sl2':
        dsl2_= 1/(2*gl**3)*(2*gl*t-3+4*Exp(-gl*t)-Exp(-gl*t)**2)
    if key=='mq':
        dmq_ = 0
    if key=='gq':
        dgq_ = 0
    if key=='sq2':
        dsq2_ = 0
    if key=='sx2':
        dsx2_ = 0
    if key=='sdx2':
        dsdx2_ = 0
    if key == 'sdg2':
        dsdg2_ = 0
    if key == 'sg2':
        dsg2_ = 0
    if key == 'b':
        db_ = 0
    return {'ml':dml_,'gl':dgl_,'sl2':dsl2_,'mq':dmq_,'gq':dgq_,'sq2':dsq2_\
		,'sx2':dsx2_,'sg2':dsg2_,'b':db_,'sdx2':dsdx2_,'sdg2':dsdg2_}
def cov_xg(t,bx,bg,bl,bq,Cxx,Cxg,Cxl,Cxq,Cgg,Cgl,Cgq,Cll,Clq,Cqq,ml,gl,sl2,mq,gq,sq2,b,nm):
    return ((bg*bl + Cgl)/Exp(b*t) - (bg*bl + Cgl)/Exp((b + gl)*t) + ((bg*bx + Cxg)*gl)/Exp(b*t) - (bg*ml)/Exp(b*t) + (bg*ml)/Exp((b + gl)*t) + \
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
def d_cov_xg(t,bx,bg,bl,bq,Cxx,Cxg,Cxl,Cxq,Cgg,Cgl,Cgq,Cll,Clq,Cqq,ml,gl,sl2,mq,gq,sq2,b,nm,dnm_x,dnm_g,key):
    db_=None; dsdg2_=None; dsdx2_=None; dsg2_=None; dsx2_=None;
    dsq2_=None; dgq_=None; dmq_=None; dsl2_=None; dml_=None; dgl_=None;
    if key=='ml':
        dml_ =(-(bg/Exp(b*t)) + bg/Exp((b + gl)*t) + (bg*gl*t)/Exp(b*t) +\
            quad(lambda tau:Exp(bx + Cxx/2. - b*t - gl*t + (b + bl + Cxl - gq)*tau + (Cll*tau**2)/2.)*(1 + Exp(gl*t)*(-1 + gl*t))*\
            (bq + Cxq - mq + Exp(gq*tau)*mq + Clq*tau),0,t)[0])/gl - nm[1,0]*dnm_x['ml']- nm[0,0]*dnm_g['ml']
    if key=='gl':
        dgl_ = (Cgl*Exp((b + gl)*t)*(1 - Exp(gl*t) + gl*t) + bg*Exp((b + gl)*t)*(bl - ml)*(1 - Exp(gl*t) + gl*t) -\
            quad(lambda tau: Exp(bx + Cxx/2. + gl*t + bl*tau + Cxl*tau - gq*tau + (Cll*tau**2)/2. + b*(t + tau))*(-1 + Exp(gl*t) - gl*t)*\
            ((bq + Cxq + (-1 + Exp(gq*tau))*mq)*(Cxl - ml + Cll*tau) + bl*(bq + Cxq - mq + Exp(gq*tau)*mq + Clq*tau) +\
                Clq*(1 + Cxl*tau - ml*tau + Cll*tau**2)),0,t)[0])/(Exp(2*(b + gl)*t)*gl**2)- nm[1,0]*dnm_x['gl']- nm[0,0]*dnm_g['gl']
    if key=='sl2':
        dsl2_ =0
    if key=='mq':
        dmq_ =quad(lambda tau: (Exp(bx + Cxx/2. - b*t - gl*t + (b + bl + Cxl - gq)*tau + (Cll*tau**2)/2.)*(-1 + Exp(gq*tau))*\
            (bl*(-1 + Exp(gl*t)) + bx*Exp(gl*t)*gl + Cxx*Exp(gl*t)*gl + ml - Exp(gl*t)*ml + Exp(gl*t)*gl*ml*t - Cll*tau + Cll*Exp(gl*t)*tau +\
            Cxl*(-1 + Exp(gl*t)*(1 + gl*tau))))/gl,0,t)[0] - nm[1,0]*dnm_x['mq']- nm[0,0]*dnm_g['mq']
    if key=='gq':
        dgq_ = quad(lambda tau: -((Exp(bx + Cxx/2. - b*t - gl*t + (b + bl + Cxl - gq)*tau + (Cll*tau**2)/2.)*tau*\
            (-(bq*Cxl) - Cxl*Cxq + bq*Cxl*Exp(gl*t) + Cxl*Cxq*Exp(gl*t) + bq*bx*Exp(gl*t)*gl + Cxq*Exp(gl*t)*gl + bx*Cxq*Exp(gl*t)*gl + \
                bq*Cxx*Exp(gl*t)*gl + Cxq*Cxx*Exp(gl*t)*gl + bq*ml + Cxq*ml - bq*Exp(gl*t)*ml - Cxq*Exp(gl*t)*ml + Cxl*mq - Cxl*Exp(gl*t)*mq - \
                bx*Exp(gl*t)*gl*mq - Cxx*Exp(gl*t)*gl*mq - ml*mq + Exp(gl*t)*ml*mq + bq*Exp(gl*t)*gl*ml*t + Cxq*Exp(gl*t)*gl*ml*t - \
                Exp(gl*t)*gl*ml*mq*t - bq*Cll*tau - Cll*Cxq*tau + bq*Cll*Exp(gl*t)*tau + Cll*Cxq*Exp(gl*t)*tau + bq*Cxl*Exp(gl*t)*gl*tau + \
                Cxl*Cxq*Exp(gl*t)*gl*tau + Cll*mq*tau - Cll*Exp(gl*t)*mq*tau - Cxl*Exp(gl*t)*gl*mq*tau + \
                bl*(-1 + Exp(gl*t))*(bq + Cxq - mq + Clq*tau) + \
                Clq*(-1 - Cxl*tau + ml*tau - Cll*tau**2 + Exp(gl*t)*\
                (1 + bx*gl*tau + Cxx*gl*tau - ml*tau + gl*ml*t*tau + Cll*tau**2 + Cxl*tau*(1 + gl*tau)))))/gl),0,t)[0]- nm[1,0]*dnm_x['gq']- nm[0,0]*dnm_g['gq']
    if key=='sq2':
        dsq2_ = 0
    if key=='sx2':
        dsx2_ = 0
    if key=='sg2':
        dsg2_ = 0
    if key=='sdx2':
        dsdx2_ = 0
    if key=='sdg2':
        dsdg2_ = 0
    if key=='b':
        db_ =  (-(((bg*bl + Cgl)*t)/Exp(b*t)) + ((bg*bl + Cgl)*t)/Exp((b + gl)*t) - ((bg*bx + Cxg)*gl*t)/Exp(b*t) + (bg*ml*t)/Exp(b*t) - \
             (bg*ml*t)/Exp((b + gl)*t) - (bg*gl*ml*t**2)/Exp(b*t) + \
              quad(lambda tau: Exp(bx + Cxx/2. - b*t - gl*t + (b + bl + Cxl - gq)*tau + (Cll*tau**2)/2.)*(-t + tau)*\
         (-(bq*Cxl) - Cxl*Cxq + bq*Cxl*Exp(gl*t) + Cxl*Cxq*Exp(gl*t) + bq*bx*Exp(gl*t)*gl + Cxq*Exp(gl*t)*gl + bx*Cxq*Exp(gl*t)*gl + \
         bq*Cxx*Exp(gl*t)*gl + Cxq*Cxx*Exp(gl*t)*gl + bq*ml + Cxq*ml - bq*Exp(gl*t)*ml - Cxq*Exp(gl*t)*ml + Cxl*mq - Cxl*Exp(gl*t)*mq - \
         Cxl*Exp(gq*tau)*mq + Cxl*Exp(gl*t + gq*tau)*mq - bx*Exp(gl*t)*gl*mq - Cxx*Exp(gl*t)*gl*mq + bx*Exp(gl*t + gq*tau)*gl*mq + \
         Cxx*Exp(gl*t + gq*tau)*gl*mq - ml*mq + Exp(gl*t)*ml*mq + Exp(gq*tau)*ml*mq - Exp(gl*t + gq*tau)*ml*mq + bq*Exp(gl*t)*gl*ml*t + \
         Cxq*Exp(gl*t)*gl*ml*t - Exp(gl*t)*gl*ml*mq*t + Exp(gl*t + gq*tau)*gl*ml*mq*t - bq*Cll*tau - Cll*Cxq*tau + bq*Cll*Exp(gl*t)*tau + \
         Cll*Cxq*Exp(gl*t)*tau + bq*Cxl*Exp(gl*t)*gl*tau + Cxl*Cxq*Exp(gl*t)*gl*tau + Cll*mq*tau - Cll*Exp(gl*t)*mq*tau - \
         Cll*Exp(gq*tau)*mq*tau + Cll*Exp(gl*t + gq*tau)*mq*tau - Cxl*Exp(gl*t)*gl*mq*tau + Cxl*Exp(gl*t + gq*tau)*gl*mq*tau + \
         bl*(-1 + Exp(gl*t))*(bq + Cxq - mq + Exp(gq*tau)*mq + Clq*tau) + \
         Clq*(-1 - Cxl*tau + ml*tau - Cll*tau**2 + Exp(gl*t)*\
         (1 + bx*gl*tau + Cxx*gl*tau - ml*tau + gl*ml*t*tau + Cll*tau**2 + Cxl*tau*(1 + gl*tau)))),0,t)[0])/gl
    return {'ml':dml_,'gl':dgl_,'sl2':dsl2_,'mq':dmq_,'gq':dgq_,'sq2':dsq2_\
		,'sx2':dsx2_,'sg2':dsg2_,'b':db_,'sdx2':dsdx2_,'sdg2':dsdg2_}
def cov_xl(t,bx,bg,bl,bq,Cxx,Cxg,Cxl,Cxq,Cgg,Cgl,Cgq,Cll,Clq,Cqq,ml,gl,sl2,mq,gq,sq2,b):
    cxl = sl2/(2*gl**2)*(1-Exp(-gl*t))**2
    return cxl+Cll*Exp(-gl*t)*(1-Exp(-gl*t))/gl+Cxl*Exp(-gl*t)
def cov_xq(t,bx,bg,bl,bq,Cxx,Cxg,Cxl,Cxq,Cgg,Cgl,Cgq,Cll,Clq,Cqq,ml,gl,sl2,mq,gq,sq2,b):
    return Clq*(1-Exp(-gl*t))*Exp(-gq*t)/gl+Cxq*Exp(-gq*t)
def cov_gg(t,bx,bg,bl,bq,Cxx,Cxg,Cxl,Cxq,Cgg,Cgl,Cgq,Cll,Clq,Cqq,ml,gl,sl2,mq,gq,sq2,b,nm):
    def covqq(t1,t2):
        return sq2/(2*gq)*(np.exp(-np.abs(t1-t2)*gq)-np.exp(-gq*(t1+t2)))
    def vargt_z0(t1,t2):
        return covqq(t1,t2)*Exp(2*bx+2*Cxx-2*b*t+(b+bl+2*Cxl)*(t1+t2)+(Cll*(t1+t2)**2)/2.)
    def mean_gtz0_part1(tau1):
        return  Exp(bx + (Cxx + tau1*(2*bl + 2*Cxl + Cll*tau1))/2.)*\
        (Exp(b*(-2*t + tau1))*mq*(bg + Cxg + Cgl*tau1) - Exp(-2*b*t + b*tau1 - gq*tau1)*mq*(bg + Cxg + Cgl*tau1) + \
          Exp(-2*b*t + b*tau1 - gq*tau1)*(Cgq + bg*(bq + Cxq + Clq*tau1) + (Cxg + Cgl*tau1)*(bq + Cxq + Clq*tau1)))
    def mean_gtz0_part2(tau1,tau2):
        return Exp(2*bx+2*Cxx-gq*(tau1+tau2)+b*(-2*t + tau1 + tau2) + ((tau1 + tau2)*(2*bl + 4*Cxl + Cll*(tau1 + tau2)))/2.)*\
        (bq**2 + Cqq + (2*Cxq + (-1 + Exp(gq*tau1))*mq + Clq*(tau1 + tau2))*(2*Cxq + (-1 + Exp(gq*tau2))*mq + Clq*(tau1 + tau2)) + \
          bq*(4*Cxq + (-2 + Exp(gq*tau1) + Exp(gq*tau2))*mq + 2*Clq*(tau1 + tau2)))
    vgtz0 = dblquad(vargt_z0, 0, t, lambda x: 0, lambda x: t)[0]
    m2gtz0 = (Cgg+bg**2)/Exp(2*b*t) +2*quad(mean_gtz0_part1, 0, t)[0] +\
              dblquad(mean_gtz0_part2, 0, t, lambda x: 0, lambda x: t)[0]
    return vgtz0 +m2gtz0- nm[1,0]**2
def d_cov_gg(t,bx,bg,bl,bq,Cxx,Cxg,Cxl,Cxq,Cgg,Cgl,Cgq,Cll,Clq,Cqq,ml,gl,sl2,mq,gq,sq2,b,nm,dnm_g,key):
    db_=None; dsdg2_=None; dsdx2_=None; dsg2_=None; dsx2_=None;
    dsq2_=None; dgq_=None; dmq_=None; dsl2_=None; dml_=None; dgl_=None;
    if key=='ml':
        dml_ = 0
    if key=='gl':
        dgl_ = 0
    if key=='sl2':
        dsl2_ = 0
    if key=='mq':
        dmqpart1 = quad(lambda tau1: Exp(bx + Cxx/2. - 2*b*t - gq*tau1)*(Exp((b + bl + Cxl)*tau1 + (Cll*tau1**2)/2.)*(-bg - Cxg - Cgl*tau1) +\
                        Exp((b + bl + Cxl + gq)*tau1 + (Cll*tau1**2)/2.)*(bg + Cxg + Cgl*tau1)),0,t)[0]
        dmqpart2 = dblquad(lambda tau1,tau2:         Exp(2*(bx + Cxx) - 2*b*t + (b + bl + 2*Cxl - gq)*(tau1 + tau2) + (Cll*(tau1 + tau2)**2)/2.)*\
                    (bq*(-2 + Exp(gq*tau1) + Exp(gq*tau2)) + (-1 + Exp(gq*tau2))*(2*Cxq + (-1 + Exp(gq*tau1))*mq + Clq*(tau1 + tau2)) +\
                    (-1 + Exp(gq*tau1))*(2*Cxq + (-1 + Exp(gq*tau2))*mq + Clq*(tau1 + tau2))) ,0,t,lambda x:0,lambda x:t)[0]
        m2gtz0 = 2*dmqpart1 + dmqpart2
        dmq_ = m2gtz0- 2*nm[1,0]*dnm_g['mq']
    if key=='gq':
        dgqvgtz0 =dblquad(lambda t1,t2: (sq2*(-1 - np.abs(t1-t2)*gq + Exp(gq*(np.abs(t1-t2)- t1 - t2))*(1 + gq*(t1 + t2))))\
                    /(2.*Exp(np.abs(t1-t2)*gq)*gq**2)*Exp(2*bx+2*Cxx-2*b*t+(b+bl+2*Cxl)*(t1+t2)+(Cll*(t1+t2)**2)/2.),0,t,lambda x:0,lambda x:t)[0]
        dgqpart1= quad(lambda tau1:        Exp(bx + Cxx/2. - 2*b*t - gq*tau1 + (b + bl + Cxl + gq)*tau1 + (Cll*tau1**2)/2.)*mq*tau1*(bg + Cxg + Cgl*tau1) -\
                    Exp(bx + Cxx/2. - 2*b*t - gq*tau1)*tau1*(Exp((b + bl + Cxl + gq)*tau1 + (Cll*tau1**2)/2.)*mq*(bg + Cxg + Cgl*tau1) +\
                    Exp((b + bl + Cxl)*tau1 + (Cll*tau1**2)/2.)*(Cgq + bg*(bq + Cxq - mq + Clq*tau1) + (Cxg + Cgl*tau1)*(bq + Cxq - mq + Clq*tau1))),0,t)[0]
        dgqpart2 = dblquad(lambda tau1,tau2:         Exp(2*(bx + Cxx) - 2*b*t + (b + bl + 2*Cxl - gq)*(tau1 + tau2) + (Cll*(tau1 + tau2)**2)/2.)*\
            (bq*mq*(Exp(gq*tau1)*tau1 + Exp(gq*tau2)*tau2) + Exp(gq*tau2)*mq*tau2*(2*Cxq + (-1 + Exp(gq*tau1))*mq + Clq*(tau1 + tau2)) + \
            Exp(gq*tau1)*mq*tau1*(2*Cxq + (-1 + Exp(gq*tau2))*mq + Clq*(tau1 + tau2))) - \
            Exp(2*(bx + Cxx) - 2*b*t + (b + bl + 2*Cxl - gq)*(tau1 + tau2) + (Cll*(tau1 + tau2)**2)/2.)*(tau1 + tau2)*\
            (bq**2 + Cqq + (2*Cxq + (-1 + Exp(gq*tau1))*mq + Clq*(tau1 + tau2))*(2*Cxq + (-1 + Exp(gq*tau2))*mq + Clq*(tau1 + tau2)) + \
            bq*(4*Cxq + (-2 + Exp(gq*tau1) + Exp(gq*tau2))*mq + 2*Clq*(tau1 + tau2)))  ,0,t,lambda x:0,lambda x:t)[0]
        m2gtz0 = 2*dgqpart1+dgqpart2
        dgq_ = dgqvgtz0+m2gtz0- 2*nm[1,0]*dnm_g['gq']
    if key=='sq2':
        dsq2_ =dblquad(lambda t1,t2:1/(2*gq)*(np.exp(-np.abs(t1-t2)*gq)-np.exp(-gq*(t1+t2)))\
            *Exp(2*bx+2*Cxx-2*b*t+(b+bl+2*Cxl)*(t1+t2)+(Cll*(t1+t2)**2)/2.),0,t,lambda\
                       x:0,lambda x:t)[0]-2*dnm_g['sq2']*nm[1,0]
    if key=='sx2':
        dsx2_ = 0
    if key=='sg2':
        dsg2_ = 0
    if key=='sdx2':
        dsdx2_ = 0
    if key=='sdg2':
        dsdg2_ = 0
    if key=='b':
        dbvgtz0 = dblquad(lambda tau1,tau2: sq2/(2*gq)*(np.exp(-np.abs(tau1-tau2)*gq)-np.exp(-gq*(tau1+tau2)))\
            *Exp(2*bx + 2*Cxx - 2*b*t + (b + bl + 2*Cxl)*(tau1 + tau2) + (Cll*(tau1 + tau2)**2)/2.)*(-2*t + tau1 + tau2)  ,0,t,lambda x:0,lambda x:t)[0]
        dbpart1 = quad(lambda tau1:        -2*Exp(bx + Cxx/2. - 2*b*t - gq*tau1)*t*(Exp((b + bl + Cxl + gq)*tau1 + (Cll*tau1**2)/2.)*mq*(bg + Cxg + Cgl*tau1) +\
     	        Exp((b + bl + Cxl)*tau1 + (Cll*tau1**2)/2.)*(Cgq + bg*(bq + Cxq - mq + Clq*tau1) + (Cxg + Cgl*tau1)*(bq + Cxq - mq + Clq*tau1)))\
       	        + Exp(bx + Cxx/2. - 2*b*t - gq*tau1)*(Exp((b + bl + Cxl + gq)*tau1 + (Cll*tau1**2)/2.)*mq*tau1*(bg + Cxg + Cgl*tau1) +\
                Exp((b + bl + Cxl)*tau1 + (Cll*tau1**2)/2.)*tau1*\
       	        (Cgq + bg*(bq + Cxq - mq + Clq*tau1) + (Cxg + Cgl*tau1)*(bq + Cxq - mq + Clq*tau1))) ,0,t)[0]
        dbpart2 = dblquad(lambda tau1,tau2:         Exp(2*(bx + Cxx) - 2*b*t + (b + bl + 2*Cxl - gq)*(tau1 + tau2) + (Cll*(tau1 + tau2)**2)/2.)*(-2*t + tau1 + tau2)*\
                (bq**2 + Cqq + (2*Cxq + (-1 + Exp(gq*tau1))*mq + Clq*(tau1 + tau2))*(2*Cxq + (-1 + Exp(gq*tau2))*mq + Clq*(tau1 + tau2)) +\
                bq*(4*Cxq + (-2 + Exp(gq*tau1) + Exp(gq*tau2))*mq + 2*Clq*(tau1 + tau2))) ,0,t,lambda x:0,lambda x:t)[0]
        m2gtz0 = (Cgg+bg**2)/Exp(2*b*t)*(-2*t) + 2*dbpart1+dbpart2
        db_ =  dbvgtz0 +m2gtz0- 2*nm[1,0]*dnm_g['b']
    return {'ml':dml_,'gl':dgl_,'sl2':dsl2_,'mq':dmq_,'gq':dgq_,'sq2':dsq2_\
		,'sx2':dsx2_,'sg2':dsg2_,'b':db_,'sdx2':dsdx2_,'sdg2':dsdg2_}
def cov_gl(t,bx,bg,bl,bq,Cxx,Cxg,Cxl,Cxq,Cgg,Cgl,Cgq,Cll,Clq,Cqq,ml,gl,sl2,mq,gq,sq2,b,nm):
    return  (bg*bl + Cgl)/Exp((b + gl)*t) + (bg*ml)/Exp(b*t) - (bg*ml)/Exp((b + gl)*t) +\
            quad(lambda tau: Exp(bx + Cxx/2. - b*t - gl*t + (b + bl + Cxl - gq)*tau + (Cll*tau**2)/2.)*\
            ((bq + Cxq + (-1 + Exp(gq*tau))*mq)*(Cxl + (-1 + Exp(gl*t))*ml + Cll*tau) + bl*(bq + Cxq - mq + Exp(gq*tau)*mq + Clq*tau) +\
            Clq*(1 + Cxl*tau + (-1 + Exp(gl*t))*ml*tau + Cll*tau**2)),0,t)[0]-nm[1,0]*nm[2,0]
def cov_gq(t,bx,bg,bl,bq,Cxx,Cxg,Cxl,Cxq,Cgg,Cgl,Cgq,Cll,Clq,Cqq,ml,gl,sl2,mq,gq,sq2,b,nm):
    return  quad(lambda tau:(Exp(bx - b*t + b*tau + bl*tau + (Cxx + tau*(2*Cxl + Cll*tau))/2.)*(Exp(gq*(-t + tau)) - Exp(-(gq*(t + tau))))*sq2)/(2.*gq),0,t)[0]+\
            (bg*bq + Cgq)/Exp((b + gq)*t) + (bg*mq)/Exp(b*t) - (bg*mq)/Exp((b + gq)*t) + quad(lambda tau:\
            Exp(bx + Cxx/2. - b*t - gq*t + b*tau + bl*tau + Cxl*tau - gq*tau + (Cll*tau**2)/2.)*\
            (bq**2 + Cqq + (Cxq + (-1 + Exp(gq*t))*mq + Clq*tau)*(Cxq + (-1 + Exp(gq*tau))*mq + Clq*tau) +\
            bq*(2*Cxq + (-2 + Exp(gq*t) + Exp(gq*tau))*mq + 2*Clq*tau)),0,t)[0]-nm[1,0]*nm[3,0]
def cov_ll(t,bx,bg,bl,bq,Cxx,Cxg,Cxl,Cxq,Cgg,Cgl,Cgq,Cll,Clq,Cqq,ml,gl,sl2,mq,gq,sq2,b):
    vl = sl2/(2*gl)*(1-Exp(-gl*t)**2)
    return Cll*Exp(-gl*t)**2 + vl
def cov_lq(t,bx,bg,bl,bq,Cxx,Cxg,Cxl,Cxq,Cgg,Cgl,Cgq,Cll,Clq,Cqq,ml,gl,sl2,mq,gq,sq2,b):
    return  Clq*Exp(-gl*t)*Exp(-gq*t)
def cov_qq(t,bx,bg,bl,bq,Cxx,Cxg,Cxl,Cxq,Cgg,Cgl,Cgq,Cll,Clq,Cqq,ml,gl,sl2,mq,gq,sq2,b):
    vq = sq2/(2*gq)*(1-Exp(-gq*t)**2)
    return vq + Cqq*Exp(-gq*t)**2
def mean_cov_model(m,C,t,ml,gl,sl2,mq,gq,sq2,b):
    """Given p(z0)=n(m,C) find p(z1) with no cell division"""
    nC = np.zeros((4,4))
    nm = np.zeros((4,1))
    bx=m[0,0]; bg=m[1,0]; bl=m[2,0]; bq=m[3,0]
    Cxx=C[0,0];Cxg=C[0,1];Cxl=C[0,2];Cxq=C[0,3];Cgg=C[1,1];Cgl=C[1,2];Cgq=C[1,3];Cll=C[2,2];Clq=C[2,3];Cqq=C[3,3]
    #####################
    ######   Mean     ###
    #####################
    nm[0,0] = mean_x(t,bx,bg,bl,bq,Cxx,Cxg,Cxl,Cxq,Cgg,Cgl,Cgq,Cll,Clq,Cqq,ml,gl,sl2,mq,gq,sq2,b)
    nm[1,0] = mean_g(t,bx,bg,bl,bq,Cxx,Cxg,Cxl,Cxq,Cgg,Cgl,Cgq,Cll,Clq,Cqq,ml,gl,sl2,mq,gq,sq2,b)
    nm[2,0] = mean_l(t,bx,bg,bl,bq,Cxx,Cxg,Cxl,Cxq,Cgg,Cgl,Cgq,Cll,Clq,Cqq,ml,gl,sl2,mq,gq,sq2,b)
    nm[3,0] = mean_q(t,bx,bg,bl,bq,Cxx,Cxg,Cxl,Cxq,Cgg,Cgl,Cgq,Cll,Clq,Cqq,ml,gl,sl2,mq,gq,sq2,b)
    ####################
    ####### Cov ########
    ####################
    nC[0,0] = cov_xx(t,bx,bg,bl,bq,Cxx,Cxg,Cxl,Cxq,Cgg,Cgl,Cgq,Cll,Clq,Cqq,ml,gl,sl2,mq,gq,sq2,b)
    nC[0,1] = cov_xg(t,bx,bg,bl,bq,Cxx,Cxg,Cxl,Cxq,Cgg,Cgl,Cgq,Cll,Clq,Cqq,ml,gl,sl2,mq,gq,sq2,b,nm)
    nC[0,2] = cov_xl(t,bx,bg,bl,bq,Cxx,Cxg,Cxl,Cxq,Cgg,Cgl,Cgq,Cll,Clq,Cqq,ml,gl,sl2,mq,gq,sq2,b)
    nC[0,3] = cov_xq(t,bx,bg,bl,bq,Cxx,Cxg,Cxl,Cxq,Cgg,Cgl,Cgq,Cll,Clq,Cqq,ml,gl,sl2,mq,gq,sq2,b)
    nC[1,1] = cov_gg(t,bx,bg,bl,bq,Cxx,Cxg,Cxl,Cxq,Cgg,Cgl,Cgq,Cll,Clq,Cqq,ml,gl,sl2,mq,gq,sq2,b,nm)
    nC[1,2] = cov_gl(t,bx,bg,bl,bq,Cxx,Cxg,Cxl,Cxq,Cgg,Cgl,Cgq,Cll,Clq,Cqq,ml,gl,sl2,mq,gq,sq2,b,nm)
    nC[1,3] = cov_gq(t,bx,bg,bl,bq,Cxx,Cxg,Cxl,Cxq,Cgg,Cgl,Cgq,Cll,Clq,Cqq,ml,gl,sl2,mq,gq,sq2,b,nm)
    nC[2,2] = cov_ll(t,bx,bg,bl,bq,Cxx,Cxg,Cxl,Cxq,Cgg,Cgl,Cgq,Cll,Clq,Cqq,ml,gl,sl2,mq,gq,sq2,b)
    nC[2,3] = cov_lq(t,bx,bg,bl,bq,Cxx,Cxg,Cxl,Cxq,Cgg,Cgl,Cgq,Cll,Clq,Cqq,ml,gl,sl2,mq,gq,sq2,b)
    nC[3,3] = cov_qq(t,bx,bg,bl,bq,Cxx,Cxg,Cxl,Cxq,Cgg,Cgl,Cgq,Cll,Clq,Cqq,ml,gl,sl2,mq,gq,sq2,b)
    ####################
    #### return     ####
    ####################
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
    log_lik = -1/2*y.T@M@y-1/2*np.log(np.linalg.det(M))-2*np.log(2*Pi)
    return log_lik
def dD():
    dsdx2_ = np.zeros((2,2))
    dsdx2_[0,0]=1
    dsdx2_ = np.zeros((2,2))
    dsdg2_[1,1]=1
    return {'ml':0,'gl':0,'sl2':0,'mq':0,'gq':0,'sq2':0\
		,'sx2':0,'sg2':0,'b':0,'sdx2':dsdx2_,'sdg2':dsdg2_}
def d_log_likelihood(y,m,C,dm,dC,sx2,sg2,keys):
    D = np.zeros((2,2))
    D[0,0]=sx2;D[1,1]=sg2
    M = np.linalg.inv(C[:2,:2]+D)
    y = y-m[:2,:]
    for key in keys:
        d_log_lik[key] = 1/2*y.T@M@(dC[key]+dD[key])@M@y\
                -1/2*np.linalg.trace(M@(dC[key]+dD[key]))\
                +dm[key].T@M@y
    return d_log_likelihood
def posterior(y,m,C,sx2,sg2):
    """Given y_t and the p(z_t) prior, find p(z_t|y_t). Return mean and cov"""
    D = np.zeros((2,2))
    D[0,0]=sx2;D[1,1]=sg2
    M = np.linalg.inv(C[:2,:2]+D)
    K = C[:2,:]
    return m+K.T@M@(y-m[:2,:]), C-K.T@M@K
def posterior_likelihood(Y,m,C,ml,gl,sl2,mq,gq,sq2,b,sx2,sg2,sdx2,sdg2,Likelihood,Posterior):
    """df must be connected in genealogy, time ordered and
    maximun 1 division apart i.e. mother daugther relationship """
    post_mean=[];post_error=[];log_lik=0
    XG = Y[:,:2].astype('float64');T=Y[:,2:3].astype('float64');ID=Y[:,3:4]
    T = T-T[0,0]
    for i in range(T.shape[0]):
        if i==0:nm=m;nC=C
        if i!=0 and ID[i,0]!=ID[i-1,0]:
            # If cell division happen
            nm,nC = division(nm,nC,sdx2,sdg2)
        y = XG[i:i+1,:].T
        # COMPUTE LIKELIHOOD
        if Likelihood:
            log_lik+=log_likelihood(y,nm,nC,sx2,sg2)
        #COMPUTE POSTERIOR 
        nm,nC = posterior(y,nm,nC,sx2,sg2)
        if Posterior:
            # Use forward backward algorithm and multiplication
            # identity of Gaussian dist.

            post_mean.append(nm.reshape(-1))
            post_error.append(np.diag(nC).reshape(-1))
        # NEXT TIME POINT PRIOR NO DIVISION
        if i<T.shape[0]-1:
            dt = T[i+1,0]-T[i,0]
        nm,nC = mean_cov_model(nm,nC,dt,ml,gl,sl2,mq,gq,sq2,b)
    if Likelihood:
        return log_lik
    else:
        return post_mean, post_error
########################################################################################################################################################
########################################################################################################################################################
########################################################################################################################################################
########################################################################################################################################################
########################################################################################################################################################
########################################################################################################################################################
########################################################################################################################################################
########################################################################################################################################################
########################################################################################################################################################
########################################################################################################################################################
########################################################################################################################################################
########################################################################################################################################################
########################################################################################################################################################
########################################################################################################################################################
########################################################################################################################################################
########################################################################################################################################################
########################################################################################################################################################
########################################################################################################################################################
########################################################################################################################################################
########################################################################################################################################################
########################################################################################################################################################
########################################################################################################################################################
########################################################################################################################################################
########################################################################################################################################################
########################################################################################################################################################
########################################################################################################################################################
########################################################################################################################################################
########################################################################################################################################################
########################################################################################################################################################
########################################################################################################################################################
########################################################################################################################################################
########################################################################################################################################################
########################################################################################################################################################
########################################################################################################################################################
########################################################################################################################################################
########################################################################################################################################################
########################################################################################################################################################
########################################################################################################################################################
########################################################################################################################################################
########################################################################################################################################################
########################################################################################################################################################
########################################################################################################################################################
########################################################################################################################################################
########################################################################################################################################################
########################################################################################################################################################
def mean(t,m,C,ml,gl,sl2,mq,gq,sq2,b):
    nm = np.zeros((4,1))
    bx=m[0,0]; bg=m[1,0]; bl=m[2,0]; bq=m[3,0]
    Cxx=C[0,0];Cxg=C[0,1];Cxl=C[0,2];Cxq=C[0,3];Cgg=C[1,1];Cgl=C[1,2];Cgq=C[1,3];Cll=C[2,2];Clq=C[2,3];Cqq=C[3,3]
    nm[0,0] = mean_x(t,bx,bg,bl,bq,Cxx,Cxg,Cxl,Cxq,Cgg,Cgl,Cgq,Cll,Clq,Cqq,ml,gl,sl2,mq,gq,sq2,b)
    nm[1,0] = mean_g(t,bx,bg,bl,bq,Cxx,Cxg,Cxl,Cxq,Cgg,Cgl,Cgq,Cll,Clq,Cqq,ml,gl,sl2,mq,gq,sq2,b)
    nm[2,0] = mean_l(t,bx,bg,bl,bq,Cxx,Cxg,Cxl,Cxq,Cgg,Cgl,Cgq,Cll,Clq,Cqq,ml,gl,sl2,mq,gq,sq2,b)
    nm[3,0] = mean_q(t,bx,bg,bl,bq,Cxx,Cxg,Cxl,Cxq,Cgg,Cgl,Cgq,Cll,Clq,Cqq,ml,gl,sl2,mq,gq,sq2,b)
    return nm
#@np.vectorize
def cov_ll2(t,s,m,C,ml,gl,sl2,mq,gq,sq2,b):
    Cll=C[2,2]
    return sl2/(2*gl)*(Exp(-gl*np.abs(t-s))-Exp(-gl*(t+s)))+Cll*Exp(-gl*(t+s))
def cov_qq2(t,s,m,C,ml,gl,sl2,mq,gq,sq2,b):
    Cqq=C[3,3]
    return sq2/(2*gq)*(Exp(-gq*np.abs(t-s))-Exp(-gq*(t+s)))+Cqq*Exp(-gq*(t+s))
def cov_lq2(t,s,m,C,ml,gl,sl2,mq,gq,sq2,b):
    Clq=C[2,3];
    return Clq*Exp(-gl*t-gq*s)
def cov_xx2(t,s,m,C,ml,gl,sl2,mq,gq,sq2,b):
    Cxl=C[0,2];Cll=C[2,2];Cxx=C[0,0]
    integ = dblquad(lambda t1,s1:sl2/(2*gl)*(Exp(-gl*np.abs(t1-s1))-Exp(-gl*(t1+s1)))\
                    ,0,t,lambda x:0,lambda x:s)[0]
    return Cll*(1-Exp(-gl*t))*(1-Exp(-gl*s))/gl**2+Cxl*(1-Exp(-gl*t))/gl\
            +Cxl*(1-Exp(-gl*s))/gl+Cxx+integ
def cov_xl2(t,s,m,C,ml,gl,sl2,mq,gq,sq2,b):
    Cxl=C[0,2];Cll=C[2,2]
    integ = quad(lambda t1:sl2/(2*gl)*(Exp(-gl*np.abs(t1-s))-Exp(-gl*(t1+s))),0,t)[0]
    return integ+Cll*Exp(-gl*s)*(1-Exp(-gl*t))/gl+Cxl*Exp(-gl*s)
def cov_xq2(t,s,m,C,ml,gl,sl2,mq,gq,sq2,b):
    Clq=C[2,3];Cxq=C[0,3];
    return Clq*(1-Exp(-gl*t))*Exp(-gq*s)/gl+Cxq*Exp(-gq*s)
def cov_gl2(t,s,m,C,ml,gl,sl2,mq,gq,sq2,b):
    bx=m[0,0]; bg=m[1,0]; bl=m[2,0]; bq=m[3,0]
    Cxx=C[0,0];Cxg=C[0,1];Cxl=C[0,2];Cxq=C[0,3];Cgg=C[1,1];Cgl=C[1,2];Cgq=C[1,3];Cll=C[2,2];Clq=C[2,3];Cqq=C[3,3]
    mgml = mean_g(t,bx,bg,bl,bq,Cxx,Cxg,Cxl,Cxq,Cgg,Cgl,Cgq,Cll,Clq,Cqq,ml,gl,sl2,mq,gq,sq2,b)*mean_l(s,bx,bg,bl,bq,Cxx,Cxg,Cxl,Cxq,Cgg,Cgl,Cgq,Cll,Clq,Cqq,ml,gl,sl2,mq,gq,sq2,b)
    return((bg*bl + Cgl)/Exp(gl*s) + bg*ml - (bg*ml)/Exp(gl*s) + \
         quad(lambda tau:Exp(bx + Cxx/2. - gl*s + (b + bl + Cxl - gq)*tau + (Cll*tau**2)/2.)*\
         ((bq + Cxq + (-1 + Exp(gq*tau))*mq)*(Cxl + (-1 + Exp(gl*s))*ml + Cll*tau) + bl*(bq + Cxq - mq + Exp(gq*tau)*mq + Clq*tau) +\
          Clq*(1 + Cxl*tau + (-1 + Exp(gl*s))*ml*tau + Cll*tau**2)),0,t)[0])/Exp(b*t)-mgml
def cov_xg2(s,t,m,C,ml,gl,sl2,mq,gq,sq2,b):
    bx=m[0,0]; bg=m[1,0]; bl=m[2,0]; bq=m[3,0]
    Cxx=C[0,0];Cxg=C[0,1];Cxl=C[0,2];Cxq=C[0,3];Cgg=C[1,1];Cgl=C[1,2];Cgq=C[1,3];Cll=C[2,2];Clq=C[2,3];Cqq=C[3,3]
    mgmx = mean_g(t,bx,bg,bl,bq,Cxx,Cxg,Cxl,Cxq,Cgg,Cgl,Cgq,Cll,Clq,Cqq,ml,gl,sl2,mq,gq,sq2,b)*mean_x(s,bx,bg,bl,bq,Cxx,Cxg,Cxl,Cxq,Cgg,Cgl,Cgq,Cll,Clq,Cqq,ml,gl,sl2,mq,gq,sq2,b)
    return (bg*bl + Cgl - (bg*bl + Cgl)/Exp(gl*s) + (bg*bx + Cxg)*gl - bg*ml + (bg*ml)/Exp(gl*s) + bg*gl*ml*s + \
         quad(lambda tau:Exp(bx + Cxx/2. - gl*s + (b + bl + Cxl - gq)*tau + (Cll*tau**2)/2.)*\
         (-(bq*Cxl) - Cxl*Cxq + bq*Cxl*Exp(gl*s) + Cxl*Cxq*Exp(gl*s) + bq*bx*Exp(gl*s)*gl + Cxq*Exp(gl*s)*gl + bx*Cxq*Exp(gl*s)*gl + \
         bq*Cxx*Exp(gl*s)*gl + Cxq*Cxx*Exp(gl*s)*gl + bq*ml + Cxq*ml - bq*Exp(gl*s)*ml - Cxq*Exp(gl*s)*ml + Cxl*mq - Cxl*Exp(gl*s)*mq - \
         Cxl*Exp(gq*tau)*mq + Cxl*Exp(gl*s + gq*tau)*mq - bx*Exp(gl*s)*gl*mq - Cxx*Exp(gl*s)*gl*mq + bx*Exp(gl*s + gq*tau)*gl*mq + \
         Cxx*Exp(gl*s + gq*tau)*gl*mq - ml*mq + Exp(gl*s)*ml*mq + Exp(gq*tau)*ml*mq - Exp(gl*s + gq*tau)*ml*mq + bq*Exp(gl*s)*gl*ml*s + \
         Cxq*Exp(gl*s)*gl*ml*s - Exp(gl*s)*gl*ml*mq*s + Exp(gl*s + gq*tau)*gl*ml*mq*s - bq*Cll*tau - Cll*Cxq*tau + \
         bq*Cll*Exp(gl*s)*tau + Cll*Cxq*Exp(gl*s)*tau + bq*Cxl*Exp(gl*s)*gl*tau + Cxl*Cxq*Exp(gl*s)*gl*tau + Cll*mq*tau - \
         Cll*Exp(gl*s)*mq*tau - Cll*Exp(gq*tau)*mq*tau + Cll*Exp(gl*s + gq*tau)*mq*tau - Cxl*Exp(gl*s)*gl*mq*tau + \
         Cxl*Exp(gl*s + gq*tau)*gl*mq*tau + bl*(-1 + Exp(gl*s))*(bq + Cxq - mq + Exp(gq*tau)*mq + Clq*tau) + \
         Clq*(-1 - Cxl*tau + ml*tau - Cll*tau**2 + Exp(gl*s)*\
         (1 + bx*gl*tau + Cxx*gl*tau - ml*tau + gl*ml*s*tau + Cll*tau**2 + Cxl*tau*(1 + gl*tau))) ),0,t)[0] )/(Exp(b*t)*gl)-mgmx
def cov_gq2(t,s,m,C,ml,gl,sl2,mq,gq,sq2,b):
    bx=m[0,0]; bg=m[1,0]; bl=m[2,0]; bq=m[3,0]
    Cxx=C[0,0];Cxg=C[0,1];Cxl=C[0,2];Cxq=C[0,3];Cgg=C[1,1];Cgl=C[1,2];Cgq=C[1,3];Cll=C[2,2];Clq=C[2,3];Cqq=C[3,3]
    mgml = mean_g(t,bx,bg,bl,bq,Cxx,Cxg,Cxl,Cxq,Cgg,Cgl,Cgq,Cll,Clq,Cqq,ml,gl,sl2,mq,gq,sq2,b)*mean_q(s,bx,bg,bl,bq,Cxx,Cxg,Cxl,Cxq,Cgg,Cgl,Cgq,Cll,Clq,Cqq,ml,gl,sl2,mq,gq,sq2,b)
    first = quad(lambda tau:(Exp(bx-b*t+b*tau+bl*tau+(Cxx+tau*(2*Cxl+Cll*tau))/2.)*(Exp(gq*(-s+tau))-Exp(-(gq*(s+tau))))*sq2)/(2.*gq),0,t)[0]
    second =((bg*bq + Cgq)/Exp(gq*s) + bg*mq - (bg*mq)/Exp(gq*s) +\
         quad(lambda tau: Exp(bx + Cxx/2. - gq*s + b*tau + bl*tau + Cxl*tau - gq*tau + (Cll*tau**2)/2.)*\
         (bq**2 + Cqq + (Cxq + (-1 + Exp(gq*s))*mq + Clq*tau)*(Cxq + (-1 + Exp(gq*tau))*mq + Clq*tau) +\
         bq*(2*Cxq + (-2 + Exp(gq*s) + Exp(gq*tau))*mq + 2*Clq*tau)),0,t)[0])/Exp(b*t)
    return first+second-mgml
def cov_gg2(t,s,m,C,ml,gl,sl2,mq,gq,sq2,b):
    bx=m[0,0]; bg=m[1,0]; bl=m[2,0]; bq=m[3,0]
    Cxx=C[0,0];Cxg=C[0,1];Cxl=C[0,2];Cxq=C[0,3];Cgg=C[1,1];Cgl=C[1,2];Cgq=C[1,3];Cll=C[2,2];Clq=C[2,3];Cqq=C[3,3]
    mgmg = mean_g(t,bx,bg,bl,bq,Cxx,Cxg,Cxl,Cxq,Cgg,Cgl,Cgq,Cll,Clq,Cqq,ml,gl,sl2,mq,gq,sq2,b)*mean_g(s,bx,bg,bl,bq,Cxx,Cxg,Cxl,Cxq,Cgg,Cgl,Cgq,Cll,Clq,Cqq,ml,gl,sl2,mq,gq,sq2,b)
    def vargt_z0(t1,t2):
        return sq2/(2*gq)*(np.exp(-np.abs(t1-t2)*gq)-np.exp(-gq*(t1+t2)))*Exp(2*bx+2*Cxx-b*(t+s)+(b+bl+2*Cxl)*(t1+t2)+(Cll*(t1+t2)**2)/2.)
    def mean_gtz0_part1(tau1):
        return  Exp(bx + (Cxx + tau1*(2*bl + 2*Cxl + Cll*tau1))/2.)*\
        (Exp(b*(-(t+s) + tau1))*mq*(bg + Cxg + Cgl*tau1) - Exp(-b*(t+s) + b*tau1 - gq*tau1)*mq*(bg + Cxg + Cgl*tau1) + \
          Exp(-b*(t+s) + b*tau1 - gq*tau1)*(Cgq + bg*(bq + Cxq + Clq*tau1) + (Cxg + Cgl*tau1)*(bq + Cxq + Clq*tau1)))
    def mean_gtz0_part2(tau1,tau2):
        return Exp(2*bx + 2*Cxx - gq*(tau1 + tau2) + b*(-(t+s) + tau1 + tau2) + ((tau1 + tau2)*(2*bl + 4*Cxl + Cll*(tau1 + tau2)))/2.)*\
        (bq**2 + Cqq + (2*Cxq + (-1 + Exp(gq*tau1))*mq + Clq*(tau1 + tau2))*(2*Cxq + (-1 + Exp(gq*tau2))*mq + Clq*(tau1 + tau2)) + \
          bq*(4*Cxq + (-2 + Exp(gq*tau1) + Exp(gq*tau2))*mq + 2*Clq*(tau1 + tau2)))
    vgtz0 = dblquad(vargt_z0, 0, t, lambda x: 0, lambda x: s)[0]
    m2gtz0 = (Cgg+bg**2)/Exp(b*(t+s)) +quad(mean_gtz0_part1, 0, t)[0] +quad(mean_gtz0_part1, 0, s)[0] +\
              dblquad(mean_gtz0_part2, 0, t, lambda x: 0, lambda x: s)[0]
    return vgtz0 +m2gtz0-mgmg
def likelihood_posteriori_1cc_full(Y,m,C,ml,gl,sl2,mq,gq,sq2,b,sx2,sg2,posterior,likelihood,nproc):
    """Compute the likelihood or posterior over 1cc (no cell division). The
    vector Y must be in the form Y=[log_length,gfp,time] of shape (n,3) time
    ordered"""
    ##########################
    ######  FUNCTIONS     ####
    ##########################
    def mean_(t):
        return mean(t,m,C,ml,gl,sl2,mq,gq,sq2,b)
    def covxx(two_time):
        return cov_xx2(two_time[0],two_time[1],m,C,ml,gl,sl2,mq,gq,sq2,b)
    def covxg(two_time):
        return cov_xg2(two_time[0],two_time[1],m,C,ml,gl,sl2,mq,gq,sq2,b)
    def covgg(two_time):
        return cov_gg2(two_time[0],two_time[1],m,C,ml,gl,sl2,mq,gq,sq2,b)
    def covxl(two_time):
        return cov_xl2(two_time[0],two_time[1],m,C,ml,gl,sl2,mq,gq,sq2,b)
    def covxq(two_time):
        return cov_xq2(two_time[0],two_time[1],m,C,ml,gl,sl2,mq,gq,sq2,b)
    def covgl(two_time):
        return cov_gl2(two_time[0],two_time[1],m,C,ml,gl,sl2,mq,gq,sq2,b)
    def covgq(two_time):
        return cov_gq2(two_time[0],two_time[1],m,C,ml,gl,sl2,mq,gq,sq2,b)
    ##########################
    ######      MAIN      ####
    ##########################
    assert likelihood==True or posterior==True,"compute one of the two"
    #p1 = Pool(nproc)
    p = path_pool(nproc)
    p1=p
    X = Y[:,:1]; G=Y[:,1:2];T=Y[:,2:3]-Y[0,2]
    # ERROR MATRIX
    if type(sx2) is float:
        Dx = np.zeros((Y.shape[0],Y.shape[0]))
        np.fill_diagonal(Dx,[sx2]*Y.shape[0])
    else: Dx = sx2
    if type(sg2) is float:
        Dg = np.zeros((Y.shape[0],Y.shape[0]))
        np.fill_diagonal(Dg,[sg2]*Y.shape[0])
    else: Dg = sg2
    Ds = np.concatenate((Dx,np.zeros_like(Dx)),axis=1)
    Di = np.concatenate((np.zeros_like(Dx),Dg),axis=1)
    D = np.concatenate((Ds,Di),axis=0)
    # COMPUTE MEAN VECTOR AND SOME  MATRICES
    s,t = np.meshgrid(T,T)# t = [[0,0],[1,1]]; s = [[0,1],[0,1]]
    Mt = p.map(mean_,T)
    # Compute Covgg in parallel way
    # Do not vecotrize so reduce number of computations
    upper_time=[]
    for k in T:
        for j in T:
            if j>=k:
                upper_time.append((k,j))
        else:continue
    tmp_kxx = p1.map(covxx,upper_time)
    tmp_kxg = p1.map(covxg,upper_time)
    tmp_kxl = p1.map(covxl,upper_time)
    tmp_kxq = p1.map(covxq,upper_time)
    tmp_kgg = p1.map(covgg,upper_time)
    tmp_kgl = p1.map(covgl,upper_time)
    tmp_kgq = p1.map(covgq,upper_time)
    Kgg,Kxx,Kxg,Kxl,Kxq,Kgl,Kgq  = [np.zeros((T.shape[0],T.shape[0]))]*7
    s=0
    for k,_ in enumerate(T):
        for j,_ in enumerate(T):
            if j>=k:
                Kxx[k,j] = tmp_kxx[s]
                Kxg[k,j] = tmp_kxg[s]
                Kxl[k,j] = tmp_kxl[s]
                Kxq[k,j] = tmp_kxq[s]
                Kgg[k,j] = tmp_kgg[s]
                Kgl[k,j] = tmp_kgl[s]
                Kgq[k,j] = tmp_kgq[s]
                s+=1
            else:continue
    myfunc = lambda K: K + np.triu(K,1).T
    Kgg,Kxx,Kxg,Kxl,Kxq,Kgl,Kgq = list(map(myfunc,[Kgg,Kxx,Kxg,Kxl,Kxq,Kgl,Kgq]))
    # MEAN FUNCTION COMPUTATION
    XmM = np.vstack([x-m[0] for x,m in zip(X,Mt)])
    GmM = np.vstack([g-m[1] for g,m in zip(G,Mt)])
    Vm = np.concatenate((XmM,GmM),axis=0)
    # K0 MATRIX AND INVERSE COMPUTATION
    K0s = np.concatenate((Kxx,Kxg),axis=1)
    K0i = np.concatenate((Kxg.T,Kgg),axis=1)
    K0 = np.concatenate((K0s,K0i),axis=0)
    print("inversion")
    start=datetime.now()
    K0Di = np.linalg.inv(K0+D)
    K0DiVm = K0Di@Vm
    #K TILDE MATRIX COMPUTATION
    print(datetime.now()-start)
    def KtilT_C_t(i):
        """Find Ktilde and C for time i"""
        kll = cov_ll2(T[i,0],T[i,0],m,C,ml,gl,sl2,mq,gq,sq2,b)
        klq = cov_lq2(T[i,0],T[i,0],m,C,ml,gl,sl2,mq,gq,sq2,b)
        kqq = cov_qq2(T[i,0],T[i,0],m,C,ml,gl,sl2,mq,gq,sq2,b)
        KtilTl=np.concatenate((Kxx[:,i:i+1].T,Kxg[:,i:i+1].T,Kxl[:,i:i+1].T,Kxq[:,i:i+1].T),axis=0)
        KtilTr=np.concatenate((Kxg.T[:,i:i+1].T,Kgg[:,i:i+1].T,Kgl[:,i:i+1].T,Kgq[:,i:i+1].T),axis=0)
        KtilT = np.concatenate((KtilTl,KtilTr),axis=1)
        C_t = np.array([[Kxx[i,i],Kxg[i,i],Kxl[i,i],Kxq[i,i]],\
                           [Kxg[i,i],Kgg[i,i],Kgl[i,i],Kgq[i,i]],\
                           [Kxl[i,i],Kgl[i,i],kll,klq],\
                           [Kxq[i,i],Kgq[i,i],klq,kqq],\
                          ])
        return KtilT,C_t
    # COMPUTE POSTERIOR
    def posterior(Mt,C,KtilT):
        return Mt+KtilT@K0DiVm, C-KtilT@K0Di@KtilT.T
    if posterior:
        # Ktil (list for t*=0,1,..,n)
        KtilT =[]; C_t=[]
        print("Ktild")
        start=datetime.now()
        tmp = p.map(KtilT_C_t,range(Kxx.shape[1]))
        print(datetime.now()-start)
        my_function = lambda x: posterior(x[0],x[1][1],x[1][0])
        print("post")
        start=datetime.now()
        POST = p.map(my_function,zip(Mt,tmp))
        print(datetime.now()-start)
        #POST = [ posterior(m,c,ktilT) for m,c,ktilT in zip(Mt,C_t,KtilT)]
        # We also return POST[-1] for the next cell computation
        return POST,POST[-1]
    # COMPUTE LIKELIHOOD
    if likelihood:
        LIK = -1/2*Vm.T@K0DiVm-1/2*np.linalg.det(K0DiVm)-2*np.log((2*Pi))
        # Compute last cell mean and cov
        ktilt,ct = KtilT_C_t(Kxx.shape[1]-1)
        POSTm1 = posterior(Mt[-1],ct,ktilt)
        # We also return POSTm1 for next cell computation
        return LIK, POSTm1
def full_likelihood_posterior(df,m,C,ml,gl,sl2,mq,gq,sq2,b,sx2,sg2,sdx2,sdg2,posterior\
                          ,likelihood,inf_points=5,nproc=4,xlabel='log_length_um',\
                          glabel='gfp_nb',tlabel='time_min',cell_lab='cell_id'):
    """Y[x,g,time,cellid] Y must be connected in genealogy, time ordered and
    maximun 1 division apart i.e. mother daugther relationship """
    post=[];log_lik=0
    df =\
    df[['{}'.format(xlabel),'{}'.format(glabel),'{}'.format(tlabel),'{}'.format(cell_lab)]]
    oldc = df['{}'.format(cell_lab)].iloc[0]
    delta_div =\
    (df.groupby('cell')['time_min'].first().values[1:]-df.groupby('cell')['time_min'].last().values[:-1])
    i=0
    for cel,Y in df.groupby('{}'.format(cell_lab)):
        if cel==oldc: nm=m;nC=C
        else:
            # If cell division happen
            nm,nC = division(nm,nC,sdx2,sdg2)
        Y = df[['{}'.format(xlabel),'{}'.format(glabel),'{}'.format(tlabel)]].values
        # COMPUTE LIKELIHOOD
        if likelihood:
            tmp=likelihood_posteriori_1cc_full(Y,nm,nC,ml,gl,sl2,mq,gq,sq2,b,sx2,sg2,False,True,nproc)
            log_lik+=tmp[0]
            nm,nC = tmp[1]
        #COMPUTE POSTERIOR
        if posterior:
            tmp=likelihood_posteriori_1cc_full(Y,nm,nC,ml,gl,sl2,mq,gq,sq2,b,sx2,sg2,True,False,nproc)
            post.append(tmp[0])
            nm,nC = tmp[1]
        # NEXT TIME POINT PRIOR BEFORE DIVISION
        dt = delta_div[i]
        nm,nC = mean_cov_model(nm,nC,dt,ml,gl,sl2,mq,gq,sq2,b)
    return post,log_lik
