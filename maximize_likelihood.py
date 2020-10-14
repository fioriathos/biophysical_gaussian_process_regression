import mean_and_cov_matrix as mc
import scipy
import numpy as np
class maximize_likelihood(object):
    def __init__(self,free,boundary,fixed={}):
        """These 3 dictionary fixed the free param, constrained param, boundary"""
        assert type(fixed)==dict
        assert type(free)==dict;assert free!={}
        self.fixed = fixed; self.free = free
        self.boundary = boundary
        assert set(fixed.keys())|set((free.keys()))==\
	set(('ml','gl','sl2','mq','gq','sq2','b','sx2','sg2','sdx2','sdg2'))
        # Fix them as model parameters
        def set_att(key,val):
            if key=='ml':self.ml=val
            if key=='gl':self.gl=val
            if key=='sl2':self.sl2=val
            if key=='mq':self.mq=val
            if key=='gq':self.gq=val
            if key=='sq2':self.sq2=val
            if key=='b':self.b=val
            if key=='sx2':self.sx2=val
            if key=='sg2':self.sg2=val
            if key=='sdx2':self.sdx2=val
            if key=='sdg2':self.sdg2=val
        #set attributes
        for key,val in free.items():
            set_att(key,val)
        for key,val in fixed.items():
            set_att(key,val)
    def fix_par(self,vec, **kwargs):
        """From np.array vec divide in array the non fixed and dict the fix by giving fixed"""
        from collections import OrderedDict
        vecout = {}
        tmp = OrderedDict([('ml',vec[0]),('gl',vec[1]),('sl2',vec[2]),\
	      ('mq',vec[3]),('gq',vec[4]),('sq2',vec[5]),('b',vec[6])\
	      ('sx2',vec[7]),('sg2',vec[8]),('sdx2',vec[9]),('sdg2',vec[10])])
        for key in kwargs:
            vecout[key]=tmp[key]
            del tmp[key]
        return np.array([tmp[key] for key in tmp]), vecout
    def rebuild_param(self,vec,**kwargs):
        """ Inverse operation than fix_par"""
        from collections import OrderedDict
        tmp = OrderedDict([('ml',None),('gl',None),('sl2',None),\
                           ('mq',None),('gq',None),('sq2',None),('b',None),\
                           ('sx2',None),('sg2',None),('sdx2',None),('sdg2',None)])
        for key,val in kwargs.items():
            assert val!=None, "Can't have None as fixed values"
            tmp[key]=val
        for key,val in tmp.items():
            if val==None:
                tmp[key]=vec[0]
                vec = np.delete(vec,0)
        return np.array([tmp[key] for key in tmp])
    def initialize(self):
        """Return the x np array"""
        x0 = [None]*11
        for i in self.free:
            if i=='ml':x0[0]=self.free[i]
            if i=='gl':x0[1]=self.free[i]
            if i=='sl2':x0[2]=self.free[i]
            if i=='mq':x0[3]=self.free[i]
            if i=='gq':x0[4]=self.free[i]
            if i=='sq2':x0[5]=self.free[i]
            if i=='b':x0[6]=self.free[i]
            if i=='sx2':x0[7]=self.free[i]
            if i=='sg2':x0[8]=self.free[i]
            if i=='sdx2':x0[9]=self.free[i]
            if i=='sdg2':x0[10]=self.free[i]
        x0 = [x for x in x0 if x is not None]
        return np.array(x0)
    def objective(self,x0,Y,m,C):
        """Minus log lik"""
        x = self.rebuild_param(x0,**self.fixed)
        return -mc.posterior_likelihood(Y,m,C,*x,likelihood=True)
    def minimize(self,Y,m,C):
        """x0 start point, m&C mean and cov at time 0, Y[n,4] data"""
        x0 = self.initialize()
        return scipy.optimize.minimize(self.objective, x0, args=(Y,m,C),\
                                       method='Powell', bounds=self.bounds)
