import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
from scipy.integrate import dblquad

#

class constents(object):
    """useful constants"""
    # constant of nature
    speed_of_light = 2.99792458e5                            # in km/s
    H0 = 100.0                                               # in units of h * km/sec/Mpc
    # c_by_H0=0.05*speed_of_light*0.0032615637967311         #present hubble radius in Giga year/h
    c_by_H0 = 0.05 * speed_of_light                          # present hubble radius in  Mpc/h
    omegam0 = 0.315                                         # in units of 1/h^2
    omegal0 = 0.6911                                         # in units of 1/h^2
    omegar0 = 8.24e-5                                        # in units of 1/h^2
    omegak = 1 - omegam0 - omegal0 - omegar0                 # in units of 1/h^2
    G = 4.301e-9                                             # km^2 Mpc M_sun^-1 s^-2 gravitational constant
    rho_crit0 = 3*H0**2/(8.0*np.pi*G)                                # in units of  h^2 * Msun / Mpc^3     ( = 3 * H0^2 / 8*PI* G)


#



class cosmology_distances(constents):
    """useful functions for cosmology """

#

    def scale_to_redshift(self, scale):                      #converts from scale factor to redshift
        redshift = (1.0 / scale) - 1.0
        return redshift

#

    def redshift_to_scale(self, redshift):                   #convert from redshift to scale
        scale = 1.0 / (1.0 + redshift)
        return scale

#

    def dHub0(self, redshift):                               #distanace according to hubble law in units of Mpc/h

        dist = self.c_by_H0 * redshift
        return dist

#

    def E_z(self, redshift):                                 # H(z) = H0 * E(z)
        om0 = self.omegam0 * self.redshift_to_scale(redshift) ** -3
        or0 = self.omegar0 * self.redshift_to_scale(redshift) ** -4
        ok0 = self.omegak * self.redshift_to_scale(redshift) ** -2
        ol0 = self.omegal0
        e_z = np.sqrt(om0 + or0 + ok0 + ol0)
        return e_z

#

    def H_z(self, redshift):                                  # hubble parameter at redshift z
        return self.H0 * self.E_z(redshift)

#

    def dconformal_los(self, z):                              # comoving line of sight

        def integrand(x):
            return 1.0 / self.E_z(x)


        integration = quad(integrand, 0, z)
        # return  integration*self.c_by_H0
        return integration[0] * self.c_by_H0                #integration[0] since quad return array as output

#

    def dconformal_transverse(self, z):                     # comoving distance transverse

        if (self.omegak > 0):
            f = self.c_by_H0 / np.sqrt(self.omegak)
            tcd = f * np.sinh(self.dconformal_los(z) / f)
        elif self.omegak == 0:
            tcd = self.dconformal_los(z)
        elif self.omegak < 0:
            f = self.c_by_H0 / np.sqrt(abs(self.omegak))
            tcd = f * np.sin(self.dconformal_los(z) / f)
        return tcd

#

    def dangular(self, z):                                   # angular diamter
        angdist = self.dconformal_transverse(z) / (1.0 + z)
        return angdist

#

    def dlumonicity(self, z):                                # lumonicity distance
        return (1.0 + z) * self.dconformal_transverse(z)

#

    def dlighttravel(self, z):                               # light travel distance

        def integrand(x):
            return 1.0 / (1.0 + z) * self.E_z(x)


        integration = quad(integrand, 0, z)
        return integration * self.c_by_H0

#





class nfw_projection:                                           # takes parameters(mass, c)
                                                                # mass in units of Msun , c is concentration parameter
#

    def __init__(self,mass,c):
        self.mass=mass
        self.c=c
        self.omegam = constents.omegam0
        self.rho_crit = constents.rho_crit0
        self.PI = np.pi
        self.r200 = ((self.mass) / (200 * self.rho_crit * self.omegam * (4.0 * self.PI / 3.0))) ** (1.0 / 3.0)  # in Mpc
        self.bool_splinesigma=False
        self.bool_less_than_pointzeroone=False
        self.bool_spline_avg = False
        self.bool_spline_avg_lessthanpointzeroone=False




#


    def sigma_R(self,R):

        PI = np.pi
        rs = self.r200 / self.c
        log_term = (np.log((rs + self.r200) / rs) - self.r200 / (rs + self.r200))
        rho_not = self.mass / ((4.0 * PI) * (rs ** 3.0) * log_term)  # in units of Msun/Mpc^3

#

        if R>self.r200:
            return 0
        else:

            def integrand2(s,R):                        # s in z axis ....since we used z for redshift so we used s here
                r = np.sqrt(R ** 2.0 + s ** 2.0)
                return 1.0 / ((r / rs) * (1.0 + r / rs) ** 2)

            #ll = -(np.sqrt(r200 ** 2.0 - R ** 2.0))  # lower limit of intigration
            ul = (np.sqrt((  ( 8.0*self.r200) ** 2.0 ) - R ** 2.0))  # upper limit of intigration

            integration2 = quad(integrand2, 0, ul, args=R)

            return integration2[0] *2.0* rho_not          # intigration2 is array of intigration value and error


#


    def spline_fitted_sigma_R(self,R):                    #getting sigma_R by cubic spline fitting


#


        if R<0.01  and self.bool_less_than_pointzeroone==False:
            self.value=self.sigma_R(0.01)
            self.bool_less_than_pointzeroone=True
            return self.value
        elif R<0.01 and self.bool_less_than_pointzeroone==True:
            return self.value
        if R>self.r200:
            return 0

#

        if self.bool_splinesigma==False:

            x = []  # to fit cubic spline
            y = []  # to fit cubic spline
            sampling=np.logspace(-2,np.log10(5*self.r200),30)
            for kk in range (0,len(sampling)):
                x.append(sampling[kk])  # storing data to fit cubic spline
                y.append(self.sigma_R(sampling[kk]))  # storing data to fit cubic spline
            from scipy.interpolate import CubicSpline
            self.cs=CubicSpline(x,y)
            self.bool_splinesigma = True

            return  self.cs(R)
        else:
            return self.cs(R)

#

    def sigma_mean_daughter(self,Rmax):
        value = 2 * np.pi * quad(lambda Rp: Rp * self.spline_fitted_sigma_R(Rp), 0.0, Rmax)[0] / (np.pi * Rmax ** 2)
        return value

#

    def sigma_average_daughter(self,Rmax):                                #sigma averaged overa a constant radius Rmax
        return self.spline_fitted_sigma_R(Rmax)

#

    def delta_sigma_daughter(self,R):
        return self.sigma_mean_daughter(R)-self.sigma_average_daughter(R)

#

    def sigma_average_parent(self,R,R0):

        if R<0.01:
            return 0
        elif R>self.r200:
            return 0
        else:

            def integrand3(theta,R,R0):
                r = np.sqrt(R ** 2 + R0 ** 2 + 2 * R * R0 * np.cos(theta))

                return self.spline_fitted_sigma_R(r)

            self.int3 = quad(integrand3,0,2*np.pi,args=(R,R0))
            return self.int3 [0] /(2.0*np.pi)

#
    def sigma_mean_parent(self, R, R0):
        def integrand5( radius, R0):
            return self.spline_fitted_avg_parent(radius,R0)*radius

        integration5 = quad(integrand5, 0, R, args=(R0))

        return 2*integration5[0]/R**2

#

    def spline_fitted_avg_parent(self,R,R0):

        if R<0.01  and self.bool_spline_avg_lessthanpointzeroone==False:
            self.value=self.sigma_average_parent(0.01,R0)
            self.bool_spline_avg_lessthanpointzeroone=True
            return self.value
        elif R<0.01 and self.bool_spline_avg_lessthanpointzeroone==True:
            return self.value
        if R>self.r200:
            return 0
        if self.bool_spline_avg == False:

            x = []  # to fit cubic spline
            y = []  # to fit cubic spline
            sampling=np.logspace(-2,5*self.r200,30)
            for kk in range(0,len(sampling)):
                x.append(sampling[kk])  # storing data to fit cubic spline
                y.append(self.sigma_average_parent(sampling[kk],R0))  # storing data to fit cubic spline
            from scipy.interpolate import CubicSpline
            self.css = CubicSpline(x, y)
            self.bool_spline_avg = True
            return self.css(R,R0)

        else:
            return self.css(R,R0)

#


    def delta_sigma_parent(self,R,R0):
        return self.sigma_mean_parent(R, R0) - self.spline_fitted_avg_parent(R, R0)






"""


            #scale=1.0
            #redshift=cosmo.scale_to_redshift(scale)
            #dist=cosmo.dHub0(redshift)
            #print ("for given scale factor = {} correspond to redshift = {}".format(scale, redshift))
            #print ("hubble distance  = {} corresponding  to redshift  = {}".format(dist,redshift))
            #print(cosmo.speed_of_light)
            #print(constents.speed_of_light)
            #print cosmo.omegak
            #print  cosmo.dconformal(1)

            for ii in np.arange(0,1000,0.001):

            print "{0:.3f}   {1:.3f}".format(ii,cosmo.dconformal_los(ii))
            file.write("%3.5f \t %f  \n"% (ii,cosmo.dconformal_los(ii)))
            #file.write("z kg")
            #file = open('cosmo.txt', 'w')
            i = 0
            while i <1000:
            print(i)
            file.write("%3.5f \t %f \t %f \t %f  \n" % (i,cosmo.dconformal_los(i),cosmo.dangular(i),cosmo.dlumonicity(i)))
            if i<2:
                i+=0.0001
            elif i>=2 and i<100:
                i += 1
            elif i>=100 and i < 100:
                i += 1
            else:
                i += 10





            #file.close()    #print cosmo.omegak

        """




if __name__ == "__main__":


    settalite=nfw_projection(2e14, 10)                      #definifn a of nfw_projection type
    parent=nfw_projection(2e14,10)



    settalite_dist = 0.0#parent.r200/np.sqrt(2)

    rpbin = np.logspace(-2.6, np.log10(4 *parent.r200), 50)

    rdbin = np.logspace(-2.6, np.log10(4 * settalite.r200), 50)

    #plot for  diff_sett_distances

    for dist in np.arange( parent.r200/2 , 2.0*parent.r200,(2.0*parent.r200-parent.r200)/5):
        x_d = []
        y_d = []
        x_p = []
        y_p = []

        for kk in range(0, len(rdbin)):
            x_d.append(rdbin[kk])
            y_d.append(settalite.delta_sigma_daughter(rdbin[kk])/1e12)
            y_p.append(parent.delta_sigma_parent(rdbin[kk],0)/1e12)

        xd=np.array(x_d)
        xp=np.array(x_p)
        yp=np.array(y_p)
        yd = np.array(y_d)
        yresult=yp+yd
        plt.plot(xd, yp, 'r*-', label='parent halo contribution')
        plt.plot(xd, yd,'b+--', label='daughter halo contribution')
        #plt.plot(xd, yresult,'ob', label='addition of both')
        plt.xlim(0.05, )
        #plt.ylim(-100,600)
        plt.xscale('log')
        plt.xlabel('R (Mpc h-1)')
        plt.ylabel(r'$\Delta \Sigma (R)$  ')
        #print(yp)

        # plt.yscale('log')
        plt.legend()
        plt.savefig("%.2f.png" %dist)
        plt.clf()