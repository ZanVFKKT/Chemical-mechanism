import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

#PARAMETERS

a = 2.040
b = 0.835
c = 16.647
d = 0.0420
h = 9.459

I = 3
sign = 1      

#INITIAL CONDITIONS

theta_i0 = 0.5255 
phi_i0 = 0.2630

theta_j0 = 0.5793
phi_j0 = 0.2630

#PHYSICAL CHECK

if theta_i0 + phi_i0 > 1:
    raise ValueError("Particle i initial condition not physical")

if theta_j0 + phi_j0 > 1:
    raise ValueError("Particle j initial condition not physical")

#DIFFERENTIAL EQUATIONS

def system(t, y):

    theta_i, phi_i, theta_j, phi_j = y

    s_i = 1.0 - theta_i - phi_i
    s_j = 1.0 - theta_j - phi_j

    dtheta_i = (
        a*h*s_i
        - b*theta_i
        - c*h*theta_i*s_i**2
        - theta_i*s_i
    )

    dtheta_j = (
        a*h*s_j
        - b*theta_j
        - c*h*theta_j*s_j**2
        - theta_j*s_j
    )

    dphi_i = (
        theta_i*s_i
        - d*h*phi_i
        + sign*I*dtheta_j
    )

    dphi_j = (
        theta_j*s_j
        - d*h*phi_j
        + sign*I*dtheta_i
    )

    return [
        dtheta_i,
        dphi_i,
        dtheta_j,
        dphi_j
    ]


#TIME SETTINGS


t_start = 0
t_end = 500

dt = 0.0001      #time step

t_eval = np.arange(
    t_start,
    t_end + dt,
    dt
)

#SOLVE SYSTEM

y0 = [
    theta_i0,
    phi_i0,
    theta_j0,
    phi_j0
]

def project(theta, phi):
    theta=max(theta,0.0)
    phi=max(phi,0.0)
    if theta+phi>1.0:
        s=theta+phi
        theta/=s
        phi/=s
    return theta,phi

Y=np.zeros((4,len(t_eval)))
Y[:,0]=y0

for k in range(1,len(t_eval)):
    sol=solve_ivp(
        system,
        (t_eval[k-1],t_eval[k]),
        Y[:,k-1],
        method='BDF',
        rtol=1e-8,
        atol=1e-10
    )
    y=sol.y[:,-1]
    y[0],y[1]=project(y[0],y[1])
    y[2],y[3]=project(y[2],y[3])
    Y[:,k]=y

t=t_eval
theta_i=Y[0]
phi_i=Y[1]
theta_j=Y[2]
phi_j=Y[3]

#THETA(t)

plt.figure(figsize=(10,6))

plt.plot(
    t,
    theta_i,
    linewidth=2,
    label=r'$\theta_i(t)$'
)

plt.plot(
    t,
    theta_j,
    linewidth=2,
    label=r'$\theta_j(t)$'
)

plt.xlabel(r'$\tau$')
plt.ylabel(r'$\theta$')

plt.title(
    f'Time evolution of theta (I={I})'
)

plt.grid(True)
plt.legend()

plt.show()
