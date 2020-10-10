"""
Author : Kailash Nagarajan 
Date : 07-Oct-2020
e-Mail : kailashnagarajan@gmail.com

"""


import numpy as np 
import matplotlib.pyplot as plt 
import math
from UKF_Estimation import UKF 
from parameters import parameters

"""
This main script, Generates Simulated data, Initializes Parameters, Animation
and runs UKF on the simulated data.
"""

"""
Parameters for Simulation, Noise Parameters, UKF Parameters.
"""

params = parameters.read_params()

INPUT_NOISE = np.diag([0.1,np.deg2rad(30.0)]) ** 2 # Input Process Noise 
SENSOR_NOISE = np.diag([0.1,0.1]) ** 2 # Sensor Noise 


Q = np.diag([
    0.1,  # variance of location on x-axis
    0.1,  # variance of location on y-axis
    np.deg2rad(1.0),  # variance of yaw angle
    1.0  # variance of velocity
]) ** 2  # predict state covariance

R = np.diag([1.0, 1.0]) ** 2  # Observation x,y position covariance

DT = params.DT # Time Interval 

SIM_TIME = params.SIM_TIME # Simulation Time 

ALPHA = params.ALPHA # Alpha Parameter
BETA = params.BETA # Beta Parameter 
KAPPA = params.KAPPA # Kappa Parameter 


ukf = UKF() # UKF Object 


def calc_input():

    """
    This Function calculates the control input array 'u'. Velocity and yaw rate 

    Returns : 
    
    u - Control array - Array (2,)

    """

    v = 1.0 # Velocity 

    yaw_rate = 0.1 # Yaw Rate 

    u = np.array([[v,yaw_rate]]).T # Control Input Array 

    return u 

def observation_model(x):
    
    """
    This function takes the state and returns an 
    observation matrix based on the Observation Model.

    Parameters : 
    
    x - state array - Array - (4,)
    
    Returns : 

    z - observation array - Array - (2,)

    """
    # Observation model  

    H = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ]) 


    # Observation Equation
    
    z = H @ x

    return z 

def motion_model(x,u):

    """
    This function takes the state and control at time - t  
    as the input and returns the next state at - t+1  
    based on a motion model (A and B - State Matrices)

    Parameters :

    x - state array - Array - (4,)
    u - control array - Array - (2,)

    Returns : 

    x - Next state - Array (4,)

    """

    # State Matrix

    A = np.array([[1.0, 0, 0, 0], 
                  [0, 1.0, 0, 0], 
                  [0, 0, 1.0, 0], 
                  [0, 0, 0, 0]])

    # Control Matrix

    B = np.array([[DT * math.cos(x[2]), 0], 
                  [DT * math.sin(x[2]), 0], 
                  [0.0, DT], 
                  [1.0, 0.0]])

    # State Equation 

    x = A @ x + B @ u

    return x


def observation(xTrue,xd,u):

    """
    Generates true states and observed states and dead reckoning based on 
    the motion model, observation model, input noise and sensor noise.

    Parameters : 
    xTrue : True States - Array - (4,)
    xd : DeadReckoning States - Array - (4,)
    u : Control Input - Array (2,)

    Returns : 
    xTrue : Next True States - Array - (4,)
    z : Next Observed States - Array - (4,)
    xd : Next Deadreckoning States - Array - (4,)
    ud : Next Deadreckoning Control Inputs - Array - (2,)

    """
    
    xTrue = motion_model(xTrue,u) # True states 

    z = observation_model(xTrue) + SENSOR_NOISE @ np.random.randn(2,1) # Observed States 

    ud = u + INPUT_NOISE @ np.random.randn(2,1) # Dead Reckoning control input

    xd = motion_model(xd,ud) # Dead Reckoning states 

    return xTrue, z, xd, ud


def setup_parameters(N):

    """
    This function generates parameters for UKF, gamma, wm, wc

    Parameters : 
    N - Number of Dimensions - integer val. 

    Returns : 
    wm - Weight Matrix (Mean)
    wc - Weight Matrix (Covarince)
    gamma - Gamma Parameter (sqrt(N+Kappa))

    """

    lam = ALPHA ** 2 * (N + KAPPA) - N # Lambda 

    wm = [lam / (lam+N)] # mean weight matrix

    wc = [(lam / (lam+N)) + (1-ALPHA**2 + BETA)] # mean covariance matrix

    for i in range(2*N):
        wm.append(1.0/(2 * (N+lam))) 
        wc.append(1.0/(2 * (N+lam)))

    gamma = math.sqrt(N+lam) # gamma parameter

    wm = np.array([wm])
    wc = np.array([wc])

    return wm, wc, gamma


if __name__ == '__main__':
    
    time = 0.0 # Initial time

    nx = 4 # Number of states 

    xTrue = np.zeros((nx, 1)) # Intialized True States 

    x_est = np.zeros((nx,1)) # Intialized Estimated States 

    P_est = np.eye(nx) # Initialized Co-variance

    xDR = np.zeros((nx, 1)) # Intialized Dead Reckoning states

    wm,wc,gamma = setup_parameters(nx) # Intializing the UKF Parameters

    hxTrue = xTrue # Intialized True States Array

    hxDR = xTrue # Intialized Dead Reckoning States Array 

    hz = np.zeros((2, 1)) # Intialized Observed states 

    hxest = x_est # Intialized Estimated states array

    

    while SIM_TIME >= time :

        time += DT 

        

        u = calc_input() # Calculates Control Input Array 

        xTrue, z, xDR, ud = observation(xTrue, xDR, u) 
        # Observation Values (True and Dead Reckoning)


        xPred, PPred = ukf.prediction_step(x_est,P_est,wm,wc,gamma,motion_model,u,Q)
        # Prediction Step of UKF. Estimates xPred and PPred. 


        x_est, P_est   = ukf.correction_step(xPred,PPred,gamma,wc,wm,R,z,observation_model)
        # Correction Step of UKF. Final Estimated value X_est and P_est


        hxest = np.hstack((hxest, x_est)) # Estimated States

        hxDR = np.hstack((hxDR, xDR)) # Dead Reckoning States

        hxTrue = np.hstack((hxTrue, xTrue)) # True States 

        hz = np.hstack((hz, z)) # Observed States

        # Plotting The Trajectory 

        plt.cla()

        plt.plot(hz[0, :], hz[1, :], ".g") # Plots Observed Trajectory.

        # Plots True Trajectory 
        plt.plot(np.array(hxTrue[0, :]).flatten(),
                    np.array(hxTrue[1, :]).flatten(), "b", 
                    marker ='x' ,label='Sensor Reading')
        
        # Plot Dead Reckoning Trajectory
        plt.plot(np.array(hxDR[0, :]).flatten(),
                    np.array(hxDR[1, :]).flatten(), "-k", 
                    label = 'Dead Reckoning')
        
        # Plot Estimated Trajectory
        plt.plot(np.array(hxest[0, :]).flatten(),
                    np.array(hxest[1, :]).flatten(), "-r", 
                    label='Estimated Trajectory(UKF)')

        plt.xlabel("Position X[m]")
        plt.ylabel("Position Y[m]")

        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
           ncol=2, mode="expand", borderaxespad=0.)


        
        plt.axis("equal")

        plt.grid(True)

        plt.pause(0.001)








