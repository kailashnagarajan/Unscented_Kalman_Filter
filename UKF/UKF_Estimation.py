"""
Author : Kailash Nagarajan 
Date : 07-Oct-2020
e-Mail : kailashnagarajan@gmail.com

"""

import numpy as np 
import math
from scipy.linalg import sqrtm

"""
Main UKF Class which handles calculating sigma points, 
Propagate sigma points over nonlinear functions,
Predict States, 
Calculate Co-variance between states and observation,
Corrected States 
"""

class UKF : 

	def __init__(self):
		pass 

	def calculate_sigma_points(self,x_est, P_est, gamma):

		"""
		This function calculates the sigma points of the states, covariance and gamma.

		Parameters: 

		x_est : Estimated States.
		P_est : Estimated Covariance. 
		gamma : Gamma Value.

		Returns :

		sigma : Sigma Points - Array - (4,)

		"""

		sigma = x_est 

		n = len(x_est[:,0])

		L = sqrtm(P_est)

		for i in range(n):
			sigma = np.hstack((sigma,x_est+gamma * L[:,i:i+1]))

		for i in range(n):
			sigma = np.hstack((sigma,x_est-gamma * L[:,i:i+1]))

		return sigma 

	def predict_sigma_motion_model(self,sigma,u,motion_model):
		"""
		This function propogates the sigma points through 
		the non linear motion model

		Parameters : 
		
		sigma : Sigma values - Array - (4,)
		u : Control Array - Array - (2,)
		motion_model : Function which returns the next state when 
					   the current state is entered  - Function

		Returns : 

		sigma : Sigma values - Array - (4,)

		"""

		for i in range(sigma.shape[1]):
			sigma[:, i:i+1] = motion_model(sigma[:, i:i+1],u) 
			# Propogating the sigma points through the non-linear motion model

		return sigma 

	def prediction_step(self,x_est,P_est,wm,wc,gamma,motion_model,u,Q):

		"""
		This function predicts the set of states based on the motion model 
		and the sigma points. 

		Parameters : 
		
		x_est : Estimated States - Array - (4,)
		P_est : Estimated State Co-Variance Matrix - Array - (4,4)
		wm : Weights for Mean matrix - List - (2)
		wc : Weights for Co-variance matrix - (2)
		gamma : Gamma parameter - floatval 
		motion_model : Function that takes the current state 
					   and returns the next state. - Function 
		u : Control array : Array : (2,)
		Q : Process Noise Matrix : Array : (4,4)
		
		Returns : 
		xPred : Predicted State : Array : (4,)
		PPred : Predicted State Co-Variance : Array : (4,4)

		"""

		sigma = self.calculate_sigma_points(x_est,P_est,gamma) # Calculates Sigma

		sigma = self.predict_sigma_motion_model(sigma,u,motion_model) # Propogates Sigma 

		xPred = (wm @ sigma.T).T # Predicted States

		nsigma = sigma.shape[1] # Number of sigma values

		d = sigma - xPred[0:sigma.shape[0]] 
		# distance between the predicted state and sigma

		for i in range(nsigma):
			PPred = Q + wc[0,i] * d[:,i:i+1] @ d[:,i:i+1].T
			# Predicted State Co-Variance

		return xPred, PPred 

	def predict_sigma_measurement_model(self,sigma,measurement_model):
		"""
		This function propogates the sigma points through 
		the non linear measurement model 

		Parameters : 

		sigma : sigma points - Array - (4,)
		
		measurement_model : This function takes the state and 
							returns the observation - Array - (4,)
		Returns : 

		sigma : Propogated sigma points through the non-linear 
				measurement model. - Array - (2,)

		"""
		for i in range(sigma.shape[1]):
			sigma[0:2, i] = measurement_model(sigma[:, i])
			# Propogating the sigma points through the non-linear measurement model


		sigma = sigma [0:2, :]

		return sigma 

	def calculate_covar_pxz(self,sigma,x,z_sigma,zb,wc):
		"""
		Calculates the covariance matrix between states and observation 

		Parameters : 

		sigma : sigma values - Array - (4,)
		x : State Matrix - Array - (4,)
		z_sigma : Propogated Sigma points through measurement model - (2,)
		zb : Observed values propgated through the mean. - (2,)
		wc : Weight Matrix (Co-Variance) - List - (2)

		Returns : 

		Pxz : Co-Variance matrix between state and co-variance - Array - (4,4)

		"""
		
		nSigma = sigma.shape[1] # Number of Sigma Values 
		
		dx = sigma - x # Difference between sigma and states
		
		dz = z_sigma - zb[0:2] # Difference between z_sigma and observations

		Pxz = np.zeros((dx.shape[0],dz.shape[0])) 
		# Co-variance between states and observations

		for i in range(nSigma):
			Pxz = Pxz + wc[0,i] * dx[:,i:i+1] @ dz[:,i:i+1].T 
			# Weighted Co-Variance between x and z 

		return Pxz


	def correction_step(self,xPred,PPred,gamma,wc,wm,R,z,measurement_model):

		"""
		Correction step of UKF, takes the predicted states. 

		Parameters : 
		
		xPred : Predicted States - Array - (4,)
		PPred : Predicted State Co-Variance - (4,4)
		gamma : Gamma Parameter - floatval
		wc : Weight Matrix (Co-Variance) - List - (2)
		wm : Weight Matrix (Mean) - List - (2)
		R : Measurement Noise Co-Variance - Array - (2,2)
		z : Observation - Array - (2,)
		measurement_model : This function takes the current observation 
							and propogates it through time. Function 
		
		Returns : 

		x_est : Estimated States - Array - (4,)
		P_est : Estimated State Co-Variance - Array (4,4) 

		"""

		zpred = measurement_model(xPred) # Predicted Observations

		y = z - zpred # Difference between actual and predicted observations

		sigma = self.calculate_sigma_points(xPred,PPred,gamma) 
		# sigma points calculation

		zb = (wm @ sigma.T).T # Calculating the observed states from sigma and wm

		z_sigma = self.predict_sigma_measurement_model(sigma,measurement_model)
		# sigma points propogated through non-linear measurement model.

		nsigma = z_sigma.shape[1] # number of sigma points. 

		d = z_sigma - zb[0:z_sigma.shape[0]] 

		Pz = R  # Covariance initialization.

		for i in range(nsigma):
			Pz = Pz + wc[0,i] * d[:, i:i+1] @ d[:, i:i+1].T
			# calculating Pz using co-variance weight matrix

		Pxz = self.calculate_covar_pxz(sigma,xPred,z_sigma,zb,wc) 
		# Calculating Pxz (Co-variance between x and z)

		
		K_k = Pxz @ np.linalg.inv(Pz) # Kalman Gain

		xest = xPred + K_k @ y # Estimated State  

		Pest = PPred - K_k @ Pz @ K_k.T # Estimated State Co-variance

		return xest, Pest

















