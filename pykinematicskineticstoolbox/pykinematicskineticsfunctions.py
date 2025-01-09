# Copyright (C) 2021 NTNU
# MIT License. See the LICENSE.md file for details
# Author: John Martin Kleven Godø <john.martin.godo@ntnu.no;
# john.martin.kleven.godo@gmail.com>

import numpy as np

# Kinematics functions

def Rln():
	Rlb = np.zeros((3,3))
	Rlb[0, :] = [-1, 0, 0]
	Rlb[1, :] = [0, 1, 0]
	Rlb[2, :] = [0, 0, -1]
	return Rlb

def Tln():
	Tlb = np.zeros((3,3))
	Tlb[0, :] = [-1, 0, 0]
	Tlb[1, :] = [0, 1, 0]
	Tlb[2, :] = [0, 0, -1]
	return Tlb

def Rzyx(phi, theta, psi):
	Rbn 		= np.zeros((3,3))
	Rbn[0,0] 	= np.cos(psi)*np.cos(theta)
	Rbn[0,1] 	= -np.sin(psi)*np.cos(phi) + np.cos(psi)*np.sin(theta)*np.sin(phi)
	Rbn[0,2] 	= np.sin(psi)*np.sin(phi) + np.cos(psi)*np.cos(phi)*np.sin(theta)
	Rbn[1,0] 	= np.sin(psi)*np.cos(theta)
	Rbn[1,1] 	= np.cos(psi)*np.cos(phi) + np.sin(phi)*np.sin(theta)*np.sin(psi)
	Rbn[1,2] 	= -np.cos(psi)*np.sin(phi) + np.sin(theta)*np.sin(psi)*np.cos(phi)
	Rbn[2,0] 	= -np.sin(theta)
	Rbn[2,1] 	= np.cos(theta)*np.sin(phi)
	Rbn[2,2] 	= np.cos(theta)*np.cos(phi)

	return Rbn

def Smtrx(vector):
	lambda_1 	= vector[0]
	lambda_2	= vector[1]
	lambda_3 	= vector[2]

	S 		= np.zeros((3,3))
	S[0,0] 	= 0
	S[0,1] 	= -lambda_3
	S[0,2] 	= lambda_2
	S[1,0] 	= lambda_3
	S[1,1] 	= 0
	S[1,2] 	= -lambda_1
	S[2,0]	= -lambda_2
	S[2,1] 	= lambda_1
	S[2,2] 	= 0

	return S

def R_axis_theta(k_in, theta):
	k = k_in/np.sqrt(np.sum(k_in**2))
	R = np.identity(3) + np.sin(theta)*Smtrx(k) + (1-np.cos(theta))*np.dot(Smtrx(k), Smtrx(k))
	return R


# Kinetics functions

def calculate_Ig(m, r_xx_CG):
	''' Inertia matrix about CG. m denotes vessel mass, r_xx_CG is a (3x3)
	array denoting all radii of inertia about CG.'''
	Ig 					= np.zeros((3, 3))
	for i in range(3):
		for j in range(3):
			Ig[i, j] 		= m*r_xx_CG[i, j]**2
	return Ig

def calculate_Ib(m, r_xx_CG, rg_b):
	''' Inertia matrix about the origin of the vessel coordinate system.
	* m: vessel mass
	* r_xx_CG: (3x3) matrix of radii of inertia about CG
	* rg_b: position of CG expressed in BODY coordinates. Ref. Fossen 2011, page 50 (1)'''
	Ig 					= calculate_Ig(m, r_xx_CG)
	Ib 					= Ig - m*Smtrx(rg_b)**2
	return Ib

def calculate_MRB(m, r_xx_CG, rg_b):
	''' Calculate M matrix for a vessel with center of gravity at rg_b from CO, defined in BODY coordinates.
	rg_b denotes the position of COG expressed in BODY coordinates ("vector to b expressed in BODY coordinates". See Fossen 2011 page 51)
	'''
	Ib 						= calculate_Ib(m, r_xx_CG, rg_b)
	MRB 					= np.zeros((6, 6))
	MRB[0:3, 0:3] 			= m*np.identity(3)
	MRB[3:7, 0:3]			= m*Smtrx(rg_b)
	MRB[0:3, 3:7]			= -m*Smtrx(rg_b)
	MRB[3:7, 3:7] 			= Ib
	return MRB, Ib

# def M_matrix_m_radiiOfGyration(m, r44, r55, r66, r45 = 0, r46 = 0, r56 = 0, rg_b = np.zeros(3)):
# 	''' Calculate M matrix for a vessel with center of gravity at rg_b from CO, defined in body coordinates.
# 	NB: Not tested for rg_b != 0 and off-diagonal radii of gyration != 0
# 	Update: Found a bug when rg_b is not zero. Then there is no automatic update of off-diagonal terms in Ib. M_matrix_m_COG should then
# 	be used instead'''
# 	M_matrix 	= np.zeros((6, 6))
# 	Ib 			= np.zeros((3,3))
#
# 	Ib[0, 0] 	= m*r44**2
# 	Ib[1, 1] 	= m*r55**2
# 	Ib[2, 2] 	= m*r66**2
# 	Ib[0, 1] 	= -m*r45**2
# 	Ib[0, 2]   	= -m*r46**2
# 	Ib[1, 2] 	= -m*r56**2
# 	Ib[1, 0] 	= Ib[0, 1]
# 	Ib[2, 0] 	= Ib[0, 2]
# 	Ib[2, 1] 	= Ib[1, 2]
#
#
# 	M_matrix[0:3, 0:3] 		= m*np.identity(3)
# 	M_matrix[3:7, 0:3]		= m*Smtrx(rg_b)
# 	M_matrix[0:3, 3:7]		= -m*Smtrx(rg_b)
# 	M_matrix[3:7, 3:7] 		= Ib
#
# 	return M_matrix, Ib

def calculate_CRB(m, Ib, nu2, rg_b):
	''' Calculate coriolis matrix for a vessel with center of gravity at
	rg_b from CO, defined in BODY coordinates.
	* m: Mass
	* Ib: Inertia matrix about the BODY coordinate system origin
	* nu2: vector of BODY frame angular velocities in roll, pitch and yaw respectively
	* rg_b: vector to CG in BODY coordinate system '''
	CRB 			= np.zeros((6, 6))
	CRB[0:3, 0:3]	= m*Smtrx(nu2)
	CRB[0:3, 3:7] 	= -m*np.dot(Smtrx(nu2), Smtrx(rg_b))
	CRB[3:7, 0:3] 	= m*np.dot(Smtrx(rg_b), Smtrx(nu2))
	#CRB[0:3, 3:7] 	= -m*Smtrx(nu2)*Smtrx(rg_b)
	#CRB[3:7, 0:3] 	= m*Smtrx(rg_b)*Smtrx(nu2)
	CRB[3:7, 3:7] 	= -Smtrx(np.dot(Ib, nu2))

	return CRB

def calculate_TTheta(euler_angles):
	'''Calculate the transformation matrix from body frame angular velocities to
	time derivatives of Euler angles'''
	phi  	= euler_angles[0]
	theta 	= euler_angles[1]
	psi 	= euler_angles[2]

	T 		= np.zeros((3, 3))
	T[0, 0] = 1
	T[0, 1] = np.sin(phi)*np.tan(theta)
	T[0, 2] = np.cos(phi)*np.tan(theta)
	T[1, 1] = np.cos(phi)
	T[1, 2] = -np.sin(phi)
	T[2, 1] = np.sin(phi)/np.cos(theta)
	T[2, 2] = np.cos(phi)/np.cos(theta)

	return T


# References
# (1) 2011, Fossen, T.I. Handbook of Marine Craft Hydrodynamics and Motion Control
