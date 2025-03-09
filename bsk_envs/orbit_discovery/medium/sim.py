import os
import numpy as np
import matplotlib.pyplot as plt
from Basilisk.simulation import spacecraft
from Basilisk.utilities import (SimulationBaseClass, macros, simIncludeGravBody, vizSupport)
from Basilisk.utilities import orbitalMotion
from Basilisk.utilities import unitTestSupport


class OrbitDiscovery3DOFSimulator(SimulationBaseClass.SimBaseClass):

    def __init__(self, mu=4.463e5, radius=8000, render_mode=None):
        super(OrbitDiscovery3DOFSimulator, self).__init__()

        self.simTaskName = "simTask"
        self.simProcessName = "simProcess"
        dynProcess = self.CreateNewProcess(self.simProcessName)

        self.simulationTimeStep = macros.min2nano(5.0)
        self.simulationTime = 0
        dynProcess.addTask(self.CreateNewTask(self.simTaskName, self.simulationTimeStep))

        self.scObject = spacecraft.Spacecraft()
        self.scObject.ModelTag = "bsk-Sat"
        self.AddModelToTask(self.simTaskName, self.scObject)

        self.gravFactory = simIncludeGravBody.gravBodyFactory()
        self.gravBody = self.gravFactory.createCustomGravObject("body", mu, radEquator=radius)
        self.gravBody.isCentralBody = True
        self.gravFactory.addBodiesTo(self.scObject)

        oe = orbitalMotion.ClassicElements()
        oe.a = np.random.uniform(radius + 10 * 1000, 3 * radius)
        oe.e = 0 * np.random.uniform(0, 1)
        oe.i = 0 * np.random.uniform(0, 2 * np.pi)
        oe.Omega = 0 * np.random.uniform(0, 2 * np.pi)
        oe.omega = 0 * np.random.uniform(0, 2 * np.pi)
        oe.f = np.random.uniform(0, 2 * np.pi)
        r_S_N, v_S_N = orbitalMotion.elem2rv(self.gravBody.mu, oe)
        self.scObject.hub.r_CN_NInit = r_S_N 
        self.scObject.hub.v_CN_NInit = v_S_N

        samplingTime = self.simulationTimeStep
        self.dataLog = self.scObject.scStateOutMsg.recorder(samplingTime)
        self.AddModelToTask(self.simTaskName, self.dataLog)

        self.InitializeSimulation()

    def init(self):
        r_S_N = np.array(self.scObject.hub.r_CN_NInit).reshape(-1)
        v_S_N = np.array(self.scObject.hub.v_CN_NInit).reshape(-1)
        return {
            'r_S_N': r_S_N,
            'v_S_N': v_S_N
        }
    
    def run(self, action):
        posRef = self.scObject.dynManager.getStateObject('hubPosition')
        velRef = self.scObject.dynManager.getStateObject('hubVelocity')
        
        v_S_N = unitTestSupport.EigenVector3d2np(velRef.getState())
        v_S_N = v_S_N + action
        velRef.setState(unitTestSupport.np2EigenVectorXd(v_S_N))

        self.simulationTime += self.simulationTimeStep
        self.ConfigureStopTime(self.simulationTime)
        self.ExecuteSimulation()

        r_S_N = unitTestSupport.EigenVector3d2np(posRef.getState())
        v_S_N = unitTestSupport.EigenVector3d2np(velRef.getState())
        return {
            'r_S_N': r_S_N,
            'v_S_N': v_S_N
        }
    
