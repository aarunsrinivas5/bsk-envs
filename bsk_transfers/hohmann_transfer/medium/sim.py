import math
import os

import matplotlib.pyplot as plt
import numpy as np
# The path to the location of Basilisk
# Used to get the location of supporting data.
from Basilisk import __path__
# import message declarations
from Basilisk.architecture import messaging
# import FSW Algorithm related support
from Basilisk.fswAlgorithms import attTrackingError
from Basilisk.fswAlgorithms import mrpFeedback
from Basilisk.fswAlgorithms import velocityPoint
from Basilisk.simulation import extForceTorque
from Basilisk.simulation import simpleNav
# import simulation related support
from Basilisk.simulation import spacecraft
from Basilisk.simulation import svIntegrators
from Basilisk.simulation import thrusterStateEffector
# import general simulation support files
from Basilisk.utilities import SimulationBaseClass
from Basilisk.utilities import macros
from Basilisk.utilities import orbitalMotion
from Basilisk.utilities import simIncludeGravBody
from Basilisk.utilities import simIncludeThruster
from Basilisk.utilities import unitTestSupport  # general support file with common unit test functions
# attempt to import vizard
from Basilisk.utilities import vizSupport



THRESHOLD = 1000 * 1000
R1 = 7000 * 1000
R2 = 42000 * 1000
RMIN = R1 - THRESHOLD
RMAX = R2 + THRESHOLD * 10


class HohmannTransfer3DOFSimulator(SimulationBaseClass.SimBaseClass):

    def __init__(self, render_mode=None):
        super(HohmannTransfer3DOFSimulator, self).__init__()

        self.render_mode = render_mode

        self.simTaskName = "simTask"
        self.simProcessName = "simProcess"
        dynProcess = self.CreateNewProcess(self.simProcessName)

        self.simulationTimeStep = macros.sec2nano(10)
        self.simulationTime = 0
        dynProcess.addTask(self.CreateNewTask(self.simTaskName, self.simulationTimeStep))

        self.scObject = spacecraft.Spacecraft()
        self.scObject.ModelTag = "bsk-Sat"
        self.AddModelToTask(self.simTaskName, self.scObject)

        self.gravFactory = simIncludeGravBody.gravBodyFactory()
        self.earth = self.gravFactory.createEarth()
        self.earth.isCentralBody = True
        self.gravFactory.addBodiesTo(self.scObject)

        oe = orbitalMotion.ClassicElements()
        oe.a = R1
        oe.e = 0.0001
        oe.i = 0.0 * macros.D2R
        oe.Omega = 48.2 * macros.D2R
        oe.omega = 347.8 * macros.D2R
        oe.f = 85.3 * macros.D2R
        r_S_N, v_S_N = orbitalMotion.elem2rv(self.earth.mu, oe)
        self.scObject.hub.r_CN_NInit = r_S_N 
        self.scObject.hub.v_CN_NInit = v_S_N

        samplingTime = self.simulationTimeStep
        self.dataLog = self.scObject.scStateOutMsg.recorder(samplingTime)
        self.AddModelToTask(self.simTaskName, self.dataLog)

        if self.render_mode:
            viz = vizSupport.enableUnityVisualization(self, self.simTaskName,  self.scObject, saveFile=__file__)
            vizSupport.setActuatorGuiSetting(viz, showThrusterLabels=True)
            viz.settings.mainCameraTarget = 'earth'
            viz.settings.trueTrajectoryLinesOn = 1

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









