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
from Basilisk.fswAlgorithms import attRefCorrection
from Basilisk.simulation import extForceTorque
from Basilisk.simulation import simpleNav, planetEphemeris
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


RE = 149.78e6 * 1000
RM = 228e6 * 1000
PHI = 44.267551176032


class InterplanetaryTransfer3DOFSimulator(SimulationBaseClass.SimBaseClass):

    def __init__(self, render_mode=None):
        super(InterplanetaryTransfer3DOFSimulator, self).__init__()

        self.render_mode = render_mode

        self.simTaskName = "simTask"
        self.simProcessName = "simProcess"

        self.simulationTime = 0
        self.thrustStopTime = 0

        self.dynProcess = self.CreateNewProcess(self.simProcessName)
        self.simulationTimeStep = macros.sec2nano(1 * 7 * 24 * 60 * 60)
        self.dynProcess.addTask(self.CreateNewTask(self.simTaskName, self.simulationTimeStep))

        self.gravBodyEphem = planetEphemeris.PlanetEphemeris()
        self.gravBodyEphem.ModelTag = 'planetEphemeris'
        self.AddModelToTask(self.simTaskName, self.gravBodyEphem)
        self.gravBodyEphem.setPlanetNames(planetEphemeris.StringVector(["mars", "earth"]))

        self.oeEarth = planetEphemeris.ClassicElements()
        self.oeEarth.a = RE
        self.oeEarth.e = 0.0001
        self.oeEarth.i = 0 * macros.D2R
        self.oeEarth.Omega = 0 * macros.D2R
        self.oeEarth.omega = 0 * macros.D2R
        self.oeEarth.f = 270 * macros.D2R

        self.oeMars = planetEphemeris.ClassicElements()
        self.oeMars.a = RM
        self.oeMars.e = 0.0001
        self.oeMars.i = 0 * macros.D2R
        self.oeMars.Omega = 0 * macros.D2R
        self.oeMars.omega = 0 * macros.D2R
        self.oeMars.f = (270 + PHI) * macros.D2R

        self.gravBodyEphem.planetElements = planetEphemeris.classicElementVector([self.oeMars, self.oeEarth])
        self.gravBodyEphem.rightAscension = planetEphemeris.DoubleVector([0.0 * macros.D2R, 0.0 * macros.D2R])
        self.gravBodyEphem.declination = planetEphemeris.DoubleVector([0.0 * macros.D2R, 0.0 * macros.D2R])
        self.gravBodyEphem.lst0 = planetEphemeris.DoubleVector([0.0 * macros.D2R, 0.0 * macros.D2R])
        self.gravBodyEphem.rotRate = planetEphemeris.DoubleVector(
            [360 * macros.D2R / (24.6229 * 3600.), 360 * macros.D2R / (24. * 3600.)])

        self.gravFactory = simIncludeGravBody.gravBodyFactory()
        self.earth = self.gravFactory.createEarth()
        self.earth.planetBodyInMsg.subscribeTo(self.gravBodyEphem.planetOutMsgs[1])
        self.mars = self.gravFactory.createMarsBarycenter()
        self.mars.planetBodyInMsg.subscribeTo(self.gravBodyEphem.planetOutMsgs[0])
        self.sun = self.gravFactory.createSun()
        self.sun.isCentralBody = True

        self.scObject = spacecraft.Spacecraft()
        self.scObject.ModelTag = "bsk-Sat"
        self.AddModelToTask(self.simTaskName, self.scObject)
        self.gravFactory.addBodiesTo(self.scObject)

        self.oe = orbitalMotion.ClassicElements()
        self.oe.a = RE
        self.oe.e = 0.0001
        self.oe.i = 0 * macros.D2R
        self.oe.Omega = 0 * macros.D2R
        self.oe.omega = 0 * macros.D2R
        self.oe.f = 270.01 * macros.D2R
        r_S_N, v_S_N = orbitalMotion.elem2rv(self.sun.mu, self.oe)
        self.scObject.hub.r_CN_NInit = r_S_N
        self.scObject.hub.v_CN_NInit = v_S_N


        samplingTime = self.simulationTimeStep
        self.scRec = self.scObject.scStateOutMsg.recorder(samplingTime)
        self.marsRec = self.gravBodyEphem.planetOutMsgs[0].recorder(samplingTime)
        self.earthRec = self.gravBodyEphem.planetOutMsgs[1].recorder(samplingTime)
        self.AddModelToTask(self.simTaskName, self.scRec)
        self.AddModelToTask(self.simTaskName, self.marsRec)
        self.AddModelToTask(self.simTaskName, self.earthRec)

        if self.render_mode:
            viz = vizSupport.enableUnityVisualization(
                self, 
                self.simTaskName, 
                self.scObject,
                oscOrbitColorList=[vizSupport.toRGBA255('yellow')],
                trueOrbitColorList=[vizSupport.toRGBA255('turquoise')],
                saveFile='-'.join(__file__.split('/')[-3:])
            )
            vizSupport.setActuatorGuiSetting(viz, showThrusterLabels=True)
            viz.settings.mainCameraTarget = 'sun'
            viz.settings.trueTrajectoryLinesOn = 1

        self.InitializeSimulation()

    def init(self):
        r_S_N, v_S_N = orbitalMotion.elem2rv(self.sun.mu, self.oe)
        r_M_N, v_M_N = orbitalMotion.elem2rv(self.sun.mu, self.oeMars)
        return {
            'r_S_N': r_S_N,
            'v_S_N': v_S_N,
            'r_M_N': r_M_N,
            'v_M_N': v_M_N
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
        r_M_N = self.marsRec.PositionVector[-1]
        v_M_N = self.marsRec.VelocityVector[-1]
        return {
            'r_S_N': r_S_N,
            'v_S_N': v_S_N,
            'r_M_N': r_M_N,
            'v_M_N': v_M_N
        }








