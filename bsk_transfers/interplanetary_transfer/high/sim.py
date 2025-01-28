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


class InterplanetaryTransfer6DOFSimulator(SimulationBaseClass.SimBaseClass):

    def __init__(self, render_mode=None):
        super(InterplanetaryTransfer6DOFSimulator, self).__init__()

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
        I = [900., 0., 0.,
                0., 800., 0.,
                0., 0., 600.]
        self.scObject.hub.mHub = 750.0
        self.scObject.hub.IHubPntBc_B = unitTestSupport.np2EigenMatrix3d(I)
        self.AddModelToTask(self.simTaskName, self.scObject)
        self.gravFactory.addBodiesTo(self.scObject)

        oe = orbitalMotion.ClassicElements()
        oe.a = RE
        oe.e = 0.0001
        oe.i = 0 * macros.D2R
        oe.Omega = 0 * macros.D2R
        oe.omega = 0 * macros.D2R
        oe.f = 270.01 * macros.D2R
        r_S_N, v_S_N = orbitalMotion.elem2rv(self.sun.mu, oe)
        self.scObject.hub.r_CN_NInit = r_S_N
        self.scObject.hub.v_CN_NInit = v_S_N

        self.sNavObject = simpleNav.SimpleNav()
        self.sNavObject.ModelTag = "SimpleNavigation"
        self.AddModelToTask(self.simTaskName, self.sNavObject)
        self.sNavObject.scStateInMsg.subscribeTo(self.scObject.scStateOutMsg)

        self.attRefMsgData = messaging.AttRefMsgPayload()
        self.attRefMsg = messaging.AttRefMsg()

        self.attRefCor = attRefCorrection.attRefCorrection()
        self.attRefCor.ModelTag = "attRefCor"
        self.AddModelToTask(self.simTaskName, self.attRefCor)
        self.attRefCor.sigma_BcB = [np.tan(np.pi/8), 0, 0]
        self.attRefCor.attRefInMsg.subscribeTo(self.attRefMsg)
        self.scObject.attRefInMsg.subscribeTo(self.attRefCor.attRefOutMsg)

        self.thrusterSet = thrusterStateEffector.ThrusterStateEffector()
        self.AddModelToTask(self.simTaskName, self.thrusterSet)

        self.integratorObject = svIntegrators.svIntegratorRKF78(self.scObject)
        self.scObject.setIntegrator(self.integratorObject)

        self.thFactory = simIncludeThruster.thrusterFactory()

        self.MaxThrust = 3000.0
        location = [[0, 0, 1]]
        direction = [[0, 0, 1]]
        for pos_B, dir_B in zip(location, direction):
            self.thFactory.create('Blank_Thruster', pos_B, dir_B, cutoffFrequency=0.00001, MaxThrust=self.MaxThrust,
                                areaNozzle=0.046759, steadyIsp=318.0)

        thrModelTag = "GTOThrusterDynamics"
        self.thFactory.addToSpacecraft(thrModelTag, self.thrusterSet, self.scObject)

        self.ThrOnTimeMsgData = messaging.THRArrayOnTimeCmdMsgPayload()
        self.ThrOnTimeMsgData.OnTimeRequest = [0]
        self.thrOnTimeMsg = messaging.THRArrayOnTimeCmdMsg().write(self.ThrOnTimeMsgData)

        samplingTime = self.simulationTimeStep
        self.attGuidLog = self.attRefMsg.recorder(samplingTime)
        self.snTransLog = self.sNavObject.transOutMsg.recorder(samplingTime)
        self.dataRec = self.scObject.scStateOutMsg.recorder(samplingTime)
        self.thrCmdRec0 = self.thrusterSet.thrusterOutMsgs[0].recorder()
        self.scRec = self.scObject.scStateOutMsg.recorder(samplingTime)
        self.marsRec = self.gravBodyEphem.planetOutMsgs[0].recorder(samplingTime)
        self.earthRec = self.gravBodyEphem.planetOutMsgs[1].recorder(samplingTime)
        self.AddModelToTask(self.simTaskName, self.thrCmdRec0)
        self.AddModelToTask(self.simTaskName, self.dataRec)
        self.AddModelToTask(self.simTaskName, self.snTransLog)
        self.AddModelToTask(self.simTaskName, self.attGuidLog)
        self.AddModelToTask(self.simTaskName, self.scRec)
        self.AddModelToTask(self.simTaskName, self.marsRec)
        self.AddModelToTask(self.simTaskName, self.earthRec)

        self.sNavObject.scStateInMsg.subscribeTo(self.scObject.scStateOutMsg)
        self.thrusterSet.cmdsInMsg.subscribeTo(self.thrOnTimeMsg)

        if self.render_mode:
            viz = vizSupport.enableUnityVisualization(
                self, 
                self.simTaskName, 
                self.scObject,
                oscOrbitColorList=[vizSupport.toRGBA255('yellow')],
                trueOrbitColorList=[vizSupport.toRGBA255('turquoise')],
                thrEffectorList=self.thrusterSet,
                thrColors=vizSupport.toRGBA255("red"),
                saveFile=__file__
            )
            vizSupport.setActuatorGuiSetting(viz, showThrusterLabels=True)
            viz.settings.mainCameraTarget = 'sun'
            viz.settings.trueTrajectoryLinesOn = 1

        self.InitializeSimulation()

    def init(self):
        r_S_N, v_S_N = orbitalMotion.elem2rv(self.sun.mu, self.oe)
        r_M_N, v_M_N = orbitalMotion.elem2rv(self.sun.mu, self.oeMars)
        sigma_S_N = np.array(self.scObject.hub.sigma_BNInit).reshape(-1)
        omega_S_N = np.array(self.scObject.hub.omega_BN_BInit).reshape(-1)
        return {
            'r_S_N': r_S_N,
            'v_S_N': v_S_N,
            'r_M_N': r_M_N,
            'v_M_N': v_M_N,
            'sigma_S_N': sigma_S_N,
            'omega_S_N': omega_S_N
        }


    def run(self, action):
        if self.thrustStopTime <= self.simulationTime:
            t0Burn = action[0] / (self.MaxThrust / self.scObject.hub.mHub)
            self.ThrOnTimeMsgData.OnTimeRequest = [t0Burn]
            self.thrOnTimeMsg.write(self.ThrOnTimeMsgData, time=self.simulationTime)
            self.thrustStopTime = self.simulationTime + macros.sec2nano(t0Burn)
        self.attRefMsgData.sigma_RN = action[1:]
        self.attRefMsg.write(self.attRefMsgData)

        self.simulationTime += self.simulationTimeStep
        self.ConfigureStopTime(self.simulationTime)
        self.ExecuteSimulation()

        posRef = self.scObject.dynManager.getStateObject('hubPosition')
        velRef = self.scObject.dynManager.getStateObject('hubVelocity')
        sigmaRef = self.scObject.dynManager.getStateObject('hubSigma')
        omegaRef = self.scObject.dynManager.getStateObject('hubOmega')

        r_S_N = unitTestSupport.EigenVector3d2np(posRef.getState())
        v_S_N = unitTestSupport.EigenVector3d2np(velRef.getState())
        sigma_S_N = unitTestSupport.EigenVector3d2np(sigmaRef.getState())
        omega_S_N = unitTestSupport.EigenVector3d2np(omegaRef.getState())

        r_M_N = self.marsRec.PositionVector[-1]
        v_M_N = self.marsRec.VelocityVector[-1]

        return {
            'r_S_N': r_S_N,
            'v_S_N': v_S_N,
            'r_M_N': r_M_N,
            'v_M_N': v_M_N,
            'sigma_S_N': sigma_S_N,
            'omega_S_N': omega_S_N
        }



delta_v = 2936.7657449941544

sim = InterplanetaryTransfer6DOFSimulator(render_mode='human')

headings = np.load('headings_interplanetary.npy')
sim.run([delta_v, *headings[0]])
for i, heading in enumerate(headings[1:]):
    sim.run(np.concatenate(([0], heading)))







