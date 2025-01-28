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


class HohmannTransfer6DOFSimulator(SimulationBaseClass.SimBaseClass):

    def __init__(self, render_mode=None):
        super(HohmannTransfer6DOFSimulator, self).__init__()

        self.render_mode = render_mode

        self.simTaskName = "simTask"
        self.simProcessName = "simProcess"
        dynProcess = self.CreateNewProcess(self.simProcessName)
        self.simulationTimeStep = macros.sec2nano(2.0)
        self.simulationTime = 0
        self.thrustStopTime = 0
        dynProcess.addTask(self.CreateNewTask(self.simTaskName, self.simulationTimeStep))

        self.scObject = spacecraft.Spacecraft()
        self.scObject.ModelTag = "bsk-Sat"
        I = [900., 0., 0.,
                0., 800., 0.,
                0., 0., 600.]
        self.scObject.hub.mHub = 750.0  # kg - spacecraft mass
        self.scObject.hub.IHubPntBc_B = unitTestSupport.np2EigenMatrix3d(I)
        self.AddModelToTask(self.simTaskName, self.scObject)

        self.gravFactory = simIncludeGravBody.gravBodyFactory()
        self.earth = self.gravFactory.createEarth()
        self.earth.isCentralBody = True  # ensure this is the central gravitational body
        self.gravFactory.addBodiesTo(self.scObject)

        oe = orbitalMotion.ClassicElements()
        oe.a = R1
        oe.e = 0.0001
        oe.i = 0.0 * macros.D2R
        oe.Omega = 48.2 * macros.D2R
        oe.omega = 347.8 * macros.D2R
        oe.f = 85.3 * macros.D2R
        rN, vN = orbitalMotion.elem2rv(self.earth.mu, oe)
        self.scObject.hub.r_CN_NInit = rN  # m - r_CN_N
        self.scObject.hub.v_CN_NInit = vN  # m - v_CN_N
        self.scObject.hub.sigma_BNInit = [[0.0], [0.0], [0.0]]  # sigma_BN_B
        self.scObject.hub.omega_BN_BInit = [[0.0], [0.03], [0.01]]  # rad/s - omega_BN_B

        self.extFTObject = extForceTorque.ExtForceTorque()
        self.extFTObject.ModelTag = "externalTorque"
        self.scObject.addDynamicEffector(self.extFTObject)
        self.AddModelToTask(self.simTaskName, self.extFTObject)

        self.sNavObject = simpleNav.SimpleNav()
        self.sNavObject.ModelTag = "SimpleNavigation"
        self.AddModelToTask(self.simTaskName, self.sNavObject)
        self.sNavObject.scStateInMsg.subscribeTo(self.scObject.scStateOutMsg)

        self.thrusterSet = thrusterStateEffector.ThrusterStateEffector()
        self.AddModelToTask(self.simTaskName, self.thrusterSet)

        self.integratorObject = svIntegrators.svIntegratorRKF45(self.scObject)
        self.scObject.setIntegrator(self.integratorObject)

        self.thFactory = simIncludeThruster.thrusterFactory()

        self.MaxThrust = 3000.0
        location = [[0, 0, 1]]
        direction = [[0, 0, 1]]
        for pos_B, dir_B in zip(location, direction):
            self.thFactory.create('Blank_Thruster', pos_B, dir_B, cutoffFrequency=.1, MaxThrust=self.MaxThrust,
                                areaNozzle=0.046759, steadyIsp=318.0)

        self.thrModelTag = "GTOThrusterDynamics"
        self.thFactory.addToSpacecraft(self.thrModelTag, self.thrusterSet, self.scObject)

        self.ThrOnTimeMsgData = messaging.THRArrayOnTimeCmdMsgPayload()
        self.ThrOnTimeMsgData.OnTimeRequest = [0]
        self.thrOnTimeMsg = messaging.THRArrayOnTimeCmdMsg().write(self.ThrOnTimeMsgData)

        self.attRefMsgData = messaging.AttRefMsgPayload()
        self.attRefMsg = messaging.AttRefMsg()

        self.attError = attTrackingError.attTrackingError()
        self.attError.ModelTag = "attErrorVelocityPoint"
        self.AddModelToTask(self.simTaskName, self.attError)
        self.attError.sigma_R0R = [np.tan(np.pi/8), 0,  0]
        self.attError.attRefInMsg.subscribeTo(self.attRefMsg)
        self.attError.attNavInMsg.subscribeTo(self.sNavObject.attOutMsg)

        self.mrpControl = mrpFeedback.mrpFeedback()
        self.mrpControl.ModelTag = "mrpFeedback"
        self.AddModelToTask(self.simTaskName, self.mrpControl)
        self.mrpControl.K = 3.5
        self.mrpControl.Ki = -1.0  # make value negative to turn off integral feedback
        self.mrpControl.P = 30.0
        self.mrpControl.integralLimit = 2. / self.mrpControl.Ki * 0.1

        self.extFTObject.cmdTorqueInMsg.subscribeTo(self.mrpControl.cmdTorqueOutMsg)

        samplingTime = self.simulationTimeStep
        self.mrpLog = self.mrpControl.cmdTorqueOutMsg.recorder(samplingTime)
        self.attGuidLog = self.attRefMsg.recorder(samplingTime)
        self.attErrLog = self.attError.attGuidOutMsg.recorder(samplingTime)
        self.snAttLog = self.sNavObject.attOutMsg.recorder(samplingTime)
        self.snTransLog = self.sNavObject.transOutMsg.recorder(samplingTime)
        self.dataRec = self.scObject.scStateOutMsg.recorder(samplingTime)
        self.thrCmdRec0 = self.thrusterSet.thrusterOutMsgs[0].recorder()
        # self.thrCmdRec1 = self.thrusterSet.thrusterOutMsgs[1].recorder()
        self.AddModelToTask(self.simTaskName, self.thrCmdRec0)
        # self.AddModelToTask(self.simTaskName, self.thrCmdRec1)
        self.AddModelToTask(self.simTaskName, self.dataRec)
        self.AddModelToTask(self.simTaskName, self.mrpLog)
        self.AddModelToTask(self.simTaskName, self.attErrLog)
        self.AddModelToTask(self.simTaskName, self.snAttLog)
        self.AddModelToTask(self.simTaskName, self.snTransLog)
        self.AddModelToTask(self.simTaskName, self.attGuidLog)

        # create the FSW vehicle configuration message
        self.vehicleConfigOut = messaging.VehicleConfigMsgPayload()
        self.vehicleConfigOut.ISCPntB_B = I  # use the same inertia in the FSW algorithm as in the simulation
        self.vcMsg = messaging.VehicleConfigMsg().write(self.vehicleConfigOut)

        # connect messages
        self.sNavObject.scStateInMsg.subscribeTo(self.scObject.scStateOutMsg)
        self.attError.attNavInMsg.subscribeTo(self.sNavObject.attOutMsg)
        self.attError.attRefInMsg.subscribeTo(self.attRefMsg)
        self.mrpControl.guidInMsg.subscribeTo(self.attError.attGuidOutMsg)
        self.mrpControl.vehConfigInMsg.subscribeTo(self.vcMsg)
        self.thrusterSet.cmdsInMsg.subscribeTo(self.thrOnTimeMsg)

        if self.render_mode:
            viz = vizSupport.enableUnityVisualization(self, self.simTaskName,  self.scObject
                                                        , saveFile='-'.join(__file__.split('/')[-3:])
                                                        , thrEffectorList=self.thrusterSet
                                                        , thrColors=vizSupport.toRGBA255("red")
                                                        )
            vizSupport.setActuatorGuiSetting(viz, showThrusterLabels=True)
            viz.settings.mainCameraTarget = 'earth'
            viz.settings.trueTrajectoryLinesOn = 1


        self.InitializeSimulation()
        self.SetProgressBar(True)

    def init(self):
        r_S_N, v_S_N = orbitalMotion.elem2rv(self.sun.mu, self.oe)
        sigma_S_N = np.array(self.scObject.hub.sigma_BNInit).reshape(-1)
        omega_S_N = np.array(self.scObject.hub.omega_BN_BInit).reshape(-1)
        return {
            'r_S_N': r_S_N,
            'v_S_N': v_S_N,
            'sigma_S_N': sigma_S_N,
            'omega_S_N': omega_S_N
        }


    def run(self, action):
        if self.thrustStopTime < self.simulationTime:
            t0Burn = action[0] / (self.MaxThrust / self.scObject.hub.mHub)
            self.ThrOnTimeMsgData.OnTimeRequest = [t0Burn]
            self.thrOnTimeMsg.write(self.ThrOnTimeMsgData, time=self.simulationTime)
            self.thrustStopTime = self.simulationTime + macros.sec2nano(t0Burn)
        self.attRefMsgData.sigma_RN = action[1:]
        self.attRefMsg.write(self.attRefMsgData)

        self.simulationTime += self.simulationTimeStep * 100
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
        return {
            'r_S_N': r_S_N,
            'v_S_N': v_S_N,
            'sigma_S_N': sigma_S_N,
            'omega_S_N': omega_S_N
        }







