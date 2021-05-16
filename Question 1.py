import numpy as np
import openravepy as orpy
from openravepy import *
import time
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf

env = orpy.Environment()
env.SetViewer('qtcoin')
env.Load('osr_openrave/worlds/pick_and_place.env.xml')
def create_box(T, color = [0, 0.6, 0]):
  box = orpy.RaveCreateKinBody(env, '')
  box.SetName('box')
  box.InitFromBoxes(np.array([[0,0,0,0.035,0.03,0.005]]), True)
  g = box.GetLinks()[0].GetGeometries()[0]
  g.SetAmbientColor(color)
  g.SetDiffuseColor(color)
  box.SetTransform(T)
  env.Add(box,True)
  return box
T = np.eye(4)
container_center = np.array([0.4, 0.2, 0.195])
# Destination
T[:3, 3] = container_center + np.array([0, -0.5, 0])
destination0 = create_box(T, color = [0, 0, 0.6])
T[:3, 3] = container_center + np.array([0, -0.6, 0])
destination1 = create_box(T, color = [0, 0, 0.6])
# Generate random box positions
boxes = []
nbox_per_layer = 2
n_layer = 20
h = container_center[2]
for i in range(n_layer):
  nbox_current_layer = 0
  while nbox_current_layer < nbox_per_layer:
    theta = np.random.rand()*np.pi
    T[0, 0] = np.cos(theta)
    T[0, 1] = -np.sin(theta)
    T[1, 0] = np.sin(theta)
    T[1, 1] = np.cos(theta)
    T[0, 3] = container_center[0] + (np.random.rand()-0.5)*0.2
    T[1, 3] = container_center[1] + (np.random.rand()-0.5)*0.1
    T[2, 3] = h
    box = create_box(T)
    if env.CheckCollision(box):
      env.Remove(box)
    else:
      boxes.append(box)
      nbox_current_layer += 1
  h += 0.011

'''----------------------------------Initial Condition Setup Done-------------------------------------------------------------------------------------------------------------------'''

# Robot setup #
robot = env.GetRobots()[0]
manipulator = robot.SetActiveManipulator('gripper')
robot.SetActiveDOFs(manipulator.GetArmIndices())
manipprob = interfaces.BaseManipulation(robot)

np.set_printoptions(precision=6, suppress=True)

ikmodel = orpy.databases.inversekinematics.InverseKinematicsModel(robot, iktype=orpy.IkParameterization.Type.Transform6D)
if not ikmodel.load():
  ikmodel.autogenerate()

lmodel = databases.linkstatistics.LinkStatisticsModel(robot)
if not lmodel.load():
  lmodel.autogenerate()

lmodel.setRobotResolutions(0.01)
lmodel.setRobotWeights()

# Functions declaration #
def get_Tgrasp(input_box, side = 1):
  T = input_box.GetTransform()
  T[:3,2] *= -1
  T[:3,1] = T[:3,side]
  T[:3,0] = np.cross(T[:3,1], T[:3,2])
  T[2,3] += 0.005
  return T

def get_Tgoal(destination, n, side, tilt_tries = 0, slide_tries = 0):
  T = destination.GetTransform()
  T[:3,2] *= -1
  T[:3,1] = T[:3,side]
  T[:3,0] = np.cross(T[:3,1], T[:3,2])
  T[2,3] += (0.016 + n*0.012)

  if(tilt_tries != 0):
    for tries in range(tilt_tries):
      T = tilt(T)

  if(slide_tries != 0):
    for tries in range(slide_tries):
      T = slide(T)
  
  return T

def tilt(T, direction = 1):
  t = np.eye(4)
  if direction == 1:
    angle = 5 * np.pi / 180
  else:
    angle = -5 * np.pi / 180
  t[1,1] = np.cos(angle)
  t[1,2] = -np.sin(angle)
  t[2,1] = np.sin(angle)
  t[2,2] = np.cos(angle)
  return(np.dot(T,t))

def slide(T):
  t = np.eye(4)
  t[0,3] = 0.003
  return(np.dot(T,t))

def to_pick(qgrasp):
  traj = manipprob.MoveManipulator(goal = qgrasp, execute = False, outputtrajobj = True)
  return(traj)

def to_put(qgoal):
  traj = manipprob.MoveManipulator(goal = qgoal, execute = False, outputtrajobj = True)
  return(traj)

'''------------------------------------Preparation Done---------------------------------------------------------------------------------------------'''

# Main code #
n = 0
tilt_tries = 0
slide_tries = 0
grab_side = 1
successful = 0
cmi = 0

all_time_tick = []
all_tilt_tries = []
all_put_traj = []
all_box_id = []

start_time = time.time()

for i in range(-1, -41, -1):
  selected_box = boxes[i]
  box_id = i + 40
  Tgrasp = get_Tgrasp(selected_box)
  qgrasp = manipulator.FindIKSolutions(Tgrasp, orpy.IkFilterOptions.CheckEnvCollisions)

  print('successful = ', successful)
  print('cmi = ', cmi)

  if(not len(qgrasp)):
    for tries in range(9):
      Tgrasp = tilt(Tgrasp)
      qgrasp = manipulator.FindIKSolutions(Tgrasp, orpy.IkFilterOptions.CheckEnvCollisions)
      tilt_tries = tries+1
      if(len(qgrasp)):
        break

  if(not len(qgrasp)):
    Tgrasp = get_Tgrasp(selected_box)
    tilt_tries = 0
    for tries in range(5):
      Tgrasp = slide(Tgrasp)
      qgrasp = manipulator.FindIKSolutions(Tgrasp, orpy.IkFilterOptions.CheckEnvCollisions)
      slide_tries = tries+1
      if(len(qgrasp)):
        break

  if(not len(qgrasp)):
    grab_side = 0
    slide_tries = 0
    Tgrasp = get_Tgrasp(selected_box, grab_side)
    for tries in range(10):
      qgrasp = manipulator.FindIKSolutions(Tgrasp, orpy.IkFilterOptions.CheckEnvCollisions)
      if(len(qgrasp)):
        break

      else:
        Tgrasp = tilt(Tgrasp)
        tilt_tries = tries

  if(not len(qgrasp)):
    Tgrasp = get_Tgrasp(selected_box, grab_side)
    tilt_tries = 0
    for tries in range(5):
      Tgrasp = slide(Tgrasp)
      qgrasp = manipulator.FindIKSolutions(Tgrasp, orpy.IkFilterOptions.CheckEnvCollisions)
      slide_tries = tries + 1
      if(len(qgrasp)):
        break

  if(not len(qgrasp)):
    cmi += 1
    env.Remove(selected_box)
    print('Box with index', box_id, 'cannot be picked up and is removed.')
    print('successful = ', successful)
    print('cmi = ', cmi)
    tilt_tries = 0
    slide_tries = 0
    grab_side = 1
    continue

  if(abs(successful) % 2 == 0):
    Tgoal = get_Tgoal(destination1, n, grab_side, tilt_tries, slide_tries)
  else:
    Tgoal = get_Tgoal(destination0, n, grab_side, tilt_tries, slide_tries)
    n += 1

  qgoal = manipulator.FindIKSolutions(Tgoal, orpy.IkFilterOptions.CheckEnvCollisions)

  if(not len(qgoal)):
    cmi += 1
    env.Remove(selected_box)
    print('Box with index', box_id, 'cannot be placed down and is removed.')
    print('successful = ', successful)
    print('cmi = ', cmi)
    continue

  pick_traj = to_pick(qgrasp[0])
  robot.GetController().SetPath(pick_traj)
  robot.WaitForController(0)
  robot.Grab(selected_box)
  
  put_traj = to_put(qgoal[0])
  robot.GetController().SetPath(put_traj)
  robot.WaitForController(0)
  robot.Release(selected_box)

  # Question 1 ends here #

  # Question 2

  time_tick = np.arange(0, put_traj.GetDuration(), 0.01)
  
  all_time_tick.append(time_tick)
  all_tilt_tries.append(tilt_tries)
  all_put_traj.append(put_traj)
  all_box_id.append(box_id)

  successful += 1
  
  tilt_tries = 0
  slide_tries = 0
  grab_side = 1

end_time = time.time()

pdf = matplotlib.backends.backend_pdf.PdfPages('for_testing.pdf')

for i in range(len(all_time_tick)):
  time_tick = all_time_tick[i]
  tilt_tries = all_tilt_tries[i]
  put_traj = all_put_traj[i]
  box_id = all_box_id[i]
  spec = put_traj.GetConfigurationSpecification()
  q_tick = np.zeros((len(time_tick), robot.GetActiveDOF()))
  print('box id = ', box_id)

  for tick in range(len(time_tick)):
    trajdata = put_traj.Sample(time_tick[tick])
    q_tick[tick,:] = spec.ExtractJointValues(trajdata, robot, manipulator.GetArmIndices(), 0)

    angles = np.zeros(len(time_tick))
    with robot:
      for tick in range(len(time_tick)):
        robot.SetActiveDOFValues(q_tick[tick,:])
        T = manipulator.GetEndEffectorTransform()
        if(tilt_tries != 0):
          for tries in range(tilt_tries):
            T = tilt(T,-1)
            if abs(T[2, 2]) > 1:
                T[2, 2] = T[2, 2] / abs(T[2, 2])
        angles[tick] = (180 - np.arccos(T[2, 2]) * 180 / np.pi)
  print('box id', box_id, 'angles = ', angles)
  fig = plt.figure(i)
  ax = fig.gca()
  plt.plot(time_tick, angles, 'r', linewidth = 2, label='box %d'%box_id)
  ax.set_xlabel('Time [s]')
  ax.set_ylabel('Tilt angles [deg]')
  plt.grid()
  plt.legend()
  pdf.savefig(fig)

pdf.close()

print('successful = ', successful)
print('cmi = ', cmi)
print('It is done!')
print('Time Taken = ', (end_time - start_time)/60)

raw_input('Press enter to quit')
