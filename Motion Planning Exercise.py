import numpy as np
import openravepy as orpy
from openravepy import *
import time
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
from osr_openrave import kinematics, planning

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

'''----------------------------------Initial Condition Setup Done----------------------------------------------------'''

''' Flow and logic of the code
1.) Locate the box at upper most layer.

2.) Find the IK solution to grab the box.
    i) It will firstly try to grab the box vertically along the long side.
   ii) If no solution is found, it will try to grab by tilting the gripper.
  iii) If tilting gripper does not work, gripper will slide along the long side.
   iv) Steps i to iii are repeated on the short side of the box.
    v) If no solution is found, the box will be removed.

3.) Find the IK solution to put down the box.
    i) If no solution is found, the box will be removed.

4.) Move the gripper to grab the box.
    i) If trajectory cannot be generated, the box will be removed.

5.) Move the gripper to destination and put down the box.
    i) If trajectory cannot be generated, the box will be removed.

6.) Steps 1 to 5 are repeated until all the movable boxes are placed at the destination.
'''

'''----------------------------------Declaring Functions ----------------------------------------------------'''

def get_Tgrasp(input_box, grab_side = 1):
# Usage: Obtain the transformation matrix of gripper when picking up boxes from container
# Params:
#   input_box: Rave body information of the box
#   side: It determines the gripper to grab which side of the box. Input 0 for short side, 1 for long side

  T = input_box.GetTransform()
  T[:3,2] *= -1
  T[:3,1] = T[:3, grab_side]
  T[:3,0] = np.cross(T[:3,1], T[:3,2])
  T[2,3] += 0.005
  return T

def get_Tgoal(destination, n, grab_side, tilt_tries = 0, slide_tries = 0):
# Usage: Obtain the transformation matrix of gripper when putting down the box
# Params:
#   destination: Which destination to place the box
#   n: To compensate the height difference after putting down the boxes.
#   side: It determines the gripper to grab which side of the box. Input 0 for short side, 1 for long side
#   tilt_tries: Tilted angle of the gripple when picking up the box
#   slide_tries: Gripple slide along the grabbing side when picking up

# If gripple tilted or slided when grabbing the box, the same amount of tilted angle or slide distance need to be compensate when putting down the box.

  T = destination.GetTransform()
  T[:3,2] *= -1
  T[:3,1] = T[:3, grab_side]
  T[:3,0] = np.cross(T[:3,1], T[:3,2])
  T[2,3] += (0.016 + int(n)*0.012)

  if(tilt_tries != 0):
    for tries in range(tilt_tries):
      T = tilt(T)

  if(slide_tries != 0):
    for tries in range(slide_tries):
      T = slide(T)
  
  return T

def tilt(T, direction = 1):
# Usage: Compute the transformation matrix of gripple when it is tilted by 5 degree.
# Params:
#   T: Transformation matrix of the gripple
#   direction: It determines which direction to tilt the gripple

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
# Usage: Compute the transformation matrix of gripple when it slides along the box.
# Params:
#   T: Transformation matrix of the gripple

  t = np.eye(4)
  t[0,3] = 0.003
  return(np.dot(T,t))

def to_pick(robot, qgrasp):
# Usage: compute the trajectory of gripper to pick up the box from container
# Params:
#   robot: Information of the robot body
#   qgrasp: IK solution of the gripple to pick up the box

  traj = planning.plan_to_joint_configuration(robot, qgrasp)
  return(traj)

def to_put(robot, qgoal):

# Usage: compute the trajectory of gripper to put down the box at destination
# Params:
#   robot: Information of the robot body
#   qgoal: IK solution of the gripple to put down the box

  traj = planning.plan_to_joint_configuration(robot, qgoal)
  return(traj)

'''------------------------------------Preparation Done-------------------------------------------------------------------'''

'''------------------------------------Main Code--------------------------------------------------------------------------'''

# Robot Setup #
robot = env.GetRobots()[0]
manipulator = robot.SetActiveManipulator('gripper')
robot.SetActiveDOFs(manipulator.GetArmIndices())
manipprob = interfaces.BaseManipulation(robot)

np.set_printoptions(precision=6, suppress=True)

iktype = orpy.IkParameterization.Type.Transform6D

ikmodel = databases.inversekinematics.InverseKinematicsModel(robot, iktype=iktype)
if not ikmodel.load():
  ikmodel.autogenerate()

lmodel = databases.linkstatistics.LinkStatisticsModel(robot)
if not lmodel.load():
  lmodel.autogenerate()

lmodel.setRobotResolutions(0.01)
lmodel.setRobotWeights()

# Declare Variables #
n = 0
successful = 0
cmi = 0

updir = np.array((0,0,1))
closedir = np.array((-1, 0, 0))
sidedir = np.array((0, -1, 0))

all_time_tick = []
all_tilt_tries = []
all_put_traj = []
all_box_id = []

start_time = time.time()

for i in range(-1, -41, -1):
  selected_box = boxes[i] # Box at the upper most layer
  box_id = i + 40

  tilt_tries = 0
  slide_tries = 0
  grab_side = 1

  Tgrasp = get_Tgrasp(selected_box)
  qgrasp = kinematics.find_closest_iksolution(robot, Tgrasp, iktype)
  print('successful = ', successful)
  print('cmi = ', cmi)

  # Try to tilt the gripper
  if(qgrasp is None):
    for tries in range(9):
      Tgrasp = tilt(Tgrasp)
      qgrasp = kinematics.find_closest_iksolution(robot, Tgrasp, iktype)
      tilt_tries = tries+1
      if(qgrasp is not None):
        break

  # Try to slide the gripper along long side
  if(qgrasp is None):
    Tgrasp = get_Tgrasp(selected_box)
    tilt_tries = 0
    for tries in range(5):
      Tgrasp = slide(Tgrasp)
      qgrasp = kinematics.find_closest_iksolution(robot, Tgrasp, iktype)
      slide_tries = tries+1
      if(qgrasp is not None):
        break

  # Try to grab the box by the short side
  if(qgrasp is None):
    grab_side = 0
    slide_tries = 0
    Tgrasp = get_Tgrasp(selected_box, grab_side)
    for tries in range(10):
      qgrasp = kinematics.find_closest_iksolution(robot, Tgrasp, iktype)
      if(qgrasp is not None):
        break

      else:
        Tgrasp = tilt(Tgrasp)
        tilt_tries = tries

  # Try to slide the gripper along short side
  if(qgrasp is None):
    Tgrasp = get_Tgrasp(selected_box, grab_side)
    tilt_tries = 0
    for tries in range(5):
      Tgrasp = slide(Tgrasp)
      qgrasp = kinematics.find_closest_iksolution(robot, Tgrasp, iktype)
      slide_tries = tries + 1
      if(qgrasp is not None):
        break

  # Remove the box if it cannot be picked up
  if(qgrasp is None):
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

  qgoal = kinematics.find_closest_iksolution(robot, Tgoal, iktype)

  # Remove the box if it cannot be placed
  if(qgoal is None):
    cmi += 1
    env.Remove(selected_box)
    print('Box with index', box_id, 'cannot be placed down and is removed.')
    print('successful = ', successful)
    print('cmi = ', cmi)
    continue

  try:
    pick_traj = to_pick(robot, qgrasp)
    robot.GetController().SetPath(pick_traj)
    robot.WaitForController(0)
    robot.Grab(selected_box)

  except:
    cmi += 1
    env.Remove(selected_box)
    print('Box with index', box_id, 'is removed because no collision free path to pick up.')
    print('successful = ', successful)
    print('cmi = ', cmi)
    continue

  # The gripper will try to move the box higher than the container for a smoother trajectory when putting down the box #
  try:
    z = manipulator.GetEndEffectorTransform()[2,3]
    while(z < 0.435):
      manipprob.MoveHandStraight(direction=updir, stepsize=0.01, minsteps=1, maxsteps=20)
      robot.WaitForController(0)

      manipprob.MoveHandStraight(direction=closedir,stepsize=0.01, minsteps=1, maxsteps=10)
      robot.WaitForController(0)

      manipprob.MoveHandStraight(direction=sidedir, stepsize=0.01, minsteps=1, maxsteps=30)
      robot.WaitForController(0)

      z = manipulator.GetEndEffectorTransform()[2,3]

    put_traj = to_put(robot, qgoal)
    robot.GetController().SetPath(put_traj)
    robot.WaitForController(0)
    robot.Release(selected_box)

  except:
    cmi += 1
    robot.Release(selected_box)
    env.Remove(selected_box)
    print('Box with index', box_id, 'is removed because no collision free path to put down.')
    print('successful = ', successful)
    print('cmi = ', cmi)
    continue

  successful += 1
  n += 0.5  
  # Question 1 ends here #

  # Question 2 starts here #
  time_tick = np.arange(0, put_traj.GetDuration(), 0.01)

  all_time_tick.append(time_tick)
  all_tilt_tries.append(tilt_tries)
  all_put_traj.append(put_traj)
  all_box_id.append(box_id)
  
end_time = time.time()

pdf = matplotlib.backends.backend_pdf.PdfPages('for_testing.pdf')

for i in range(len(all_time_tick)):
  time_tick = all_time_tick[i]
  tilt_tries = all_tilt_tries[i]
  put_traj = all_put_traj[i]
  box_id = all_box_id[i]
  spec = put_traj.GetConfigurationSpecification()
  q_tick = np.zeros((len(time_tick), robot.GetActiveDOF()))

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
            if(abs(T[2, 2]) > 1):
                T[2, 2] = T[2, 2] / abs(T[2, 2])
        angles[tick] = (180 - np.arccos(T[2, 2]) * 180 / np.pi)
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