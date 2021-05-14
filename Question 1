import numpy as np
import openravepy as orpy
import time
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
np.set_printoptions(precision=6, suppress=True)

ikmodel = orpy.databases.inversekinematics.InverseKinematicsModel(robot, iktype=orpy.IkParameterization.Type.Transform6D)
if not ikmodel.load():
  ikmodel.autogenerate()

# bi-RRT setup #
planner = orpy.RaveCreatePlanner(env, 'birrt')
params = orpy.Planner.PlannerParameters()
params.SetPostProcessing('ParabolicSmoother', '<_nmaxiterations>40</_nmaxiterations>')

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

def tilt(T):
  t = np.eye(4)
  angle = 5 * np.pi/180
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
  params.SetRobotActiveJoints(robot)
  params.SetGoalConfig(qgrasp)
  planner.InitPlan(robot, params)
  traj = orpy.RaveCreateTrajectory(env, '')
  planner.PlanPath(traj)
  controller = robot.GetController()
  controller.SetPath(traj)

def to_put(qgoal):
  params.SetRobotActiveJoints(robot)
  params.SetGoalConfig(qgoal)
  planner.InitPlan(robot, params)
  traj = orpy.RaveCreateTrajectory(env, '')
  planner.PlanPath(traj)
  controller = robot.GetController()
  controller.SetPath(traj)

'''------------------------------------Preparation Done---------------------------------------------------------------------------------------------'''

# Main code #
n = 0
theta = 5 * np.pi/180
tilt_tries = 0
slide_tries = 0
grab_side = 1
successful = 0
cmi = 0

for i in range(-1, -41, -1):
  selected_box = boxes[i]
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
    print('Box with index', i+40, 'cannot be picked up and is removed.')
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
    print('Box with index', i+40, 'cannot be placed down and is removed.')
    print('successful = ', successful)
    print('cmi = ', cmi)
    continue

  to_pick(qgrasp[0])
  robot.WaitForController(0)
  robot.Grab(selected_box)
  time.sleep(0)
  
  to_put(qgoal[0])
  robot.WaitForController(0)
  robot.Release(selected_box)
  time.sleep(0)

  successful += 1
  
  tilt_tries = 0
  slide_tries = 0
  grab_side = 1

print('successful = ', successful)
print('cmi = ', cmi)
print('It is done!')

raw_input('Press anykey to quit')
