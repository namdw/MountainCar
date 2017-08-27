# import openai gym
import gym

# import neural net moodule
import sys
sys.path.append('../../NN')
import base

import pickle
import os.path

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import copy

env = gym.make('MountainCar-v0')
env.reset()

observation_limit = [l for l in env.observation_space.low]+[h for h in env.observation_space.high]
print(observation_limit)

filename = "value_net.pk"

if os.path.isfile(filename):
	f = open(filename,'rb')
	value_net = pickle.load(f)
	f.close()
else:
	value_net = base.NN(2,1,[128,64],func='relu', dropout=0.8, weight=20)

num_episode = 10
num_epoch = 3

TRAINING = True

# matplotlib.interactive(True)
plt.ion()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

X = np.arange(observation_limit[0], observation_limit[2], 0.05)
Y = np.arange(observation_limit[1], observation_limit[3], 0.005)
X,Y = np.meshgrid(X, Y)
Z = np.zeros(X.shape)

print(X.shape, Y.shape)
for i in range(X.shape[0]):
	for j in range(X.shape[1]):
		Z[i, j] = value_net.forward([X[i, j], Y[i, j]])
surf = ax.plot_surface(X,Y,Z)
plt.draw()
plt.pause(0.02)
ax.cla()

done = False
num_success = 0
for i in range(num_episode):
# i = 0
# while(num_success < 10):
	print('Episode', i)
	# i+=1
	observ_list = []
	action_list = []
	sum_reward = 0

	env.reset()

	action = env.action_space.sample()
	for _ in range(300):
		env.render()
		if(not TRAINING):
			env.render()
		observation, reward, done, info = env.step(action)
		sum_reward += reward
		# print(observation, reward, done, info)
		if(TRAINING):
			observ_list.append([o for o in observation])
			action_list.append([action])

		action = env.action_space.sample()

		value_net.train(observ_list[len(observ_list)-1],sum_reward,0.001)
		if(done and sum_reward > -200):
			# num_success+=1
			# print('Task Done : ', sum_reward)
			# if(TRAINING):
			# 	LR = 0.001
			# 	for j in range(len(observ_list)):
			# 		for _ in range(num_epoch):
			# 			# print(observ_list[j], action_list[j])
			# 			car_net.train(observ_list[j], action_list[j],LR)
			break

	for i in range(X.shape[0]):
		for j in range(X.shape[1]):
			Z[i, j] = value_net.forward([X[i, j], Y[i, j]])
	surf = ax.plot_surface(X,Y,Z)
	plt.draw()
	plt.pause(0.02)
	ax.cla()
	# fig.canvas.draw()




f = open(filename,'wb')
pickle.dump(value_net,f)
f.close()

print("done with pickle")