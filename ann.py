from random import gauss, shuffle, seed
from math import fabs

from os import listdir

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

class Net(nn.Module):
	def __init__(self, channels=10):
		super(Net, self).__init__()

		self.inps = 3

		self.lay1 = channels
		self.lay2 = channels
		self.lay3 = channels
		self.outs = 1

		self.fc1 = nn.Linear(self.inps, self.lay1)
		#self.d1 = nn.Dropout(p=0.04)
		self.fc2 = nn.Linear(self.lay1, self.lay2)
		#self.d2 = nn.Dropout(p=0.04)
		self.fc3 = nn.Linear(self.lay2, self.lay3)
		#self.d3 = nn.Dropout(p=0.04)
		self.fc4 = nn.Linear(self.lay3, self.outs)

	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = F.relu(self.fc3(x))
		x = self.fc4(x)

		return x

def get_itd(left, right, sample_rate=1.0/44100.0, p=0.9):
	left_mag = [fabs(el) for el in left]
	right_mag = [fabs(el) for el in right]
	l_max = max(left_mag)
	r_max = max(right_mag)

	right_onset = None
	left_onset = None

	assert len(left) == len(right)
	for i in range(len(left)):
		if left_mag[i] >= 0.9*l_max and left_onset==None:
			left_onset = i*sample_rate
		if right_mag[i] >= 0.9*r_max and right_onset==None:
			right_onset = i*sample_rate

		if left_onset!=None and right_onset!=None:
			break

	return left_onset-right_onset

def get_head_size(id):
	return 1.0 # TODO: Write.

def get_data(ids):
	print("Reading data")
	itds = []
	inps = []
	for id in ids:
		print(id)
		head_size = get_head_size(id)
		files = listdir("itd_data/"+id)
		for file in files:
			with open("itd_data/"+id+"/"+file) as f:
				left = []
				right = []
				f.readline()
				line = f.readline().split(" ")
				azi = float(line[1])
				line = f.readline().split(" ")
				ele = float(line[1])

				f.readline()
				f.readline()

				while True:
					line = f.readline().split("\t")
					if len(line)==2:
						left.append(float(line[0]))
						right.append(float(line[1]))
					else:
						break
			itds.append([get_itd(left, right)])
			inps.append([azi, ele, head_size])

	assert len(itds) == len(inps)
	n = len(inps)
	idxs = [i for i in range(n)]
	shuffle(idxs)

	itds = [itds[idx] for idx in idxs]
	inps = [inps[idx] for idx in idxs]

	return inps, itds


if __name__ == '__main__':
	seed(0)
	IDs = ['003', '008', '009', '010', '011', '012', '015', '017', '018', '019', '020', '021', '027', '028', '033', '040', '044', '048', '050', '051', '058', '059', '060', '061', '065', '119', '124', '126', '127', '131', '133', '134', '135', '137', '147', '148', '152', '153', '154', '155', '156', '158', '162', '163', '165']

	idxs = [i for i in range(len(IDs))]
	shuffle(idxs)
	assert len(IDs) == 45
	train_IDS = [IDs[idxs[i]] for i in range(30)]
	test_IDS = [IDs[idxs[i]] for i in range(30, len(IDs))]

	inps, itds = get_data(train_IDS)

	from math import sin

	errors = []
	for channels in range(1, 10):
		net = Net(channels)

		#print(sum([fabs(el[0]) for el in itds])/len(itds))

		crit = nn.MSELoss()
		optimizer = optim.Adam(net.parameters())
		print("Starting training")

		c = 0
		batchsize = 32
		inps_tensor = []
		outs_true = []
		for epoch in range(100):
			loss_sum = 0.0
			for i in range(len(inps)):
				# TODO: Batchsize

				inps_tensor.append(inps[i])
				outs_true.append(itds[i])

				c += 1
				if c==batchsize:
					optimizer.zero_grad()
					
					inps_tensor = torch.FloatTensor(inps_tensor)
					outs_true = torch.FloatTensor(outs_true)
					outs = net(inps_tensor)

					loss = crit(outs, outs_true)
					loss_sum += loss.item()


					loss.backward()
					optimizer.step()


					inps_tensor = []
					outs_true = []
					c = 0


			#if epoch%1==0:
			#	print(loss_sum)

		# Validation
		print("Starting Validation")
		net.train(False)
		error = 0.0
		for i in range(len(inps)):
			inps_tensor = torch.FloatTensor(inps[i])
			outs_true = torch.FloatTensor(itds[i])
			outs = net(inps_tensor)

			error += fabs(outs.item()-outs_true)/ len(inps)
		errors.append(error)
		print("Error of", channels, "channels:", error)
	print(errors)


