from random import gauss

import torch

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()

		self.inps = 3

		self.lay1 = 10
		self.lay2 = 10
		self.outs = 1

		self.fc1 = nn.Linear(self.inps, self.lay1)
		self.d1 = nn.Dropout(p=0.4)
		self.fc2 = nn.Linear(self.lay1, self.lay2)
		self.d2 = nn.Dropout(p=0.4)
		self.fc3 = nn.Linear(self.lay2, self.outs)

	def forward(self, x):
		x = self.d1(F.relu(self.fc1(x)))
		x = self.d2(F.relu(self.fc2(x)))
		x = self.fc3(x)

		return x



if __name__ == '__main__':
	from math import sin
	x = torch.FloatTensor([0.1, 1.2, 0.0])

	in_data = [torch.FloatTensor([gauss(0,1), gauss(0,1), gauss(0,1)]) for _ in range(30)]
	out_data = [torch.FloatTensor([sin(x[0])+x[1]-x[0]]) for x in in_data]

	net = Net()

	crit = nn.MSELoss()
	optimizer = optim.Adam(net.parameters())

	for epoch in range(1000):
		loss_sum = 0.0
		for i in range(len(in_data)):
			# TODO: Batchsize

			inps = in_data[i]
			outs_true = out_data[i]

			optimizer.zero_grad()

			outs = net(inps)

			loss = crit(outs, outs_true)
			loss_sum += loss.item()

			loss.backward()
			optimizer.step()
		if epoch%10==0:
			print(loss_sum)
