import os
import glob

def data_prep():
	f = open('../input/driver_imgs_list.csv', 'r')
	g = open('data/statefarm/train.txt', 'w')
	h = open('data/statefarm/test.txt', 'w')
	test_ix = ['p015','p041','p051','p075']
	line = f.readline()
	while 1:
		line = f.readline()
		if line == "":
			break
		arr = line.strip().split(",")
		if arr[0] in test_ix:
			h.write('data/statefarm/images/c'+arr[1][-1]+"/"+arr[2]+' '+arr[1][-1]+'\n')
		else:
			g.write('data/statefarm/images/c'+arr[1][-1]+"/"+arr[2]+' '+arr[1][-1]+'\n')
	f.close()
	g.close()
	h.close()

data_prep()


