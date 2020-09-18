import shutil

with open('meta.txt') as f:
    lines = f.read().splitlines()

lines.sort()
src = [i.split('\t', 1)[0] for i in lines]
dest = [i.split('\t', 1)[1] for i in lines]
dest = [i.split('\t', 1)[0] for i in dest]
filename = [i.split('/', 1)[1] for i in src]

l = list(zip(filename, dest))

with open('evaluation_setup/fold1_test.txt') as f:
    lines = f.read().splitlines()

lines = [i.split('/', 1)[1] for i in lines]
out = [l[filename.index(i)] for i in lines]

filename, dest = zip(*out)

src_path = '../feat/mel_out/background/'
dest_path = '../feat/cross_val/background/fold1/test/'

for i in range(len(filename)):
	f = filename[i].split('.')[0]
	src_pth = src_path+dest[i]+'/'+f+'.png'
	dst_pth = dest_path+dest[i]
	shutil.copy(src_pth, dst_pth)

with open('meta.txt') as f:
    lines = f.read().splitlines()

lines.sort()
src = [i.split('\t', 1)[0] for i in lines]
dest = [i.split('\t', 1)[1] for i in lines]
dest = [i.split('\t', 1)[0] for i in dest]
filename = [i.split('/', 1)[1] for i in src]

l = list(zip(filename, dest))

with open('evaluation_setup/fold2_test.txt') as f:
    lines = f.read().splitlines()

lines = [i.split('/', 1)[1] for i in lines]
out = [l[filename.index(i)] for i in lines]

filename, dest = zip(*out)

src_path = '../feat/mel_out/background/'
dest_path = '../feat/cross_val/background/fold2/test/'

for i in range(len(filename)):
	f = filename[i].split('.')[0]
	src_pth = src_path+dest[i]+'/'+f+'.png'
	dst_pth = dest_path+dest[i]
	shutil.copy(src_pth, dst_pth)


with open('meta.txt') as f:
    lines = f.read().splitlines()

lines.sort()
src = [i.split('\t', 1)[0] for i in lines]
dest = [i.split('\t', 1)[1] for i in lines]
dest = [i.split('\t', 1)[0] for i in dest]
filename = [i.split('/', 1)[1] for i in src]

l = list(zip(filename, dest))

with open('evaluation_setup/fold3_test.txt') as f:
    lines = f.read().splitlines()

lines = [i.split('/', 1)[1] for i in lines]
out = [l[filename.index(i)] for i in lines]

filename, dest = zip(*out)

src_path = '../feat/mel_out/background/'
dest_path = '../feat/cross_val/background/fold3/test/'

for i in range(len(filename)):
	f = filename[i].split('.')[0]
	src_pth = src_path+dest[i]+'/'+f+'.png'
	dst_pth = dest_path+dest[i]
	shutil.copy(src_pth, dst_pth)

with open('meta.txt') as f:
    lines = f.read().splitlines()

lines.sort()
src = [i.split('\t', 1)[0] for i in lines]
dest = [i.split('\t', 1)[1] for i in lines]
dest = [i.split('\t', 1)[0] for i in dest]
filename = [i.split('/', 1)[1] for i in src]

l = list(zip(filename, dest))

with open('evaluation_setup/fold4_test.txt') as f:
    lines = f.read().splitlines()

lines = [i.split('/', 1)[1] for i in lines]
out = [l[filename.index(i)] for i in lines]

filename, dest = zip(*out)

src_path = '../feat/mel_out/background/'
dest_path = '../feat/cross_val/background/fold4/test/'

for i in range(len(filename)):
	f = filename[i].split('.')[0]
	src_pth = src_path+dest[i]+'/'+f+'.png'
	dst_pth = dest_path+dest[i]
	shutil.copy(src_pth, dst_pth)