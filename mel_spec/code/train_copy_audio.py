import shutil

with open('evaluation_setup/fold1_train.txt') as f:
    lines = f.read().splitlines()

lines.sort()
src = [i.split('\t', 1)[0] for i in lines]
dest = [i.split('\t', 1)[1] for i in lines]
dest = [i.split('\t', 1)[0] for i in dest]
filename = [i.split('/', 1)[1] for i in src]


src_path = '../feat/mel_out/audio/'
dest_path = '../feat/cross_val/audio/fold1/train/'

for i in range(len(filename)):
	f = filename[i].split('.')[0]
	src_pth = src_path+dest[i]+'/'+f+'.png'
	dst_pth = dest_path+dest[i]
	shutil.copy(src_pth, dst_pth)

with open('evaluation_setup/fold2_train.txt') as f:
    lines = f.read().splitlines()

lines.sort()
src = [i.split('\t', 1)[0] for i in lines]
dest = [i.split('\t', 1)[1] for i in lines]
dest = [i.split('\t', 1)[0] for i in dest]
filename = [i.split('/', 1)[1] for i in src]

src_path = '../feat/mel_out/audio/'
dest_path = '../feat/cross_val/audio/fold2/train/'

for i in range(len(filename)):
	f = filename[i].split('.')[0]
	src_pth = src_path+dest[i]+'/'+f+'.png'
	dst_pth = dest_path+dest[i]
	shutil.copy(src_pth, dst_pth)


with open('evaluation_setup/fold3_train.txt') as f:
    lines = f.read().splitlines()

lines.sort()
src = [i.split('\t', 1)[0] for i in lines]
dest = [i.split('\t', 1)[1] for i in lines]
dest = [i.split('\t', 1)[0] for i in dest]
filename = [i.split('/', 1)[1] for i in src]

src_path = '../feat/mel_out/audio/'
dest_path = '../feat/cross_val/audio/fold3/train/'

for i in range(len(filename)):
	f = filename[i].split('.')[0]
	src_pth = src_path+dest[i]+'/'+f+'.png'
	dst_pth = dest_path+dest[i]
	shutil.copy(src_pth, dst_pth)


with open('evaluation_setup/fold4_train.txt') as f:
    lines = f.read().splitlines()

lines.sort()
src = [i.split('\t', 1)[0] for i in lines]
dest = [i.split('\t', 1)[1] for i in lines]
dest = [i.split('\t', 1)[0] for i in dest]
filename = [i.split('/', 1)[1] for i in src]

src_path = '../feat/mel_out/audio/'
dest_path = '../feat/cross_val/audio/fold4/train/'

for i in range(len(filename)):
	f = filename[i].split('.')[0]
	src_pth = src_path+dest[i]+'/'+f+'.png'
	dst_pth = dest_path+dest[i]
	shutil.copy(src_pth, dst_pth)