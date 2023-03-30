import os
import glob
import shutil

path = 'dataset/dataset/bra'
save_path = 'dataset/dataset/bra/'

cnt = len(os.listdir(save_path))
files = glob.glob(path + '/*')
cnt = 0
for i, f in enumerate(files):
    ftitle, fext = os.path.splitext(f)
    fname = ftitle.split('_')[0]  # 파일이름

    # 파일이름을 fname_0000.jpg 이런 식으로 변경
    filename = save_path + 'bras' + '_' + '{0:04d}'.format(cnt) + fext
    print(filename)

    # Use shutil.move to overwrite existing files
    shutil.move(f, filename)
    cnt += 1

print(cnt)
