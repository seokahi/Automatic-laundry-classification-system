import os
import glob

# path = '0529/contract_image/short_sleeve'
path = '0529/all/long_pants'
# path = 'week13/modi_dataset/long_pants'
save_path = '0529/dataset/long_pants/'
# save_path = '0529/all/short_sleeve/'

cnt = len(os.listdir(save_path))
files = glob.glob(path + '/*')
for i, f in enumerate(files):
    ftitle, fext = os.path.splitext(f)
    fname = ftitle.split('_')[0]  # 파일이름
    # 파일이름을 fname_0000.jpg 이런 식으로 변경
    filename = save_path + 'long_pants' + '_' + '{0:04d}'.format(cnt) + fext
    print(filename)
    os.rename(f, filename)
    cnt = len(os.listdir(save_path))
print(cnt)
