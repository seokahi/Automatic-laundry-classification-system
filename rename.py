import os
import glob

# path = '0529/contract_image/short_sleeve'
path = 'dataset/dataset/skirt'
# path = 'week13/modi_dataset/long_pants'
save_path = 'dataset/dataset/skirt/'
# save_path = '0529/all/short_sleeve/'

cnt = len(os.listdir(save_path))
files = glob.glob(path + '/*')
for i, f in enumerate(files):
    ftitle, fext = os.path.splitext(f)
    fname = ftitle.split('_')[0]  # 파일이름
    # 파일이름을 fname_0000.jpg 이런 식으로 변경
    filename = save_path + '_' + '{0:04d}'.format(cnt) + fext
    print(filename)
    os.rename(f, filename)
    cnt = len(os.listdir(save_path))
print(cnt)
