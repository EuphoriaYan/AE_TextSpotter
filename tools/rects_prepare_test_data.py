
import os
import sys
root_path = "/".join(os.path.realpath(__file__).split("/")[:-2])
print(f'root_path is {root_path}')
if root_path not in sys.path:
    sys.path.insert(0, root_path)

import mmcv
import os.path as osp

data_root = 'data/ReCTS/'

test_img_root = osp.join(data_root, 'test/img/')


def prepare_test_img_infos(cache_path):
    img_names = [img_name for img_name in mmcv.utils.scandir(test_img_root, '.jpg')]

    img_infos = []
    print('Loading images...')
    for i, img_name in enumerate(img_names):
        if i % 1000 == 0:
            print('%d / %d' % (i, len(img_names)))
        img_path = test_img_root + img_name
        ann_file = None

        try:
            h, w, _ = mmcv.imread(img_path).shape
            img_info = dict(
                filename=img_name,
                height=h,
                width=w,
                annfile=ann_file)
            img_infos.append(img_info)
        except:
            print('Load image error when generating img_infos: %s' % img_path)

    with open(cache_path, 'w') as f:
        mmcv.dump(img_infos, f, file_format='json', ensure_ascii=False)


# def tmp(file_path):
#     img_infos = mmcv.load(file_path)
#     for i in range(len(img_infos)):
#         annpath = img_infos[i]['annpath']
#         annpath = annpath.split('/')[-1]
#         img_infos[i]['annfile'] = annpath
#         del img_infos[i]['annpath']
#
#     with open(file_path + '1', 'w') as f:
#         mmcv.dump(img_infos, f, file_format='json', ensure_ascii=False)


if __name__ == '__main__':
    # prepare img infos
    prepare_test_img_infos(osp.join(data_root, 'tda_rects_test_cache_file.json'))

    # # combine img infos
    img_infos = mmcv.load(osp.join(data_root, 'tda_rects_train_cache_file.json')) + \
                mmcv.load(osp.join(data_root, 'tda_rects_val_cache_file.json'))
    with open(osp.join(data_root, 'train_cache_file.json'), 'w') as f:
        mmcv.dump(img_infos, f, file_format='json', ensure_ascii=False)

