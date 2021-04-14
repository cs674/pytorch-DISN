import argparse
from datetime import datetime
import numpy as np
import os
import cv2
import sys
import time

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))
print(os.path.join(BASE_DIR, 'models'))
sys.path.append(BASE_DIR) # model
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
sys.path.append(os.path.join(BASE_DIR, 'data'))
sys.path.append(os.path.join(BASE_DIR, 'preprocessing'))

# import model_normalization as model
import data_sdf_h5_queue # as data
# import output_utils
import create_file_lst
lst_dir, cats, all_cats, raw_dirs = create_file_lst.get_all_info()


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default='1', help='GPU to use [default: GPU 0]')
parser.add_argument('--category', type=str, default="all", help='Which single class to train on [default: None]')
parser.add_argument('--log_dir', default='checkpoint', help='Log dir [default: log]')
parser.add_argument('--num_points', type=int, default=1, help='Point Number [default: 2048]')
parser.add_argument("--beta1", type=float, dest="beta1",
                    default=0.5, help="beta1 of adams")
parser.add_argument('--num_sample_points', type=int, default=256, help='Sample Point Number [default: 2048]')
# parser.add_argument('--sdf_points_num', type=int, default=32, help='Sample Point Number [default: 2048]')
parser.add_argument('--max_epoch', type=int, default=200, help='Epoch to run [default: 201]')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 32]')
parser.add_argument('--img_h', type=int, default=137, help='Image Height')
parser.add_argument('--img_w', type=int, default=137, help='Image Width')
parser.add_argument('--sdf_res', type=int, default=256, help='sdf grid')
parser.add_argument('--num_classes', type=int, default=1024, help='vgg dim')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--restore_model', default='', help='restore_model') #checkpoint/sdf_2d3d_sdfbasic2_nowd
parser.add_argument('--restore_modelpn', default='', help='restore_model')#checkpoint/sdf_3dencoder_sdfbasic2/latest.ckpt
parser.add_argument('--restore_modelcnn', default='', help='restore_model')#../../models/CNN/pretrained_model/vgg_16.ckpt

parser.add_argument('--train_lst_dir', default=lst_dir, help='train mesh data list')
parser.add_argument('--valid_lst_dir', default=lst_dir, help='test mesh data list')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.9, help='Decay rate for lr decay [default: 0.7]')
parser.add_argument('--mask_weight', type=float, default=4.0)
parser.add_argument('--threedcnn', action='store_true')
parser.add_argument('--volimp', action='store_true')
parser.add_argument('--img_feat_onestream', action='store_true')
parser.add_argument('--img_feat_twostream', action='store_true')
parser.add_argument('--binary', action='store_true')
parser.add_argument('--alpha', action='store_true')
parser.add_argument('--augcolorfore', action='store_true')
parser.add_argument('--augcolorback', action='store_true')
parser.add_argument('--backcolorwhite', action='store_true')
parser.add_argument('--rot', action='store_true')
parser.add_argument('--tanh', action='store_true')
parser.add_argument('--cam_est', action='store_true')
parser.add_argument('--cat_limit', type=int, default=168000, help="balance each category, 1500 * 24 = 36000")
parser.add_argument('--multi_view', action='store_true')

FLAGS = parser.parse_args()
print(FLAGS)

# Change this to switch between train and test
train = False
if train:
    split = 'train'
else:
    split = 'test'


TRAIN_LISTINFO = []
cats_limit = {}

cat_ids = []
if FLAGS.category == "all":
    for key, value in cats.items():
        cat_ids.append(value)
        cats_limit[value] = 0
else:
    cat_ids.append(cats[FLAGS.category])
    cats_limit[cats[FLAGS.category]] = 0

for cat_id in cat_ids:
    train_lst = os.path.join(FLAGS.train_lst_dir, cat_id+"_{}.lst".format(split))
    with open(train_lst, 'r') as f:
        lines = f.read().splitlines()
        for line in lines:
            for render in range(24):
                cats_limit[cat_id]+=1
                TRAIN_LISTINFO += [(cat_id, line.strip(), render)]


info = {'rendered_dir': raw_dirs["renderedh5_dir"],
            'sdf_dir': raw_dirs['sdf_dir']}
print(info)

TRAIN_DATASET = data_sdf_h5_queue.Pt_sdf_img(FLAGS, listinfo=TRAIN_LISTINFO, info=info, cats_limit=cats_limit)

TRAIN_DATASET.start()

# Use fetch to get random, use get_batch for the same everytime
# batch_data = TRAIN_DATASET.fetch()
batch_data = TRAIN_DATASET.get_batch(0)

print(batch_data.keys())

for key in batch_data:
    print(key)
    try:
        print(batch_data[key].shape)
    except:
        print(batch_data[key])
    print()

# show image
# plt.imshow(batch_data['img'][0])
# plt.show()

# show point clouds
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# points = batch_data['sdf_pt'][0, :, :]
# points_rot = batch_data['sdf_pt_rot'][0, :, :] + 0.01
# ax.scatter(points[:, 0], points[:, 1], points[:, 2])
# ax.scatter(points_rot[:, 0], points_rot[:, 1], points_rot[:, 2])
# plt.show()

fig = plt.figure(figsize=plt.figaspect(0.5))
points = batch_data['sdf_pt'][0, :, :]
trans_mat = batch_data['trans_mat'][0]

# Transform points
points_trans = np.matmul(np.concatenate((points, np.ones([points.shape[0], 1])), axis=1), trans_mat)
# Project points
points_xy = points_trans[:, :2]/(np.expand_dims(points_trans[:, 2], axis=-1))
# Force all points to be within image space
points_xy = np.clip(points_xy, 0, 136)

# set colors of points
colors = cm.rainbow(np.linspace(0, 0.5, points.shape[0]))
colors[-1, :] = np.array([1, 0, 0, 1])

ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors)
# Make the shapes upright
ax1.view_init(120, -80)
# Make all axes the same size
bound_size = 1
ax1.set_xlim(-bound_size, bound_size)
ax1.set_ylim(-bound_size, bound_size)
ax1.set_zlim(-bound_size, bound_size)
# Label axes
# ax1.set_xlabel
ax1.set_xlabel('X axis')
ax1.set_ylabel('Y axis')
ax1.set_zlabel('Z axis')

ax2 = fig.add_subplot(1, 2, 2)
ax2.imshow(batch_data['img'][0])
ax2.scatter(points_xy[:, 0], points_xy[:, 1], c=colors)
plt.show()

TRAIN_DATASET.shutdown()
