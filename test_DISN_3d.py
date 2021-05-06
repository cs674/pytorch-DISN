import argparse
from datetime import datetime
import numpy as np
import os
import cv2
import sys
import time
import torch

from sdfnet import sdfnet

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

# Shuffle the dataset?
shuffle = False
# Change this to switch between train and test
train = True
if train:
    split = 'train'
else:
    split = 'test'

two_stream = False

TEST_LISTINFO = []
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
    test_lst = os.path.join(lst_dir, cat_id+"_{}.lst".format(split))
    with open(test_lst, 'r') as f:
        lines = f.read().splitlines()
        for line in lines:
            for render in range(24):
                cats_limit[cat_id]+=1
                TEST_LISTINFO += [(cat_id, line.strip(), render)]


info = {'rendered_dir': raw_dirs["renderedh5_dir"],
            'sdf_dir': raw_dirs['sdf_dir']}
print(info)

with torch.no_grad():
    net = sdfnet()
    # Here we would like to load a pre trained model
    if two_stream:
        net.load_state_dict(torch.load('models/sdfmodel_two_stream_99.torch', map_location='cpu'))
    else:
        net.load_state_dict(torch.load('models/sdfmodel_one_stream_99.torch', map_location='cpu'))
    net.eval()


    TEST_DATASET = data_sdf_h5_queue.Pt_sdf_img(FLAGS, listinfo=TEST_LISTINFO, info=info, cats_limit=cats_limit, shuffle=shuffle)

    TEST_DATASET.start()

    print()
    print(len(TEST_DATASET)/FLAGS.batch_size)

    # Use fetch to get random, use get_batch for the same everytime
    batch_data = TEST_DATASET.get_batch(650)

    # show image
    # plt.imshow(batch_data['img'][0])
    # plt.show()

    # Generate grid
    N = 64
    dist = 0.9
    max_dimensions = np.array([dist, dist, dist])
    min_dimensions = np.array([-dist, -dist, -dist])
    bounding_box_dimensions = max_dimensions - min_dimensions
    grid_spacing = max(bounding_box_dimensions)/N
    X, Y, Z = np.meshgrid(list(np.arange(min_dimensions[0], max_dimensions[0], grid_spacing)),
    list(np.arange(min_dimensions[1], max_dimensions[1], grid_spacing)),
    list(np.arange(min_dimensions[2], max_dimensions[2], grid_spacing)))

    X = X.reshape(-1)
    Y = Y.reshape(-1)
    Z = Z.reshape(-1)
    points = np.array([X, Y, Z])
    


    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')

    # ax.scatter(points[0, :], points[1, :], points[2, :])

    # plt.show()
    # exit()


    # Run out network here

    # Convert the numpy data to torch
    image = torch.from_numpy(batch_data['img']).permute(0, 3, 1, 2)[0]
    points = torch.from_numpy(points.astype('Float32')).permute(1,0)
    trans_mat = torch.from_numpy(batch_data['trans_mat'])[0]

    if two_stream:
        pred_sdf = net(image.unsqueeze(0), points.unsqueeze(0), trans_mat.unsqueeze(0))
    else:
        pred_sdf = net(image.unsqueeze(0), points.unsqueeze(0))
    # pred_sdf = net(image.unsqueeze(0), points.unsqueeze(0))

    np_sdf = pred_sdf.numpy()




    import plotly
    import plotly.figure_factory as ff
    from skimage import measure
    
    IF = pred_sdf.reshape(N,N,N)
    verts, simplices = measure.marching_cubes_classic(IF, 0.01)
    x, y, z = zip(*verts)
    colormap = ['rgb(255,105,180)', 'rgb(255,255,51)', 'rgb(0,191,255)']
    fig = ff.create_trisurf(x=x,
                        y=y,
                        z=z,
                        plot_edges=False,
                        colormap=colormap,
                        simplices=simplices,
                        title="Isosurface")
    plotly.offline.plot(fig)

    # show image
    plt.imshow(batch_data['img'][0])
    plt.show()

    
    np.save('sdf.npy', np_sdf)


    TEST_DATASET.shutdown()

