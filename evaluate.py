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

# Evaluation 
from eval_util import CD, EMD, obj_data_to_mesh3d, get_normalize_mesh

import trimesh

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

# PC  : (218,3)
# PC_T: (256,3)

    

OBJ_NO = 31







lst_dir, cats, all_cats, raw_dirs = create_file_lst.get_all_info()


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default='1', help='GPU to use [default: GPU 0]')
parser.add_argument('--category', type=str, default="all", help='Which single class to train on [default: None]')
parser.add_argument('--log_dir', default='checkpoint', help='Log dir [default: log]')
parser.add_argument('--num_points', type=int, default=1, help='Point Number [default: 2048]')
parser.add_argument("--beta1", type=float, dest="beta1",
                    default=0.5, help="beta1 of adams")
#parser.add_argument('--num_sample_points', type=int, default=256, help='Sample Point Number [default: 2048]')
parser.add_argument('--num_sample_points', type=int, default=2048, help='Sample Point Number [default: 2048]')
#parser.add_argument('--sdf_points_num', type=int, default=32, help='Sample Point Number [default: 2048]')
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
train = False
if train:
    split = 'train'
else:
    split = 'test'


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
    net.load_state_dict(torch.load('models/sdfmodel99.torch', map_location='cpu'))
    net.eval()


    TEST_DATASET = data_sdf_h5_queue.Pt_sdf_img(FLAGS, listinfo=TEST_LISTINFO, info=info, cats_limit=cats_limit, shuffle=shuffle)

    TEST_DATASET.start()

    # Use fetch to get random, use get_batch for the same everytime
    batch_data = TEST_DATASET.get_batch(0)

    # print(batch_data.keys())
    # for k in batch_data.keys():
    #     print(k)
    #     for i in range(32):
    #         print(batch_data[k])
    print(batch_data['obj_nm'][OBJ_NO])
    # print(batch_data['obj_nm'])
    # print(batch_data['sdf_pt'].shape)  # BATCH_SIZE(32), NUM_SAPMLES(4096), 3
    #PC_T = batch_data['sdf_pt'][0, :, :]
    # print('PC_T(Ground Truth)')
    # print(PC_T.shape)
#    print(batch_data['img'][0].shape)


    
    # show image
    # plt.imshow(batch_data['img'][0])
    # plt.show()

    # Generate grid
    N = 65
    #N = 16
    dist = 1
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

    # print(X[:10])
    # print(Y[:10])
    # print(Z[:10])
    points = np.array([X, Y, Z])
    


    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # ax.scatter(points[0, :], points[1, :], points[2, :])
    # plt.show()
    # exit()


    # Demo Image: comment out if not using demo image
    # img_file = "03001627_17e916fc863540ee3def89b32cef8e45_20.png" # 
    # img_arr = cv2.imread(img_file, cv2.IMREAD_UNCHANGED).astype(np.uint8)[:, :, :3]
    # batch_img = np.asarray([img_arr.astype(np.float32) / 255.])
    # batch_data = {}
    # print('batch_img')
    # print(batch_img.shape)
    # batch_data['img'] = batch_img


    
    image = torch.from_numpy(batch_data['img']).permute(0, 3, 1, 2)[OBJ_NO]
    points = torch.from_numpy(points.astype('Float32')).permute(1,0)
    pred_sdf = net(image.unsqueeze(0), points.unsqueeze(0))
    np_sdf = pred_sdf.numpy()

    import plotly
    import plotly.figure_factory as ff
    from skimage import measure
    
    IF = pred_sdf.reshape(N,N,N)
    IF = IF.permute(1,0,2)
    
    import mcubes
    verts_pred, simplices_pred = mcubes.marching_cubes(np.asarray(IF), 0)
    mcubes.export_obj(verts_pred, simplices_pred, "out/chair_pred.obj")
    obj_file_pred_norm, centroid, m = get_normalize_mesh('out/chair_pred.obj', 'out/')
    # mcubes.export_obj(vertices, triangles, "demo/chair_pred_norm.obj")
    
#    verts_pred, simplices_pred = measure.marching_cubes_classic(IF, 0, spacing=(grid_spacing, grid_spacing, grid_spacing))
    with open(obj_file_pred_norm, 'r', encoding='utf8') as f:
        obj_data = f.read()
        verts_pred, simplices_pred = obj_data_to_mesh3d(obj_data)

    choice_pred = np.random.randint(verts_pred.shape[0], size=FLAGS.num_sample_points)
    sampled_verts_pred = verts_pred[choice_pred, ...]
    PC = sampled_verts_pred

    ################################################################################
    # Evaluation for Mesh
    ################################################################################
    # Ground Truth Mesh
    #obj_file_gt = '../ssd1/datasets/ShapeNet/ShapeNetCore.v1/03001627/ed751e0c20f48b3226fc87e2982c8a2b/model.obj'
    obj_file_gt = '../ssd1/datasets/ShapeNet/mesh/03001627/d72f27e4240bd7d0283b00891f680579/isosurf.obj'

    # using surface sample
    # mesh = trimesh.load_mesh(obj_file_gt, process=False)
    # verts_gt, simplices_gt = trimesh.sample.sample_surface(mesh, FLAGS.num_sample_points)

    # using vertices
    with open(obj_file_gt, 'r', encoding='utf8') as f:
        obj_data = f.read()
        verts_gt, simplices_gt = obj_data_to_mesh3d(obj_data)

    ################################################################################
    # Evaluation for Sampled Point Cloud
    ################################################################################
    choice_gt = np.random.randint(verts_gt.shape[0], size=FLAGS.num_sample_points)
    sampled_verts_gt = verts_gt[choice_gt, ...]
    PC_T = sampled_verts_gt

    print('Ground Truth')
    print(PC_T[:10])
    print('Prediction')
    print(PC[:10])
    
    cd   = CD(PC, PC_T)
    print('Chamfer Distance      : %f' % cd)
    emd  = EMD(PC, PC_T)
    print('Earth Mover\'s Distance: %f' % emd)


    ################################################################################
    x_gt, y_gt, z_gt = zip(*verts_gt)
    colormap = ['rgb(255,105,180)', 'rgb(255,255,51)', 'rgb(0,191,255)']
    fig = ff.create_trisurf(x=x_gt,
                        y=y_gt,
                        z=z_gt,
                        plot_edges=False,
                        colormap=colormap,
                        simplices=simplices_gt,
                        title="Ground Truth")
    plotly.offline.plot(fig, auto_open=False, filename="chair_gt.html")

    
    # #Pred Mesh
    x_pred, y_pred, z_pred = zip(*verts_pred)
    colormap = ['rgb(255,105,180)', 'rgb(255,255,51)', 'rgb(0,191,255)']
    fig = ff.create_trisurf(x=x_pred,
                        y=y_pred,
                        z=z_pred,
                        plot_edges=False,
                        colormap=colormap,
                        simplices=simplices_pred,
                        title="Isosurface")
    plotly.offline.plot(fig, auto_open=False, filename="chair_pred.html")
    
    TEST_DATASET.shutdown()
