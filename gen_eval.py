from tqdm import tqdm
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
from eval_util import CD, EMD, obj_data_to_mesh3d, get_normalize_mesh, HTML_rendering, FSCORE
import trimesh
from skimage import measure
import mcubes
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))
print(os.path.join(BASE_DIR, 'models'))
sys.path.append(BASE_DIR) # model
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
sys.path.append(os.path.join(BASE_DIR, 'data'))
sys.path.append(os.path.join(BASE_DIR, 'preprocessing'))
import data_sdf_h5_queue # as data
import create_file_lst
OBJ_NO = 0
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
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during training [default: 32]')
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

#two_stream = False
two_stream = True


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
    batch_no_vec = [1,100,201,300,400,500,550,600,80,900,1150]
    for i in tqdm(range(len(batch_no_vec))):
        batch_data = TEST_DATASET.get_batch(batch_no_vec[i])

        # GT (Already Normalized)
        print('Collect GT surface samples...', end=' ')
        obj_file_gt    = '../ssd1/datasets/ShapeNet/mesh/03001627/' + str(batch_data['obj_nm'][OBJ_NO])  + '/isosurf.obj'
        print(obj_file_gt)
        #obj_file_gt    = 'isosurf.obj'
        if not os.path.isfile(obj_file_gt):
            print('NOT EXIST: ' + obj_file_gt)
            continue
        mesh_gt        = trimesh.load_mesh(obj_file_gt, process=False)
        pc_gt_surf, _  = trimesh.sample.sample_surface(mesh_gt, FLAGS.num_sample_points)
        choice_gt      = np.random.randint(pc_gt_surf.shape[0], size=FLAGS.num_sample_points)
        PC_GT          = pc_gt_surf[choice_gt, ...]
        print('done.')




        
        # Generate grid
        N = FLAGS.sdf_res + 1
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
        points = np.array([X, Y, Z])
        print('Load a sample image...')
        # Demo Image: comment out below five lines unless you use the specified image
        # img_file = "./03001627_953a6c4d742f1e44d1dcc55e36186e4e_02.png"
        # img_arr = cv2.imread(img_file, cv2.IMREAD_UNCHANGED).astype(np.uint8)[:, :, :3]
        # batch_img = np.asarray([img_arr.astype(np.float32) / 255.])
        # batch_data['img'] = batch_img
        # print(batch_data.keys())
        # print(batch_data['trans_mat'])
        # Prediction & obj generation
        print('Predict and generate .obj file...', end=' ')
        image = torch.from_numpy(batch_data['img']).permute(0, 3, 1, 2)[OBJ_NO]
        points = torch.from_numpy(points.astype('Float32')).permute(1,0)
        trans_mat = torch.from_numpy(batch_data['trans_mat'])[OBJ_NO]
        # ours_sdf = net(image.unsqueeze(0), points.unsqueeze(0))
        max_num_points = 300000
        num_chunks = int(np.ceil(points.shape[0]/max_num_points))
        points_chunks = torch.chunk(points, num_chunks, dim=0)
        pred_sdf = []
        print('num_chunks')
        print(num_chunks)
        for c in tqdm(range(num_chunks)):
            if two_stream:
                pred_sdf_chunk = net(image.unsqueeze(0), points_chunks[c].unsqueeze(0), trans_mat.unsqueeze(0))
            else:
                pred_sdf_chunk = net(image.unsqueeze(0), points_chunks[c].unsqueeze(0))
            pred_sdf.append(pred_sdf_chunk)
        pred_sdf = torch.cat(pred_sdf, dim=1)
        ours_sdf = pred_sdf
        np_sdf = ours_sdf.numpy()
        IF = ours_sdf.reshape(N,N,N)
        IF = IF.permute(1,0,2)
        verts_ours, simplices_ours = mcubes.marching_cubes(np.asarray(IF), 0)
        mcubes.export_obj(verts_ours, simplices_ours, "obj/chair_ours_" + str(batch_no_vec[i]).zfill(4) + ".obj")
        print('done.')
        # OURS
        print('Collect OURS surface samples...', end=' ')
        obj_file_ours                     = "obj/chair_ours_" + str(batch_no_vec[i]).zfill(4) + ".obj"
        obj_file_ours_norm, centroid, m   = get_normalize_mesh(obj_file_ours, "obj/chair_ours_norm_" + str(batch_no_vec[i]).zfill(4) + ".obj")
        with open(obj_file_ours_norm, 'r', encoding='utf8') as f_ours:
            obj_data_ours = f_ours.read()
            pc_ours_surf,_                = obj_data_to_mesh3d(obj_data_ours)
        choice_ours                       = np.random.randint(pc_ours_surf.shape[0], size=FLAGS.num_sample_points)
        PC_OURS                           = pc_ours_surf[choice_ours, ...]
        print('done.')












        
        # THEIRS
        print('Collect OURS surface samples...', end=' ')
        #    obj_file_theirs                   = '../DISN_xar/demo/chair_theirs.obj'
        obj_file_theirs                   = "../DISN/demo/chair_theirs_" + str(batch_no_vec[i]).zfill(4) + ".obj"
        obj_file_theirs_norm, centroid, m = get_normalize_mesh(obj_file_theirs, "obj/chair_theirs_norm_" + str(batch_no_vec[i]).zfill(4) + ".obj")
        with open(obj_file_theirs_norm, 'r', encoding='utf8') as f_theirs:
            obj_data_theirs = f_theirs.read()
            pc_theirs_surf,_                = obj_data_to_mesh3d(obj_data_theirs)
        choice_theirs                     = np.random.randint(pc_theirs_surf.shape[0], size=FLAGS.num_sample_points)
        PC_THEIRS                         = pc_theirs_surf[choice_theirs, ...]
        PC_THEIRS[:, [0,2]]               = PC_THEIRS[:, [2,0]]
        print('done.')






        print('Ground Truth')
        print(PC_GT[:10])
        print('Ours')
        print(PC_OURS[:10])
        print('Theirs')
        print(PC_THEIRS[:10])
        print('--------------------------------------------------------------------------------')
        print('FSCORE')
        SIDE_LEN = 1.4
        ratios = [0.01, 0.02, 0.05, 0.10, 0.20]
        print('Threshold(%)', end='\t')
        for r in ratios:
            pr = 100*r
            print('%3.1f' % pr, end='\t')
        print()
        print('--------------------------------------------------------------------------------')        
        print('OURS        ', end='\t')
        for r in ratios:
            print('%5.3f' % FSCORE(PC_OURS  , PC_GT, SIDE_LEN * r), end='\t')
        print()
        print('THEIRS      ', end='\t')    
        for r in ratios:
            print('%5.3f' % FSCORE(PC_THEIRS, PC_GT, SIDE_LEN * r), end='\t')
        print('\n--------------------------------------------------------------------------------')
        #ours
        print('Our Distances:')
        cd   = CD(PC_OURS, PC_GT)
        print('Chamfer Distance      : %f' % cd)
        emd  = EMD(PC_OURS, PC_GT)
        print('Earth Mover\'s Distance: %f' % emd)
        print('--------------------------------------------------------------------------------')
        #Theirs
        print('Their Distances:')
        cd   = CD(PC_THEIRS, PC_GT)
        print('Chamfer Distance      : %f' % cd)
        emd  = EMD(PC_THEIRS, PC_GT)
        print('Earth Mover\'s Distance: %f' % emd)
        print('--------------------------------------------------------------------------------')
        # Renderings
        with open(obj_file_gt, 'r', encoding='utf8') as f_gt:
            obj_data_gt = f_gt.read()
            verts_gt    , simplices_gt     = obj_data_to_mesh3d(obj_data_gt)
        print('GT shape')
        print(verts_gt.shape)
        print(simplices_gt.shape)        
        with open(obj_file_ours_norm, 'r', encoding='utf8') as f_ours:
            obj_data_ours = f_ours.read()
            verts_ours  , simplices_ours   = obj_data_to_mesh3d(obj_data_ours)
        print('OURS shape')
        print(verts_ours.shape)
        print(simplices_ours.shape)
        with open(obj_file_theirs_norm, 'r', encoding='utf8') as f_theirs:
            obj_data_theirs = f_theirs.read()
            verts_theirs, simplices_theirs = obj_data_to_mesh3d(obj_data_theirs)
        print('THEIRS shape')
        print(verts_theirs.shape)
        print(simplices_theirs.shape)
        verts_theirs[:, [0,2]] = verts_theirs[:, [2,0]]
        HTML_rendering('GT_'     + str(batch_no_vec[i]).zfill(4), verts_gt    , simplices_gt    )
        HTML_rendering('OURS_'   + str(batch_no_vec[i]).zfill(4), verts_ours  , simplices_ours  )
        HTML_rendering('THEIRS_' + str(batch_no_vec[i]).zfill(4), verts_theirs, simplices_theirs)
        print('done.')
    TEST_DATASET.shutdown()
