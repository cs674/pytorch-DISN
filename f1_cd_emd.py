from tqdm import tqdm
import os
import trimesh
from eval_util import obj_data_to_mesh3d, FSCORE, CD, EMD, get_normalize_mesh
import numpy as np
import sys

num_sample_points = 2048
batch_no_vec = [100,201,300,400,500,550, 1150]
for i in tqdm(range(len(batch_no_vec))):
    print('Collect GT surface samples...', end=' ')
    obj_file_gt    = './obj/gt/chair_gt_' + str(batch_no_vec[i]).zfill(4) + '.obj'
    # if not os.path.isfile(obj_file_gt):
    #     print('NOT EXIST: ' + obj_file_gt)
    #     continue
    with open(obj_file_gt, 'r', encoding='utf8') as f_gt:
        obj_data_gt = f_gt.read()
        pc_gt_surf,_                = obj_data_to_mesh3d(obj_data_gt)
    choice_gt      = np.random.randint(pc_gt_surf.shape[0], size=num_sample_points)
    PC_GT          = pc_gt_surf[choice_gt, ...]
    print('done.')


    print('Collect OURS surface samples...', end=' ')
    # NORMALIZED and CLEANED
    #obj_file_ours_norm                     = "./obj_cleaned/ours/chair_ours_norm_" + str(batch_no_vec[i]).zfill(4) + ".obj"
    # with open(obj_file_ours_norm, 'r', encoding='utf8') as f_ours:
    #     obj_data_ours = f_ours.read()
    #     pc_ours_surf,_                = obj_data_to_mesh3d(obj_data_ours)    

    # One more normalization
    obj_file_ours                     = "obj_cleaned/ours/chair_ours_norm_" + str(batch_no_vec[i]).zfill(4) + ".obj"
    obj_file_ours_cleaned_norm, centroid, m   = get_normalize_mesh(obj_file_ours, "obj/chair_ours_cleaned_normalized_" + str(batch_no_vec[i]).zfill(4) + ".obj")
    with open(obj_file_ours_cleaned_norm, 'r', encoding='utf8') as f_ours:
         obj_data_ours = f_ours.read()
         pc_ours_surf,_                = obj_data_to_mesh3d(obj_data_ours)    
    choice_ours                       = np.random.randint(pc_ours_surf.shape[0], size=num_sample_points)
    PC_OURS                           = pc_ours_surf[choice_ours, ...]
    print('done.')



    
    print('Collect THEIRS surface samples...', end=' ')
    #    obj_file_theirs                   = '../DISN_xar/demo/chair_theirs.obj'
    obj_file_theirs_norm                   = "./obj/theirs/chair_theirs_norm_" + str(batch_no_vec[i]).zfill(4) + ".obj"

    with open(obj_file_theirs_norm, 'r', encoding='utf8') as f_theirs:
        obj_data_theirs = f_theirs.read()
        pc_theirs_surf,_                = obj_data_to_mesh3d(obj_data_theirs)
    choice_theirs                     = np.random.randint(pc_theirs_surf.shape[0], size=num_sample_points)
    PC_THEIRS                         = pc_theirs_surf[choice_theirs, ...]
    PC_THEIRS[:, [0,2]]               = PC_THEIRS[:, [2,0]]


    print('Ground Truth')
    print(PC_GT[:10])
    print('Ours')
    print(PC_OURS[:10])
    print('Theirs')
    print(PC_THEIRS[:10])
    print('--------------------------------------------------------------------------------')
    print('FSCORE')
    #SIDE_LEN = 1.4


    x_side = np.max(PC_GT[:,0]) - np.min(PC_GT[:,0])
    y_side = np.max(PC_GT[:,1]) - np.min(PC_GT[:,1])
    z_side = np.max(PC_GT[:,2]) - np.min(PC_GT[:,2])
    SIDE_LEN = max(x_side, y_side, z_side)
    #print(x_side, y_side, z_side, SIDE_LEN)
    #sys.exit()


    
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
    # with open(obj_file_gt, 'r', encoding='utf8') as f_gt:
    #     obj_data_gt = f_gt.read()
    #     verts_gt    , simplices_gt     = obj_data_to_mesh3d(obj_data_gt)
    # print('GT shape')
    # print(verts_gt.shape)
    # print(simplices_gt.shape)        
    # with open(obj_file_ours_norm, 'r', encoding='utf8') as f_ours:
    #     obj_data_ours = f_ours.read()
    #     verts_ours  , simplices_ours   = obj_data_to_mesh3d(obj_data_ours)
    # print('OURS shape')
    # print(verts_ours.shape)
    # print(simplices_ours.shape)
    # with open(obj_file_theirs_norm, 'r', encoding='utf8') as f_theirs:
    #     obj_data_theirs = f_theirs.read()
    #     verts_theirs, simplices_theirs = obj_data_to_mesh3d(obj_data_theirs)
    # print('THEIRS shape')
    # print(verts_theirs.shape)
    # print(simplices_theirs.shape)
    # verts_theirs[:, [0,2]] = verts_theirs[:, [2,0]]
    # HTML_rendering('GT_'     + str(batch_no_vec[i]).zfill(4), verts_gt    , simplices_gt    )
    # HTML_rendering('OURS_'   + str(batch_no_vec[i]).zfill(4), verts_ours  , simplices_ours  )
    # HTML_rendering('THEIRS_' + str(batch_no_vec[i]).zfill(4), verts_theirs, simplices_theirs)
    print('done.')
