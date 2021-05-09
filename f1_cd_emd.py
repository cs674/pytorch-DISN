from tqdm import tqdm
import os
import trimesh
from eval_util import obj_data_to_mesh3d

batch_no_vec = [1,100,201,300,400,500,550,600,80,900,1150]
for i in tqdm(range(len(batch_no_vec))):
    print('Collect GT surface samples...', end=' ')
    obj_file_gt    = './obj/gt/chair_gt_' + str(batch_no_vec[i]).zfill(4) + '.obj'
    if not os.path.isfile(obj_file_gt):
        print('NOT EXIST: ' + obj_file_gt)
        continue
    with open(obj_file_gt, 'r', encoding='utf8') as f_gt:
        obj_data_gt = f_gt.read()
        pc_gt_surf,_                = obj_data_to_mesh3d(obj_data_gt)
    choice_gt      = np.random.randint(pc_gt_surf.shape[0], size=FLAGS.num_sample_points)
    PC_GT          = pc_gt_surf[choice_gt, ...]
    print('done.')


    print('Collect OURS surface samples...', end=' ')
    obj_file_ours                     = "./obj_cleaned/chair_ours_" + str(batch_no_vec[i]).zfill(4) + ".obj"
    with open(obj_file_ours_norm, 'r', encoding='utf8') as f_ours:
        obj_data_ours = f_ours.read()
        pc_ours_surf,_                = obj_data_to_mesh3d(obj_data_ours)
    choice_ours                       = np.random.randint(pc_ours_surf.shape[0], size=FLAGS.num_sample_points)
    PC_OURS                           = pc_ours_surf[choice_ours, ...]
    print('done.')

    print('Collect OURS surface samples...', end=' ')
    #    obj_file_theirs                   = '../DISN_xar/demo/chair_theirs.obj'
    obj_file_theirs                   = "./obj/theirs/chair_theirs_norm_" + str(batch_no_vec[i]).zfill(4) + ".obj"

    with open(obj_file_theirs_norm, 'r', encoding='utf8') as f_theirs:
        obj_data_theirs = f_theirs.read()
        pc_theirs_surf,_                = obj_data_to_mesh3d(obj_data_theirs)
    choice_theirs                     = np.random.randint(pc_theirs_surf.shape[0], size=FLAGS.num_sample_points)
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
