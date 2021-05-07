import os
import pymesh
import trimesh
import numpy as np
import mcubes
import plotly
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.offline import iplot
from scipy.optimize import linear_sum_assignment
################################################################################
# Evaluation for Sampled Point Cloud
################################################################################
# Chamfer Distance
def CD(PC, PC_T):
    # print('PC')
    # print(PC.shape)
    # print('PC_T')
    # print(PC_T.shape)
    ret1 = 0
    ret2 = 0
    for p1 in PC:
        diff1 = p1 - PC_T  # (1,3) - (N,3) = (N,3)
        dist1 = np.sum(diff1**2, axis=1)  # (N,1)
        ret1 += np.min(dist1)
    for p2 in PC_T:
        diff2 = p2 - PC
        dist2 = np.sum(diff2**2, axis=1)
        ret2 += np.min(dist2)
    return (ret1 + ret2)

# Earth Mover's Distance: Bijection Minimum Cost Matching using Hungarian Algorithm
def EMD(PC, PC_T):
    N = PC.shape[0]
    M = PC_T.shape[0]
    C = np.zeros((N,M))
    for i in range(N):
        for j in range(M):
            C[i][j] = np.sqrt(np.sum((PC[i] - PC_T[j])**2))
    row_ind, col_ind = linear_sum_assignment(C)
    return C[row_ind, col_ind].sum()


def FSCORE(PC, PC_T, thresh):
    min_dists1 = []
    min_dists2 = []
    for p1 in PC:
        diff1 = p1 - PC_T # (N,3)
        distances1 = np.sqrt(np.sum(diff1**2, axis=1))  # (N,1)
        min_d1 = np.min(distances1)
        min_dists1.append(min_d1)
    num_correct1 = 0
    for d1 in min_dists1:
        if d1 < thresh:
            num_correct1 += 1
    precision = num_correct1 / len(PC)

    for p2 in PC_T:
        diff2 = p2 - PC # (N,3)
        distances2 = np.sqrt(np.sum(diff2**2, axis=1))  # (N,1)
        min_d2 = np.min(distances2)
        min_dists2.append(min_d2)
    num_correct2 = 0
    for d2 in min_dists2:
        if d2 < thresh:
            num_correct2 += 1
    recall = num_correct2 / len(PC_T)
    return 2 * (precision*recall) / (precision+recall)

################################################################################



################################################################################
# Evaluation for Mesh
################################################################################
# def iou_pymesh(mesh_src, mesh_pred, dim=110):
#     try:
#         mesh1 = pymesh.load_mesh(mesh_src)
#         grid1 = pymesh.VoxelGrid(2./dim)
#         grid1.insert_mesh(mesh1)
#         grid1.create_grid()

#         ind1 = ((grid1.mesh.vertices + 1.1) / 2.4 * dim).astype(np.int)
#         v1 = np.zeros([dim, dim, dim])
#         v1[ind1[:,0], ind1[:,1], ind1[:,2]] = 1


#         mesh2 = pymesh.load_mesh(mesh_pred)
#         grid2 = pymesh.VoxelGrid(2./dim)
#         grid2.insert_mesh(mesh2)
#         grid2.create_grid()

#         ind2 = ((grid2.mesh.vertices + 1.1) / 2.4 * dim).astype(np.int)
#         v2 = np.zeros([dim, dim, dim])
#         v2[ind2[:,0], ind2[:,1], ind2[:,2]] = 1

#         intersection = np.sum(np.logical_and(v1, v2))
#         union = np.sum(np.logical_or(v1, v2))
#         return [float(intersection) / union, mesh_pred]
#     except:
#         print("error mesh {} / {}".format(mesh_src, mesh_pred))


# def export_obj_from_sdf(pred_sdf_val):
#     new_dim = int(pred_sdf_val.shape[0] ** (1.0 / 3)) + 1
#     u = pred_sdf_val.reshape(new_dim, new_dim, new_dim)
#     vertices, triangles = mcubes.marching_cubes(u, 0)
#     mcubes.export_obj(vertices, triangles, "obj/chair_pred.obj")




def obj_data_to_mesh3d(odata):
    # odata is the string read from an obj file
    vertices = []
    faces = []
    lines = odata.splitlines()

    for line in lines:
        slist = line.split()
        if slist:
            if slist[0] == 'v':
                vertex = np.array(slist[1:], dtype=float)
                vertices.append(vertex)
            elif slist[0] == 'f':
                face = []
                for k in range(1, len(slist)):
                    face.append([int(s) for s in slist[k].replace('//','/').split('/')])
                if len(face) > 3: # triangulate the n-polyonal face, n>3
                    faces.extend([[face[0][0]-1, face[k][0]-1, face[k+1][0]-1] for k in range(1, len(face)-1)])
                else:    
                    faces.append([face[j][0]-1 for j in range(len(face))])
            else: pass
    return np.array(vertices), np.array(faces)    





#def get_normalize_mesh(model_file, norm_mesh_sub_dir):
def get_normalize_mesh(model_file, norm_file):
    total = 16384 
    print("trimesh_load:", model_file)
    mesh_list = trimesh.load_mesh(model_file, process=False)
    if not isinstance(mesh_list, list):
        mesh_list = [mesh_list]
    area_sum = 0
    area_lst = []
    for idx, mesh in enumerate(mesh_list):
        area = np.sum(mesh.area_faces)
        area_lst.append(area)
        area_sum+=area
    area_lst = np.asarray(area_lst)
    amount_lst = (area_lst * total / area_sum).astype(np.int32)
    points_all=np.zeros((0,3), dtype=np.float32)
    for i in range(amount_lst.shape[0]):
        mesh = mesh_list[i]
        print("start sample surface of ", mesh.faces.shape[0])
        points, index = trimesh.sample.sample_surface(mesh, amount_lst[i])
        print("end sample surface")
        points_all = np.concatenate([points_all,points], axis=0)
    centroid = np.mean(points_all, axis=0)
    points_all = points_all - centroid
    m = np.max(np.sqrt(np.sum(points_all ** 2, axis=1)))
#    obj_file = os.path.join(norm_mesh_sub_dir, "pc_norm.obj")
    obj_file = norm_file
    ori_mesh = pymesh.load_mesh(model_file)
    print("centroid, m", centroid, m)
    pymesh.save_mesh_raw(obj_file, (ori_mesh.vertices - centroid) / float(m), ori_mesh.faces);
    print("export_mesh", obj_file)
    return obj_file, centroid, m



def HTML_rendering(name, verts, simplices):
    print('HTML Rendering: ' + name)
    _x, _y, _z = zip(*verts)
    colormap = ['rgb(255,105,180)', 'rgb(255,255,51)', 'rgb(0,191,255)']
    fig = ff.create_trisurf(x = _x,
                            y = _y,
                            z = _z,
                            simplices=simplices,
                            plot_edges=False,
                            colormap=colormap,
                            title=name)
    plotly.offline.plot(fig, auto_open=False, filename="html/chair_" + name + ".html")    
