import open3d as o3d
import numpy as np
import torch

from EJRGF import EJRGF_register

pcd_list = []
tensor_list = []
color_list = []
for i in range(10):
    filename = f"view{i+1}.txt"

    data = np.loadtxt(filename)

    tensor = torch.from_numpy(data[:, :3]).float().cuda()
    tensor_list.append(tensor)
    color_list.append(data[:, 3:6]/255.0)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(data[:, 3:6]/255.0)
    pcd_list.append(pcd)
o3d.visualization.draw_geometries(pcd_list)

subgroup_size = 10
gmm_mean_local_num = 500
gmm_mean_global_num = 0
epsilon = 1e-6
local_sigma = 0
local_iteration_num = 200
g_list = EJRGF_register(tensor_list, subgroup_size, gmm_mean_local_num, gmm_mean_global_num, epsilon, local_sigma, local_iteration_num)
print(g_list)
T_pcd_list = []
for i, mat in enumerate(g_list):
    cloud = tensor_list[i] @ mat[:3, :3].T + mat[:3, 3]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud.cpu().numpy())
    pcd.colors = o3d.utility.Vector3dVector(color_list[i])
    T_pcd_list.append(pcd)
o3d.visualization.draw_geometries(T_pcd_list)
