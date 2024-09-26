import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

if __name__ == "__main__":

    point = np.load('semantic_3d_pointcloud/point.npy')
    color = np.load('semantic_3d_pointcloud/color01.npy')

    # remove the ceiling and the floor
    point_cut = point[np.where((point[:,1]>(-0.03)) & (point[:,1]<(-0.005)))]
    color_cut = color[np.where((point[:,1]>(-0.03)) & (point[:,1]<(-0.005)))]
    point_cut = point_cut*10000./255.

    # plt.figure(figsize=(15,11))
    plt.scatter(point_cut[:,2], point_cut[:,0], s=5, c=color_cut, alpha=1)
    plt.axis('equal')
    plt.ylim(-4,7)
    plt.xlim(-5,10)
    plt.axis('off')
    plt.savefig("map.png", bbox_inches='tight',pad_inches = 0)
    plt.show()

    


    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cut)
    pcd.colors = o3d.utility.Vector3dVector(color_cut)
    # o3d.visualization.draw_geometries([pcd])