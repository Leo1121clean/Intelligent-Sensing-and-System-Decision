import numpy as np
import open3d as o3d
import cv2
import math

final_count = 0 #counting point_cloud reconstruction
img2pcd_count = 0 #counting image_to_point_cloud
voxel_size = 0.05  # means 5cm for this dataset

#####depth_image_to_point_cloud#####
save_rgb = []
save_pixel = []
def get_pointcloud(img2pcd_count):
    
    save_rgb.clear()
    save_pixel.clear()
    
    frame_number = f"{img2pcd_count:05d}.png"
    img_sem = cv2.imread('semantic-segmentation-pytorch/reconstruct/apartment_' + frame_number)
    # img_sem = cv2.imread('semantic-segmentation-pytorch/reconstruct_gt/apartment_' + frame_number)
    img_depth = cv2.imread('semantic-segmentation-pytorch/data/apartment0_new/depths/depth_' + frame_number)
    
    fov = 90
    f = float(512/2*(1/math.tan(fov/180*math.pi/2)))
    
    for i in range(0,512):
        for j in range(0,512):
            x = (img_depth[i,j][0]/25.5)*(j-256)/f
            y = (img_depth[i,j][0]/25.5)*(i-256)/f
            
            if y>(-0.6):  #去除天花板
                save_pixel.append([x,y,img_depth[i,j][0]/25.5])
                save_rgb.append([img_sem[i][j][2]/255,img_sem[i][j][1]/255,img_sem[i][j][0]/255])
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(save_pixel)
    pcd.colors = o3d.utility.Vector3dVector(save_rgb)
    o3d.io.write_point_cloud('pcd/' + str(img2pcd_count) +  '.pcd',pcd)
    print("Save Point Cloud " + str(img2pcd_count))


#####初始化點雲#####
def prepare_dataset(voxel_size,source,target):
    
    print(":: Load two point clouds and disturb initial pose. " + str(final_count))

    demo_icp_pcds = o3d.data.DemoICPPointClouds()
    # source.estimate_normals()
    # target.estimate_normals()
    trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    source.transform(trans_init)
    #draw_registration_result(source, target, np.identity(4))
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source, target, source_down, target_down, source_fpfh, target_fpfh


#####點雲前處理#####
def preprocess_point_cloud(pcd, voxel_size):
    #print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    #print(":: Estimate normal with search radius %.3f." % radius_normal)
    #加上法向量(後面point to plane ICP會用到)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    #print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh


#####global registration#####
def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    #print(":: RANSAC registration on downsampled point clouds.")
    #print("   Since the downsampling voxel size is %.3f," % voxel_size)
    #print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result


#####Local refinement, 這裡做point-to-plane ICP#####
def refine_registration(source, target, source_fpfh, target_fpfh, voxel_size, result_ransac):
    distance_threshold = voxel_size * 0.4
    #print(":: Point-to-plane ICP registration is applied on original point")
    #print("   clouds to refine the alignment. This time we use a strict")
    #print("   distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return result

#####將轉移後的source存到下一次使用的target#####
def new_target(source_down,T):
    return source_down.transform(T)


#####整合過的獨立ICP function#####
def local_icp(voxel_size, total_count):
    
    global final_count
    final = []

    while True:
        
        source = o3d.io.read_point_cloud('pcd/' + str(final_count) + '.pcd')
        if final_count == 0:
            target = o3d.io.read_point_cloud('pcd/0.pcd')
        
        source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(voxel_size,source,target)
        result_ransac = execute_global_registration(source_down, target_down,source_fpfh, target_fpfh,voxel_size)
        result_icp = refine_registration(source_down, target_down, source_fpfh, target_fpfh,voxel_size, result_ransac)
        T = result_icp.transformation
        
        if final_count == 0:
            final.append(target_down)
            
        target = new_target(source_down,T)
        final.append(source_down)
        
        if final_count == total_count-1:
            break
        
        final_count = final_count + 1
    
    return final


if __name__ == "__main__":

    #讀取圖片張數
    with open('count.txt', 'r') as infile:
        total_count = int(infile.read())
    
    #生成點雲
    while img2pcd_count < total_count:
        get_pointcloud(img2pcd_count)
        img2pcd_count = img2pcd_count+1    
    
    #執行ICP重建函式(包含estimated trajectory)
    final = local_icp(voxel_size, total_count)

    #建立3D點雲資料
    final_pcd = o3d.geometry.PointCloud()
    for i in range(len(final)):
        final_pcd = final_pcd + final[i]
    

    #######################custom voxel down#######################
    xyz = np.asarray(final_pcd.points)
    xyz_color = np.asarray(final_pcd.colors)
    
    # 建立bounding box
    box = final_pcd.get_axis_aligned_bounding_box()
    box.color = (1,0,0)
    center = box.get_center() # 取質心
    margin = box.get_extent() # 取長寬高
    print("center: ", center, ", margin: ", margin)
    x_min = round(center[0] - margin[0]/2, 2)
    x_max = round(center[0] + margin[0]/2, 2)
    y_min = round(center[1] - margin[1]/2, 2)
    y_max = round(center[1] + margin[1]/2, 2)
    z_min = round(center[2] - margin[2]/2, 2)
    z_max = round(center[2] + margin[2]/2, 2)
    print("total point: ", len(xyz))
    print("x_min:", x_min, "x_max:", x_max, "y_min:", y_min, "y_max:", y_max, "z_min:", z_min, "z_max:", z_max)

    # # 顯示被voxel down的方格(觀察用可不打開)
    # box_z = [[x_min+0.2, y_min, z_min+1],[x_min+0.4, y_min, z_min+1],[x_min+0.2, y_min+0.2, z_min+1],[x_min+0.2, y_min, z_min+1.2]]
    # pcd_voxel = o3d.geometry.PointCloud()
    # pcd_voxel.points=o3d.utility.Vector3dVector(box_z)
    # box_voxel = pcd_voxel.get_axis_aligned_bounding_box()
    # box_voxel.color = (0,1,0)

    # 宣告降維後的pcd
    pcd_custom = o3d.geometry.PointCloud()
    point = []
    color = []

    voxel = 0.1
    count = 1
    
    # 最主要的voxel down函式，i,j,k分別代表x,y,z軸
    for i in range(int(x_min*100), int(x_max*100), int(voxel*100)):
        for j in range(int(y_min*100), int(y_max*100), int(voxel*100)):
            for k in range(int(z_min*100), int(z_max*100), int(voxel*100)):
                # print("step:", str(count))
                print("Success: ", round(count/(margin[0]*margin[1]*margin[2])*(voxel**3)*100, 1), "%")
                count = count + 1

                #在x軸取box範圍
                box_x = xyz[np.where((xyz[:,0]>=(i/100)) & (xyz[:,0]<=(i/100+voxel)))]
                box_color_voxel = xyz_color[np.where((xyz[:,0]>=(i/100)) & (xyz[:,0]<=(i/100+voxel)))]
                # print("After voxel x: ", box_x.shape[0])

                #在y軸取box範圍
                box_y = box_x[np.where((box_x[:,1]>=(j/100)) & (box_x[:,1]<=(j/100+voxel)))]
                box_color_voxel = box_color_voxel[np.where((box_x[:,1]>=(j/100)) & (box_x[:,1]<=(j/100+voxel)))]
                # print("After voxel y: ",box_y.shape[0])

                #在z軸取box範圍
                box_z = box_y[np.where((box_y[:,2]>=(k/100)) & (box_y[:,2]<=(k/100+voxel)))]
                box_color_voxel = box_color_voxel[np.where((box_y[:,2]>=(k/100)) & (box_y[:,2]<=(k/100+voxel)))]
                # print("numbers of picked points: ",box_z.shape[0])

                #計算出現次數最多的顏色,並新增到point, color
                box_color_voxel = np.around(box_color_voxel, 2)
                unique, counts = np.unique(box_color_voxel, axis=0, return_counts=True)
                if counts != []:
                    major_color = unique[counts.tolist().index(max(counts))]
                    # print("major color:", major_color)
                    point.append([(2*(i/100)+voxel)/2, (2*(j/100)+voxel)/2, (2*(k/100)+voxel)/2])
                    color.append(major_color)

    pcd_custom.points = o3d.utility.Vector3dVector(point)
    pcd_custom.colors = o3d.utility.Vector3dVector(color)

    #視覺化所有點雲資料及存檔
    o3d.visualization.draw_geometries([final_pcd])
    # o3d.visualization.draw_geometries([final_pcd, box, box_voxel])
    o3d.visualization.draw_geometries([pcd_custom])
    # o3d.io.write_point_cloud('final.pcd',final_pcd)