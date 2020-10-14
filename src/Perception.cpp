#include "Perception.h"

Perception::Perception(ros::NodeHandle nh){
    double distance = 70.0;

    //initial value
    m_max.x = distance; //ROI value
    m_max.y = distance;
    m_max.z = 3.0;

    m_min.x = (-1)*distance;
    m_min.y = (-1)*distance;
    m_min.z = -3.0;

    m_grid_dim = distance*10.0; //HeightMap value
    m_per_cell = 0.2;
    m_height_diff_threshold = 0.05;

    m_cluster_Range = 0.6; //Clustering value
    m_cluster_min = 4;
    m_cluster_max = 400;

    //sub & pub
    sub_scan = nh.subscribe<sensor_msgs::PointCloud2Ptr> ("/pandar", 100, &Perception::HesaiCallback, this); //Subscribe Pandar 40M

    //bbox_check_publish
    polygon_pub = nh.advertise<visualization_msgs::MarkerArray>("/polygon_cube", 1);
    clustering_pub = nh.advertise<sensor_msgs::PointCloud2>("cluster_points", 1);
}

void Perception::HesaiCallback(const sensor_msgs::PointCloud2Ptr scan)
{
    m_lidar_point.clear();
    m_hull.clear();

    pcl::PointCloud<velodyne_pointcloud::PointXYZIR> arr;
    pcl::fromROSMsg(*scan, arr);

    for(int i = 0; i < arr.size(); i++){
        pcl::PointXYZI pt;
        pt.x         = arr[i].x;
        pt.y         = arr[i].y;
        pt.z         = arr[i].z;
        pt.intensity = arr[i].intensity;

        m_lidar_point.push_back(pt);
    }

    init();
}

void Perception::init(){
    //Function
    pcl::PointCloud<pcl::PointXYZI> roi_point = f_lidar_Passthrough(m_lidar_point);
    pcl::PointCloud<pcl::PointXYZI> hm_point = f_lidar_HeightMap(roi_point);
    pcl::PointCloud<pcl::PointXYZI> cluster_point = f_lidar_Euclidean_Clustering(hm_point);
    visualization_msgs::MarkerArray marker_arr = f_lidar_Drawing_polygon();

    polygon_pub.publish(marker_arr); //Polygon Publish
    sensor_msgs::PointCloud2 output;
    pcl::toROSMsg(cluster_point, output);
    sensor_msgs::PointCloud output_arr;
    sensor_msgs::convertPointCloud2ToPointCloud(output, output_arr);
    output.header.frame_id = "pos";
    clustering_pub.publish(output); //Clustering Data Publish
}

// LiDAR Points ROI setting
pcl::PointCloud<pcl::PointXYZI> Perception::f_lidar_Passthrough(pcl::PointCloud<pcl::PointXYZI> point){ //장애물의 전체적인 범위 설정을 통해서 필요없는 부분 제거
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZI>);
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_filter (new pcl::PointCloud<pcl::PointXYZI>);
    pcl::PointCloud<pcl::PointXYZI> filter;
    pcl::PassThrough <pcl::PointXYZI> pass;

    *cloud = point;
    pass.setInputCloud(cloud);
    pass.setFilterFieldName("z");
    pass.setFilterLimits(m_min.z, m_max.z); // z-axis
    pass.filter(*cloud_filter);
    pass.setInputCloud(cloud_filter);
    pass.setFilterFieldName("x");
    pass.setFilterLimits(m_min.x, m_max.x); // x-axis
    pass.filter(*cloud_filter);
    pass.setInputCloud(cloud_filter);
    pass.setFilterFieldName("y");
    pass.setFilterLimits(m_min.y, m_max.y); // y-axis
    pass.filter(*cloud_filter);
    filter = *cloud_filter;

    return filter;
}

pcl::PointCloud<pcl::PointXYZI> Perception::f_lidar_HeightMap(pcl::PointCloud<pcl::PointXYZI> point){
    pcl::PointCloud<pcl::PointXYZI> obstacle_cloud_;
    // pass along original time stamp and frame ID
    obstacle_cloud_.header.stamp = point.header.stamp;
    obstacle_cloud_.header.frame_id = point.header.frame_id;

    // set the exact point cloud size -- the vectors should already have enough space
    int num_point = point.size();
    int obj_count = 0;
    obstacle_cloud_.points.resize(num_point);

    std::vector<std::vector<float>> min;
    std::vector<std::vector<float>> max;
    std::vector<std::vector<unsigned int>> type;
    std::vector<std::vector<unsigned int>> num;
    std::vector<std::vector<bool>> init;

    min.assign(m_grid_dim, std::vector<float>(m_grid_dim, 0));
    max.assign(m_grid_dim, std::vector<float>(m_grid_dim, 0));
    num.assign(m_grid_dim, std::vector<unsigned int>(m_grid_dim, 0));
    type.assign(m_grid_dim, std::vector<unsigned int>(m_grid_dim, 0));
    init.assign(m_grid_dim, std::vector<bool>(m_grid_dim, false));

    // build height map
    for(unsigned i = 0; i < num_point; ++i){
        int x = ((m_grid_dim/2)+point.at(i)._PointXYZI::x/m_per_cell);
        int y = ((m_grid_dim/2)+point.at(i)._PointXYZI::y/m_per_cell);

        cv::Point2f arr;
        arr.x = point.at(i).x;
        arr.y = point.at(i).y;

        if(x >= 0 && x < m_grid_dim && y >= 0 && y < m_grid_dim){
            num[x][y] += 1;
            if(!init[x][y]){
                min[x][y] = point.at(i)._PointXYZI::z;
                max[x][y] = point.at(i)._PointXYZI::z;
                init[x][y] = true;
            }
            else{
                min[x][y] = MIN(min[x][y], point.at(i)._PointXYZI::z);
                max[x][y] = MAX(max[x][y], point.at(i)._PointXYZI::z);
            }
        }
    }

    // calculate number of obstacles, clear and unknown in each cell
    for(int x = 0; x < m_grid_dim; x++){
        for(int y = 0; y < m_grid_dim; y++){
            if(num[x][y] >= 1){
                if (max[x][y] - min[x][y] > m_height_diff_threshold /*|| (min[x][y] >= -1.0 && max[x][y] <= -0.2)*/){
                    type[x][y] = 2;
                }
                else{type[x][y] = 1;}
            }
        }
    }

    // create clouds from geid
    double grid_offset = m_grid_dim/2*m_per_cell;
    for (int x = 0; x < m_grid_dim; x++){
        for (int y = 0; y < m_grid_dim; y++){
            if(type[x][y] == 2){
                obstacle_cloud_.points[obj_count].x = -grid_offset + (x*m_per_cell+m_per_cell/2.0);
                obstacle_cloud_.points[obj_count].y = -grid_offset + (y*m_per_cell+m_per_cell/2.0);
                obstacle_cloud_.points[obj_count].z = m_height_diff_threshold;
                obj_count ++;
            }
        }
    }
    obstacle_cloud_.points.resize(obj_count);
    return obstacle_cloud_;
}


pcl::PointCloud<pcl::PointXYZI> Perception::f_lidar_Euclidean_Clustering(pcl::PointCloud<pcl::PointXYZI> point)
{
    pcl::PointCloud <pcl::PointXYZI> output;

    ///////////////////////// Clustering ////////////////////////////
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_obstacle(new pcl::PointCloud<pcl::PointXYZI>);
    cloud_obstacle = point.makeShared();

    pcl::search::KdTree<pcl::PointXYZI>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZI>);
    tree->setInputCloud(cloud_obstacle);

    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZI> ec;
    ec.setClusterTolerance(m_cluster_Range);
    ec.setMinClusterSize(m_cluster_min);
    ec.setMaxClusterSize(m_cluster_max);
    ec.setSearchMethod(tree);
    ec.setInputCloud(cloud_obstacle);
    ec.extract(cluster_indices);

    int j = 0;
    for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); ++it)
    {
        std::vector<cv::Point> points;
        for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); ++pit)
        {
            pcl::PointXYZI pt = cloud_obstacle->points[*pit];
            int v = m_max.y*10.0 - pt.y*10.0;
            int w = m_max.x*10.0 - pt.x*10.0;
            cv::Point point(w,v);
            points.push_back(point);

            pt._PointXYZI::intensity = j;
            output.push_back(pt);
        }

        std::vector<cv::Point> hull(points.size());
        cv::convexHull(points,hull);
        m_hull.push_back(hull);

        j++;
    }

    return output;
}

visualization_msgs::MarkerArray Perception::f_lidar_Drawing_polygon(){
    visualization_msgs::MarkerArray marker_result;
    std::vector<std::vector<cv::Point>> hull;
    hull = m_hull;

    //Drawing Rviz
    for(int i = 0; i < hull.size(); i++){
        visualization_msgs::Marker marker;
        marker.header.stamp = ros::Time::now();
        marker.header.frame_id = "pos";
        marker.ns = "adaptive_clustering";
        marker.id = i;
        marker.type = visualization_msgs::Marker::LINE_STRIP;

        geometry_msgs::Point p[hull[i].size()+1];

        for(int j = 0; j < hull[i].size(); j++){
            p[j].x = m_max.x - hull[i][j].x/10.0;
            p[j].y = m_max.y - hull[i][j].y/10.0;
            p[j].z = 0.05;
        }
        p[hull[i].size()].x = m_max.x - hull[i][0].x/10.0;
        p[hull[i].size()].y = m_max.y - hull[i][0].y/10.0;
        p[hull[i].size()].z = 0.05;

        for(int k =  0; k < hull[i].size()+1; k++){
            marker.points.push_back(p[k]);
        }
        marker.scale.x = 0.05;
        marker.color.a = 1.0;
        marker.color.r = 1.0;
        marker.color.g = 1.0;
        marker.color.b = 1.0;
        marker.lifetime = ros::Duration(0.1);

        marker_result.markers.push_back(marker);
    }

    return marker_result;
}
