#include "Perception.h"

LiDAR_Percept::LiDAR_Percept(){
    //initial value
    m_max_x = 50.0; //ROI value
    m_max_y = 50.0;
    m_max_z = -0.2;

    m_min_x = -50.0;
    m_min_y = -50.0;
    m_min_z = -3.0;

    m_grid_dim = 500; //HeightMap value
    m_per_cell = 0.2;
    m_height_diff_threshold = 0.05;

    m_cluster_Range = 0.5; //Clustering value
    m_cluster_min = 5;
    m_cluster_max = 600;

    m_Image_map = cv::imread("/home/a/av_ws/src/ImageMap(edit_center_lane).png",cv::IMREAD_GRAYSCALE);
}

LiDAR_Percept::LiDAR_Percept(ros::NodeHandle nh){
    //initial value
    m_max_x = 50.0; //ROI value
    m_max_y = 50.0;
    m_max_z = -0.2;

    m_min_x = -50.0;
    m_min_y = -50.0;
    m_min_z = -3.0;

    m_grid_dim = 500; //HeightMap value
    m_per_cell = 0.2;
    m_height_diff_threshold = 0.05;

    m_cluster_Range = 0.5; //Clustering value
    m_cluster_min = 5;
    m_cluster_max = 600;

    //sub & pub
    sub_scan = nh.subscribe<sensor_msgs::PointCloud2Ptr> ("/pandar", 100, &LiDAR_Percept::HesaiCallback, this); //Subscribe Pandar 40M
    //bbox_check_publish
    boundingbox_pub = nh.advertise<visualization_msgs::MarkerArray>("/boundingbox_cube", 1);
    clustering_pub = nh.advertise<sensor_msgs::PointCloud2> ("/clustering_debug", 1);
}

void LiDAR_Percept::HesaiCallback(const sensor_msgs::PointCloud2Ptr scan)
{
    PPointCloud arr;
    pcl::fromROSMsg(*scan, arr);

    m_lidar_Point = Hesai_Transform(arr);

    //Function
    pcl::PointCloud<pcl::PointXYZI> roi_point = f_lidar_Passthrough(m_lidar_Point);
    pcl::PointCloud<pcl::PointXYZI> hm_point = f_lidar_HeightMap(roi_point);
    visualization_msgs::MarkerArray bbox = f_lidar_Euclidean_Clustering(hm_point);

    boundingbox_pub.publish(bbox);
    //Bounding Box Debug
    pcl::PointCloud <pcl::PointXYZI> debug_point = debug();
    sensor_msgs::PointCloud2 output;
    pcl::toROSMsg(debug_point, output);
    sensor_msgs::PointCloud output_arr;
    sensor_msgs::convertPointCloud2ToPointCloud(output, output_arr);
    output.header.frame_id = "Pandar40M";
    clustering_pub.publish(output);
}

pcl::PointCloud<pcl::PointXYZI> LiDAR_Percept::Hesai_Transform(PPointCloud arr){
    pcl::PointCloud<pcl::PointXYZI> new_point;

    for(int i = 0; i < arr.size(); i++){
        if(arr.at(i).ring != 39){
            pcl::PointXYZI pt;

            pt._PointXYZI::x = arr.at(i).x;
            pt._PointXYZI::y = arr.at(i).y;
            pt._PointXYZI::z = arr.at(i).z;
            pt._PointXYZI::intensity = arr.at(i).intensity;

            new_point.push_back(pt);
        }
    }

    return new_point;
}

//Function to make
pcl::PointCloud<pcl::PointXYZI> LiDAR_Percept::f_lidar_Passthrough(pcl::PointCloud<pcl::PointXYZI> point){ //장애물의 전체적인 범위 설정을 통해서 필요없는 부분 제거
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZI>);
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_filter (new pcl::PointCloud<pcl::PointXYZI>);
    pcl::PointCloud<pcl::PointXYZI> filter;
    pcl::PassThrough <pcl::PointXYZI> pass;

    *cloud = point;
    pass.setInputCloud(cloud);
    pass.setFilterFieldName("z");
    pass.setFilterLimits(m_min_z, m_max_z);
    pass.filter(*cloud_filter);
    pass.setInputCloud(cloud_filter);
    pass.setFilterFieldName("x");
    pass.setFilterLimits(m_min_x, m_max_x);
    pass.filter(*cloud_filter);
    pass.setInputCloud(cloud_filter);
    pass.setFilterFieldName("y");
    pass.setFilterLimits(m_min_y, m_max_y);
    pass.filter(*cloud_filter);
    filter = *cloud_filter;

    return filter;
}

pcl::PointCloud<pcl::PointXYZI> LiDAR_Percept::f_lidar_HeightMap(pcl::PointCloud<pcl::PointXYZI> point){
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
                if (max[x][y] - min[x][y] > m_height_diff_threshold){type[x][y] = 2;}
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

visualization_msgs::MarkerArray LiDAR_Percept::f_lidar_Euclidean_Clustering(pcl::PointCloud<pcl::PointXYZI> point){
    pcl::PointCloud <pcl::PointXYZI> output;
    visualization_msgs::MarkerArray marker_result;
    clustering_check.resize(0);

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
    cv::Mat drawing = cv::Mat::zeros(1000, 1000, CV_8UC1);
    for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); ++it)
    {
        std::vector<cv::Point> points;

        for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); ++pit)
        {
            pcl::PointXYZI pt = cloud_obstacle->points[*pit];
            int v = 500 - pt.y*10.0;
            int w = 500 - pt.x*10.0;
            cv::Point point(w,v);
            points.push_back(point);

            pt._PointXYZI::intensity = j;
            output.push_back(pt);
        }
        std::vector<cv::Point> hull(points.size());
        cv::convexHull(points,hull);
        for(int i = 0 ; i < hull.size() ; i++ )
        {
            if(i > 0)
                cv::line(drawing,hull[i-1],hull[i],cv::Scalar(255),1);
            if(i == hull.size()-1)
                cv::line(drawing,hull[0],hull[i],cv::Scalar(255),1);
        }

        visualization_msgs::Marker marker;
        marker.header.stamp = ros::Time::now();
        marker.header.frame_id = "Pandar40M";
        marker.ns = "adaptive_clustering";
        marker.id = j;
        marker.type = visualization_msgs::Marker::LINE_STRIP;

        geometry_msgs::Point p[hull.size()+1];

        for(int i = 0; i < hull.size(); i++){
            p[i].x = 50 - hull.at(i).x/10.0;
            p[i].y = 50 - hull.at(i).y/10.0;
            p[i].z = 0.05;
        }
        p[hull.size()].x = 50 - hull.at(0).x/10.0;
        p[hull.size()].y = 50 - hull.at(0).y/10.0;
        p[hull.size()].z = 0.05;

        for(int k =  0; k < hull.size()+1; k++){
            marker.points.push_back(p[k]);
        }
        marker.scale.x = 0.3;
        marker.color.a = 1.0;
        marker.color.r = 1.0;
        marker.color.g = 0.0;
        marker.color.b = 0.0;
        marker.lifetime = ros::Duration(0.1);
        marker_result.markers.push_back(marker);
        j++;
    }
    ////////////////////////////////////////////////////////////////////////
    // cv::imshow("drawing",drawing);
    // cv::waitKey(10);
    clustering_check = output;

    return marker_result;
}

//Function Merge
cv::Mat getPaddedROI(const cv::Mat &input, int top_left_x, int top_left_y, int width, int height) {
    int bottom_right_x = top_left_x + width;
    int bottom_right_y = top_left_y + height;

    cv::Mat output;
    if (top_left_x < 0 || top_left_y < 0 || bottom_right_x > input.cols || bottom_right_y > input.rows) {
        // border padding will be required
        int border_left = 0, border_right = 0, border_top = 0, border_bottom = 0;

        if (top_left_x < 0) {
            width = width + top_left_x;
            border_left = -1 * top_left_x;
            top_left_x = 0;
        }
        if (top_left_y < 0) {
            height = height + top_left_y;
            border_top = -1 * top_left_y;
            top_left_y = 0;
        }
        if (bottom_right_x > input.cols) {
            width = width - (bottom_right_x - input.cols);
            border_right = bottom_right_x - input.cols;
        }
        if (bottom_right_y > input.rows) {
            height = height - (bottom_right_y - input.rows);
            border_bottom = bottom_right_y - input.rows;
        }

        cv::Rect R(top_left_x, top_left_y, width, height);
        copyMakeBorder(input(R), output, border_top, border_bottom, border_left, border_right, cv::BORDER_CONSTANT);
    }
    else {
        // no border padding required
        cv::Rect R(top_left_x, top_left_y, width, height);
        output = input(R);
    }
    return output;
}

visualization_msgs::MarkerArray LiDAR_Percept::detection_imgMap(pcl::PointCloud<pcl::PointXYZI> raw_point, double x, double y, double yaw){
    visualization_msgs::MarkerArray result;
    pcl::PointCloud<pcl::PointXYZI> passed_raw_point = f_lidar_Passthrough(raw_point);

    double res = 0.102655;
    int offset_x = (x/res)+1882;
    int offset_y = 8008 - (y/res+724);
    int result_x = offset_x - 487;
    int result_y = offset_y - 487;
    int result_w = 974; int result_h = 974;

    cv::Mat map_ROI = getPaddedROI(m_Image_map, result_x, result_y, result_w, result_h);
    cv::Point2f c_pt(result_w/2,result_h/2);
    double angle = yaw*180/3.1415926535897;
    cv::Mat r = cv::getRotationMatrix2D(c_pt,(angle*-1),1);
    cv::Mat map_rot_ROI;
    cv::warpAffine(map_ROI, map_rot_ROI, r, map_ROI.size());
    cv::Mat map_flip_ROI;
    cv::flip(map_rot_ROI, map_flip_ROI, 1);

    pcl::PointCloud<pcl::PointXYZI> roi_point;
    unsigned char *roi_map_data = (unsigned char *)map_flip_ROI.data;

    cv::Mat img; img = cv::Mat::zeros(974, 974, CV_8UC1);
    for(int i = 0 ; i < passed_raw_point.size() ; i++)
    {
        int ix = 487 - passed_raw_point[i]._PointXYZI::x*9.72;
        int iy = 487 - passed_raw_point[i]._PointXYZI::y*9.72;

        int img_index = iy*map_ROI.cols+ix;
        if(roi_map_data[img_index] != 0){
            roi_point.push_back(passed_raw_point[i]);
            img.at<uchar>(iy,ix) = 128;
        }
    }

    pcl::PointCloud<pcl::PointXYZI> hMap = f_lidar_HeightMap(roi_point);
    result = f_lidar_Euclidean_Clustering(hMap);

    return result;
}

//debug
pcl::PointCloud<pcl::PointXYZI> LiDAR_Percept::debug(){
    return clustering_check;
}
