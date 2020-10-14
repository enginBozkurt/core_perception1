#include "Perception.h"


LiDAR_Percept::LiDAR_Percept(){
    double distance = 70.0;

    //initial value
    m_max.x = distance; //ROI value
    m_max.y = distance;
    m_max.z = -0.2;

    m_min.x = (-1)*distance;
    m_min.y = (-1)*distance;
    m_min.z = -3.0;

    m_grid_dim = distance*10.0; //HeightMap value
    m_per_cell = 0.2;
    m_height_diff_threshold = 0.05;
    std::cout << "hi" << std::endl;

    m_cluster_Range = 0.6; //Clustering value
    m_cluster_min = 4;
    m_cluster_max = 400;

    std::cout << "hi" << std::endl;

    m_Image_map = cv::imread("/home/a/av_ws/src/ImageMap(edit_center_lane).png",cv::IMREAD_GRAYSCALE);
}

LiDAR_Percept::LiDAR_Percept(ros::NodeHandle nh){
    double distance = 70.0;

    //initial value
    m_max.x = distance; //ROI value
    m_max.y = distance;
    m_max.z = -0.2;

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
    sub_scan = nh.subscribe<sensor_msgs::PointCloud2Ptr> ("/pandar", 100, &LiDAR_Percept::HesaiCallback, this); //Subscribe Pandar 40M
    ctrl_speed = nh.subscribe<ctrl_msgs::AvanteData>("/avante_data", 100, &LiDAR_Percept::speedCallback, this);

    //bbox_check_publish
    boundingbox_pub = nh.advertise<visualization_msgs::MarkerArray>("/boundingbox_cube", 1);
    ID_pub = nh.advertise<visualization_msgs::MarkerArray>("/boundingbox_ID", 1);
    clustering_pub = nh.advertise<sensor_msgs::PointCloud2> ("/clustering_debug", 1);
    pose_pub = nh.advertise<geometry_msgs::PoseArray>("poses", 100);
}

void LiDAR_Percept::speedCallback(const ctrl_msgs::AvanteData avante){
    m_speed += avante.wheel_speed.rear_left;
    m_speed += avante.wheel_speed.front_left;
    m_speed += avante.wheel_speed.rear_right;
    m_speed += avante.wheel_speed.front_right;

    m_speed = m_speed*(0.25);
}

void LiDAR_Percept::HesaiCallback(const sensor_msgs::PointCloud2Ptr scan)
{
    //m_speed = -1.0;

    raw_grid_data.resize(m_grid_dim*m_grid_dim);
    speed_result.markers.resize(0);

//    PPointCloud arr;
//    pcl::PointCloud<pcl::PointXYZI> arr;
    pcl::PointCloud<velodyne_pointcloud::PointXYZIR> arr;
    pcl::fromROSMsg(*scan, arr);

    m_lidar_Point = Hesai_Transform(arr);
    ros::Time t_total = ros::Time::now();
    //Function
    pcl::PointCloud<pcl::PointXYZI> roi_point = f_lidar_Passthrough(m_lidar_Point);
    pcl::PointCloud<pcl::PointXYZI> hm_point = f_lidar_HeightMap(roi_point);
    visualization_msgs::MarkerArray bbox = f_lidar_Euclidean_Clustering(hm_point);
    ros::Duration d_total = ros::Time::now() - t_total;

    //std::cout << d_total.toSec()*1000.0 << "ms" << std::endl;

    boundingbox_pub.publish(bbox);
    //Bounding Box Debug
    pcl::PointCloud <pcl::PointXYZI> debug_point = debug();
    sensor_msgs::PointCloud2 output;
    pcl::toROSMsg(hm_point, output);
    sensor_msgs::PointCloud output_arr;
    sensor_msgs::convertPointCloud2ToPointCloud(output, output_arr);
    output.header.frame_id = "pos";
    clustering_pub.publish(output);
}

pcl::PointCloud<pcl::PointXYZI> LiDAR_Percept::Hesai_Transform(pcl::PointCloud<velodyne_pointcloud::PointXYZIR> arr){
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
    pass.setFilterLimits(m_min.z, m_max.z);
    pass.filter(*cloud_filter);
    pass.setInputCloud(cloud_filter);
    pass.setFilterFieldName("x");
    pass.setFilterLimits(m_min.x, m_max.x);
    pass.filter(*cloud_filter);
    pass.setInputCloud(cloud_filter);
    pass.setFilterFieldName("y");
    pass.setFilterLimits(m_min.y, m_max.y);
    pass.filter(*cloud_filter);
    filter = *cloud_filter;

    return filter;
}

pcl::PointCloud<pcl::PointXYZI> LiDAR_Percept::f_lidar_HeightMap(pcl::PointCloud<pcl::PointXYZI> point){
    raw_grid_data.resize(m_grid_dim*m_grid_dim);
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

        raw_grid_data[(x*m_grid_dim)+y].push_back(arr);

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

    std::vector<Point3f> track_present;
    int j = 0;

    for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); ++it)
    {
        std::vector<cv::Point> points;

        double min_dis = 255.0;
        cv::Point2f min_point;

        for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); ++pit)
        {
            pcl::PointXYZI pt = cloud_obstacle->points[*pit];
            int v = m_max.y*10.0 - pt.y*10.0;
            int w = m_max.x*10.0 - pt.x*10.0;
            cv::Point point(w,v);
            points.push_back(point);

            pt._PointXYZI::intensity = j;
            output.push_back(pt);

            int grid_x = (pt.x + m_grid_dim/2.0*m_per_cell - m_per_cell/2.0)/m_per_cell;
            int grid_y = (pt.y + m_grid_dim/2.0*m_per_cell - m_per_cell/2.0)/m_per_cell;

            int check_real_num = grid_y+(grid_x*m_grid_dim);

            for(int i = 0; i < raw_grid_data.at(check_real_num).size(); i++){
                double dis = sqrt(pow(raw_grid_data[check_real_num][i].x, 2) + pow(raw_grid_data[check_real_num][i].y, 2));

                if(min_dis > dis){
                    min_dis = dis;
                    min_point.x = raw_grid_data[check_real_num][i].x;
                    min_point.y = raw_grid_data[check_real_num][i].y;
                }
            }


        }

        std::vector<cv::Point> hull(points.size());
        cv::convexHull(points,hull);

        // Convex Hull에서 추출된 중심점을 이용하여 현재 분류된 객체에

        cv::Point3f track_aarr;
        track_aarr.x = min_point.x;
        track_aarr.y = min_point.y;
        track_aarr.z = j;

        track_present.push_back(track_aarr);

        //Drawing Rviz
        visualization_msgs::Marker marker;
        marker.header.stamp = ros::Time::now();
        marker.header.frame_id = "pos";
        marker.ns = "adaptive_clustering";
        marker.id = j;
        marker.type = visualization_msgs::Marker::LINE_STRIP;

        geometry_msgs::Point p[hull.size()+1];

        for(int i = 0; i < hull.size(); i++){
            p[i].x = m_max.x - hull.at(i).x/10.0;
            p[i].y = m_max.y - hull.at(i).y/10.0;
            p[i].z = 0.05;
        }
        p[hull.size()].x = m_max.x - hull.at(0).x/10.0;
        p[hull.size()].y = m_max.y - hull.at(0).y/10.0;
        p[hull.size()].z = 0.05;

        for(int k =  0; k < hull.size()+1; k++){
            marker.points.push_back(p[k]);
        }
        marker.scale.x = 0.05;
        marker.color.a = 1.0;
        marker.color.r = 1.0;
        marker.color.g = 1.0;
        marker.color.b = 1.0;
//        marker.lifetime = ros::Duration(0.1);

        marker_result.markers.push_back(marker);
        j++;
    }
    ////////////////////////////////////////////////////////////////////////
    Track.Track(track_present);
    clustering_check = output;


//    for(int i = 0 ; i < Track.vc_groups.size() ; i++)
//    {
//        if(Track.vc_groups[i].life_time >= 18 && Track.vc_groups[i].m_obj.x != 0.0 && Track.vc_groups[i].m_obj.x != 0.0){
//            if(Track.vc_groups[i].m_obj.z < marker_result.markers.size()){
//                marker_result.markers.at(Track.vc_groups[i].m_obj.z).color.a = (m_max.x - Track.vc_groups[i].m_obj.x)/(m_max.x*2);
//                marker_result.markers.at(Track.vc_groups[i].m_obj.z).color.r = (m_max.y - Track.vc_groups[i].m_obj.y)/(m_max.y*2);
//                marker_result.markers.at(Track.vc_groups[i].m_obj.z).color.g = (Track.vc_groups[i].theta)/6.28319;
//                marker_result.markers.at(Track.vc_groups[i].m_obj.z).color.b = (127 - Track.vc_groups[i].speed)/255.0;
//            }
//        }
//    }

    pcl::PointCloud <pcl::PointXYZI> arr;
    for(int i = 0 ; i < Track.vc_groups.size() ; i++)
    {
        if(Track.vc_groups[i].life_time >= 18 && Track.vc_groups[i].m_obj.x != 0.0 && Track.vc_groups[i].m_obj.x != 0.0){
            string ID;
            cv::Point3f pt2d= Track.vc_groups[i].m_obj;
            pcl::PointXYZI pt;
            pt.x = pt2d.x;
            pt.y = pt2d.y;
            pt.z = (20 - Track.vc_groups[i].life_time)/20.0;
            pt.intensity = Track.vc_groups[i].ID%255;

            arr.push_back(pt);
            ID = std::to_string(Track.vc_groups[i].ID);
            ID += "\n";
            ID += std::to_string(Track.vc_groups[i].life_time);
            ID += "\n";
            ID += std::to_string(Track.vc_groups[i].speed + m_speed);
            ID.erase(ID.size()-4, ID.size());

            visualization_msgs::Marker marker;
            marker.header.frame_id = "pos";
            marker.header.stamp = ros::Time::now();
            marker.ns = "basic_shapes";
            marker.id = i;
            marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
            marker.action = visualization_msgs::Marker::ADD;

            marker.pose.position.x = pt.x;
            marker.pose.position.y = pt.y;
            marker.pose.position.z = 1.0;
            marker.pose.orientation.x = 0.0;
            marker.pose.orientation.y = 0.0;
            marker.pose.orientation.z = 0.0;
            marker.pose.orientation.w = 1.0;

            marker.text = ID;

            marker.scale.x = 1.0;
            marker.scale.y = 1.0;
            marker.scale.z = 1.0;

            marker.color.r = 1.0f;
            marker.color.g = 1.0f;
            marker.color.b = 1.0f;
            marker.color.a = 1.0;

            marker.lifetime = ros::Duration(0.1);
            speed_result.markers.push_back(marker);
        }
    }
    ID_pub.publish(speed_result);

//    clustering_check = arr;

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
    speed_result.markers.resize(0);

    double res = 0.102655;
    int offset_x = (x/res)+1882;
    int offset_y = 8008 - (y/res+724);
    int result_x = offset_x - 9.74*m_max.x;
    int result_y = offset_y - 9.74*m_max.y;
    int result_w = 9.74*m_max.x*2; int result_h = 9.74*m_max.y*2;
//    int result_x = offset_x - 487;
//    int result_y = offset_y - 487;
//    int result_w = 974; int result_h = 974;

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

    cv::Mat img; img = cv::Mat::zeros(result_w, result_h, CV_8UC1);
    for(int i = 0 ; i < passed_raw_point.size() ; i++)
    {
        int ix = result_w/2.0 - passed_raw_point[i]._PointXYZI::x*9.72;
        int iy = result_h/2.0 - passed_raw_point[i]._PointXYZI::y*9.72;

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
