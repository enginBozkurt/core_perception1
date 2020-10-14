#pragma once

#include <iostream>
#include <ros/ros.h>
#include <vector>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/passthrough.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/search/search.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/filters/extract_indices.h>
#include <pcl_ros/point_cloud.h>

#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud_conversion.h>

#include <visualization_msgs/MarkerArray.h>
#include <geometry_msgs/PoseArray.h>
#include <pcl/features/moment_of_inertia_estimation.h>

#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

#include <velodyne_pointcloud/point_types.h>
#include <velodyne_pointcloud/pointcloudXYZIR.h>

class Perception
{
private:
    ros::NodeHandle nh;
    ros::Subscriber sub_scan; //LiDAR Sub

    ros::Publisher clustering_pub;
    ros::Publisher polygon_pub;

    //initial var
    pcl::PointCloud<pcl::PointXYZI> m_lidar_point;

    cv::Point3d m_max; //ROI Range
    cv::Point3d m_min;

    int m_grid_dim; //Make HeightMap Value
    double m_per_cell;
    double m_height_diff_threshold;
    std::vector<std::vector<cv::Point2f>> raw_grid_data;

    double m_cluster_Range; //Clustering value
    int m_cluster_min;
    int m_cluster_max;

    std::vector<std::vector<cv::Point>> m_hull; //Convex Hull

public:
    Perception(ros::NodeHandle nh); //Constructor

    void init();
    void HesaiCallback(const sensor_msgs::PointCloud2Ptr scan); //LiDAR(Hesai) Raw Data

    // Function
    pcl::PointCloud<pcl::PointXYZI> f_lidar_Passthrough( pcl::PointCloud<pcl::PointXYZI> point); //pcl ROI
    pcl::PointCloud<pcl::PointXYZI> f_lidar_HeightMap(pcl::PointCloud<pcl::PointXYZI> point);
    pcl::PointCloud<pcl::PointXYZI> f_lidar_Euclidean_Clustering(pcl::PointCloud<pcl::PointXYZI> point);
    visualization_msgs::MarkerArray f_lidar_Drawing_polygon();
};
