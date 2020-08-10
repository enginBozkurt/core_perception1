#ifndef INCLUDE_POINT_TYPES_H_
#define INCLUDE_POINT_TYPES_H_

#include <iostream>
#include <ros/ros.h>
#include <vector>
#include <time.h>

#include <sensor_msgs/PointCloud2.h>
#include <pcl_ros/point_cloud.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud_conversion.h>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/passthrough.h>

#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/search/search.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/filters/extract_indices.h>
#include <visualization_msgs/MarkerArray.h>

#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <tf/transform_datatypes.h>

struct PointXYZIT {
  PCL_ADD_POINT4D
  uint8_t intensity;
  double timestamp;
  uint16_t ring;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT(
    PointXYZIT,
    (float, x, x)(float, y, y)(float, z, z)(uint8_t, intensity, intensity)(
        double, timestamp, timestamp)(uint16_t, ring, ring))

typedef PointXYZIT PPoint;
typedef pcl::PointCloud<PPoint> PPointCloud;

#endif  // INCLUDE_POINT_TYPES_H_

class LiDAR_Percept
{
private:
    ros::NodeHandle nh;
    ros::Subscriber sub_scan; //LiDAR Sub
    //bbox_check_publish
    ros::Publisher clustering_pub;
    ros::Publisher boundingbox_pub;

    //initial var
    cv::Mat m_Image_map; // Boundary Filter
    pcl::PointCloud<pcl::PointXYZI> m_lidar_Point;
    pcl::PointCloud<pcl::PointXYZI> clustering_check; //clustering debug

    double m_max_x; //ROI Range
    double m_max_y;
    double m_max_z;
    double m_min_x;
    double m_min_y;
    double m_min_z;

    int m_grid_dim; //Make HeightMap Value
    double m_per_cell;
    double m_height_diff_threshold;

    double m_cluster_Range; //Clustering value
    int m_cluster_min;
    int m_cluster_max;
public:
    //Basic Setting
    LiDAR_Percept(); //Constructor
    LiDAR_Percept(ros::NodeHandle nh); //Constructor

    void LiDARCallback(const sensor_msgs::PointCloud2Ptr scan); //LiDAR(Velodyne...) Raw Data

    void HesaiCallback(const sensor_msgs::PointCloud2Ptr scan); //LiDAR(Hesai) Raw Data
    pcl::PointCloud<pcl::PointXYZI> Hesai_Transform(PPointCloud arr); //Hesai PointCloud Transform

    // Function
    pcl::PointCloud<pcl::PointXYZI> f_lidar_Passthrough( pcl::PointCloud<pcl::PointXYZI> point); //pcl ROI
    pcl::PointCloud<pcl::PointXYZI> f_lidar_HeightMap(pcl::PointCloud<pcl::PointXYZI> point);
    visualization_msgs::MarkerArray f_lidar_Euclidean_Clustering(pcl::PointCloud<pcl::PointXYZI> point);

    //Function Merge code
    visualization_msgs::MarkerArray detection_imgMap(pcl::PointCloud<pcl::PointXYZI> raw_point, double x, double y, double yaw);

    //debug
    pcl::PointCloud<pcl::PointXYZI> debug();
};
