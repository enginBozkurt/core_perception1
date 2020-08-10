#include "Perception.h"

int main(int argc, char **argv)
{
    ros::init(argc, argv, "lidar_perception");
    ros::NodeHandle nh;
    LiDAR_Percept a(nh);
//
    ros::spin();
    return 0;
}
