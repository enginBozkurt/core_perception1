#include "Perception.h"

int main(int argc, char **argv)
{
    ros::init(argc, argv, "lidar_perception");
    ros::NodeHandle nh;
    Perception node(nh);

    ros::spin();
    return 0;
}
