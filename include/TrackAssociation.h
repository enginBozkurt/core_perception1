#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

#define MAXN 1000
#define INF 1e9;
#define MAXLIFETIME 20
#define MAXCOST 255.0

static int TrackID = 0;

using namespace cv;
using namespace std;

float euclideanDist2(Point2f& p, Point2f& q);

class Group{
public:
    int ID;
    //std::string class_name;
    int life_time;
    cv::Point3f m_obj;
    std::vector<cv::Point3f> m_obj_arr;
    double speed;
    std::vector<float> speed_arr;
    double theta;

public:
    Group();
    void miss_Group();
    int get_life_time();
    void track(Point3f obj);

};

class CHungarianAlgorithm_pt
{
public:
    int n, Match_num;                            // worker 수
    float label_x[MAXN], label_y[MAXN];           // label x, y
    int yMatch[MAXN];                           // y와 match되는 x
    int xMatch[MAXN];                           // x와 match되는 y
    bool S[MAXN], T[MAXN];                      // 알고리즘 상에 포함되는 vertex.
    float slack[MAXN];
    float slackx[MAXN];
    int parent[MAXN];                           // alternating path
    float cost[MAXN][MAXN];                       // cost
    float init_cost[MAXN][MAXN];                       // 초기 cost

    void init_labels();
    void update_labels();
    void add_to_tree(int x, int parent_x);
    void augment();
    void hungarian();


public:

    void HAssociation(vector<Group> &vc_tracks, vector<Point3f> &candi,float DIST_TH);

    CHungarianAlgorithm_pt();
    ~CHungarianAlgorithm_pt();
};



class TrackAssociation_pt
{
public:
     vector<Group> vc_groups;
public:
    void Track(vector<Point3f> &candi);
    TrackAssociation_pt(){
    }
    ~TrackAssociation_pt(){

    }
};

