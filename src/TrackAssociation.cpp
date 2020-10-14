#include "TrackAssociation.h"
//#include "HungarianAlgorithm.h"

float euclideanDist2(Point3f &p, Point3f &q) {
    Point2f a;
    a.x= p.x;
    a.y = p.y;

    Point2f b;
    b.x = q.x;
    b.y = q.y;

    Point2f diff = a - b;
    return cv::sqrt(diff.x*diff.x + diff.y*diff.y);
}

Group::Group()
{
    life_time = -1;
}
void Group::miss_Group(){
    if(life_time > -1)
        life_time--;

    if (life_time == -1)
        return;
}
int Group::get_life_time(){
    return life_time;
}
void Group::track(Point3f obj)
{
    m_obj = obj;

    if (life_time == -1)
    {
        ID = TrackID++;
        life_time = 13;
    }
    else
    {
        if (life_time < MAXLIFETIME)
        {
            life_time++;
        }
    }
}

CHungarianAlgorithm_pt::CHungarianAlgorithm_pt()
{
}
CHungarianAlgorithm_pt::~CHungarianAlgorithm_pt()
{
}

void CHungarianAlgorithm_pt::HAssociation(vector<Group> &vc_tracks, vector<Point3f> &candi, float DIST_TH)
{
    float xmax = 0;

    if (vc_tracks.size() > candi.size()) // 기존 트랙의 수가 더많을때
    {
        n = vc_tracks.size();
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                if (j >= candi.size())
                {
                    cost[i][j] = 255;
                    init_cost[i][j] = 255;
                    xmax = max(xmax, cost[i][j]);
                }
                else
                {
                    Point3f predict_candi_pt = vc_tracks[i].m_obj;
                    Point3f candi_pt = candi[j];
                    //float inv_IOU = MAXCOST - calcIOU(predict_candi_rect, candi_rect); // 최소값을 찾아야하기 때문에 1에서 빼줌
                    float Dist = euclideanDist2(predict_candi_pt,candi_pt);
                    cost[i][j] = Dist;
                    init_cost[i][j] = Dist;
                    xmax = max(xmax, cost[i][j]);
                }
            }
        }
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                cost[i][j] = xmax - cost[i][j];
            }
        }
        hungarian();
        for (int x = 0; x < n; x++)
        {
            if (init_cost[x][xMatch[x]] > DIST_TH) //  이하이면
            {
                vc_tracks[x].miss_Group();
                if (xMatch[x] < candi.size()) // 둘다 유효할떄
                {
                    Group newTrack;
                    newTrack.track(candi[xMatch[x]]);
                    vc_tracks.push_back(newTrack);
                }
            }
            else
            {
                vc_tracks[x].track(candi[xMatch[x]]);
            }
        }

    }
    else // 기존 트랙수가 더적을때
    {
        n = candi.size();

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                if (i >= vc_tracks.size())
                {
                    cost[i][j] = 255;
                    init_cost[i][j] = 255;
                    xmax = max(xmax, cost[i][j]);
                }
                else
                {
                    Point3f predict_candi_pt = vc_tracks[i].m_obj;
                    Point3f candi_pt = candi[j];
                    float Dist = euclideanDist2(predict_candi_pt,candi_pt);
                    cost[i][j] = Dist;
                    init_cost[i][j] = Dist;
                    xmax = max(xmax, cost[i][j]);
                }
            }
        }
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                cost[i][j] = xmax - cost[i][j];
            }
        }
        hungarian();

        for (int y = 0; y < n; y++)
        {
            if (init_cost[yMatch[y]][y] > DIST_TH) // IOU 0.2 이하이면
            {
                if (yMatch[y] < vc_tracks.size()) // 둘다 유효할때
                {
                    vc_tracks[yMatch[y]].miss_Group();
                }
                Group newTrack;
                newTrack.track(candi[y]);
                vc_tracks.push_back(newTrack);
            }
            else
            {
                vc_tracks[yMatch[y]].track(candi[y]);
            }
        }
    }
}

void CHungarianAlgorithm_pt::init_labels()
{
    memset(label_x, 0, sizeof(label_x));
    memset(label_y, 0, sizeof(label_y));      // y label은 모두 0으로 초기화.

    for (int x = 0; x < n; x++)
        for (int y = 0; y < n; y++)
            label_x[x] = max(label_x[x], cost[x][y]);    // cost중에 가장 큰 값을 label 값으로 잡음.
}

void CHungarianAlgorithm_pt::update_labels()
{
    float delta = (float)INF;

    // slack통해서 delta값 계산함.
    for (int y = 0; y < n; y++)
        if (!T[y]) delta = min(delta, slack[y]);

    for (int x = 0; x < n; x++)
        if (S[x]) label_x[x] -= delta;
    for (int y = 0; y < n; y++) {
        if (T[y]) label_y[y] += delta;
        else slack[y] -= delta;
    }
}

void CHungarianAlgorithm_pt::add_to_tree(int x, int parent_x)
{
    S[x] = true;            // S집합에 포함.
    parent[x] = parent_x;   // augmenting 할때 필요.

    for (int y = 0; y < n; y++) {                                   // 새 노드를 넣었으니, slack 갱신해야함.
        if (label_x[x] + label_y[y] - cost[x][y] < slack[y]) {
            slack[y] = label_x[x] + label_y[y] - cost[x][y];
            slackx[y] = x;
        }
    }
}

void CHungarianAlgorithm_pt::augment()
{
    if (Match_num == n) return;
    int root;   // 시작지점.
    queue<int> q;

    memset(S, false, sizeof(S));
    memset(T, false, sizeof(T));
    memset(parent, -1, sizeof(parent));

    // root를 찾음. 아직 매치안된 y값을 찾음ㅇㅇ.
    for (int x = 0; x < n; x++) {
        if (xMatch[x] == -1) {
            q.push(root = x);
            parent[x] = -2;
            S[x] = true;
            break;
        }
    }

    // slack 초기화.
    for (int y = 0; y < n; y++) {
        slack[y] = label_x[root] + label_y[y] - cost[root][y];
        slackx[y] = root;
    }

    int x, y;
    // augment function
    while (1) {
        // bfs cycle로 tree building.
        while (!q.empty()) {
            x = q.front(); q.pop();
            for (y = 0; y < n; y++) {
                if (cost[x][y] == label_x[x] + label_y[y] && !T[y]) {
                    if (yMatch[y] == -1) break;
                    T[y] = true;
                    q.push(yMatch[y]);
                    add_to_tree(yMatch[y], x);
                }
            }
            if (y < n) break;
        }
        if (y < n) break;

        while (!q.empty()) q.pop();

        update_labels(); // 증가경로가 없다면 label 향상ㄱ.

                         // label 향상을 통해서 equality graph의 새 edge를 추가함.
                         // !T[y] && slack[y]==0 인 경우에만 add 할 수 있음.
        for (y = 0; y < n; y++) {
            if (!T[y] && slack[y] == 0) {
                if (yMatch[y] == -1) {          // 증가경로 존재.
                    x = slackx[y];
                    break;
                }
                else {
                    T[y] = true;
                    if (!S[yMatch[y]]) {
                        q.push(yMatch[y]);
                        add_to_tree(yMatch[y], slackx[y]);
                    }
                }
            }
        }
        if (y < n) break;  // augment path found;
    }

    if (y < n) {        // augment path exist
        Match_num++;

        for (int cx = x, cy = y, ty; cx != -2; cx = parent[cx], cy = ty) {
            ty = xMatch[cx];
            yMatch[cy] = cx;
            xMatch[cx] = cy;
        }
        augment();  // 새 augment path 찾음.
    }
}

void CHungarianAlgorithm_pt::hungarian()
{
    Match_num = 0;

    memset(xMatch, -1, sizeof(xMatch));
    memset(yMatch, -1, sizeof(yMatch));

    init_labels();
    augment();
}

void TrackAssociation_pt::Track(vector<Point3f> &candi) // 현재 측정치
{
    std::vector<Group> prev_datas;

    if (vc_groups.size() == 0) // 기존 트랙이 없는경우
    {
        for (int i = 0; i < candi.size(); i++)
        {
            Group newTrack;
            newTrack.track(candi[i]);
            vc_groups.push_back(newTrack);
        }
    }
    else // 기존 트랙이 있는경우
    {
        float L_minCost = 5.0;
        if (candi.size() == 1 && vc_groups.size() == 1)
        {
            cv::Point3f candi0_pt = candi[0];
            cv::Point3f prev_candi_pt = vc_groups[0].m_obj;

            float cost = euclideanDist2(candi0_pt,prev_candi_pt);

            if (cost < L_minCost)
            {
                Group newTrack;
                newTrack.track(candi[0]);
                vc_groups.push_back(newTrack);
                vc_groups[0].miss_Group();
                if (vc_groups[0].life_time == -1)
                    vc_groups.erase(vc_groups.begin());
            }
            else
            {
                vc_groups[0].track(candi[0]);
                double theta = atan2(prev_candi_pt.y - candi0_pt.y, prev_candi_pt.x - candi0_pt.x);

                if((theta >= 0.0 && theta <= 1.5708) || (theta >= 4.71239 && theta <= 6.28319) ){
                    vc_groups[0].speed = (-1)*sqrt(pow(prev_candi_pt.y - candi0_pt.y, 2) + pow(prev_candi_pt.x - candi0_pt.x, 2))/0.1;
                }
                else{
                    vc_groups[0].speed = sqrt(pow(prev_candi_pt.y - candi0_pt.y, 2) + pow(prev_candi_pt.x - candi0_pt.x, 2))/0.1;
                }
            }
        }
        else
        {
            for(int i = 0; i < vc_groups.size(); i++){
                prev_datas.push_back(vc_groups.at(i));
            }

            CHungarianAlgorithm_pt h;
            h.HAssociation(vc_groups, candi, L_minCost);

            for (int i = 0; i < vc_groups.size(); i++)
            {
                if (vc_groups[i].life_time == -1)
                {
                    vc_groups.erase(vc_groups.begin() + i);
                    i--;
                }
            }

            for(int i = 0; i < vc_groups.size(); i++){
                for(int j = 0; j < prev_datas.size(); j++){
                    if(vc_groups.at(i).ID == prev_datas.at(j).ID){
                        // speed Median filter
                        double speed_pt = sqrt(pow(prev_datas.at(j).m_obj.y - vc_groups.at(i).m_obj.y, 2)+ pow(prev_datas.at(j).m_obj.x - vc_groups.at(i).m_obj.x, 2))*10.0;
                        double theta = atan2(vc_groups.at(i).m_obj.y - prev_datas.at(i).m_obj.y, vc_groups.at(i).m_obj.x - prev_datas.at(i).m_obj.x);
                        if(theta < 0.0){
                            theta = 6.28319 + theta;
                        }
                        vc_groups[i].theta = theta;

                        cv::Point3f xy_arr;

                        if((theta >= 0.0 && theta <= 1.5708) || (theta >= 4.71239 && theta <= 6.28319) ){
                            if(vc_groups[i].speed_arr.size() < 3){
                                vc_groups[i].speed = speed_pt;
                                vc_groups[i].speed_arr.push_back(speed_pt);
                            }
                            else{
                                vc_groups[i].speed_arr.push_back(speed_pt);
                                speed_pt = 0.0;
                                for(int K = 0; K < vc_groups[i].speed_arr.size(); K++){
                                    speed_pt += (K+1)*vc_groups[i].speed_arr.at(K);
                                }
                                vc_groups[i].speed_arr.erase(vc_groups[i].speed_arr.begin());
                                speed_pt = speed_pt/10.0;
                                vc_groups[i].speed = speed_pt;
                            }
                        }
                        else{
                            if(vc_groups[i].speed_arr.size() < 3){
                                vc_groups[i].speed = (-1)*speed_pt;
                                vc_groups[i].speed_arr.push_back((-1)*speed_pt);
                            }
                            else{
                                vc_groups[i].speed_arr.push_back((-1)*speed_pt);
                                speed_pt = 0.0;
                                for(int K = 0; K < vc_groups[i].speed_arr.size(); K++){
                                    speed_pt += (K+1)*vc_groups[i].speed_arr.at(K);
                                }
                                vc_groups[i].speed_arr.erase(vc_groups[i].speed_arr.begin());

                                speed_pt = speed_pt/10.0;
                                vc_groups[i].speed = speed_pt;
                            }
                        }

                        // Position Median filter
                        if(vc_groups[i].m_obj_arr.size() < 3){
                            vc_groups[i].m_obj_arr.push_back(vc_groups[i].m_obj);
                        }
                        else{
                            xy_arr.x = 0.0;
                            xy_arr.y = 0.0;
                            xy_arr.z = vc_groups[i].m_obj.z;

                            vc_groups[i].m_obj_arr.push_back(vc_groups[i].m_obj);

                            for(int K = 0; K < vc_groups[i].m_obj_arr.size(); K++){
                                xy_arr.x += (K + 1)*vc_groups[i].m_obj_arr.at(K).x;
                                xy_arr.y += (K + 1)*vc_groups[i].m_obj_arr.at(K).y;
                            }
                            vc_groups[i].m_obj_arr.erase(vc_groups[i].m_obj_arr.begin());

                            xy_arr.x = xy_arr.x/10.0;
                            xy_arr.y = xy_arr.y/10.0;

                            vc_groups[i].m_obj = xy_arr;
                        }
                    }
                }
            }
        }
    }
}
