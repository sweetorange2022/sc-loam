//这个部分主要是对激光雷达数据进行预处理。包括激光雷达数据获取、点云数据分割、点云类别标注、数据发布。
#include "utility.h"

class ImageProjection{
private:

    ros::NodeHandle nh;

    //订阅雷达话题
    ros::Subscriber subLaserCloud;

    //发布话题
    ros::Publisher pubFullCloud;                                              //二维形式存储深度图对应的点云
    ros::Publisher pubFullInfoCloud;                                   ///带有到激光雷达的相对距离信息      的非地面点云信息

    ros::Publisher pubGroundCloud;                                  //地面点云
    ros::Publisher pubSegmentedCloud;                         //分割后的点云,包含降采样的地面点云
    ros::Publisher pubSegmentedCloudPure;               //纯分割后的点云
    ros::Publisher pubSegmentedCloudInfo;                 //segmsg
    ros::Publisher pubOutlierCloud;                                   //经过降采样的未被分割的点＋噪点之类的废弃点

    //typedef pcl::PointXYZI  PointType;
    pcl::PointCloud<PointType>::Ptr laserCloudIn;
    pcl::PointCloud<PointXYZIR>::Ptr laserCloudInRing;

    pcl::PointCloud<PointType>::Ptr fullCloud; // projected velodyne raw cloud, but saved in the form of 1-D matrix
    pcl::PointCloud<PointType>::Ptr fullInfoCloud; // same as fullCloud, but with intensity - range

    pcl::PointCloud<PointType>::Ptr groundCloud;
    pcl::PointCloud<PointType>::Ptr segmentedCloud;
    pcl::PointCloud<PointType>::Ptr segmentedCloudPure;
    pcl::PointCloud<PointType>::Ptr outlierCloud;

    PointType nanPoint; // fill in fullCloud at each iteration

    cv::Mat rangeMat; // range matrix for range image
    cv::Mat labelMat; // label matrix for segmentaiton marking
    cv::Mat groundMat; // ground matrix for ground cloud marking
    int labelCount;

    float startOrientation;
    float endOrientation;

    cloud_msgs::cloud_info segMsg; // info of segmented cloud
    std_msgs::Header cloudHeader;

    std::vector<std::pair<int8_t, int8_t> > neighborIterator; // neighbor iterator for segmentaiton process

    uint16_t *allPushedIndX; // array for tracking points of a segmented object
    uint16_t *allPushedIndY;

    uint16_t *queueIndX; // array for breadth-first search process of segmentation, for speed
    uint16_t *queueIndY;

public:
    ImageProjection():nh("~"){
        // subscriber
        subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2>(pointCloudTopic, 1, &ImageProjection::cloudHandler, this);

        // publisher
        pubFullCloud = nh.advertise<sensor_msgs::PointCloud2> ("/full_cloud_projected", 1);
        pubFullInfoCloud = nh.advertise<sensor_msgs::PointCloud2> ("/full_cloud_info", 1);

        pubGroundCloud = nh.advertise<sensor_msgs::PointCloud2> ("/ground_cloud", 1);
        pubSegmentedCloud = nh.advertise<sensor_msgs::PointCloud2> ("/segmented_cloud", 1);
        pubSegmentedCloudPure = nh.advertise<sensor_msgs::PointCloud2> ("/segmented_cloud_pure", 1);
        pubSegmentedCloudInfo = nh.advertise<cloud_msgs::cloud_info> ("/segmented_cloud_info", 1);
        pubOutlierCloud = nh.advertise<sensor_msgs::PointCloud2> ("/outlier_cloud", 1);         //离群点或异常点

        nanPoint.x = std::numeric_limits<float>::quiet_NaN();
        nanPoint.y = std::numeric_limits<float>::quiet_NaN();
        nanPoint.z = std::numeric_limits<float>::quiet_NaN();
        nanPoint.intensity = -1;// 无效点的标志

        // new内存
        allocateMemory();
        // 清空,初始化
        resetParameters();
    }

//使用allocateMemory对点云进行reset、resize、assign等重置、赋值等操作
    void allocateMemory(){

        laserCloudIn.reset(new pcl::PointCloud<PointType>());
        laserCloudInRing.reset(new pcl::PointCloud<PointXYZIR>());

        fullCloud.reset(new pcl::PointCloud<PointType>());
        fullInfoCloud.reset(new pcl::PointCloud<PointType>());

        groundCloud.reset(new pcl::PointCloud<PointType>());
        segmentedCloud.reset(new pcl::PointCloud<PointType>());
        segmentedCloudPure.reset(new pcl::PointCloud<PointType>());
        outlierCloud.reset(new pcl::PointCloud<PointType>());

        fullCloud->points.resize(N_SCAN*Horizon_SCAN);
        fullInfoCloud->points.resize(N_SCAN*Horizon_SCAN);

        segMsg.startRingIndex.assign(N_SCAN, 0);
        segMsg.endRingIndex.assign(N_SCAN, 0);

        segMsg.segmentedCloudGroundFlag.assign(N_SCAN*Horizon_SCAN, false);
        segMsg.segmentedCloudColInd.assign(N_SCAN*Horizon_SCAN, 0);
        segMsg.segmentedCloudRange.assign(N_SCAN*Horizon_SCAN, 0);

        std::pair<int8_t, int8_t> neighbor;
        neighbor.first = -1; neighbor.second =  0; neighborIterator.push_back(neighbor);
        neighbor.first =  0; neighbor.second =  1; neighborIterator.push_back(neighbor);
        neighbor.first =  0; neighbor.second = -1; neighborIterator.push_back(neighbor);
        neighbor.first =  1; neighbor.second =  0; neighborIterator.push_back(neighbor);

        allPushedIndX = new uint16_t[N_SCAN*Horizon_SCAN];
        allPushedIndY = new uint16_t[N_SCAN*Horizon_SCAN];

        queueIndX = new uint16_t[N_SCAN*Horizon_SCAN];
        queueIndY = new uint16_t[N_SCAN*Horizon_SCAN];
    }

//reset 参数
    void resetParameters(){
        laserCloudIn->clear();
        groundCloud->clear();
        segmentedCloud->clear();
        segmentedCloudPure->clear();
        outlierCloud->clear();

        rangeMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_32F, cv::Scalar::all(FLT_MAX));
        groundMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_8S, cv::Scalar::all(0));
        labelMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_32S, cv::Scalar::all(0));
        labelCount = 1;

        std::fill(fullCloud->points.begin(), fullCloud->points.end(), nanPoint);
        std::fill(fullInfoCloud->points.begin(), fullInfoCloud->points.end(), nanPoint);
    }
    //析构函数
    ~ImageProjection(){}

    //通过pcl::fromROSMsg函数将ROS的PointCloud保存成PCL的PointCloud，
    //通过pcl::removeNaNFromPointCloud函数对nan噪点（距离传感器过远过近或者是由于表面反射造成的无效点云，不是数的）进行滤除，避免了后续计算中出现各种异常情况。
    void copyPointCloud(const sensor_msgs::PointCloud2ConstPtr& laserCloudMsg){
        cloudHeader = laserCloudMsg->header;
        cloudHeader.stamp = ros::Time::now(); // Ouster lidar users may need to uncomment this line
        pcl::fromROSMsg(*laserCloudMsg, *laserCloudIn);
        // Remove Nan points
        std::vector<int> indices;
        // pcl::removeNaNFromPointCloud(laserCloudIn, laserCloudIn, indices);函数有三个参数，分别为输入点云，输出点云及对应保留的索引。
        pcl::removeNaNFromPointCloud(*laserCloudIn, *laserCloudIn, indices); 
        // have "ring" channel in the cloud
        if (useCloudRing == true){
            pcl::fromROSMsg(*laserCloudMsg, *laserCloudInRing);
            if (laserCloudInRing->is_dense == false) {
                ROS_ERROR("Point cloud is not in dense format, please remove NaN points first!");
                ros::shutdown();
            }  
        }
    }
    
    void cloudHandler(const sensor_msgs::PointCloud2ConstPtr& laserCloudMsg){

        // 1. 将传感器格式的点云转化成PCL格式的点云,还去除了nan点
        copyPointCloud(laserCloudMsg);
        // 2. 计算起止角度范围，将差值放在segMsg变量中
        findStartEndAngle();
        // 3. 投影成图片并计算点云所在行列和深度
        projectPointCloud();
        // 4. 地面过滤
        groundRemoval();
        // 5. 非地面点云聚类
        cloudSegmentation();
        // 6. 发布所有话题
        publishCloud();
        // 7. 为下一次（下一帧）迭代重置参数
        resetParameters();
    }

    // start and end orientation of this cloud
    //segMsg保存了当前帧的一些重要信息，包括起至角度，每个线的起至序号，及成员变量fullCloud中每个点的状态。
    void findStartEndAngle(){
        segMsg.startOrientation = -atan2(laserCloudIn->points[0].y, laserCloudIn->points[0].x);//第一个点相对于激光雷达的水平角度
        segMsg.endOrientation   = -atan2(laserCloudIn->points[laserCloudIn->points.size() - 1].y,laserCloudIn->points[laserCloudIn->points.size() - 1].x) + 2 * M_PI;
        //差值合理化
        if (segMsg.endOrientation - segMsg.startOrientation > 3 * M_PI) {
            segMsg.endOrientation -= 2 * M_PI;
        } else if (segMsg.endOrientation - segMsg.startOrientation < M_PI)
            segMsg.endOrientation += 2 * M_PI;
        //记录差值：计算雷达转过的角度
        segMsg.orientationDiff = segMsg.endOrientation - segMsg.startOrientation;
    }

    //该函数中将激光点云按照角度展开成图像的形式，计算所在行列和深度
    //行表示激光线束数量，列表示每个线上同一时刻扫描到的点,以x轴的负方向开始逆时针列序列号逐渐递增，即图像中的从左到右
    void projectPointCloud(){
        // range image projection
        float verticalAngle, horizonAngle, range;
        size_t rowIdn, columnIdn, index, cloudSize; 

        PointType thisPoint;
        cloudSize = laserCloudIn->points.size();

        for (size_t i = 0; i < cloudSize; ++i){
            thisPoint.x = laserCloudIn->points[i].x;
            thisPoint.y = laserCloudIn->points[i].y;
            thisPoint.z = laserCloudIn->points[i].z;
            // find the row and column index in the iamge for this point,计算所在行列和深度
            //计算竖直方向上的点的角度以及在整个雷达点云中的哪一条雷达线rowIdn上
            if (useCloudRing == true){
                rowIdn = laserCloudInRing->points[i].ring;
            }
            else{
            //计算垂直夹角
                verticalAngle = atan2(thisPoint.z, sqrt(thisPoint.x * thisPoint.x + thisPoint.y * thisPoint.y)) * 180 / M_PI;
            //计算第几条线
                rowIdn = (verticalAngle + ang_bottom) / ang_res_y;
            }
            //忽略异常值，正常应该在0-15之间，共16条
            if (rowIdn < 0 || rowIdn >= N_SCAN)
                continue;
            //计算水平方向上的角度，一行1800个像素点
            horizonAngle = atan2(thisPoint.x, thisPoint.y) * 180 / M_PI;
             
             //按照指定的小数位数进行四舍五入运算的结果
            columnIdn = -round((horizonAngle-90.0)/ang_res_x) + Horizon_SCAN/2;

            if (columnIdn >= Horizon_SCAN)   columnIdn -= Horizon_SCAN;

            if (columnIdn < 0 || columnIdn >= Horizon_SCAN)  continue;
            // 每个点的深度(到雷达的距离)
            range = sqrt(thisPoint.x * thisPoint.x + thisPoint.y * thisPoint.y + thisPoint.z * thisPoint.z);
            //剔除过近点(<1)
            if (range < sensorMinimumRange)  continue;
            
            rangeMat.at<float>(rowIdn, columnIdn) = range;
           //不是很懂为什么÷10000
            thisPoint.intensity = (float)rowIdn + (float)columnIdn / 10000.0;
            //给每一个点标好序号
            index = columnIdn  + rowIdn * Horizon_SCAN;

            fullCloud->points[index] = thisPoint;
            fullInfoCloud->points[index] = thisPoint;
            fullInfoCloud->points[index].intensity = range; // 和雷达的相对距离 is saved as "intensity"
        }
    }

    //滤除地面:从贴近地面的7个线中提取地面点。每两个圈之间进行一次比较，角度相差10°以内的我们可以看做是平地。
    //并且将地面点加入到groundCloud点云中。地面的点云会在groundMat中标记为1，labelMat中标记-1 ，不会参与后面的分割。
    void groundRemoval(){
        size_t lowerInd, upperInd;
        float diffX, diffY, diffZ, angle;
        for (size_t j = 0; j < Horizon_SCAN; ++j){
            for (size_t i = 0; i < groundScanInd; ++i){
                lowerInd = j + ( i )*Horizon_SCAN;
                upperInd = j + (i+1)*Horizon_SCAN;
                if (fullCloud->points[lowerInd].intensity == -1 ||fullCloud->points[upperInd].intensity == -1){
                    // no info to check, invalid points
                    groundMat.at<int8_t>(i,j) = -1;
                    // groundMat
                    // -1, no valid info to check if ground of not（无效点、未知点、被抛弃的点）
                    //  0, initial value, after validation, means not ground　非地面点
                    //  1, ground
                    continue;
                }                    
                diffX = fullCloud->points[upperInd].x - fullCloud->points[lowerInd].x;
                diffY = fullCloud->points[upperInd].y - fullCloud->points[lowerInd].y;
                diffZ = fullCloud->points[upperInd].z - fullCloud->points[lowerInd].z;
                angle = atan2(diffZ, sqrt(diffX*diffX + diffY*diffY) ) * 180 / M_PI;
                //如果小于10°就将前后两个点都标记为地面点，sensorMountAngle＝０
                if (abs(angle - sensorMountAngle) <= 10){
                    groundMat.at<int8_t>(i,j) = 1;
                    groundMat.at<int8_t>(i+1,j) = 1;
                }
            }
        }
        // extract ground cloud (groundMat == 1)
        // mark entry that doesn't need to label (ground and invalid point) for segmentation
        // note that ground remove is from 0~N_SCAN-1, need rangeMat for mark label matrix for the 16th scan
        for (size_t i = 0; i < N_SCAN; ++i){
            for (size_t j = 0; j < Horizon_SCAN; ++j){
                if (groundMat.at<int8_t>(i,j) == 1 || rangeMat.at<float>(i,j) == FLT_MAX){
                    labelMat.at<int>(i,j) = -1;
                }
            }
        }
        if (pubGroundCloud.getNumSubscribers() != 0){
            for (size_t i = 0; i <= groundScanInd; ++i){
                for (size_t j = 0; j < Horizon_SCAN; ++j){
                    if (groundMat.at<int8_t>(i,j) == 1)
                        groundCloud->push_back(fullCloud->points[j + i*Horizon_SCAN]);
                }
            }
        }
    }

    //对非地面点云的分割，并生成局部特征的步骤。
    //这个函数主要完成了两个任务，一是通过广度优先搜索，从非地面点中找出所有连成片的入射角比较小的patch上的点，并在labelMat标注patch的编号（从1开始）
    void cloudSegmentation(){
        // segmentation process:在排除地面点与异常点之后，逐一检测邻点特征并生成局部特征
        for (size_t i = 0; i < N_SCAN; ++i)
            for (size_t j = 0; j < Horizon_SCAN; ++j)
            // 非地面点云的分类，初始默认为0，地面点和噪声点被改为了-1
                if (labelMat.at<int>(i,j) == 0)
                    labelComponents(i, j);
        int sizeOfSegCloud = 0;
        // extract segmented cloud for lidar odometry
        for (size_t i = 0; i < N_SCAN; ++i) {
            segMsg.startRingIndex[i] = sizeOfSegCloud-1 + 5;
            for (size_t j = 0; j < Horizon_SCAN; ++j) {
                if (labelMat.at<int>(i,j) > 0 || groundMat.at<int8_t>(i,j) == 1){
                    // outliers that will not be used for optimization (always continue):噪声点和异常点永远不会被优化
                    // 如果是特征点或者是地面点，就可以纳入被分割点云
                    if (labelMat.at<int>(i,j) == 999999){
                        if (i > groundScanInd && j % 5 == 0){
                            outlierCloud->push_back(fullCloud->points[j + i*Horizon_SCAN]);
                            continue;
                        }else{
                            continue;
                        }
                    }
                    // majority of ground points are skipped：地面点云每隔5个点纳入被分割点云
                    if (groundMat.at<int8_t>(i,j) == 1){
                        if (j%5!=0 && j>5 && j<Horizon_SCAN-5)
                            continue;
                    }                    
                    // mark ground points so they will not be considered as edge features later
                    segMsg.segmentedCloudGroundFlag[sizeOfSegCloud] = (groundMat.at<int8_t>(i,j) == 1);
                    // mark the points' column index for marking occlusion later
                    segMsg.segmentedCloudColInd[sizeOfSegCloud] = j;
                    //把所有地面点和刚分割出来的labelCount上的点合并保存在segmentedCloud中
                    // save range info
                    segMsg.segmentedCloudRange[sizeOfSegCloud]  = rangeMat.at<float>(i,j);
                    // save seg cloud：把当前点纳入分割点云中
                    segmentedCloud->push_back(fullCloud->points[j + i*Horizon_SCAN]);
                    // size of seg cloud
                    ++sizeOfSegCloud;
                }
            }
            segMsg.endRingIndex[i] = sizeOfSegCloud-1 - 5;
        }        

        // extract segmented cloud for visualization：在当前有节点订阅便将分割点云的几何信息也发布出去
        if (pubSegmentedCloudPure.getNumSubscribers() != 0){
            for (size_t i = 0; i < N_SCAN; ++i){
                for (size_t j = 0; j < Horizon_SCAN; ++j){
                    if (labelMat.at<int>(i,j) > 0 && labelMat.at<int>(i,j) != 999999){
                        segmentedCloudPure->push_back(fullCloud->points[j + i*Horizon_SCAN]);
                        segmentedCloudPure->points.back().intensity = labelMat.at<int>(i,j);
                    }
                }
            }
        }
    }

    // 贴标签
    void labelComponents(int row, int col){
        float d1, d2, alpha, angle;
        int fromIndX, fromIndY, thisIndX, thisIndY; 
        bool lineCountFlag[N_SCAN] = {false};

        queueIndX[0] = row;
        queueIndY[0] = col;
        int queueSize = 1;
        int queueStartInd = 0;
        int queueEndInd = 1;

        allPushedIndX[0] = row;
        allPushedIndY[0] = col;
        int allPushedIndSize = 1;
        
        while(queueSize > 0){
            // Pop point
            fromIndX = queueIndX[queueStartInd];
            fromIndY = queueIndY[queueStartInd];
            --queueSize;
            ++queueStartInd;
            // Mark popped point
            labelMat.at<int>(fromIndX, fromIndY) = labelCount;
            // Loop through all the neighboring grids of popped grid 检查上下左右四个邻点
            for (auto iter = neighborIterator.begin(); iter != neighborIterator.end(); ++iter){
                // new index
                thisIndX = fromIndX + (*iter).first;
                thisIndY = fromIndY + (*iter).second;
                // index should be within the boundary
                if (thisIndX < 0 || thisIndX >= N_SCAN)
                    continue;
                // at range image margin (left or right side)
                if (thisIndY < 0)
                    thisIndY = Horizon_SCAN - 1;
                if (thisIndY >= Horizon_SCAN)
                    thisIndY = 0;
                // prevent infinite loop (caused by put already examined point back)
                if (labelMat.at<int>(thisIndX, thisIndY) != 0)
                    continue;
                //d1与d2分别是该点与某邻点的深度
                d1 = std::max(rangeMat.at<float>(fromIndX, fromIndY), rangeMat.at<float>(thisIndX, thisIndY));
                d2 = std::min(rangeMat.at<float>(fromIndX, fromIndY),  rangeMat.at<float>(thisIndX, thisIndY));

                if ((*iter).first == 0)
                    alpha = segmentAlphaX;
                else
                    alpha = segmentAlphaY;

                 // angle其实是该特点与某邻点的连线与XOZ平面的夹角，这个角代表了局部特征的敏感性
                angle = atan2(d2*sin(alpha), (d1 -d2*cos(alpha)));

                //如果夹角大于60°，则将这个邻点纳入到局部特征中，该邻点可以用来配准使用
                if (angle > segmentTheta){
                    queueIndX[queueEndInd] = thisIndX;
                    queueIndY[queueEndInd] = thisIndY;
                    ++queueSize;
                    ++queueEndInd;
                    labelMat.at<int>(thisIndX, thisIndY) = labelCount;
                    lineCountFlag[thisIndX] = true;
                    allPushedIndX[allPushedIndSize] = thisIndX;
                    allPushedIndY[allPushedIndSize] = thisIndY;
                    ++allPushedIndSize;
                }
            }
        }
        // check if this segment is valid:当邻点数目达到30后，则该帧雷达点云的几何特征配置成功
        bool feasibleSegment = false;
        if (allPushedIndSize >= 30)
            feasibleSegment = true;
        else if (allPushedIndSize  >= segmentValidPointNum){
            int lineCount = 0;
            for (size_t i = 0; i < N_SCAN; ++i)
                if (lineCountFlag[i] == true)
                    ++lineCount;
            if (lineCount >= segmentValidLineNum)
                feasibleSegment = true;            
        }
        // segment is valid, mark these points
        if (feasibleSegment == true){
            ++labelCount;
        }else{ // segment is invalid, mark these points
            for (size_t i = 0; i < allPushedIndSize; ++i){
                labelMat.at<int>(allPushedIndX[i], allPushedIndY[i]) = 999999;
            }
        }
    }

//ROS发布相关的话题
    void publishCloud(){
        // 1. Publish Seg Cloud Info
        segMsg.header = cloudHeader;
        pubSegmentedCloudInfo.publish(segMsg);
        // 2. Publish clouds
        // outlierCloud: 被放弃的点云：主要是经过降采样的未被分割点云。
        sensor_msgs::PointCloud2 laserCloudTemp;
        pcl::toROSMsg(*outlierCloud, laserCloudTemp);
        laserCloudTemp.header.stamp = cloudHeader.stamp;
        laserCloudTemp.header.frame_id = "base_link";
        pubOutlierCloud.publish(laserCloudTemp);
        // segmented cloud with ground点云数据:包含被分割点和经过降采样的地面点
        pcl::toROSMsg(*segmentedCloud, laserCloudTemp);
        laserCloudTemp.header.stamp = cloudHeader.stamp;
        laserCloudTemp.header.frame_id = "base_link";
        pubSegmentedCloud.publish(laserCloudTemp);
        // projected full cloud:二维形式存储深度图对应的点云
        if (pubFullCloud.getNumSubscribers() != 0){
            pcl::toROSMsg(*fullCloud, laserCloudTemp);
            laserCloudTemp.header.stamp = cloudHeader.stamp;
            laserCloudTemp.header.frame_id = "base_link";
            pubFullCloud.publish(laserCloudTemp);
        }
        // original dense ground cloud（原始的地面点云）
        if (pubGroundCloud.getNumSubscribers() != 0){
            pcl::toROSMsg(*groundCloud, laserCloudTemp);
            laserCloudTemp.header.stamp = cloudHeader.stamp;
            laserCloudTemp.header.frame_id = "base_link";
            pubGroundCloud.publish(laserCloudTemp);
        }
        // segmented cloud without ground（纯非地面点云聚类结果）
        if (pubSegmentedCloudPure.getNumSubscribers() != 0){
            pcl::toROSMsg(*segmentedCloudPure, laserCloudTemp);
            laserCloudTemp.header.stamp = cloudHeader.stamp;
            laserCloudTemp.header.frame_id = "base_link";
            pubSegmentedCloudPure.publish(laserCloudTemp);
        }
        // projected full cloud info带有到激光雷达的相对距离信息的非地面点云信息
        if (pubFullInfoCloud.getNumSubscribers() != 0){
            pcl::toROSMsg(*fullInfoCloud, laserCloudTemp);
            laserCloudTemp.header.stamp = cloudHeader.stamp;
            laserCloudTemp.header.frame_id = "base_link";
            pubFullInfoCloud.publish(laserCloudTemp);//带有到激光雷达的相对距离信息
        }
    }
};

int main(int argc, char** argv){

    ros::init(argc, argv, "lego_loam");
    
    ImageProjection IP;

    ROS_INFO("\033[1;32m---->\033[0m Image Projection Started.");

    ros::spin();
    return 0;
}
