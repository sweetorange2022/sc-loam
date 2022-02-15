/*
前两部分已经完成了一个激光雷达里程计该做的处理（点云预处理，连续帧匹配计算出激光里程计信息）。
但是这个过程中误差是逐渐累积的，为此我们需要通过回环检测来减小误差.
在地图优化部分,文中使用ScanContext来实现scan-to-map的匹配，并检测到回环以及进行优化。

mapOptmization.cpp进行的内容主要是地图优化，将得到的局部地图信息融合到全局地图中去。
主要采用了两步优化方法:
a. scan-to-model的地图优化；
b. 因子图优化，
分别由 mapOptimization::scan2MapOptimization() 和 mapOptimization::saveKeyFramesAndFactor() 两个函数承担。
*/
#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/Values.h>

#include <gtsam/nonlinear/ISAM2.h>

#include "Scancontext.h"
#include "utility.h"

using namespace gtsam;

class mapOptimization{
private:
    // 因子图优化相关变量 
    NonlinearFactorGraph gtSAMgraph;
    Values initialEstimate;
    Values optimizedEstimate;
    ISAM2 *isam;
    Values isamCurrentEstimate;

    noiseModel::Diagonal::shared_ptr priorNoise;
    noiseModel::Diagonal::shared_ptr odometryNoise;
    noiseModel::Diagonal::shared_ptr constraintNoise;
    noiseModel::Base::shared_ptr robustNoiseModel;
    // ROS句柄
    ros::NodeHandle nh;
    // 多个发布者
    ros::Publisher pubLaserCloudSurround;
    ros::Publisher pubOdomAftMapped;
    ros::Publisher pubKeyPoses;

    ros::Publisher pubHistoryKeyFrames;
    ros::Publisher pubIcpKeyFrames;
    ros::Publisher pubRecentKeyFrames;
    ros::Publisher pubRegisteredCloud;
    // 多个订阅者 
    ros::Subscriber subLaserCloudRaw;
    ros::Subscriber subLaserCloudCornerLast;
    ros::Subscriber subLaserCloudSurfLast;
    ros::Subscriber subOutlierCloudLast;
    ros::Subscriber subLaserOdometry;
    ros::Subscriber subImu;
    // 为了信息发布而定义的msg
    nav_msgs::Odometry odomAftMapped;
    tf::StampedTransform aftMappedTrans;
    tf::TransformBroadcaster tfBroadcaster;

    vector<pcl::PointCloud<PointType>::Ptr> cornerCloudKeyFrames;              //   关键帧的角点点云集合
    vector<pcl::PointCloud<PointType>::Ptr> surfCloudKeyFrames;                 //   关键帧的平面点点云集合 
    vector<pcl::PointCloud<PointType>::Ptr> outlierCloudKeyFrames;           //   关键帧的离群点点云集合

    deque<pcl::PointCloud<PointType>::Ptr> recentCornerCloudKeyFrames;
    deque<pcl::PointCloud<PointType>::Ptr> recentSurfCloudKeyFrames;
    deque<pcl::PointCloud<PointType>::Ptr> recentOutlierCloudKeyFrames;
    int latestFrameID;

    vector<int> surroundingExistingKeyPosesID;                                // 局部关键帧集合(ID集合),(随着最新关键帧的改变而逐步增减的)
    deque<pcl::PointCloud<PointType>::Ptr> surroundingCornerCloudKeyFrames;       //局部关键帧角点点云集合
    deque<pcl::PointCloud<PointType>::Ptr> surroundingSurfCloudKeyFrames;             // 局部关键帧平面点点云集合
    deque<pcl::PointCloud<PointType>::Ptr> surroundingOutlierCloudKeyFrames;       // 局部关键帧离群点点云集合
    
    PointType previousRobotPosPoint;                            // 上一帧关键帧位姿点
    PointType currentRobotPosPoint;                               //  使用经过 scan-to-model 优化的位姿创建当前位置点

    pcl::PointCloud<PointType>::Ptr cloudKeyPoses3D;                  // 所有关键帧三自由度位置集合 (x,y,z表示位置,i表示位姿点在集合中的索引)（这是经过因子图优化后的位置）
    pcl::PointCloud<PointTypePose>::Ptr cloudKeyPoses6D;         //关键帧六自由度位姿集合 (x,y,z表示位置,i表示索引,rpy表示姿态,time记录里程计时间)（这是经过因子图优化后的位置）

    
    pcl::PointCloud<PointType>::Ptr surroundingKeyPoses;         //  局部关键帧位置集合——目标点附近的位置点集合。 
    pcl::PointCloud<PointType>::Ptr surroundingKeyPosesDS;  // ↑的降采样结果 

    pcl::PointCloud<PointType>::Ptr laserCloudRaw; 
    pcl::PointCloud<PointType>::Ptr laserCloudRawDS; 
    pcl::PointCloud<PointType>::Ptr laserCloudCornerLast; // corner feature set from odoOptimization　　　　　通过Msg接收到的Less角点点云
    pcl::PointCloud<PointType>::Ptr laserCloudSurfLast; // surf feature set from odoOptimization　　　　　　　　通过Msg接收到的Less平面点点云
    pcl::PointCloud<PointType>::Ptr laserCloudCornerLastDS; // downsampled corner featuer set from odoOptimization　Less角点点云的降采样 
    pcl::PointCloud<PointType>::Ptr laserCloudSurfLastDS; // downsampled surf featuer set from odoOptimization           Less平面点点云的降采样

    pcl::PointCloud<PointType>::Ptr laserCloudOutlierLast; // corner feature set from odoOptimization                                        通过Msg接收到的离群点点云 
    pcl::PointCloud<PointType>::Ptr laserCloudOutlierLastDS; // corner feature set from odoOptimization                              离群点点云的降采样 

    pcl::PointCloud<PointType>::Ptr laserCloudSurfTotalLast; // surf feature set from odoOptimization                                           Less平面点和离群点的点云 
    pcl::PointCloud<PointType>::Ptr laserCloudSurfTotalLastDS; // downsampled corner featuer set from odoOptimization           ↑的降采样结果 

    pcl::PointCloud<PointType>::Ptr laserCloudOri;
    pcl::PointCloud<PointType>::Ptr coeffSel;

    pcl::PointCloud<PointType>::Ptr laserCloudCornerFromMap;        // 局部角点点云地图  (每轮进行清空)
    pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMap;             // 局部平面点点云地图  (每轮进行清空)
    pcl::PointCloud<PointType>::Ptr laserCloudCornerFromMapDS;    // 局部角点点云的降采样  (每轮进行清空)
    pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMapDS;          // 局部角点点云的降采样  (每轮进行清空) 

    pcl::KdTreeFLANN<PointType>::Ptr kdtreeCornerFromMap;           // 局部角点点云地图 的KD树 
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurfFromMap;                // 局部平面点点云地图 的KD树

    pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurroundingKeyPoses;   //所有关键帧位置点集合的KD树
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeHistoryKeyPoses;

    pcl::PointCloud<PointType>::Ptr RSlatestSurfKeyFrameCloud; // giseop, RS: radius search 
    pcl::PointCloud<PointType>::Ptr RSnearHistorySurfKeyFrameCloud;
    pcl::PointCloud<PointType>::Ptr RSnearHistorySurfKeyFrameCloudDS;

    pcl::PointCloud<PointType>::Ptr nearHistoryCornerKeyFrameCloud;
    pcl::PointCloud<PointType>::Ptr nearHistoryCornerKeyFrameCloudDS;
    pcl::PointCloud<PointType>::Ptr SCnearHistorySurfKeyFrameCloud;
    pcl::PointCloud<PointType>::Ptr SCnearHistorySurfKeyFrameCloudDS;

    pcl::PointCloud<PointType>::Ptr latestCornerKeyFrameCloud;
    pcl::PointCloud<PointType>::Ptr SClatestSurfKeyFrameCloud; // giseop, SC: scan context
    pcl::PointCloud<PointType>::Ptr latestSurfKeyFrameCloudDS;

    pcl::KdTreeFLANN<PointType>::Ptr kdtreeGlobalMap;
    pcl::PointCloud<PointType>::Ptr globalMapKeyPoses;
    pcl::PointCloud<PointType>::Ptr globalMapKeyPosesDS;
    pcl::PointCloud<PointType>::Ptr globalMapKeyFrames;
    pcl::PointCloud<PointType>::Ptr globalMapKeyFramesDS;

    std::vector<int> pointSearchInd;
    std::vector<float> pointSearchSqDis;

    pcl::VoxelGrid<PointType> downSizeFilterScancontext;
    pcl::VoxelGrid<PointType> downSizeFilterCorner;
    pcl::VoxelGrid<PointType> downSizeFilterSurf;
    pcl::VoxelGrid<PointType> downSizeFilterOutlier;
    pcl::VoxelGrid<PointType> downSizeFilterHistoryKeyFrames; // for histor key frames of loop closure
    pcl::VoxelGrid<PointType> downSizeFilterSurroundingKeyPoses; // for surrounding key poses of scan-to-map optimization
    pcl::VoxelGrid<PointType> downSizeFilterGlobalMapKeyPoses; // for global map visualization
    pcl::VoxelGrid<PointType> downSizeFilterGlobalMapKeyFrames; // for global map visualization


    // 信息的时间戳 
    double timeLaserCloudCornerLast;                     // Less角点的时间
    double timeLaserCloudSurfLast;                           // Less平面点的时间 
    double timeLaserOdometry;                                  // 里程计时间
    double timeLaserCloudOutlierLast;                     // 离群点点云 的时间
    double timeLastGloalMapPublish;                       //上次发布的时间

    // 信息接收的标志位 
    bool newLaserCloudCornerLast;   //Less角点的标志位
    bool newLaserCloudSurfLast;     //Less平面点的标志位
    bool newLaserOdometry;          //里程计信息的标志位
    bool newLaserCloudOutlierLast;  //离群点点云的标志位

    float transformLast[6];                                                // 上一关键帧 经过因子图优化后的位姿
    float transformSum[6];                                               // 里程计的六自由度位姿(p y r x y z)
    float transformIncre[6];                                              // 两次位姿优化时，里程计的相对位姿变化
    float transformTobeMapped[6];                             // 机器人的位姿,全局姿态优化的目标变量：位姿优化的初始值->在地图优化过程中得到更新->经过因子图优化后的位姿
    float transformBefMapped[6];                                //  未经过scan-to-model优化的位姿，即里程计位姿。
    float transformAftMapped[6];                                  //  由经过scan-to-model优化后的位姿赋值,并在因子图优化后进行修正。


    int imuPointerFront;                                                      //与里程计时间对准的IMU数据索引 
    int imuPointerLast;                                                        //最新的IMU数据索引 

    // IMU时间、数据队列
    double imuTime[imuQueLength];
    float imuRoll[imuQueLength];
    float imuPitch[imuQueLength];

    std::mutex mtx;

    double timeLastProcessing;// 上一次进行全局位姿优化的时间

    PointType pointOri, pointSel, pointProj, coeff;

    cv::Mat matA0;//存放距离平面点最近的五个点(5×3矩阵)
    cv::Mat matB0;//5*1 的矩阵 
    cv::Mat matX0;//3*1 的矩阵

    cv::Mat matA1;//3×3 的协方差矩阵
    cv::Mat matD1;//1*3 的特征值向量
    cv::Mat matV1;// 3个 特征向量

    bool isDegenerate;
    cv::Mat matP;
    // 一些点云的数量
    int laserCloudCornerFromMapDSNum;                                // 局部角点点云中点的数量(经过降采样)
    int laserCloudSurfFromMapDSNum;                                      // 局部平面点点云中点的数量(经过降采样)
    int laserCloudCornerLastDSNum;
    int laserCloudSurfLastDSNum;
    int laserCloudOutlierLastDSNum;
    int laserCloudSurfTotalLastDSNum;

    bool potentialLoopFlag;
    double timeSaveFirstCurrentScanForLoopClosure;
    int RSclosestHistoryFrameID;
    int SCclosestHistoryFrameID; // giseop 
    int latestFrameIDLoopCloure;
    float yawDiffRad;

    bool aLoopIsClosed;

    float cRoll, sRoll, cPitch, sPitch, cYaw, sYaw, tX, tY, tZ;// 点云相对于世界坐标系的关系
    float ctRoll, stRoll, ctPitch, stPitch, ctYaw, stYaw, tInX, tInY, tInZ;//用来记录变换的三角函数值

    // loop detector 
    SCManager scManager;

public:


    // 初始化了ISAM2对象，写好了订阅和发布的话题以及下采样参数，还分配了内存
    mapOptimization():nh("~")
    {
    	ISAM2Params parameters;
		parameters.relinearizeThreshold = 0.01;
		parameters.relinearizeSkip = 1;
    	isam = new ISAM2(parameters);

        pubKeyPoses = nh.advertise<sensor_msgs::PointCloud2>("/key_pose_origin", 2);
        pubLaserCloudSurround = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_surround", 2);
        pubOdomAftMapped = nh.advertise<nav_msgs::Odometry> ("/aft_mapped_to_init", 5);

        //  写给订阅的回调函数：将传感器的格式转换为ｐｃｌ格式，然后当有新的数据传入的时候就把布尔型的变量改为真
        subLaserCloudRaw = nh.subscribe<sensor_msgs::PointCloud2>(pointCloudTopic, 2, &mapOptimization::laserCloudRawHandler, this);
        subLaserCloudCornerLast = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_corner_last", 2, &mapOptimization::laserCloudCornerLastHandler, this);
        subLaserCloudSurfLast = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_surf_last", 2, &mapOptimization::laserCloudSurfLastHandler, this);
        subOutlierCloudLast = nh.subscribe<sensor_msgs::PointCloud2>("/outlier_cloud_last", 2, &mapOptimization::laserCloudOutlierLastHandler, this);
        subLaserOdometry = nh.subscribe<nav_msgs::Odometry>("/laser_odom_to_init", 5, &mapOptimization::laserOdometryHandler, this);
        subImu = nh.subscribe<sensor_msgs::Imu> (imuTopic, 50, &mapOptimization::imuHandler, this);

        pubHistoryKeyFrames = nh.advertise<sensor_msgs::PointCloud2>("/history_cloud", 2);
        pubIcpKeyFrames = nh.advertise<sensor_msgs::PointCloud2>("/corrected_cloud", 2);
        pubRecentKeyFrames = nh.advertise<sensor_msgs::PointCloud2>("/recent_cloud", 2);
        pubRegisteredCloud = nh.advertise<sensor_msgs::PointCloud2>("/registered_cloud", 2);

   
        float filter_size;
        downSizeFilterCorner.setLeafSize(0.2, 0.2, 0.2);
        filter_size = 0.5; downSizeFilterScancontext.setLeafSize(filter_size, filter_size, filter_size);
        filter_size = 0.3; downSizeFilterSurf.setLeafSize(filter_size, filter_size, filter_size); // default 0.4;
        downSizeFilterOutlier.setLeafSize(0.4, 0.4, 0.4);

        filter_size = 0.3; downSizeFilterHistoryKeyFrames.setLeafSize(filter_size, filter_size, filter_size); // default 0.4; for histor key frames of loop closure
        filter_size = 1.0; downSizeFilterSurroundingKeyPoses.setLeafSize(filter_size, filter_size, filter_size); // default 1; for surrounding key poses of scan-to-map optimization

        downSizeFilterGlobalMapKeyPoses.setLeafSize(1.0, 1.0, 1.0); // for global map visualization
        downSizeFilterGlobalMapKeyFrames.setLeafSize(0.4, 0.4, 0.4); // for global map visualization

        odomAftMapped.header.frame_id = "/camera_init";
        odomAftMapped.child_frame_id = "/aft_mapped";

        aftMappedTrans.frame_id_ = "/camera_init";
        aftMappedTrans.child_frame_id_ = "/aft_mapped";

        allocateMemory();
    }

//分配内存
    void allocateMemory(){

        cloudKeyPoses3D.reset(new pcl::PointCloud<PointType>());
        cloudKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>());

        kdtreeSurroundingKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());
        kdtreeHistoryKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());

        surroundingKeyPoses.reset(new pcl::PointCloud<PointType>());
        surroundingKeyPosesDS.reset(new pcl::PointCloud<PointType>());        

        laserCloudRaw.reset(new pcl::PointCloud<PointType>()); // corner feature set from odoOptimization
        laserCloudRawDS.reset(new pcl::PointCloud<PointType>()); // corner feature set from odoOptimization
        laserCloudCornerLast.reset(new pcl::PointCloud<PointType>()); // corner feature set from odoOptimization
        laserCloudSurfLast.reset(new pcl::PointCloud<PointType>()); // surf feature set from odoOptimization
        laserCloudCornerLastDS.reset(new pcl::PointCloud<PointType>()); // downsampled corner featuer set from odoOptimization
        laserCloudSurfLastDS.reset(new pcl::PointCloud<PointType>()); // downsampled surf featuer set from odoOptimization
        laserCloudOutlierLast.reset(new pcl::PointCloud<PointType>()); // corner feature set from odoOptimization
        laserCloudOutlierLastDS.reset(new pcl::PointCloud<PointType>()); // downsampled corner feature set from odoOptimization
        laserCloudSurfTotalLast.reset(new pcl::PointCloud<PointType>()); // surf feature set from odoOptimization
        laserCloudSurfTotalLastDS.reset(new pcl::PointCloud<PointType>()); // downsampled surf featuer set from odoOptimization

        laserCloudOri.reset(new pcl::PointCloud<PointType>());
        coeffSel.reset(new pcl::PointCloud<PointType>());

        laserCloudCornerFromMap.reset(new pcl::PointCloud<PointType>());
        laserCloudSurfFromMap.reset(new pcl::PointCloud<PointType>());
        laserCloudCornerFromMapDS.reset(new pcl::PointCloud<PointType>());
        laserCloudSurfFromMapDS.reset(new pcl::PointCloud<PointType>());

        kdtreeCornerFromMap.reset(new pcl::KdTreeFLANN<PointType>());
        kdtreeSurfFromMap.reset(new pcl::KdTreeFLANN<PointType>());

        
        nearHistoryCornerKeyFrameCloud.reset(new pcl::PointCloud<PointType>());
        nearHistoryCornerKeyFrameCloudDS.reset(new pcl::PointCloud<PointType>());
        SCnearHistorySurfKeyFrameCloud.reset(new pcl::PointCloud<PointType>());
        SCnearHistorySurfKeyFrameCloudDS.reset(new pcl::PointCloud<PointType>());

        latestCornerKeyFrameCloud.reset(new pcl::PointCloud<PointType>());
        SClatestSurfKeyFrameCloud.reset(new pcl::PointCloud<PointType>());
        latestSurfKeyFrameCloudDS.reset(new pcl::PointCloud<PointType>());

        RSlatestSurfKeyFrameCloud.reset(new pcl::PointCloud<PointType>()); // giseop
        RSnearHistorySurfKeyFrameCloud.reset(new pcl::PointCloud<PointType>());
        RSnearHistorySurfKeyFrameCloudDS.reset(new pcl::PointCloud<PointType>());

        kdtreeGlobalMap.reset(new pcl::KdTreeFLANN<PointType>());
        globalMapKeyPoses.reset(new pcl::PointCloud<PointType>());
        globalMapKeyPosesDS.reset(new pcl::PointCloud<PointType>());
        globalMapKeyFrames.reset(new pcl::PointCloud<PointType>());
        globalMapKeyFramesDS.reset(new pcl::PointCloud<PointType>());

        timeLaserCloudCornerLast = 0;
        timeLaserCloudSurfLast = 0;
        timeLaserOdometry = 0;
        timeLaserCloudOutlierLast = 0;
        timeLastGloalMapPublish = 0;

        timeLastProcessing = -1;

        newLaserCloudCornerLast = false;
        newLaserCloudSurfLast = false;

        newLaserOdometry = false;
        newLaserCloudOutlierLast = false;

        for (int i = 0; i < 6; ++i){
            transformLast[i] = 0;
            transformSum[i] = 0;
            transformIncre[i] = 0;
            transformTobeMapped[i] = 0;
            transformBefMapped[i] = 0;
            transformAftMapped[i] = 0;
        }

        imuPointerFront = 0;
        imuPointerLast = -1;

        for (int i = 0; i < imuQueLength; ++i){
            imuTime[i] = 0;
            imuRoll[i] = 0;
            imuPitch[i] = 0;
        }

        gtsam::Vector Vector6(6);
        Vector6 << 1e-6, 1e-6, 1e-6, 1e-8, 1e-8, 1e-6;
        priorNoise = noiseModel::Diagonal::Variances(Vector6);
        odometryNoise = noiseModel::Diagonal::Variances(Vector6);

        matA0 = cv::Mat (5, 3, CV_32F, cv::Scalar::all(0));
        matB0 = cv::Mat (5, 1, CV_32F, cv::Scalar::all(-1));
        matX0 = cv::Mat (3, 1, CV_32F, cv::Scalar::all(0));

        matA1 = cv::Mat (3, 3, CV_32F, cv::Scalar::all(0));
        matD1 = cv::Mat (1, 3, CV_32F, cv::Scalar::all(0));
        matV1 = cv::Mat (3, 3, CV_32F, cv::Scalar::all(0));

        isDegenerate = false;
        matP = cv::Mat (6, 6, CV_32F, cv::Scalar::all(0));

        laserCloudCornerFromMapDSNum = 0;
        laserCloudSurfFromMapDSNum = 0;
        laserCloudCornerLastDSNum = 0;
        laserCloudSurfLastDSNum = 0;
        laserCloudOutlierLastDSNum = 0;
        laserCloudSurfTotalLastDSNum = 0;

        potentialLoopFlag = false;
        aLoopIsClosed = false;

        latestFrameID = 0;
    }
    //  坐标系->map,得到可用于建图的Lidar坐标，即修改transformTobeMapped的值；从雷达坐标系转换到世界坐标系下的变换
    // 根据当前和上一次全局姿态优化时的里程计 transformSum transformBefMapped 
    // 以及上一次全局姿态优化的结果 transformAftMapped  
    // 计算当前姿态优化的初始值，赋值给 transformTobeMapped 
    void transformAssociateToMap()
    {
        float x1 = cos(transformSum[1]) * (transformBefMapped[3] - transformSum[3]) 
                 - sin(transformSum[1]) * (transformBefMapped[5] - transformSum[5]);
        float y1 = transformBefMapped[4] - transformSum[4];
        float z1 = sin(transformSum[1]) * (transformBefMapped[3] - transformSum[3]) 
                 + cos(transformSum[1]) * (transformBefMapped[5] - transformSum[5]);

        float x2 = x1;
        float y2 = cos(transformSum[0]) * y1 + sin(transformSum[0]) * z1;
        float z2 = -sin(transformSum[0]) * y1 + cos(transformSum[0]) * z1;

        transformIncre[3] = cos(transformSum[2]) * x2 + sin(transformSum[2]) * y2;
        transformIncre[4] = -sin(transformSum[2]) * x2 + cos(transformSum[2]) * y2;
        transformIncre[5] = z2;

        float sbcx = sin(transformSum[0]);
        float cbcx = cos(transformSum[0]);
        float sbcy = sin(transformSum[1]);
        float cbcy = cos(transformSum[1]);
        float sbcz = sin(transformSum[2]);
        float cbcz = cos(transformSum[2]);

        float sblx = sin(transformBefMapped[0]);
        float cblx = cos(transformBefMapped[0]);
        float sbly = sin(transformBefMapped[1]);
        float cbly = cos(transformBefMapped[1]);
        float sblz = sin(transformBefMapped[2]);
        float cblz = cos(transformBefMapped[2]);

        float salx = sin(transformAftMapped[0]);
        float calx = cos(transformAftMapped[0]);
        float saly = sin(transformAftMapped[1]);
        float caly = cos(transformAftMapped[1]);
        float salz = sin(transformAftMapped[2]);
        float calz = cos(transformAftMapped[2]);

        float srx = -sbcx*(salx*sblx + calx*cblx*salz*sblz + calx*calz*cblx*cblz)
                  - cbcx*sbcy*(calx*calz*(cbly*sblz - cblz*sblx*sbly)
                  - calx*salz*(cbly*cblz + sblx*sbly*sblz) + cblx*salx*sbly)
                  - cbcx*cbcy*(calx*salz*(cblz*sbly - cbly*sblx*sblz) 
                  - calx*calz*(sbly*sblz + cbly*cblz*sblx) + cblx*cbly*salx);
        transformTobeMapped[0] = -asin(srx);

        float srycrx = sbcx*(cblx*cblz*(caly*salz - calz*salx*saly)
                     - cblx*sblz*(caly*calz + salx*saly*salz) + calx*saly*sblx)
                     - cbcx*cbcy*((caly*calz + salx*saly*salz)*(cblz*sbly - cbly*sblx*sblz)
                     + (caly*salz - calz*salx*saly)*(sbly*sblz + cbly*cblz*sblx) - calx*cblx*cbly*saly)
                     + cbcx*sbcy*((caly*calz + salx*saly*salz)*(cbly*cblz + sblx*sbly*sblz)
                     + (caly*salz - calz*salx*saly)*(cbly*sblz - cblz*sblx*sbly) + calx*cblx*saly*sbly);
        float crycrx = sbcx*(cblx*sblz*(calz*saly - caly*salx*salz)
                     - cblx*cblz*(saly*salz + caly*calz*salx) + calx*caly*sblx)
                     + cbcx*cbcy*((saly*salz + caly*calz*salx)*(sbly*sblz + cbly*cblz*sblx)
                     + (calz*saly - caly*salx*salz)*(cblz*sbly - cbly*sblx*sblz) + calx*caly*cblx*cbly)
                     - cbcx*sbcy*((saly*salz + caly*calz*salx)*(cbly*sblz - cblz*sblx*sbly)
                     + (calz*saly - caly*salx*salz)*(cbly*cblz + sblx*sbly*sblz) - calx*caly*cblx*sbly);
        transformTobeMapped[1] = atan2(srycrx / cos(transformTobeMapped[0]), 
                                       crycrx / cos(transformTobeMapped[0]));
        
        float srzcrx = (cbcz*sbcy - cbcy*sbcx*sbcz)*(calx*salz*(cblz*sbly - cbly*sblx*sblz)
                     - calx*calz*(sbly*sblz + cbly*cblz*sblx) + cblx*cbly*salx)
                     - (cbcy*cbcz + sbcx*sbcy*sbcz)*(calx*calz*(cbly*sblz - cblz*sblx*sbly)
                     - calx*salz*(cbly*cblz + sblx*sbly*sblz) + cblx*salx*sbly)
                     + cbcx*sbcz*(salx*sblx + calx*cblx*salz*sblz + calx*calz*cblx*cblz);
        float crzcrx = (cbcy*sbcz - cbcz*sbcx*sbcy)*(calx*calz*(cbly*sblz - cblz*sblx*sbly)
                     - calx*salz*(cbly*cblz + sblx*sbly*sblz) + cblx*salx*sbly)
                     - (sbcy*sbcz + cbcy*cbcz*sbcx)*(calx*salz*(cblz*sbly - cbly*sblx*sblz)
                     - calx*calz*(sbly*sblz + cbly*cblz*sblx) + cblx*cbly*salx)
                     + cbcx*cbcz*(salx*sblx + calx*cblx*salz*sblz + calx*calz*cblx*cblz);
        transformTobeMapped[2] = atan2(srzcrx / cos(transformTobeMapped[0]), 
                                       crzcrx / cos(transformTobeMapped[0]));

        x1 = cos(transformTobeMapped[2]) * transformIncre[3] - sin(transformTobeMapped[2]) * transformIncre[4];
        y1 = sin(transformTobeMapped[2]) * transformIncre[3] + cos(transformTobeMapped[2]) * transformIncre[4];
        z1 = transformIncre[5];

        x2 = x1;
        y2 = cos(transformTobeMapped[0]) * y1 - sin(transformTobeMapped[0]) * z1;
        z2 = sin(transformTobeMapped[0]) * y1 + cos(transformTobeMapped[0]) * z1;

        transformTobeMapped[3] = transformAftMapped[3] 
                               - (cos(transformTobeMapped[1]) * x2 + sin(transformTobeMapped[1]) * z2);
        transformTobeMapped[4] = transformAftMapped[4] - y2;
        transformTobeMapped[5] = transformAftMapped[5] 
                               - (-sin(transformTobeMapped[1]) * x2 + cos(transformTobeMapped[1]) * z2);
    }

    void transformUpdate()
    {
		if (imuPointerLast >= 0) {
		    float imuRollLast = 0, imuPitchLast = 0;
		    while (imuPointerFront != imuPointerLast) {
		        if (timeLaserOdometry + scanPeriod < imuTime[imuPointerFront]) {
		            break;
		        }
		        imuPointerFront = (imuPointerFront + 1) % imuQueLength;
		    }

		    if (timeLaserOdometry + scanPeriod > imuTime[imuPointerFront]) {
		        imuRollLast = imuRoll[imuPointerFront];
		        imuPitchLast = imuPitch[imuPointerFront];
		    } else {
		        int imuPointerBack = (imuPointerFront + imuQueLength - 1) % imuQueLength;
		        float ratioFront = (timeLaserOdometry + scanPeriod - imuTime[imuPointerBack]) 
		                         / (imuTime[imuPointerFront] - imuTime[imuPointerBack]);
		        float ratioBack = (imuTime[imuPointerFront] - timeLaserOdometry - scanPeriod) 
		                        / (imuTime[imuPointerFront] - imuTime[imuPointerBack]);

		        imuRollLast = imuRoll[imuPointerFront] * ratioFront + imuRoll[imuPointerBack] * ratioBack;
		        imuPitchLast = imuPitch[imuPointerFront] * ratioFront + imuPitch[imuPointerBack] * ratioBack;
		    }

		    transformTobeMapped[0] = 0.998 * transformTobeMapped[0] + 0.002 * imuPitchLast;
		    transformTobeMapped[2] = 0.998 * transformTobeMapped[2] + 0.002 * imuRollLast;
		  }

		for (int i = 0; i < 6; i++) {
		    transformBefMapped[i] = transformSum[i];
		    transformAftMapped[i] = transformTobeMapped[i];
		}
    }

    void updatePointAssociateToMapSinCos(){
        cRoll = cos(transformTobeMapped[0]);
        sRoll = sin(transformTobeMapped[0]);

        cPitch = cos(transformTobeMapped[1]);
        sPitch = sin(transformTobeMapped[1]);

        cYaw = cos(transformTobeMapped[2]);
        sYaw = sin(transformTobeMapped[2]);

        tX = transformTobeMapped[3];
        tY = transformTobeMapped[4];
        tZ = transformTobeMapped[5];
    }

    void pointAssociateToMap(PointType const * const pi, PointType * const po)
    {
        float x1 = cYaw * pi->x - sYaw * pi->y;
        float y1 = sYaw * pi->x + cYaw * pi->y;
        float z1 = pi->z;

        float x2 = x1;
        float y2 = cRoll * y1 - sRoll * z1;
        float z2 = sRoll * y1 + cRoll * z1;

        po->x = cPitch * x2 + sPitch * z2 + tX;
        po->y = y2 + tY;
        po->z = -sPitch * x2 + cPitch * z2 + tZ;
        po->intensity = pi->intensity;
    }

    void updateTransformPointCloudSinCos(PointTypePose *tIn){

        ctRoll = cos(tIn->roll);
        stRoll = sin(tIn->roll);

        ctPitch = cos(tIn->pitch);
        stPitch = sin(tIn->pitch);

        ctYaw = cos(tIn->yaw);
        stYaw = sin(tIn->yaw);

        tInX = tIn->x;
        tInY = tIn->y;
        tInZ = tIn->z;
    }

    pcl::PointCloud<PointType>::Ptr transformPointCloud(pcl::PointCloud<PointType>::Ptr cloudIn){
	// !!! DO NOT use pcl for point cloud transformation, results are not accurate
        // Reason: unkown
        pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

        PointType *pointFrom;
        PointType pointTo;

        int cloudSize = cloudIn->points.size();
        cloudOut->resize(cloudSize);

        for (int i = 0; i < cloudSize; ++i){

            pointFrom = &cloudIn->points[i];
            float x1 = ctYaw * pointFrom->x - stYaw * pointFrom->y;
            float y1 = stYaw * pointFrom->x + ctYaw* pointFrom->y;
            float z1 = pointFrom->z;

            float x2 = x1;
            float y2 = ctRoll * y1 - stRoll * z1;
            float z2 = stRoll * y1 + ctRoll* z1;

            pointTo.x = ctPitch * x2 + stPitch * z2 + tInX;
            pointTo.y = y2 + tInY;
            pointTo.z = -stPitch * x2 + ctPitch * z2 + tInZ;
            pointTo.intensity = pointFrom->intensity;

            cloudOut->points[i] = pointTo;
        }
        return cloudOut;
    }

    pcl::PointCloud<PointType>::Ptr transformPointCloud(pcl::PointCloud<PointType>::Ptr cloudIn, PointTypePose* transformIn){

        pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

        PointType *pointFrom;
        PointType pointTo;

        int cloudSize = cloudIn->points.size();
        cloudOut->resize(cloudSize);
        
        for (int i = 0; i < cloudSize; ++i){

            pointFrom = &cloudIn->points[i];
            float x1 = cos(transformIn->yaw) * pointFrom->x - sin(transformIn->yaw) * pointFrom->y;
            float y1 = sin(transformIn->yaw) * pointFrom->x + cos(transformIn->yaw)* pointFrom->y;
            float z1 = pointFrom->z;

            float x2 = x1;
            float y2 = cos(transformIn->roll) * y1 - sin(transformIn->roll) * z1;
            float z2 = sin(transformIn->roll) * y1 + cos(transformIn->roll)* z1;

            pointTo.x = cos(transformIn->pitch) * x2 + sin(transformIn->pitch) * z2 + transformIn->x;
            pointTo.y = y2 + transformIn->y;
            pointTo.z = -sin(transformIn->pitch) * x2 + cos(transformIn->pitch) * z2 + transformIn->z;
            pointTo.intensity = pointFrom->intensity;

            cloudOut->points[i] = pointTo;
        }
        return cloudOut;
    }

    void laserCloudOutlierLastHandler(const sensor_msgs::PointCloud2ConstPtr& msg){
        timeLaserCloudOutlierLast = msg->header.stamp.toSec();
        laserCloudOutlierLast->clear();
        pcl::fromROSMsg(*msg, *laserCloudOutlierLast);
        newLaserCloudOutlierLast = true;
    }

    void laserCloudRawHandler(const sensor_msgs::PointCloud2ConstPtr& msg){
        laserCloudRaw->clear();
        pcl::fromROSMsg(*msg, *laserCloudRaw);
    }

    void laserCloudCornerLastHandler(const sensor_msgs::PointCloud2ConstPtr& msg){
        timeLaserCloudCornerLast = msg->header.stamp.toSec();
        laserCloudCornerLast->clear();
        pcl::fromROSMsg(*msg, *laserCloudCornerLast);
        newLaserCloudCornerLast = true;
    }

    void laserCloudSurfLastHandler(const sensor_msgs::PointCloud2ConstPtr& msg){
        timeLaserCloudSurfLast = msg->header.stamp.toSec();
        laserCloudSurfLast->clear();
        pcl::fromROSMsg(*msg, *laserCloudSurfLast);
        newLaserCloudSurfLast = true;
    }

    void laserOdometryHandler(const nav_msgs::Odometry::ConstPtr& laserOdometry){
        timeLaserOdometry = laserOdometry->header.stamp.toSec();
        double roll, pitch, yaw;
        geometry_msgs::Quaternion geoQuat = laserOdometry->pose.pose.orientation;
        tf::Matrix3x3(tf::Quaternion(geoQuat.z, -geoQuat.x, -geoQuat.y, geoQuat.w)).getRPY(roll, pitch, yaw);
        transformSum[0] = -pitch;
        transformSum[1] = -yaw;
        transformSum[2] = roll;
        transformSum[3] = laserOdometry->pose.pose.position.x;
        transformSum[4] = laserOdometry->pose.pose.position.y;
        transformSum[5] = laserOdometry->pose.pose.position.z;
        newLaserOdometry = true;
    }

    void imuHandler(const sensor_msgs::Imu::ConstPtr& imuIn){
        double roll, pitch, yaw;
        tf::Quaternion orientation;
        tf::quaternionMsgToTF(imuIn->orientation, orientation);
        tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw);
        imuPointerLast = (imuPointerLast + 1) % imuQueLength;
        imuTime[imuPointerLast] = imuIn->header.stamp.toSec();
        imuRoll[imuPointerLast] = roll;
        imuPitch[imuPointerLast] = pitch;
    }

// 1. 发布transformAftMapped的信息到"/camera_init"这个frame下面；
// 2. 发布transformBefMapped的信息到"/aft_mapped"这个frame下面；
// 3. 发布 tf::StampedTransform aftMappedTrans作为一个姿态变换；
    void publishTF(){

        geometry_msgs::Quaternion geoQuat = tf::createQuaternionMsgFromRollPitchYaw
                                  (transformAftMapped[2], -transformAftMapped[0], -transformAftMapped[1]);

        odomAftMapped.header.stamp = ros::Time().fromSec(timeLaserOdometry);
        odomAftMapped.pose.pose.orientation.x = -geoQuat.y;
        odomAftMapped.pose.pose.orientation.y = -geoQuat.z;
        odomAftMapped.pose.pose.orientation.z = geoQuat.x;
        odomAftMapped.pose.pose.orientation.w = geoQuat.w;
        odomAftMapped.pose.pose.position.x = transformAftMapped[3];
        odomAftMapped.pose.pose.position.y = transformAftMapped[4];
        odomAftMapped.pose.pose.position.z = transformAftMapped[5];
        odomAftMapped.twist.twist.angular.x = transformBefMapped[0];
        odomAftMapped.twist.twist.angular.y = transformBefMapped[1];
        odomAftMapped.twist.twist.angular.z = transformBefMapped[2];
        odomAftMapped.twist.twist.linear.x = transformBefMapped[3];
        odomAftMapped.twist.twist.linear.y = transformBefMapped[4];
        odomAftMapped.twist.twist.linear.z = transformBefMapped[5];
        pubOdomAftMapped.publish(odomAftMapped);

        aftMappedTrans.stamp_ = ros::Time().fromSec(timeLaserOdometry);
        aftMappedTrans.setRotation(tf::Quaternion(-geoQuat.y, -geoQuat.z, geoQuat.x, geoQuat.w));
        aftMappedTrans.setOrigin(tf::Vector3(transformAftMapped[3], transformAftMapped[4], transformAftMapped[5]));
        tfBroadcaster.sendTransform(aftMappedTrans);
    }

    PointTypePose trans2PointTypePose(float transformIn[]){
        PointTypePose thisPose6D;
        thisPose6D.x = transformIn[3];
        thisPose6D.y = transformIn[4];
        thisPose6D.z = transformIn[5];
        thisPose6D.roll  = transformIn[0];
        thisPose6D.pitch = transformIn[1];
        thisPose6D.yaw   = transformIn[2];
        return thisPose6D;
    }

    // 主题 "/key_pose_origin" 发布关键帧位姿点集合 
    // 主题 "/recent_cloud" 发布 局部平面点点云 
    // 主题 "/registered_cloud" 发布 当前帧点云 
    void publishKeyPosesAndFrames(){

        if (pubKeyPoses.getNumSubscribers() != 0){
            sensor_msgs::PointCloud2 cloudMsgTemp;
            pcl::toROSMsg(*cloudKeyPoses3D, cloudMsgTemp);
            cloudMsgTemp.header.stamp = ros::Time().fromSec(timeLaserOdometry);
            cloudMsgTemp.header.frame_id = "/camera_init";
            pubKeyPoses.publish(cloudMsgTemp);
        }

        if (pubRecentKeyFrames.getNumSubscribers() != 0){
            sensor_msgs::PointCloud2 cloudMsgTemp;
            pcl::toROSMsg(*laserCloudSurfFromMapDS, cloudMsgTemp);
            cloudMsgTemp.header.stamp = ros::Time().fromSec(timeLaserOdometry);
            cloudMsgTemp.header.frame_id = "/camera_init";
            pubRecentKeyFrames.publish(cloudMsgTemp);
        }

        if (pubRegisteredCloud.getNumSubscribers() != 0){
            pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());
            PointTypePose thisPose6D = trans2PointTypePose(transformTobeMapped);
            *cloudOut += *transformPointCloud(laserCloudCornerLastDS,  &thisPose6D);
            *cloudOut += *transformPointCloud(laserCloudSurfTotalLast, &thisPose6D);
            
            sensor_msgs::PointCloud2 cloudMsgTemp;
            pcl::toROSMsg(*cloudOut, cloudMsgTemp);
            cloudMsgTemp.header.stamp = ros::Time().fromSec(timeLaserOdometry);
            cloudMsgTemp.header.frame_id = "/camera_init";
            pubRegisteredCloud.publish(cloudMsgTemp);
        } 
    }

    void visualizeGlobalMapThread(){
        ros::Rate rate(0.2);
        while (ros::ok()){
            rate.sleep();
            publishGlobalMap();
        }
        // save final point cloud
        pcl::io::savePCDFileASCII(fileDirectory+"finalCloud.pcd", *globalMapKeyFramesDS);

        string cornerMapString = "/tmp/cornerMap.pcd";
        string surfaceMapString = "/tmp/surfaceMap.pcd";
        string trajectoryString = "/tmp/trajectory.pcd";

        pcl::PointCloud<PointType>::Ptr cornerMapCloud(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr cornerMapCloudDS(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr surfaceMapCloud(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr surfaceMapCloudDS(new pcl::PointCloud<PointType>());
        
        for(int i = 0; i < cornerCloudKeyFrames.size(); i++) {
            *cornerMapCloud  += *transformPointCloud(cornerCloudKeyFrames[i],   &cloudKeyPoses6D->points[i]);
    	    *surfaceMapCloud += *transformPointCloud(surfCloudKeyFrames[i],     &cloudKeyPoses6D->points[i]);
    	    *surfaceMapCloud += *transformPointCloud(outlierCloudKeyFrames[i],  &cloudKeyPoses6D->points[i]);
        }

        downSizeFilterCorner.setInputCloud(cornerMapCloud);
        downSizeFilterCorner.filter(*cornerMapCloudDS);
        downSizeFilterSurf.setInputCloud(surfaceMapCloud);
        downSizeFilterSurf.filter(*surfaceMapCloudDS);

        pcl::io::savePCDFileASCII(fileDirectory+"cornerMap.pcd", *cornerMapCloudDS);
        pcl::io::savePCDFileASCII(fileDirectory+"surfaceMap.pcd", *surfaceMapCloudDS);
        pcl::io::savePCDFileASCII(fileDirectory+"trajectory.pcd", *cloudKeyPoses3D);
    }

// publishGlobalMap()主要进行了3个步骤：
// 1.通过KDTree进行最近邻搜索;
// 2.通过搜索得到的索引放进队列;
// 3.通过两次下采样，减小数据量
    void publishGlobalMap(){

        if (pubLaserCloudSurround.getNumSubscribers() == 0)
            return;

        if (cloudKeyPoses3D->points.empty() == true)
            return;
	    // kd-tree to find near key frames to visualize
        std::vector<int> pointSearchIndGlobalMap;
        std::vector<float> pointSearchSqDisGlobalMap;
	    // search near key frames to visualize
        mtx.lock();
        kdtreeGlobalMap->setInputCloud(cloudKeyPoses3D);
        kdtreeGlobalMap->radiusSearch(currentRobotPosPoint, globalMapVisualizationSearchRadius, pointSearchIndGlobalMap, pointSearchSqDisGlobalMap, 0);
        mtx.unlock();

        for (int i = 0; i < pointSearchIndGlobalMap.size(); ++i)
          globalMapKeyPoses->points.push_back(cloudKeyPoses3D->points[pointSearchIndGlobalMap[i]]);
	    // downsample near selected key frames
        downSizeFilterGlobalMapKeyPoses.setInputCloud(globalMapKeyPoses);
        downSizeFilterGlobalMapKeyPoses.filter(*globalMapKeyPosesDS);
	    // extract visualized and downsampled key frames
        for (int i = 0; i < globalMapKeyPosesDS->points.size(); ++i){
			int thisKeyInd = (int)globalMapKeyPosesDS->points[i].intensity;
			*globalMapKeyFrames += *transformPointCloud(cornerCloudKeyFrames[thisKeyInd],   &cloudKeyPoses6D->points[thisKeyInd]);
			*globalMapKeyFrames += *transformPointCloud(surfCloudKeyFrames[thisKeyInd],    &cloudKeyPoses6D->points[thisKeyInd]);
			*globalMapKeyFrames += *transformPointCloud(outlierCloudKeyFrames[thisKeyInd], &cloudKeyPoses6D->points[thisKeyInd]);
        }
	    // downsample visualized points
        downSizeFilterGlobalMapKeyFrames.setInputCloud(globalMapKeyFrames);
        downSizeFilterGlobalMapKeyFrames.filter(*globalMapKeyFramesDS);
 
        sensor_msgs::PointCloud2 cloudMsgTemp;
        pcl::toROSMsg(*globalMapKeyFramesDS, cloudMsgTemp);
        cloudMsgTemp.header.stamp = ros::Time().fromSec(timeLaserOdometry);
        cloudMsgTemp.header.frame_id = "/camera_init";
        pubLaserCloudSurround.publish(cloudMsgTemp);  

        globalMapKeyPoses->clear();
        globalMapKeyPosesDS->clear();
        globalMapKeyFrames->clear();
        // globalMapKeyFramesDS->clear();     
    }


    void loopClosureThread(){
        if (loopClosureEnableFlag == false)
            return;
        // 以一定频率循环调用performLoopClosure
        ros::Rate rate(1);
        while (ros::ok()){
            rate.sleep();
            performLoopClosure();
        }
    } 

// 依据时间　距离　SC判断关键帧是否满足回环
//找到回环信息，并保存在成员变量中，然后根据形成回环的帧的点云进行匹配建立当前最新关键帧与历史关键帧之间的约束，添加到gtsam进行图优化。
    bool detectLoopClosure(){
        std::lock_guard<std::mutex> lock(mtx);//和void mapOptimization::run()不同时进行
        /* 
         * 1. xyz distance-based radius search (contained in the original LeGO LOAM code)
         * - for fine-stichting trajectories (for not-recognized nodes within scan context search) 
         * 基于目前位姿,在一定范围内(20m)内搜索最近邻,若最早的那个超过了30s,则选中为回环目标
        // 选取前后25帧组成点云,并保存当前最近一帧点云
         */
        RSlatestSurfKeyFrameCloud->clear();// 当前关键帧
        RSnearHistorySurfKeyFrameCloud->clear();// 尝试进行回环的关键帧前后一定范围帧组成的点云
        RSnearHistorySurfKeyFrameCloudDS->clear();// 上面的降采样

        // find the closest history key frame
        std::vector<int> pointSearchIndLoop; //  搜索完的邻域点对应的索引
        std::vector<float> pointSearchSqDisLoop;//  搜索完的邻域点与当前点的欧氏距离
        // 用当前的所有关键帧生成树

        kdtreeHistoryKeyPoses->setInputCloud(cloudKeyPoses3D);
        // 寻找附近20m内的历史轨迹  currentRobotPosPoint为最新一帧关键帧的位姿       
        //  0：返回的邻域个数，为0表示返回全部的邻域点
        kdtreeHistoryKeyPoses->radiusSearch(currentRobotPosPoint, historyKeyframeSearchRadius, pointSearchIndLoop, pointSearchSqDisLoop, 0);
        // jin: 选取最近邻中,时间距离30s以上,最早的那帧
        RSclosestHistoryFrameID = -1;
        int curMinID = 1000000;
        // policy: take Oldest one (to fix error of the whole trajectory)
        for (int i = 0; i < pointSearchIndLoop.size(); ++i){
            int id = pointSearchIndLoop[i];
            //需要有一定的时间间隔 30S
            if (abs(cloudKeyPoses6D->points[id].time - timeLaserOdometry) > 30.0){
                // RSclosestHistoryFrameID = id;
                // break;
                if( id < curMinID ) {
                    curMinID = id;
                    RSclosestHistoryFrameID = curMinID;
                }
            }
        }

        if (RSclosestHistoryFrameID == -1){
            // Do nothing here
            // then, do the next check: Scan context-based search 
            // not return false here;
        }
        else {
            // save latest key frames
            // 最新的一帧点云
            // jin: 回环检测的进程是单独进行的,因此这里需要确定最新帧
            //  检测到回环了会保存四种点云
            // save latest key frames
            latestFrameIDLoopCloure = cloudKeyPoses3D->points.size() - 1;
            // jin: 根据当前的位姿,对点云进行旋转和平移
            // 点云的xyz坐标进行坐标系变换(分别绕xyz轴旋转)
            *RSlatestSurfKeyFrameCloud += *transformPointCloud(cornerCloudKeyFrames[latestFrameIDLoopCloure], &cloudKeyPoses6D->points[latestFrameIDLoopCloure]);
            *RSlatestSurfKeyFrameCloud += *transformPointCloud(surfCloudKeyFrames[latestFrameIDLoopCloure],   &cloudKeyPoses6D->points[latestFrameIDLoopCloure]);
           // latestSurfKeyFrameCloud中存储的是下面公式计算后的index(intensity):
            // thisPoint.intensity = (float)rowIdn + (float)columnIdn / 10000.0;
            // 滤掉latestSurfKeyFrameCloud中index<0的点??? index值会小于0?
 pcl::PointCloud<PointType>::Ptr RShahaCloud(new pcl::PointCloud<PointType>());
            int cloudSize = RSlatestSurfKeyFrameCloud->points.size();
            for (int i = 0; i < cloudSize; ++i){
                if ((int)RSlatestSurfKeyFrameCloud->points[i].intensity >= 0){
                    RShahaCloud->push_back(RSlatestSurfKeyFrameCloud->points[i]);
                }
            }
            RSlatestSurfKeyFrameCloud->clear();
            *RSlatestSurfKeyFrameCloud = *RShahaCloud;
            //historyKeyframeSearchNum是25,
            // save history near key frames
            for (int j = -historyKeyframeSearchNum; j <= historyKeyframeSearchNum; ++j){
                if (RSclosestHistoryFrameID + j < 0 || RSclosestHistoryFrameID + j > latestFrameIDLoopCloure)
                    continue;
                //将搜索范围内的角点点云与平面点点云均叠加至nearHistorySurfKeyFrameCloud中
                *RSnearHistorySurfKeyFrameCloud += *transformPointCloud(cornerCloudKeyFrames[RSclosestHistoryFrameID+j], &cloudKeyPoses6D->points[RSclosestHistoryFrameID+j]);
                *RSnearHistorySurfKeyFrameCloud += *transformPointCloud(surfCloudKeyFrames[RSclosestHistoryFrameID+j],   &cloudKeyPoses6D->points[RSclosestHistoryFrameID+j]);
            }
            downSizeFilterHistoryKeyFrames.setInputCloud(RSnearHistorySurfKeyFrameCloud);
            downSizeFilterHistoryKeyFrames.filter(*RSnearHistorySurfKeyFrameCloudDS);
        }

        /* 
         * 2. Scan context-based global localization 
         */
        SClatestSurfKeyFrameCloud->clear();
        SCnearHistorySurfKeyFrameCloud->clear();
        SCnearHistorySurfKeyFrameCloudDS->clear();

        // std::lock_guard<std::mutex> lock(mtx);        
        latestFrameIDLoopCloure = cloudKeyPoses3D->points.size() - 1;
        SCclosestHistoryFrameID = -1; // init with -1
        
    // 这是最重要的部分，根据ScanContext确定回环的关键帧,返回的是关键帧的ID,和yaw角的偏移量，分别提取出来
        auto detectResult = scManager.detectLoopClosureID(); // first: nn index, second: yaw diff 
        SCclosestHistoryFrameID = detectResult.first;
        yawDiffRad = detectResult.second; // not use for v1 (because pcl icp withi initial somthing wrong...)

        // if all close, reject
        if (SCclosestHistoryFrameID == -1){ 
            return false;
        }
        // 以下，如果SC检测到了回环，保存回环上的帧前后25帧和当前帧，过程与上面完全一样
        // save latest key frames: query ptcloud (corner points + surface points)
        // NOTE: using "closestHistoryFrameID" to make same root of submap points to get a direct relative between
        // the query point cloud (latestSurfKeyFrameCloud) and the map (nearHistorySurfKeyFrameCloud). by giseop
        // i.e., set the query point cloud within mapside's local coordinate
        // jin: 将最新一帧激光点在回环位姿处进行投影

        // save latest key frames: query ptcloud (corner points + surface points)
        // NOTE: using "closestHistoryFrameID" to make same root of submap points to get a direct relative between the query point cloud (latestSurfKeyFrameCloud) and the map (nearHistorySurfKeyFrameCloud). by giseop
        // i.e., set the query point cloud within mapside's local coordinate
        *SClatestSurfKeyFrameCloud += *transformPointCloud(cornerCloudKeyFrames[latestFrameIDLoopCloure], &cloudKeyPoses6D->points[SCclosestHistoryFrameID]);         
        *SClatestSurfKeyFrameCloud += *transformPointCloud(surfCloudKeyFrames[latestFrameIDLoopCloure],   &cloudKeyPoses6D->points[SCclosestHistoryFrameID]); 

        pcl::PointCloud<PointType>::Ptr SChahaCloud(new pcl::PointCloud<PointType>());
        int cloudSize = SClatestSurfKeyFrameCloud->points.size();
        for (int i = 0; i < cloudSize; ++i){
            if ((int)SClatestSurfKeyFrameCloud->points[i].intensity >= 0){
                SChahaCloud->push_back(SClatestSurfKeyFrameCloud->points[i]);
            }
        }
        SClatestSurfKeyFrameCloud->clear();
        *SClatestSurfKeyFrameCloud = *SChahaCloud;
        // ScanContext确定的回环关键帧,前后一段范围内的点组成点云地图
	   // save history near key frames: map ptcloud (icp to query ptcloud)
        for (int j = -historyKeyframeSearchNum; j <= historyKeyframeSearchNum; ++j){
            if (SCclosestHistoryFrameID + j < 0 || SCclosestHistoryFrameID + j > latestFrameIDLoopCloure)
                continue;
            *SCnearHistorySurfKeyFrameCloud += *transformPointCloud(cornerCloudKeyFrames[SCclosestHistoryFrameID+j], &cloudKeyPoses6D->points[SCclosestHistoryFrameID+j]);
            *SCnearHistorySurfKeyFrameCloud += *transformPointCloud(surfCloudKeyFrames[SCclosestHistoryFrameID+j],   &cloudKeyPoses6D->points[SCclosestHistoryFrameID+j]);
        }
        downSizeFilterHistoryKeyFrames.setInputCloud(SCnearHistorySurfKeyFrameCloud);
        downSizeFilterHistoryKeyFrames.filter(*SCnearHistorySurfKeyFrameCloudDS);

        // // optional: publish history near key frames
        // if (pubHistoryKeyFrames.getNumSubscribers() != 0){
        //     sensor_msgs::PointCloud2 cloudMsgTemp;
        //     pcl::toROSMsg(*nearHistorySurfKeyFrameCloudDS, cloudMsgTemp);
        //     cloudMsgTemp.header.stamp = ros::Time().fromSec(timeLaserOdometry);
        //     cloudMsgTemp.header.frame_id = "/camera_init";
        //     pubHistoryKeyFrames.publish(cloudMsgTemp);
        // }

        return true;
    } 

// 函数流程：
// 1.先进行闭环检测detectLoopClosure()，如果返回true,则可能可以进行闭环，否则直接返回，程序结束。
// 如果ScanContext回环检测（即std::thread loopthread）对全局位姿进行了优化，需要同步下来优化后的关键帧所在的位姿。
//判断是否回环在performLoopClosure函数中。
    void performLoopClosure( void ) {
        if (cloudKeyPoses3D->points.empty() == true)
            return;
        // try to find close key frame if there are any
        // 分别根据距离和SCANCONTEXT信息查找回环帧，回环信息保存在成员变量中，包括回环帧的ID，点云，偏航角等
        if (potentialLoopFlag == false){
            if (detectLoopClosure() == true){
                std::cout << std::endl;
                potentialLoopFlag = true; // find some key frames that is old enough or close enough for loop closure
                timeSaveFirstCurrentScanForLoopClosure = timeLaserOdometry;
            }
            if (potentialLoopFlag == false){// ScanContext未能找到可以形成回环的关键帧
                return;
            }
        }
        // reset the flag first no matter icp successes or not
        potentialLoopFlag = false;
        // *****
        // 如果当前关键帧与历史关键帧确实形成了回环，开始进行优化
        // make common variables at forward
        float x, y, z, roll, pitch, yaw;
        Eigen::Affine3f correctionCameraFrame;
        float noiseScore = 0.5; // constant is ok...
        gtsam::Vector Vector6(6);
        Vector6 << noiseScore, noiseScore, noiseScore, noiseScore, noiseScore, noiseScore;
        constraintNoise = noiseModel::Diagonal::Variances(Vector6);
        robustNoiseModel = gtsam::noiseModel::Robust::Create(
            gtsam::noiseModel::mEstimator::Cauchy::Create(1), // optional: replacing Cauchy by DCS or GemanMcClure
            gtsam::noiseModel::Diagonal::Variances(Vector6)
        ); // - checked it works. but with robust kernel, map modification may be delayed (i.e,. requires more true-positive loop factors)

        bool isValidRSloopFactor = false;
        bool isValidSCloopFactor = false;

        /*
         * 1. RS loop factor (radius search) 将RS检测到的历史帧和当前帧匹配，求transform, 作为约束边
                 RS:根据距离（RangeSearch）
         */
        //icp配准
        if( RSclosestHistoryFrameID != -1 ) {
            pcl::IterativeClosestPoint<PointType, PointType> icp;
            icp.setMaxCorrespondenceDistance(100);
            icp.setMaximumIterations(100);
            icp.setTransformationEpsilon(1e-6);
            icp.setEuclideanFitnessEpsilon(1e-6);
            icp.setRANSACIterations(0);

            // Align clouds
            icp.setInputSource(RSlatestSurfKeyFrameCloud);
            icp.setInputTarget(RSnearHistorySurfKeyFrameCloudDS);
            pcl::PointCloud<PointType>::Ptr unused_result(new pcl::PointCloud<PointType>());
            icp.align(*unused_result);
            // 上面比较的两个点云都已经被投影到了世界坐标系下，所以匹配的结果应该是这段时间内，原点所发生的漂移
            // 通过score阈值判定icp是否匹配成功
            std::cout << "[RS] ICP fit score: " << icp.getFitnessScore() << std::endl;
            if ( icp.hasConverged() == false || icp.getFitnessScore() > historyKeyframeFitnessScore ) {
                std::cout << "[RS] Reject this loop (bad icp fit score, > " << historyKeyframeFitnessScore << ")" << std::endl;
                isValidRSloopFactor = false;
            }
            else {
                std::cout << "[RS] The detected loop factor is added between Current [ " << latestFrameIDLoopCloure << " ] and RS nearest [ " << RSclosestHistoryFrameID << " ]" << std::endl;
                isValidRSloopFactor = true;
            }            
            // jin: 最新帧与回环帧前后一定时间范围内的点组成的地图进行匹配,得到的坐标变换为最新帧与回环帧之间的约束
            // 因为作为地图的帧在回环帧前后很小的范围内,位姿变化很小,可以认为他们之间的相对位姿很准,地图也很准
            //  这里RS检测成功，加入约束边
            if( isValidRSloopFactor == true ) {
                correctionCameraFrame = icp.getFinalTransformation(); // get transformation in camera frame (because points are in camera frame)
                pcl::getTranslationAndEulerAngles(correctionCameraFrame, x, y, z, roll, pitch, yaw);
                Eigen::Affine3f correctionLidarFrame = pcl::getTransformation(z, x, y, yaw, roll, pitch);
                // transform from world origin to wrong pose
                // 最新关键帧在地图坐标系中的坐标，在过程中会存在误差的积累，否则匹配的结果必然是E
                // 这种误差可以被解释为地图原点发生了漂移
                Eigen::Affine3f tWrong = pclPointToAffine3fCameraToLidar(cloudKeyPoses6D->points[latestFrameIDLoopCloure]);
                // transform from world origin to corrected pose
                // 地图原点的漂移×在漂移后的地图中的坐标=没有漂移的坐标，即在回环上的关键帧时刻其应该所处的位姿
                // 这样就把当前帧的位姿转移到了回环关键帧所在时刻，没有漂移的情况下的位姿，两者再求解相对位姿
                /// 感觉以上很复杂，一开始完全没有把点云往世界坐标系投影啊！直接匹配不就是相对位姿么？
                Eigen::Affine3f tCorrect = correctionLidarFrame * tWrong; // pre-multiplying -> successive rotation about a fixed frame
                pcl::getTranslationAndEulerAngles (tCorrect, x, y, z, roll, pitch, yaw);
                gtsam::Pose3 poseFrom = Pose3(Rot3::RzRyRx(roll, pitch, yaw), Point3(x, y, z));
                gtsam::Pose3 poseTo = pclPointTogtsamPose3(cloudKeyPoses6D->points[RSclosestHistoryFrameID]);
                gtsam::Vector Vector6(6);

                std::lock_guard<std::mutex> lock(mtx);
                gtSAMgraph.add(BetweenFactor<Pose3>(latestFrameIDLoopCloure, RSclosestHistoryFrameID, poseFrom.between(poseTo), robustNoiseModel));
                isam->update(gtSAMgraph);
                isam->update();
                gtSAMgraph.resize(0);
            }
        }

        /*
         * 2. SC loop factor (scan context)SC检测成功，进行icp匹配
         */
        if( SCclosestHistoryFrameID != -1 ) {
            pcl::IterativeClosestPoint<PointType, PointType> icp;
            icp.setMaxCorrespondenceDistance(100);
            icp.setMaximumIterations(100);
            icp.setTransformationEpsilon(1e-6);
            icp.setEuclideanFitnessEpsilon(1e-6);
            icp.setRANSACIterations(0);

            // Align clouds
            // Eigen::Affine3f icpInitialMatFoo = pcl::getTransformation(0, 0, 0, yawDiffRad, 0, 0); // because within cam coord: (z, x, y, yaw, roll, pitch)
            // Eigen::Matrix4f icpInitialMat = icpInitialMatFoo.matrix();
            icp.setInputSource(SClatestSurfKeyFrameCloud);
            icp.setInputTarget(SCnearHistorySurfKeyFrameCloudDS);
            pcl::PointCloud<PointType>::Ptr unused_result(new pcl::PointCloud<PointType>());
            icp.align(*unused_result); 
            // icp.align(*unused_result, icpInitialMat); // PCL icp non-eye initial is bad ... don't use (LeGO LOAM author also said pcl transform is weird.)

            std::cout << "[SC] ICP fit score: " << icp.getFitnessScore() << std::endl;
            if ( icp.hasConverged() == false || icp.getFitnessScore() > historyKeyframeFitnessScore ) {
                std::cout << "[SC] Reject this loop (bad icp fit score, > " << historyKeyframeFitnessScore << ")" << std::endl;
                isValidSCloopFactor = false;
            }
            else {
                std::cout << "[SC] The detected loop factor is added between Current [ " << latestFrameIDLoopCloure << " ] and SC nearest [ " << SCclosestHistoryFrameID << " ]" << std::endl;
                isValidSCloopFactor = true;
            }
            // icp匹配成功也加入约束边
            if( isValidSCloopFactor == true ) {
                correctionCameraFrame = icp.getFinalTransformation(); // get transformation in camera frame (because points are in camera frame)
                pcl::getTranslationAndEulerAngles (correctionCameraFrame, x, y, z, roll, pitch, yaw);
                gtsam::Pose3 poseFrom = Pose3(Rot3::RzRyRx(roll, pitch, yaw), Point3(x, y, z));
                gtsam::Pose3 poseTo = Pose3(Rot3::RzRyRx(0.0, 0.0, 0.0), Point3(0.0, 0.0, 0.0));
                
                std::lock_guard<std::mutex> lock(mtx);
                // gtSAMgraph.add(BetweenFactor<Pose3>(latestFrameIDLoopCloure, closestHistoryFrameID, poseFrom.between(poseTo), constraintNoise)); // original 
                gtSAMgraph.add(BetweenFactor<Pose3>(latestFrameIDLoopCloure, SCclosestHistoryFrameID, poseFrom.between(poseTo), robustNoiseModel)); // giseop
                isam->update(gtSAMgraph);
                isam->update();
                gtSAMgraph.resize(0);
            }
        }
        // 在correctPoses中会导致对后端保存的pose位姿进行修改
        aLoopIsClosed = true; 

    } // performLoopClosure


    Pose3 pclPointTogtsamPose3(PointTypePose thisPoint){ // camera frame to lidar frame
    	return Pose3(Rot3::RzRyRx(double(thisPoint.yaw), double(thisPoint.roll), double(thisPoint.pitch)),
                           Point3(double(thisPoint.z),   double(thisPoint.x),    double(thisPoint.y)));
    }

    Eigen::Affine3f pclPointToAffine3fCameraToLidar(PointTypePose thisPoint){ // camera frame to lidar frame
    	return pcl::getTransformation(thisPoint.z, thisPoint.x, thisPoint.y, thisPoint.yaw, thisPoint.roll, thisPoint.pitch);
    }

        // 抽取周围关键帧
        // 函数功能：根据当前位置，提取局部关键帧集合,以及对应的三个关键帧点云集合
                // 步骤：
                //     1. 在 关键帧位置集合cloudKeyPoses3D 中  
                //     检索 当前位置currentRobotPosPoint 附近的姿态点
                //     获得局部位置点，赋值给 局部位置点集合surroundingKeyPoses 
                //     2. 根据 局部位置点集合surroundingKeyPoses 更新
                //     局部关键帧集合 surroundingExistingKeyPosesID
                //     局部关键帧 角点点云集合surroundingCornerCloudKeyFrames 
                //     局部关键帧 平面点点云集合surroundingSurfCloudKeyFrames
                //     局部关键帧 离群点点云集合surroundingOutlierCloudKeyFrames
                //     增加新进入局部的关键帧、并删除离开局部的关键帧。
                //     3. 为局部点云地图赋值 
                //     laserCloudCornerFromMap     所有局部关键帧的角点集合
                //     laserCloudSurfFromMap       所有局部关键帧平面点和离群点的几何
    void extractSurroundingKeyFrames(){
        if (cloudKeyPoses3D->points.empty() == true)
            return;	
		// 进行闭环过程
    	if (loopClosureEnableFlag == true){
    	    // only use recent key poses for graph building
                if (recentCornerCloudKeyFrames.size() < surroundingKeyframeSearchNum){ // queue is not full (the beginning of mapping or a loop is just closed)
                    // clear recent key frames queue
                    // 若recentCornerCloudKeyFrames中的点云数量不够， 清空后重新塞入新的点云直至数量够了。
                    //将角点、平面点等均填充至合格数量
                    recentCornerCloudKeyFrames. clear();
                    recentSurfCloudKeyFrames.   clear();
                    recentOutlierCloudKeyFrames.clear();                    
                    int numPoses = cloudKeyPoses3D->points.size();
                    for (int i = numPoses-1; i >= 0; --i){
                        int thisKeyInd = (int)cloudKeyPoses3D->points[i].intensity;
                        PointTypePose thisTransformation = cloudKeyPoses6D->points[thisKeyInd];
                        updateTransformPointCloudSinCos(&thisTransformation);
                        // extract surrounding map
                        recentCornerCloudKeyFrames. push_front(transformPointCloud(cornerCloudKeyFrames[thisKeyInd]));
                        recentSurfCloudKeyFrames.   push_front(transformPointCloud(surfCloudKeyFrames[thisKeyInd]));
                        recentOutlierCloudKeyFrames.push_front(transformPointCloud(outlierCloudKeyFrames[thisKeyInd]));
                        if (recentCornerCloudKeyFrames.size() >= surroundingKeyframeSearchNum)
                            break;
                    }
                }else{  //点云数量充足
                    // queue is full, pop the oldest key frame and push the latest key frame，
                    // 否则pop队列recentCornerCloudKeyFrames最前端的一个，再往队列尾部push一个
                    if (latestFrameID != cloudKeyPoses3D->points.size() - 1){ 
                         // if the robot is not moving, no need to update recent frames
                        recentCornerCloudKeyFrames. pop_front();
                        recentSurfCloudKeyFrames.   pop_front();
                        recentOutlierCloudKeyFrames.pop_front();
                        // push latest scan to the end of queue
                        latestFrameID = cloudKeyPoses3D->points.size() - 1;
                        PointTypePose thisTransformation = cloudKeyPoses6D->points[latestFrameID];
                        updateTransformPointCloudSinCos(&thisTransformation);
                        recentCornerCloudKeyFrames. push_back(transformPointCloud(cornerCloudKeyFrames[latestFrameID]));
                        recentSurfCloudKeyFrames.   push_back(transformPointCloud(surfCloudKeyFrames[latestFrameID]));
                        recentOutlierCloudKeyFrames.push_back(transformPointCloud(outlierCloudKeyFrames[latestFrameID]));
                    }
                }

                for (int i = 0; i < recentCornerCloudKeyFrames.size(); ++i){
                    *laserCloudCornerFromMap += *recentCornerCloudKeyFrames[i];
                    *laserCloudSurfFromMap   += *recentSurfCloudKeyFrames[i];
                    *laserCloudSurfFromMap   += *recentOutlierCloudKeyFrames[i];
                }
    	}else{
            /*这里不进行闭环过程*/
            // 1.进行半径surroundingKeyframeSearchRadius内的邻域搜索
            // 2.双重循环，不断对比surroundingExistingKeyPosesID和surroundingKeyPosesDS中点的index,
            // 如果能够找到一样，说明存在关键帧。然后在队列中去掉找不到的元素，留下可以找到的。
            // 3.再来一次双重循环，这部分比较有技巧，
            // 这里把surroundingExistingKeyPosesID内没有对应的点放进一个队列里，
            // 这个队列专门存放周围存在的关键帧，
            // 但是和surroundingExistingKeyPosesID的点不在同一行。
            // 关于行，需要参考intensity数据的存放格式，
            // 整数部分和小数部分代表不同意义。
            // cloudKeyPoses3D虽说是点云，但是是为了保存机器人在建图过程中的轨迹，
            // 其中的点就是定周期采样的轨迹点，这一点是在saveKeyFramesAndFactor中计算出的，在第一帧时必然是空的

            surroundingKeyPoses->clear();
            surroundingKeyPosesDS->clear();
    	    // extract all the nearby key poses and downsample them
    	    kdtreeSurroundingKeyPoses->setInputCloud(cloudKeyPoses3D);
            //surroundingKeyframeSearchRadius是50米，也就是说是在当前位置进行半径查找，得到附近的轨迹点
            //距离数据保存在pointSearchSqDis中
    	    kdtreeSurroundingKeyPoses->radiusSearch(currentRobotPosPoint, (double)surroundingKeyframeSearchRadius, pointSearchInd, pointSearchSqDis, 0);
    	    for (int i = 0; i < pointSearchInd.size(); ++i)
                surroundingKeyPoses->points.push_back(cloudKeyPoses3D->points[pointSearchInd[i]]);
            //对附近轨迹点的点云进行降采样，轨迹具有一定间隔
    	    downSizeFilterSurroundingKeyPoses.setInputCloud(surroundingKeyPoses);
    	    downSizeFilterSurroundingKeyPoses.filter(*surroundingKeyPosesDS);
    	    // delete key frames that are not in surrounding region
            int numSurroundingPosesDS = surroundingKeyPosesDS->points.size();
            for (int i = 0; i < surroundingExistingKeyPosesID.size(); ++i){
                bool existingFlag = false;
                for (int j = 0; j < numSurroundingPosesDS; ++j){
                     //也就是说，这个等式的意义是判断附近某一个关键帧等于降维后点云的第j个关键帧
                    if (surroundingExistingKeyPosesID[i] == (int)surroundingKeyPosesDS->points[j].intensity){
                        existingFlag = true;
                        break;
                    }
                }
                //这也是一个变相的降维处理
                if (existingFlag == false){
                    surroundingExistingKeyPosesID.   erase(surroundingExistingKeyPosesID.   begin() + i);
                    //这三个都是双端队列
                    surroundingCornerCloudKeyFrames. erase(surroundingCornerCloudKeyFrames. begin() + i);
                    surroundingSurfCloudKeyFrames.   erase(surroundingSurfCloudKeyFrames.   begin() + i);
                    surroundingOutlierCloudKeyFrames.erase(surroundingOutlierCloudKeyFrames.begin() + i);
                    --i;
                }
            }
    	    //add new key frames that are not in calculated existing key frames
            for (int i = 0; i < numSurroundingPosesDS; ++i) {
                bool existingFlag = false;
                for (auto iter = surroundingExistingKeyPosesID.begin(); iter != surroundingExistingKeyPosesID.end(); ++iter){
                    if ((*iter) == (int)surroundingKeyPosesDS->points[i].intensity){
                        existingFlag = true;
                        break;
                    }
                }
                if (existingFlag == true){
                    continue;
                }else{
                    //这类情况是初次处理时将点云变换到当前坐标系下
                    //cloudKeyPoses6D中的数据来源自thisPose6D，我们可以看到thisPose6D根据isam库进行先验后得到，所以在下面容器中存放的都是已经粗配准完毕的点云下标
                    int thisKeyInd = (int)surroundingKeyPosesDS->points[i].intensity;
                    PointTypePose thisTransformation = cloudKeyPoses6D->points[thisKeyInd];
                    updateTransformPointCloudSinCos(&thisTransformation);
                    surroundingExistingKeyPosesID.   push_back(thisKeyInd);
                    surroundingCornerCloudKeyFrames. push_back(transformPointCloud(cornerCloudKeyFrames[thisKeyInd]));
                    surroundingSurfCloudKeyFrames.   push_back(transformPointCloud(surfCloudKeyFrames[thisKeyInd]));
                    surroundingOutlierCloudKeyFrames.push_back(transformPointCloud(outlierCloudKeyFrames[thisKeyInd]));
                }
            }
            //累加点云
            for (int i = 0; i < surroundingExistingKeyPosesID.size(); ++i) {
                *laserCloudCornerFromMap += *surroundingCornerCloudKeyFrames[i];
                *laserCloudSurfFromMap   += *surroundingSurfCloudKeyFrames[i];
                *laserCloudSurfFromMap   += *surroundingOutlierCloudKeyFrames[i];
            }
    	}
        // 不管是否进行闭环过程，最后的输出都要进行一次下采样减小数据量的过程。
        // 最后的输出结果是laserCloudCornerFromMapDS和laserCloudSurfFromMapDS。
        // Downsample the surrounding corner key frames (or map)
        downSizeFilterCorner.setInputCloud(laserCloudCornerFromMap);
        downSizeFilterCorner.filter(*laserCloudCornerFromMapDS);
        laserCloudCornerFromMapDSNum = laserCloudCornerFromMapDS->points.size();
        // Downsample the surrounding surf key frames (or map)
        downSizeFilterSurf.setInputCloud(laserCloudSurfFromMap);
        downSizeFilterSurf.filter(*laserCloudSurfFromMapDS);
        laserCloudSurfFromMapDSNum = laserCloudSurfFromMapDS->points.size();
    }

    void downsampleCurrentScan(){

        laserCloudRawDS->clear();
        downSizeFilterScancontext.setInputCloud(laserCloudRaw);
        downSizeFilterScancontext.filter(*laserCloudRawDS);
        
        laserCloudCornerLastDS->clear();
        downSizeFilterCorner.setInputCloud(laserCloudCornerLast);
        downSizeFilterCorner.filter(*laserCloudCornerLastDS);
        laserCloudCornerLastDSNum = laserCloudCornerLastDS->points.size();
        // std::cout << "laserCloudCornerLastDSNum: " << laserCloudCornerLastDSNum << std::endl;

        laserCloudSurfLastDS->clear();
        downSizeFilterSurf.setInputCloud(laserCloudSurfLast);
        downSizeFilterSurf.filter(*laserCloudSurfLastDS);
        laserCloudSurfLastDSNum = laserCloudSurfLastDS->points.size();
        // std::cout << "laserCloudSurfLastDSNum: " << laserCloudSurfLastDSNum << std::endl;

        laserCloudOutlierLastDS->clear();
        downSizeFilterOutlier.setInputCloud(laserCloudOutlierLast);
        downSizeFilterOutlier.filter(*laserCloudOutlierLastDS);
        laserCloudOutlierLastDSNum = laserCloudOutlierLastDS->points.size();

        laserCloudSurfTotalLast->clear();
        laserCloudSurfTotalLastDS->clear();
        *laserCloudSurfTotalLast += *laserCloudSurfLastDS;
        *laserCloudSurfTotalLast += *laserCloudOutlierLastDS;
        downSizeFilterSurf.setInputCloud(laserCloudSurfTotalLast);
        downSizeFilterSurf.filter(*laserCloudSurfTotalLastDS);
        laserCloudSurfTotalLastDSNum = laserCloudSurfTotalLastDS->points.size();
    }

// 角优化
// 该函数分成了几个部分：
// 1.进行坐标变换,转换到全局坐标中去；
// 2.进行5邻域搜索，得到结果后对搜索得到的5点求平均值；
// 3.求矩阵matA1=[ax,ay,az]t*[ax,ay,az]，例如ax代表的是x-cx,表示均值与每个实际值的差值，求取5个之后再次取平均，得到matA1；
// 4.求正交阵的特征值和特征向量，特征值：matD1，特征向量：保存在矩阵matV1中。

    void cornerOptimization(int iterCount){
        updatePointAssociateToMapSinCos();
        for (int i = 0; i < laserCloudCornerLastDSNum; i++) {
            pointOri = laserCloudCornerLastDS->points[i];
            pointAssociateToMap(&pointOri, &pointSel);
            //利用kd树查找最近的5个点，接下来需要计算这五个点的协方差
            kdtreeCornerFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);
            //只有最近的点都在一定阈值内（1米）才进行计算
            if (pointSearchSqDis[4] < 1.0) {
                float cx = 0, cy = 0, cz = 0;
                for (int j = 0; j < 5; j++) {
                    cx += laserCloudCornerFromMapDS->points[pointSearchInd[j]].x;
                    cy += laserCloudCornerFromMapDS->points[pointSearchInd[j]].y;
                    cz += laserCloudCornerFromMapDS->points[pointSearchInd[j]].z;
                }
                //计算算术平均值
                cx /= 5; cy /= 5;  cz /= 5;

                float a11 = 0, a12 = 0, a13 = 0, a22 = 0, a23 = 0, a33 = 0;
                //计算协方差矩阵
                for (int j = 0; j < 5; j++) {
                    float ax = laserCloudCornerFromMapDS->points[pointSearchInd[j]].x - cx;
                    float ay = laserCloudCornerFromMapDS->points[pointSearchInd[j]].y - cy;
                    float az = laserCloudCornerFromMapDS->points[pointSearchInd[j]].z - cz;

                    a11 += ax * ax; a12 += ax * ay; a13 += ax * az;
                    a22 += ay * ay; a23 += ay * az;
                    a33 += az * az;
                }
                a11 /= 5; a12 /= 5; a13 /= 5; a22 /= 5; a23 /= 5; a33 /= 5;

                matA1.at<float>(0, 0) = a11; matA1.at<float>(0, 1) = a12; matA1.at<float>(0, 2) = a13;
                matA1.at<float>(1, 0) = a12; matA1.at<float>(1, 1) = a22; matA1.at<float>(1, 2) = a23;
                matA1.at<float>(2, 0) = a13; matA1.at<float>(2, 1) = a23; matA1.at<float>(2, 2) = a33;
                //求协方差矩阵的特征值和特征向量
                cv::eigen(matA1, matD1, matV1);
                //与里程计的计算类似，计算到直线的距离
                if (matD1.at<float>(0, 0) > 3 * matD1.at<float>(0, 1)) {
                    float x0 = pointSel.x;
                    float y0 = pointSel.y;
                    float z0 = pointSel.z;
                    float x1 = cx + 0.1 * matV1.at<float>(0, 0);
                    float y1 = cy + 0.1 * matV1.at<float>(0, 1);
                    float z1 = cz + 0.1 * matV1.at<float>(0, 2);
                    float x2 = cx - 0.1 * matV1.at<float>(0, 0);
                    float y2 = cy - 0.1 * matV1.at<float>(0, 1);
                    float z2 = cz - 0.1 * matV1.at<float>(0, 2);
                    float a012 = sqrt(((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1))
                                    * ((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) 
                                    + ((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1))
                                    * ((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1)) 
                                    + ((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1))
                                    * ((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1)));

                    float l12 = sqrt((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2) + (z1 - z2)*(z1 - z2));

                    float la = ((y1 - y2)*((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) 
                              + (z1 - z2)*((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1))) / a012 / l12;

                    float lb = -((x1 - x2)*((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) 
                               - (z1 - z2)*((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1))) / a012 / l12;

                    float lc = -((x1 - x2)*((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1)) 
                               + (y1 - y2)*((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1))) / a012 / l12;

                    float ld2 = a012 / l12;

                    float s = 1 - 0.9 * fabs(ld2);

                    coeff.x = s * la;
                    coeff.y = s * lb;
                    coeff.z = s * lc;
                    coeff.intensity = s * ld2;

                    if (s > 0.1) {
                        laserCloudOri->push_back(pointOri);
                        coeffSel->push_back(coeff);
                    }
                }
            }
        }
    }

// void surfOptimization(int)函数进行面优化，内容和函数cornerOptimization(int)的内容基本相同。
// 步骤如下：
// 1.进行坐标变换,转换到全局坐标中去；
// 2.进行5邻域搜索，得到结果后判断搜索结果是否满足条件(pointSearchSqDis[4] < 1.0)，不满足条件就不需要进行优化；
// 3.将搜索结果全部保存到matA0中，形成一个5x3的矩阵；
// 4.解这个矩阵cv::solve(matA0, matB0, matX0, cv::DECOMP_QR);
    void surfOptimization(int iterCount){
        updatePointAssociateToMapSinCos();
        for (int i = 0; i < laserCloudSurfTotalLastDSNum; i++) {
            pointOri = laserCloudSurfTotalLastDS->points[i];
            pointAssociateToMap(&pointOri, &pointSel); 
            kdtreeSurfFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);

            if (pointSearchSqDis[4] < 1.0) {
                for (int j = 0; j < 5; j++) {
                    matA0.at<float>(j, 0) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].x;
                    matA0.at<float>(j, 1) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].y;
                    matA0.at<float>(j, 2) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].z;
                }
                cv::solve(matA0, matB0, matX0, cv::DECOMP_QR);

                float pa = matX0.at<float>(0, 0);
                float pb = matX0.at<float>(1, 0);
                float pc = matX0.at<float>(2, 0);
                float pd = 1;

                float ps = sqrt(pa * pa + pb * pb + pc * pc);
                pa /= ps; pb /= ps; pc /= ps; pd /= ps;

                bool planeValid = true;
                for (int j = 0; j < 5; j++) {
                    if (fabs(pa * laserCloudSurfFromMapDS->points[pointSearchInd[j]].x +
                             pb * laserCloudSurfFromMapDS->points[pointSearchInd[j]].y +
                             pc * laserCloudSurfFromMapDS->points[pointSearchInd[j]].z + pd) > 0.2) {
                        planeValid = false;
                        break;
                    }
                }

                if (planeValid) {
                    float pd2 = pa * pointSel.x + pb * pointSel.y + pc * pointSel.z + pd;

                    float s = 1 - 0.9 * fabs(pd2) / sqrt(sqrt(pointSel.x * pointSel.x
                            + pointSel.y * pointSel.y + pointSel.z * pointSel.z));

                    coeff.x = s * pa;
                    coeff.y = s * pb;
                    coeff.z = s * pc;
                    coeff.intensity = s * pd2;

                    if (s > 0.1) {
                        laserCloudOri->push_back(pointOri);
                        coeffSel->push_back(coeff);
                    }
                }
            }
        }
    }
//高斯牛顿法优化
    bool LMOptimization(int iterCount){
        float srx = sin(transformTobeMapped[0]);
        float crx = cos(transformTobeMapped[0]);
        float sry = sin(transformTobeMapped[1]);
        float cry = cos(transformTobeMapped[1]);
        float srz = sin(transformTobeMapped[2]);
        float crz = cos(transformTobeMapped[2]);

        int laserCloudSelNum = laserCloudOri->points.size();
        if (laserCloudSelNum < 50) {
            return false;
        }

        cv::Mat matA(laserCloudSelNum, 6, CV_32F, cv::Scalar::all(0));
        cv::Mat matAt(6, laserCloudSelNum, CV_32F, cv::Scalar::all(0));
        cv::Mat matAtA(6, 6, CV_32F, cv::Scalar::all(0));
        cv::Mat matB(laserCloudSelNum, 1, CV_32F, cv::Scalar::all(0));
        cv::Mat matAtB(6, 1, CV_32F, cv::Scalar::all(0));
        cv::Mat matX(6, 1, CV_32F, cv::Scalar::all(0));
        for (int i = 0; i < laserCloudSelNum; i++) {
            pointOri = laserCloudOri->points[i];
            coeff = coeffSel->points[i];

            float arx = (crx*sry*srz*pointOri.x + crx*crz*sry*pointOri.y - srx*sry*pointOri.z) * coeff.x
                      + (-srx*srz*pointOri.x - crz*srx*pointOri.y - crx*pointOri.z) * coeff.y
                      + (crx*cry*srz*pointOri.x + crx*cry*crz*pointOri.y - cry*srx*pointOri.z) * coeff.z;

            float ary = ((cry*srx*srz - crz*sry)*pointOri.x 
                      + (sry*srz + cry*crz*srx)*pointOri.y + crx*cry*pointOri.z) * coeff.x
                      + ((-cry*crz - srx*sry*srz)*pointOri.x 
                      + (cry*srz - crz*srx*sry)*pointOri.y - crx*sry*pointOri.z) * coeff.z;

            float arz = ((crz*srx*sry - cry*srz)*pointOri.x + (-cry*crz-srx*sry*srz)*pointOri.y)*coeff.x
                      + (crx*crz*pointOri.x - crx*srz*pointOri.y) * coeff.y
                      + ((sry*srz + cry*crz*srx)*pointOri.x + (crz*sry-cry*srx*srz)*pointOri.y)*coeff.z;

            matA.at<float>(i, 0) = arx;
            matA.at<float>(i, 1) = ary;
            matA.at<float>(i, 2) = arz;
            matA.at<float>(i, 3) = coeff.x;
            matA.at<float>(i, 4) = coeff.y;
            matA.at<float>(i, 5) = coeff.z;
            matB.at<float>(i, 0) = -coeff.intensity;
        }
        cv::transpose(matA, matAt);
        matAtA = matAt * matA;
        matAtB = matAt * matB;
        cv::solve(matAtA, matAtB, matX, cv::DECOMP_QR);

        if (iterCount == 0) {
            cv::Mat matE(1, 6, CV_32F, cv::Scalar::all(0));
            cv::Mat matV(6, 6, CV_32F, cv::Scalar::all(0));
            cv::Mat matV2(6, 6, CV_32F, cv::Scalar::all(0));

            cv::eigen(matAtA, matE, matV);
            matV.copyTo(matV2);

            isDegenerate = false;
            float eignThre[6] = {100, 100, 100, 100, 100, 100};
            for (int i = 5; i >= 0; i--) {
                if (matE.at<float>(0, i) < eignThre[i]) {
                    for (int j = 0; j < 6; j++) {
                        matV2.at<float>(i, j) = 0;
                    }
                    isDegenerate = true;
                } else {
                    break;
                }
            }
            matP = matV.inv() * matV2;
        }

        if (isDegenerate) {
            cv::Mat matX2(6, 1, CV_32F, cv::Scalar::all(0));
            matX.copyTo(matX2);
            matX = matP * matX2;
        }

        transformTobeMapped[0] += matX.at<float>(0, 0);
        transformTobeMapped[1] += matX.at<float>(1, 0);
        transformTobeMapped[2] += matX.at<float>(2, 0);
        transformTobeMapped[3] += matX.at<float>(3, 0);
        transformTobeMapped[4] += matX.at<float>(4, 0);
        transformTobeMapped[5] += matX.at<float>(5, 0);

        float deltaR = sqrt(
                            pow(pcl::rad2deg(matX.at<float>(0, 0)), 2) +
                            pow(pcl::rad2deg(matX.at<float>(1, 0)), 2) +
                            pow(pcl::rad2deg(matX.at<float>(2, 0)), 2));
        float deltaT = sqrt(
                            pow(matX.at<float>(3, 0) * 100, 2) +
                            pow(matX.at<float>(4, 0) * 100, 2) +
                            pow(matX.at<float>(5, 0) * 100, 2));

        if (deltaR < 0.05 && deltaT < 0.05) {
            return true;
        }
        return false;
    }

//   是一个对代码进行优化控制的函数，主要在里面调用面优化，角优化以及L-Ｍ优化
//   使用scan-to-model位姿优化，获得当前时间点机器人的位姿transformTobeMapped，使得总体残差最小，连续循环优化多次。
//   此外该部分的优化会参考IMU消息回调所确定的roll和pitch对该位姿进行修正，对 transformTobeMapped 进行中值滤波,获得最终的机器人位姿。
//   到这里，虽然在scan-to-scan之后，又进行了scan-to-map的匹配，但是并未出现回环检测和优化，所以依然是一个误差不断积累的里程计的概念。

// 使用scan-to-model位姿优化，获得当前时间点机器人的位姿 transformTobeMapped
// 参考IMU的姿态 对 transformTobeMapped 进行中值滤波,获得最终的机器人位姿
// 为 transformBefMapped 赋值为 里程计位姿，即scan-to-model优化前的位姿 
// 为 transformAftMapped 赋值为 transformTobeMapped 即scan-to-model优化后的位姿 

    void scan2MapOptimization(){
        // laserCloudCornerFromMapDSNum是extractSurroundingKeyFrames()函数最后降采样得到的coner点云数
         // laserCloudSurfFromMapDSNum是extractSurroundingKeyFrames()函数降采样得到的surface点云数
        if (laserCloudCornerFromMapDSNum > 10 && laserCloudSurfFromMapDSNum > 100) {
             // laserCloudCornerFromMapDS和laserCloudSurfFromMapDS的来源有2个：
             // 当有闭环时，来源是：recentCornerCloudKeyFrames
             // 没有闭环时，来源是：surroundingCornerCloudKeyFrames

            kdtreeCornerFromMap->setInputCloud(laserCloudCornerFromMapDS);
            kdtreeSurfFromMap->setInputCloud(laserCloudSurfFromMapDS);
           //该函数控制了进行优化的最大次数为10次
            for (int iterCount = 0; iterCount < 10; iterCount++) {
                laserCloudOri->clear();
                coeffSel->clear();
                // 关于角特征的优化
                cornerOptimization(iterCount);
                // 关于特征平面的优化：surfOptimization;
                surfOptimization(iterCount);

                if (LMOptimization(iterCount) == true)
                    break;              
            }
            // 迭代结束更新相关的转移矩阵
            transformUpdate();
        }
    }

// 保存关键帧和进行优化的功能
               // 1选定关键帧 
                // 2根据新的关键帧更新因子图
                // 3经过因子图优化后，更新当前位姿 
                // transformAftMapped   
                // transformTobeMapped  
                // 并用因子图优化后的位姿 创建关键帧位置点和位姿点,添加到集合 
                // cloudKeyPoses3D  关键帧位置点集合 
                // cloudKeyPoses6D  关键帧位姿点集合
                // 并为 transformLast 赋值 
                // 更新 
                // cornerCloudKeyFrames 关键帧角点点云集合
                // surfCloudKeyFrames   关键帧平面点点云集合
                // outlierCloudKeyFrames关键帧离群点点云集合 
    void saveKeyFramesAndFactor(){
        // 把上次优化得到的transformAftMapped(3:5)坐标点作为当前的位置
        currentRobotPosPoint.x = transformAftMapped[3];
        currentRobotPosPoint.y = transformAftMapped[4];
        currentRobotPosPoint.z = transformAftMapped[5];

        bool saveThisKeyFrame = true;
         // save keyframe every 0.3 meter 
        //  计算和再之前的位置的欧拉距离，距离太小并且cloudKeyPoses3D不为空(初始化时为空)，则结束；
        if (sqrt((previousRobotPosPoint.x-currentRobotPosPoint.x)*(previousRobotPosPoint.x-currentRobotPosPoint.x)
                    +(previousRobotPosPoint.y-currentRobotPosPoint.y)*(previousRobotPosPoint.y-currentRobotPosPoint.y)
                    +(previousRobotPosPoint.z-currentRobotPosPoint.z)*(previousRobotPosPoint.z-currentRobotPosPoint.z)) < 0.3){ 
            saveThisKeyFrame = false;
        }
        // 如果是刚刚初始化，cloudKeyPoses3D为空，      
        // 那么NonlinearFactorGraph增加一个PriorFactor因子，      
        // initialEstimate的数据类型是Values（其实就是一个map），这里在0对应的值下面保存一个Pose3，      
        // 本次的transformTobeMapped参数保存到transformLast中去。    
        if (saveThisKeyFrame == false && !cloudKeyPoses3D->points.empty())
        	return;

        previousRobotPosPoint = currentRobotPosPoint;

        /*
         * update grsam graph
         */

        if (cloudKeyPoses3D->points.empty()){
            gtSAMgraph.add(PriorFactor<Pose3>(0, Pose3(Rot3::RzRyRx(transformTobeMapped[2], transformTobeMapped[0], transformTobeMapped[1]),
                                                       		 Point3(transformTobeMapped[5], transformTobeMapped[3], transformTobeMapped[4])), priorNoise));
            initialEstimate.insert(0, Pose3(Rot3::RzRyRx(transformTobeMapped[2], transformTobeMapped[0], transformTobeMapped[1]),
                                                  Point3(transformTobeMapped[5], transformTobeMapped[3], transformTobeMapped[4])));
            for (int i = 0; i < 6; ++i)
            	transformLast[i] = transformTobeMapped[i];
        }
        else{
            // 如果本次不是刚刚初始化，从transformLast得到上一次位姿，      
            // 从transformAftMapped得到本次位姿，      
            // gtSAMgraph.add(BetweenFactor),到它的约束中去，      
            // initialEstimate.insert(序号，位姿)
            //如果是第一帧数据则直接加入到位姿图中
            //否则按更新后的位姿对位姿图进行更新
            gtsam::Pose3 poseFrom = Pose3(Rot3::RzRyRx(transformLast[2], transformLast[0], transformLast[1]),
                                                Point3(transformLast[5], transformLast[3], transformLast[4]));
            gtsam::Pose3 poseTo   = Pose3(Rot3::RzRyRx(transformAftMapped[2], transformAftMapped[0], transformAftMapped[1]),
                                                Point3(transformAftMapped[5], transformAftMapped[3], transformAftMapped[4]));
            gtSAMgraph.add(BetweenFactor<Pose3>(cloudKeyPoses3D->points.size()-1, cloudKeyPoses3D->points.size(), poseFrom.between(poseTo), odometryNoise));
            initialEstimate.insert(cloudKeyPoses3D->points.size(), Pose3(Rot3::RzRyRx(transformAftMapped[2], transformAftMapped[0], transformAftMapped[1]),
                                                                     		   Point3(transformAftMapped[5], transformAftMapped[3], transformAftMapped[4])));
        }
        /**
         * update iSAM
         * 不管是否是初始化，都进行优化，isam->update(gtSAMgraph, initialEstimate);      
            得到优化的结果：latestEstimate = isamCurrentEstimate.at<Pose3>(isamCurrentEstimate.size()-1),      
            将结果保存，cloudKeyPoses3D->push_back(thisPose3D);      
            cloudKeyPoses6D->push_back(thisPose6D);  
         */
        isam->update(gtSAMgraph, initialEstimate);
        isam->update();
        
        gtSAMgraph.resize(0);
        initialEstimate.clear();

        /**
         * save key poses
         */
        //  cloudKeyPoses3D指当前的点云（ｘｙｚｉ），cloudKeyPoses6D指带pose的（x y z i t roll yaw pitch）
        PointType thisPose3D;
        PointTypePose thisPose6D;
        Pose3 latestEstimate;

        isamCurrentEstimate = isam->calculateEstimate();
        latestEstimate = isamCurrentEstimate.at<Pose3>(isamCurrentEstimate.size()-1);

        thisPose3D.x = latestEstimate.translation().y();
        thisPose3D.y = latestEstimate.translation().z();
        thisPose3D.z = latestEstimate.translation().x();
        thisPose3D.intensity = cloudKeyPoses3D->points.size(); // this can be used as index
        cloudKeyPoses3D->push_back(thisPose3D);

        thisPose6D.x = thisPose3D.x;
        thisPose6D.y = thisPose3D.y;
        thisPose6D.z = thisPose3D.z;
        thisPose6D.intensity = thisPose3D.intensity; // this can be used as index
        thisPose6D.roll  = latestEstimate.rotation().pitch();
        thisPose6D.pitch = latestEstimate.rotation().yaw();
        thisPose6D.yaw   = latestEstimate.rotation().roll(); // in camera frame
        thisPose6D.time = timeLaserOdometry;
        cloudKeyPoses6D->push_back(thisPose6D);

        /**
         * save updated transform       对transformAftMapped进行更新
         */
        if (cloudKeyPoses3D->points.size() > 1){
            transformAftMapped[0] = latestEstimate.rotation().pitch();
            transformAftMapped[1] = latestEstimate.rotation().yaw();
            transformAftMapped[2] = latestEstimate.rotation().roll();
            transformAftMapped[3] = latestEstimate.translation().y();
            transformAftMapped[4] = latestEstimate.translation().z();
            transformAftMapped[5] = latestEstimate.translation().x();
            //轮转更新位姿
            for (int i = 0; i < 6; ++i){
            	transformLast[i] = transformAftMapped[i];
            	transformTobeMapped[i] = transformAftMapped[i];
            }
        }

        pcl::PointCloud<PointType>::Ptr thisCornerKeyFrame(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr thisSurfKeyFrame(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr thisOutlierKeyFrame(new pcl::PointCloud<PointType>());

        pcl::copyPointCloud(*laserCloudCornerLastDS,  *thisCornerKeyFrame);
        pcl::copyPointCloud(*laserCloudSurfLastDS,    *thisSurfKeyFrame);
        pcl::copyPointCloud(*laserCloudOutlierLastDS, *thisOutlierKeyFrame);

        /* 
            Scan Context loop detector 
            - ver 1: using surface feature as an input point cloud for scan context (2020.04.01: checked it works.)
            - ver 2: using downsampled original point cloud (/full_cloud_projected + downsampling)
            */
        bool usingRawCloud = true;
        if( usingRawCloud ) { // v2 uses downsampled raw point cloud, more fruitful height information than using feature points (v1)
            //  这里对点云提取scan context特征
            pcl::PointCloud<PointType>::Ptr thisRawCloudKeyFrame(new pcl::PointCloud<PointType>());
            pcl::copyPointCloud(*laserCloudRawDS,  *thisRawCloudKeyFrame);
            scManager.makeAndSaveScancontextAndKeys(*thisRawCloudKeyFrame);
        }
        else { // v1 uses thisSurfKeyFrame, it also works. (empirically checked at Mulran dataset sequences)
            scManager.makeAndSaveScancontextAndKeys(*thisSurfKeyFrame); 
        }        
        // 最后保存最终的结果
        cornerCloudKeyFrames.push_back(thisCornerKeyFrame);
        surfCloudKeyFrames.push_back(thisSurfKeyFrame);
        outlierCloudKeyFrames.push_back(thisOutlierKeyFrame);
    } 

// void correctPoses()的调用只在回环j检测成功结束时进行(aLoopIsClosed == true),
// 回环检测成功后将位姿图的数据依次更新
// 校正位姿的过程主要是将isamCurrentEstimate的x，y，z平移坐标更新到cloudKeyPoses3D中，另外还需要更新cloudKeyPoses6D的姿态角。
    void correctPoses(){
    	if (aLoopIsClosed == true){
            recentCornerCloudKeyFrames. clear();
            recentSurfCloudKeyFrames.   clear();
            recentOutlierCloudKeyFrames.clear();

            // update key poses
            int numPoses = isamCurrentEstimate.size();
            for (int i = 0; i < numPoses; ++i){
                cloudKeyPoses3D->points[i].x = isamCurrentEstimate.at<Pose3>(i).translation().y();
                cloudKeyPoses3D->points[i].y = isamCurrentEstimate.at<Pose3>(i).translation().z();
                cloudKeyPoses3D->points[i].z = isamCurrentEstimate.at<Pose3>(i).translation().x();

                cloudKeyPoses6D->points[i].x = cloudKeyPoses3D->points[i].x;
                cloudKeyPoses6D->points[i].y = cloudKeyPoses3D->points[i].y;
                cloudKeyPoses6D->points[i].z = cloudKeyPoses3D->points[i].z;
                cloudKeyPoses6D->points[i].roll  = isamCurrentEstimate.at<Pose3>(i).rotation().pitch();
                cloudKeyPoses6D->points[i].pitch = isamCurrentEstimate.at<Pose3>(i).rotation().yaw();
                cloudKeyPoses6D->points[i].yaw   = isamCurrentEstimate.at<Pose3>(i).rotation().roll();
            }
            aLoopIsClosed = false;
        }
    }

    void clearCloud(){
        laserCloudCornerFromMap->clear();
        laserCloudSurfFromMap->clear();  
        laserCloudCornerFromMapDS->clear();
        laserCloudSurfFromMapDS->clear();   
    }

    void run(){
        //  判断是否有新的数据到来并且时间差值小于0.005(实时性＋时间戳一致 )执行后续
        // 三种点云、以及里程计信息正常接收、且时间戳一致 
        if (newLaserCloudCornerLast  && std::abs(timeLaserCloudCornerLast  - timeLaserOdometry) < 0.005 &&
            newLaserCloudSurfLast    && std::abs(timeLaserCloudSurfLast    - timeLaserOdometry) < 0.005 &&
            newLaserCloudOutlierLast && std::abs(timeLaserCloudOutlierLast - timeLaserOdometry) < 0.005 &&
            newLaserOdometry)
        {
            //标志位重新置回false 
            newLaserCloudCornerLast = false;
             newLaserCloudSurfLast = false;
              newLaserCloudOutlierLast = false; 
              newLaserOdometry = false;

              // 创建对象 lock 对mtx进行上锁,当lock被析构时,自动解锁
            std::lock_guard<std::mutex> lock(mtx);

            // 全局姿态优化的时间判断条件：距离上次优化已经过去0.3s
            if (timeLaserOdometry - timeLastProcessing >= mappingProcessInterval) {//0.3
                timeLastProcessing = timeLaserOdometry;

                transformAssociateToMap();  //  坐标系->map,得到可用于建图的Lidar坐标，即修改transformTobeMapped的值；

                extractSurroundingKeyFrames();  //抽取周围的关键帧；

                downsampleCurrentScan();//降采样当前scan；

                scan2MapOptimization();//使用scan-to-model因子图位姿优化

                saveKeyFramesAndFactor(); //保存关键帧和因子；

                correctPoses(); //更新正确的位姿 ,闭环处理矫正

                publishTF();//发布优化后的位姿和tf变换

                publishKeyPosesAndFrames();//发布所有关键帧位姿,当前的局部面点地图及当前帧中的面点/角点。

                clearCloud();//清空　重置
            }
        }
    }
};


int main(int argc, char** argv)
{
    ros::init(argc, argv, "lego_loam");

    ROS_INFO("\033[1;32m---->\033[0m Map Optimization Started.");

    mapOptimization MO;

     // std::thread 构造函数，将MO作为参数传入构造的线程中使用
    // 进行闭环检测与闭环的功能
    std::thread loopthread(&mapOptimization::loopClosureThread, &MO);

    // 该线程中进行的工作是publishGlobalMap(),将数据发布到ros中，可视化
    std::thread visualizeMapThread(&mapOptimization::visualizeGlobalMapThread, &MO);

    ros::Rate rate(200);
    while (ros::ok())
    // while ( 1 )
    {
        ros::spinOnce();
        //全局姿态优化的主要处理部分
        MO.run();
        rate.sleep();
    }
    loopthread.join();
    visualizeMapThread.join();
    return 0;
}
