//特征提取。
//代码中先初始化了lego_loam::FeatureAssociation，用来订阅了上一节点发出来的分割出来的点云，点云的属性，外点以及IMU消息，并设置了回调函数
// 其中IMU消息的订阅函数较为复杂，它从IMU数据中提取出姿态，角速度和线加速度，其中姿态用来消除重力对线加速度的影响。
// 然后函数FeatureAssociation::AccumulateIMUShiftAndRotation用来做积分，包括根据姿态，将加速度往世界坐标系下进行投影。
// 再根据匀加速度运动模型积分得到速度和位移，同时，对角速度也进行了积分。


#include "utility.h"
class FeatureAssociation{

private:
    // ROS句柄
	ros::NodeHandle nh;
    //三个点云信息订阅者 
    ros::Subscriber subLaserCloud;
    ros::Subscriber subLaserCloudInfo;
    ros::Subscriber subOutlierCloud;
    //IMU数据订阅者
    ros::Subscriber subImu;
    //多个发布者 
    ros::Publisher pubCornerPointsSharp;
    ros::Publisher pubCornerPointsLessSharp;
    ros::Publisher pubSurfPointsFlat;
    ros::Publisher pubSurfPointsLessFlat;


    //点云集合
    pcl::PointCloud<PointType>::Ptr segmentedCloud;           //主要是被分割点和经过降采样的地面点
    pcl::PointCloud<PointType>::Ptr outlierCloud;                    //被放弃的点云：主要是经过降采样的未被分割点云。

    pcl::PointCloud<PointType>::Ptr cornerPointsSharp;          //强角点
    pcl::PointCloud<PointType>::Ptr cornerPointsLessSharp;//普通角点
    pcl::PointCloud<PointType>::Ptr surfPointsFlat;                  //平面点
    pcl::PointCloud<PointType>::Ptr surfPointsLessFlat;         //除了角点的其他点

    pcl::PointCloud<PointType>::Ptr surfPointsLessFlatScan;   //当前线上的 除了角点的其他所有点 
    pcl::PointCloud<PointType>::Ptr surfPointsLessFlatScanDS; //上述点集的降采样

    pcl::VoxelGrid<PointType> downSizeFilter;   //用于对 surfPointsLessFlatScan 进行降采样

    //传感器采集到点云信息的时间。
    double timeScanCur;                                     // segmentedCloud的时间
    double timeNewSegmentedCloud;        // segmentedCloud的时间
    double timeNewSegmentedCloudInfo; //segInfo       的时间
    double timeNewOutlierCloud;                  // outlierCloud  的时间

    // 三个点云数据进入featureAssociation的标志位,接收到数据使,赋值为true,进入处理过程后,再次赋值为false
    bool newSegmentedCloud;
    bool newSegmentedCloudInfo;
    bool newOutlierCloud;

    cloud_msgs::cloud_info segInfo;          // 点云分割信息.即主题"/segmented_cloud_info"的接收载体
    std_msgs::Header cloudHeader;          // segmentedCloud进入时，记录其header作为当前点云的header 

    int systemInitCount;
    bool systemInited;

    std::vector<smoothness_t> cloudSmoothness;//点云中每个点的光滑度,每一项包含<float value,size_t ind>即粗糙度值、和该点在点云中的索引。
    float   *cloudCurvature;        //粗糙度序列
    int     *cloudNeighborPicked;   //择取特征点的标志位(如果值1,则不选为特征点),防止特征点集中地出现
    int     *cloudLabel;            //记录特征点的属性 2:强角点;1普通角点;-1 平面点

    int imuPointerFront;            //与点云数据时间轴对齐的IMU数据位置
    int imuPointerLast;             //最新的IMU数据在队列中的位置
    int imuPointerLastIteration;    //处理上一张点云时的结束IMU数据位置 

    // 每一帧点云的第一个点时的机器人的相关属性值 （初始化）
    float imuRollStart, imuPitchStart, imuYawStart;               //起始 RPY值
    float cosImuRollStart, cosImuPitchStart, cosImuYawStart, sinImuRollStart, sinImuPitchStart, sinImuYawStart;     //起始RPY值的正余弦
    float imuVeloXStart, imuVeloYStart, imuVeloZStart;//起始速度值
    float imuShiftXStart, imuShiftYStart, imuShiftZStart;//起始的位置里程计

    // 当前点时 机器人的相关属性值  
    float imuRollCur, imuPitchCur, imuYawCur;                         //RPY值 
    float imuVeloXCur, imuVeloYCur, imuVeloZCur;                    //速度值
    float imuShiftXCur, imuShiftYCur, imuShiftZCur;                // 位置里程计
    float imuAngularRotationXCur, imuAngularRotationYCur, imuAngularRotationZCur; // 角度里程计 

    float imuShiftFromStartXCur, imuShiftFromStartYCur, imuShiftFromStartZCur;   // 位置里程计变化
    float imuVeloFromStartXCur, imuVeloFromStartYCur, imuVeloFromStartZCur;      // 当前点距离当前帧起始点的 速度变化

    float imuAngularRotationXLast, imuAngularRotationYLast, imuAngularRotationZLast;   // 记录上一帧点云起始点的 角度里程计 
    float imuAngularFromStartX, imuAngularFromStartY, imuAngularFromStartZ;                    //   上一阵过程中 角度里程计(角度变化量)

    //IMU数据队列(长度是imuQueLength,循环使用)：分别存储IMU数据的 时间、RPY、加速度、速度、位置里程计、角速度里程计、角度里程计 队列,imuQueLength
    double imuTime[imuQueLength];//时间

    float imuRoll[imuQueLength];//RPY值
    float imuPitch[imuQueLength];
    float imuYaw[imuQueLength];

    float imuAccX[imuQueLength];// 加速度值
    float imuAccY[imuQueLength];
    float imuAccZ[imuQueLength];

    float imuVeloX[imuQueLength];// 速度值 
    float imuVeloY[imuQueLength];
    float imuVeloZ[imuQueLength];

    float imuShiftX[imuQueLength];// 位置里程计
    float imuShiftY[imuQueLength];
    float imuShiftZ[imuQueLength];

    float imuAngularVeloX[imuQueLength];// 角速度
    float imuAngularVeloY[imuQueLength];
    float imuAngularVeloZ[imuQueLength];

    float imuAngularRotationX[imuQueLength];// 角度里程计
    float imuAngularRotationY[imuQueLength];
    float imuAngularRotationZ[imuQueLength];


    // 用于发布经过处理后的信息 
    ros::Publisher pubLaserCloudCornerLast;      //以skipFrameNum的频率发布的 Less角点点云 
    ros::Publisher pubLaserCloudSurfLast;           //以skipFrameNum的频率发布的 Less平面点点云 
    ros::Publisher pubLaserOdometry;
    ros::Publisher pubOutlierCloudLast;

    int skipFrameNum;                                                 // 系统每过 skipFrameNum 张点云,进行一次信息发布:outlierCloud\Less角点\Less平面点
    bool systemInitedLM;                                            // 系统是否进行初始化的标志 

    int laserCloudCornerLastNum;                       // 上一帧中 less角点的数量 
    int laserCloudSurfLastNum;                            // 上一帧中 less平面点的数量

    int *pointSelCornerInd;
    float *pointSearchCornerInd1;
    float *pointSearchCornerInd2;

    int *pointSelSurfInd;                          
    float *pointSearchSurfInd1;                //最近点 序列
    float *pointSearchSurfInd2;                // 前向次近点 序列
    float *pointSearchSurfInd3;                // 后向次近点 序列 

    float transformCur[6];                         // 当前的位姿估计
    float transformSum[6];                       // 累积位姿变换量。

    float imuRollLast, imuPitchLast, imuYawLast;                                                                   //当前帧末尾  RPY值 
    float imuShiftFromStartX, imuShiftFromStartY, imuShiftFromStartZ;                    //扫描过程中的 位置里程计 变化
    float imuVeloFromStartX, imuVeloFromStartY, imuVeloFromStartZ;                       //当前帧的扫描过程 的速度变化

    pcl::PointCloud<PointType>::Ptr laserCloudCornerLast;                                          // 上一点云帧的 角点点云 
    pcl::PointCloud<PointType>::Ptr laserCloudSurfLast;                                               // 上一帧中  除了角点的其他点
    pcl::PointCloud<PointType>::Ptr laserCloudOri;                                                          // 记录 匹配点过程中  当前点序列 
    pcl::PointCloud<PointType>::Ptr coeffSel;                                                                      //   寻找匹配点过程中 获得的 数据 

    pcl::KdTreeFLANN<PointType>::Ptr kdtreeCornerLast;                                            //记录上一帧less角点的KD树 
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurfLast;                                                  //记录上一帧less平面点的KD树

    std::vector<int> pointSearchInd;
    std::vector<float> pointSearchSqDis;

    // 分别是 变换后的当前点；最近点、前向次近点、后向次近点、
    PointType pointOri, pointSel, tripod1, tripod2, tripod3, pointProj, coeff;    

    nav_msgs::Odometry laserOdometry;

    tf::TransformBroadcaster tfBroadcaster;
    tf::StampedTransform laserOdometryTrans;

    bool isDegenerate;
    cv::Mat matP;

    int frameCount;// 用来辅助 skipFrameNum 进行信息发布频率的控制 

public:

    FeatureAssociation(): nh("~")
    {
        // 包含被分割点和经过降采样的地面点
        subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2>("/segmented_cloud", 1, &FeatureAssociation::laserCloudHandler, this);
        // 订阅segMsg 
        // 主题"/segmented_cloud_info"发送的是激光数据分割信息:
        // segMsg{
        //     Header header                       //与接收到的当前帧点云数据的header一致
        //     int32[]     startRingIndex          //segMsg点云(以一维形式存储)中,每一行点云的起始和结束索引
        //     int32[]     endRingIndex      
        //     float32     startOrientation        //起始点与结束点的水平角度(atan(y,x))
        //     float32     endOrientation  
        //     float32     orientationDiff         //以上两者的差
        //     bool[]      segmentedCloudGroundFlag//segMsg中点云的地面点标志序列(true:ground point)
        //     uint32[]    segmentedCloudColInd    //segMsg中点云的cols序列
        //     float32[]   segmentedCloudRange     //segMsg中点云的range   
        // }
        subLaserCloudInfo = nh.subscribe<cloud_msgs::cloud_info>("/segmented_cloud_info", 1, &FeatureAssociation::laserCloudInfoHandler, this);
        // 经过降采样的未分割点
        subOutlierCloud = nh.subscribe<sensor_msgs::PointCloud2>("/outlier_cloud", 1, &FeatureAssociation::outlierCloudHandler, this);

        // 订阅IMU信息，保存在队列中
        // std_msgs/Header header
        // geometry_msgs/Quaternion orientation    //姿态
        // float64[9] orientation_covariance       //姿态的方差
        // geometry_msgs/Vector3 angular_velocity  //角速度
        // float64[9] angular_velocity_covariance  // 角速度的方差
        // geometry_msgs/Vector3 linear_acceleration   //线速度
        // float64[9] linear_acceleration_covariance   //线速度的方差
        subImu = nh.subscribe<sensor_msgs::Imu>(imuTopic, 50, &FeatureAssociation::imuHandler, this);
       //发布话题：
        pubCornerPointsSharp = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_sharp", 1);
        pubCornerPointsLessSharp = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_less_sharp", 1);
        pubSurfPointsFlat = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_flat", 1);
        pubSurfPointsLessFlat = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_less_flat", 1);

        pubLaserCloudCornerLast = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_corner_last", 2);
        pubLaserCloudSurfLast = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_surf_last", 2);
        pubOutlierCloudLast = nh.advertise<sensor_msgs::PointCloud2>("/outlier_cloud_last", 2);
        pubLaserOdometry = nh.advertise<nav_msgs::Odometry> ("/laser_odom_to_init", 5);
        
        initializationValue();
    }

    void initializationValue()
    {
        cloudCurvature = new float[N_SCAN*Horizon_SCAN];
        cloudNeighborPicked = new int[N_SCAN*Horizon_SCAN];
        cloudLabel = new int[N_SCAN*Horizon_SCAN];

        pointSelCornerInd = new int[N_SCAN*Horizon_SCAN];
        pointSearchCornerInd1 = new float[N_SCAN*Horizon_SCAN];
        pointSearchCornerInd2 = new float[N_SCAN*Horizon_SCAN];

        pointSelSurfInd = new int[N_SCAN*Horizon_SCAN];
        pointSearchSurfInd1 = new float[N_SCAN*Horizon_SCAN];
        pointSearchSurfInd2 = new float[N_SCAN*Horizon_SCAN];
        pointSearchSurfInd3 = new float[N_SCAN*Horizon_SCAN];

        cloudSmoothness.resize(N_SCAN*Horizon_SCAN);

        downSizeFilter.setLeafSize(0.2, 0.2, 0.2);

        segmentedCloud.reset(new pcl::PointCloud<PointType>());
        outlierCloud.reset(new pcl::PointCloud<PointType>());

        cornerPointsSharp.reset(new pcl::PointCloud<PointType>());
        cornerPointsLessSharp.reset(new pcl::PointCloud<PointType>());
        surfPointsFlat.reset(new pcl::PointCloud<PointType>());
        surfPointsLessFlat.reset(new pcl::PointCloud<PointType>());

        surfPointsLessFlatScan.reset(new pcl::PointCloud<PointType>());
        surfPointsLessFlatScanDS.reset(new pcl::PointCloud<PointType>());

        timeScanCur = 0;
        timeNewSegmentedCloud = 0;
        timeNewSegmentedCloudInfo = 0;
        timeNewOutlierCloud = 0;

        newSegmentedCloud = false;
        newSegmentedCloudInfo = false;
        newOutlierCloud = false;

        systemInitCount = 0;
        systemInited = false;

        imuPointerFront = 0;
        imuPointerLast = -1;
        imuPointerLastIteration = 0;

        imuRollStart = 0; imuPitchStart = 0; imuYawStart = 0;
        cosImuRollStart = 0; cosImuPitchStart = 0; cosImuYawStart = 0;
        sinImuRollStart = 0; sinImuPitchStart = 0; sinImuYawStart = 0;
        imuRollCur = 0; imuPitchCur = 0; imuYawCur = 0;

        imuVeloXStart = 0; imuVeloYStart = 0; imuVeloZStart = 0;
        imuShiftXStart = 0; imuShiftYStart = 0; imuShiftZStart = 0;

        imuVeloXCur = 0; imuVeloYCur = 0; imuVeloZCur = 0;
        imuShiftXCur = 0; imuShiftYCur = 0; imuShiftZCur = 0;

        imuShiftFromStartXCur = 0; imuShiftFromStartYCur = 0; imuShiftFromStartZCur = 0;
        imuVeloFromStartXCur = 0; imuVeloFromStartYCur = 0; imuVeloFromStartZCur = 0;

        imuAngularRotationXCur = 0; imuAngularRotationYCur = 0; imuAngularRotationZCur = 0;
        imuAngularRotationXLast = 0; imuAngularRotationYLast = 0; imuAngularRotationZLast = 0;
        imuAngularFromStartX = 0; imuAngularFromStartY = 0; imuAngularFromStartZ = 0;

        for (int i = 0; i < imuQueLength; ++i)
        {
            imuTime[i] = 0;
            imuRoll[i] = 0; imuPitch[i] = 0; imuYaw[i] = 0;
            imuAccX[i] = 0; imuAccY[i] = 0; imuAccZ[i] = 0;
            imuVeloX[i] = 0; imuVeloY[i] = 0; imuVeloZ[i] = 0;
            imuShiftX[i] = 0; imuShiftY[i] = 0; imuShiftZ[i] = 0;
            imuAngularVeloX[i] = 0; imuAngularVeloY[i] = 0; imuAngularVeloZ[i] = 0;
            imuAngularRotationX[i] = 0; imuAngularRotationY[i] = 0; imuAngularRotationZ[i] = 0;
        }


        skipFrameNum = 1;

        for (int i = 0; i < 6; ++i){
            transformCur[i] = 0;
            transformSum[i] = 0;
        }

        systemInitedLM = false;

        imuRollLast = 0; imuPitchLast = 0; imuYawLast = 0;
        imuShiftFromStartX = 0; imuShiftFromStartY = 0; imuShiftFromStartZ = 0;
        imuVeloFromStartX = 0; imuVeloFromStartY = 0; imuVeloFromStartZ = 0;

        laserCloudCornerLast.reset(new pcl::PointCloud<PointType>());
        laserCloudSurfLast.reset(new pcl::PointCloud<PointType>());
        laserCloudOri.reset(new pcl::PointCloud<PointType>());
        coeffSel.reset(new pcl::PointCloud<PointType>());

        kdtreeCornerLast.reset(new pcl::KdTreeFLANN<PointType>());
        kdtreeSurfLast.reset(new pcl::KdTreeFLANN<PointType>());

        laserOdometry.header.frame_id = "/camera_init";
        laserOdometry.child_frame_id = "/laser_odom";

        laserOdometryTrans.frame_id_ = "/camera_init";
        laserOdometryTrans.child_frame_id_ = "/laser_odom";
        
        isDegenerate = false;
        matP = cv::Mat(6, 6, CV_32F, cv::Scalar::all(0));

        frameCount = skipFrameNum;
    }

    void updateImuRollPitchYawStartSinCos(){
        cosImuRollStart = cos(imuRollStart);
        cosImuPitchStart = cos(imuPitchStart);
        cosImuYawStart = cos(imuYawStart);
        sinImuRollStart = sin(imuRollStart);
        sinImuPitchStart = sin(imuPitchStart);
        sinImuYawStart = sin(imuYawStart);
    }


    void ShiftToStartIMU(float pointTime)
    {
        imuShiftFromStartXCur = imuShiftXCur - imuShiftXStart - imuVeloXStart * pointTime;
        imuShiftFromStartYCur = imuShiftYCur - imuShiftYStart - imuVeloYStart * pointTime;
        imuShiftFromStartZCur = imuShiftZCur - imuShiftZStart - imuVeloZStart * pointTime;

        float x1 = cosImuYawStart * imuShiftFromStartXCur - sinImuYawStart * imuShiftFromStartZCur;
        float y1 = imuShiftFromStartYCur;
        float z1 = sinImuYawStart * imuShiftFromStartXCur + cosImuYawStart * imuShiftFromStartZCur;

        float x2 = x1;
        float y2 = cosImuPitchStart * y1 + sinImuPitchStart * z1;
        float z2 = -sinImuPitchStart * y1 + cosImuPitchStart * z1;

        imuShiftFromStartXCur = cosImuRollStart * x2 + sinImuRollStart * y2;
        imuShiftFromStartYCur = -sinImuRollStart * x2 + cosImuRollStart * y2;
        imuShiftFromStartZCur = z2;
    }

    //把速度转换回IMU的初始坐标系下
    void VeloToStartIMU()
    {
        imuVeloFromStartXCur = imuVeloXCur - imuVeloXStart;
        imuVeloFromStartYCur = imuVeloYCur - imuVeloYStart;
        imuVeloFromStartZCur = imuVeloZCur - imuVeloZStart;

        float x1 = cosImuYawStart * imuVeloFromStartXCur - sinImuYawStart * imuVeloFromStartZCur;
        float y1 = imuVeloFromStartYCur;
        float z1 = sinImuYawStart * imuVeloFromStartXCur + cosImuYawStart * imuVeloFromStartZCur;

        float x2 = x1;
        float y2 = cosImuPitchStart * y1 + sinImuPitchStart * z1;
        float z2 = -sinImuPitchStart * y1 + cosImuPitchStart * z1;

        imuVeloFromStartXCur = cosImuRollStart * x2 + sinImuRollStart * y2;
        imuVeloFromStartYCur = -sinImuRollStart * x2 + cosImuRollStart * y2;
        imuVeloFromStartZCur = z2;
    }
    //把位移转换回IMU的初始坐标系下
    void TransformToStartIMU(PointType *p)
    {
        float x1 = cos(imuRollCur) * p->x - sin(imuRollCur) * p->y;
        float y1 = sin(imuRollCur) * p->x + cos(imuRollCur) * p->y;
        float z1 = p->z;

        float x2 = x1;
        float y2 = cos(imuPitchCur) * y1 - sin(imuPitchCur) * z1;
        float z2 = sin(imuPitchCur) * y1 + cos(imuPitchCur) * z1;

        float x3 = cos(imuYawCur) * x2 + sin(imuYawCur) * z2;
        float y3 = y2;
        float z3 = -sin(imuYawCur) * x2 + cos(imuYawCur) * z2;

        float x4 = cosImuYawStart * x3 - sinImuYawStart * z3;
        float y4 = y3;
        float z4 = sinImuYawStart * x3 + cosImuYawStart * z3;

        float x5 = x4;
        float y5 = cosImuPitchStart * y4 + sinImuPitchStart * z4;
        float z5 = -sinImuPitchStart * y4 + cosImuPitchStart * z4;

        p->x = cosImuRollStart * x5 + sinImuRollStart * y5 + imuShiftFromStartXCur;
        p->y = -sinImuRollStart * x5 + cosImuRollStart * y5 + imuShiftFromStartYCur;
        p->z = z5 + imuShiftFromStartZCur;
    }

    //对数据进行积分,对速度，角速度，加速度进行积分，得到位移，角度和速度
    void AccumulateIMUShiftAndRotation()
    {
          //获得由IMUHandler函数得到该帧IMU数据的欧拉角和三轴角加速度
        float roll = imuRoll[imuPointerLast];
        float pitch = imuPitch[imuPointerLast];
        float yaw = imuYaw[imuPointerLast];
        float accX = imuAccX[imuPointerLast];
        float accY = imuAccY[imuPointerLast];
        float accZ = imuAccZ[imuPointerLast];

        //将当前时刻的加速度值绕交换过的ZXY固定轴（原XYZ）分别旋转(roll, pitch, yaw)角，转换得到世界
        //坐标系下的加速度值(右手法则)
        //绕z轴旋转(roll)
        float x1 = cos(roll) * accX - sin(roll) * accY;
        float y1 = sin(roll) * accX + cos(roll) * accY;
        float z1 = accZ;

        //绕x轴旋转(pitch)
        float x2 = x1;
        float y2 = cos(pitch) * y1 - sin(pitch) * z1;
        float z2 = sin(pitch) * y1 + cos(pitch) * z1;
        //绕y轴旋转(yaw)
        accX = cos(yaw) * x2 + sin(yaw) * z2;
        accY = y2;
        accZ = -sin(yaw) * x2 + cos(yaw) * z2;
        //上一个imu点
        int imuPointerBack = (imuPointerLast + imuQueLength - 1) % imuQueLength;
        //上一个点到当前点所经历的时间，即计算imu测量周期
        double timeDiff = imuTime[imuPointerLast] - imuTime[imuPointerBack];
        //要求imu的频率至少比lidar高，这样的imu信息才使用，后面校正也才有意义
        if (timeDiff < scanPeriod) {//（隐含从静止开始运动）
    //求每个imu时间点的位移与速度,两点之间视为匀加速直线运动

            imuShiftX[imuPointerLast] = imuShiftX[imuPointerBack] + imuVeloX[imuPointerBack] * timeDiff + accX * timeDiff * timeDiff / 2;
            imuShiftY[imuPointerLast] = imuShiftY[imuPointerBack] + imuVeloY[imuPointerBack] * timeDiff + accY * timeDiff * timeDiff / 2;
            imuShiftZ[imuPointerLast] = imuShiftZ[imuPointerBack] + imuVeloZ[imuPointerBack] * timeDiff + accZ * timeDiff * timeDiff / 2;

            imuVeloX[imuPointerLast] = imuVeloX[imuPointerBack] + accX * timeDiff;
            imuVeloY[imuPointerLast] = imuVeloY[imuPointerBack] + accY * timeDiff;
            imuVeloZ[imuPointerLast] = imuVeloZ[imuPointerBack] + accZ * timeDiff;

            imuAngularRotationX[imuPointerLast] = imuAngularRotationX[imuPointerBack] + imuAngularVeloX[imuPointerBack] * timeDiff;
            imuAngularRotationY[imuPointerLast] = imuAngularRotationY[imuPointerBack] + imuAngularVeloY[imuPointerBack] * timeDiff;
            imuAngularRotationZ[imuPointerLast] = imuAngularRotationZ[imuPointerBack] + imuAngularVeloZ[imuPointerBack] * timeDiff;
        }
    }

//　IMU的回调函数对数据进行了转存和积分（上一个函数）
   void imuHandler(const sensor_msgs::Imu::ConstPtr& imuIn)
    {
        double roll, pitch, yaw;//定义存储r\p\y的容器
        tf::Quaternion orientation;        // 提取PRY值、与真实的加速度值
        tf::quaternionMsgToTF(imuIn->orientation, orientation);//从IMU中提取出 表达姿态的四元数 
        tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw);//将姿态四元数转化为 RPY值         
        // 进行加速度坐标交换时将重力加速度去除，然后再进行xxx到zzz,yyy到xxx,zzz到yyy的变换。
        // 去除重力加速度的影响时，需要把重力加速度分解到三个坐标轴上，然后分别去除他们分量的影响，在去除的过程中需要注意加减号（默认右手坐标系的旋转方向来看）。

        // 原文链接：https://blog.csdn.net/wykxwyc/article/details/98317544
        //通过接收到的imuIn里面的四元素得到roll,pitch,yaw三个角,转换到ＩＭＵ坐标系下；

        float accX = imuIn->linear_acceleration.y - sin(roll) * cos(pitch) * 9.81;//获得真实的x方向的加速度
        float accY = imuIn->linear_acceleration.z - cos(roll) * cos(pitch) * 9.81;////获得真实的y方向的加速度
        float accZ = imuIn->linear_acceleration.x + sin(pitch) * 9.81;////获得真实的z方向的加速度
        // 接收到新的IMU数据,索引 imuPointerLast++ （imuQueLength是缓存队列的长度)，并更新：
        // imuTime(IMU时间队列) 
        // imuRoll、imuPitch、imuYaw(RPY队列) 
        // imuAccX、imuAccY、imuAccZ(加速度队列) 
        // imuAngularVeloX、imuAngularVeloY、imuAngularVeloZ(角速度队列)
        imuPointerLast = (imuPointerLast + 1) % imuQueLength;

        //imuTime 队列记录消息时间
        imuTime[imuPointerLast] = imuIn->header.stamp.toSec();

        // imuRoll、imuPitch、imuYaw 队列存储RPY值
        imuRoll[imuPointerLast] = roll;
        imuPitch[imuPointerLast] = pitch;
        imuYaw[imuPointerLast] = yaw;

        // 加速度
        imuAccX[imuPointerLast] = accX;
        imuAccY[imuPointerLast] = accY;
        imuAccZ[imuPointerLast] = accZ;

        //角度速度值
        imuAngularVeloX[imuPointerLast] = imuIn->angular_velocity.x;
        imuAngularVeloY[imuPointerLast] = imuIn->angular_velocity.y;
        imuAngularVeloZ[imuPointerLast] = imuIn->angular_velocity.z;
        // 积分计算得到位移
        // 取出前后两个IMU数据,如果时间上相邻(相差小于一次扫描周期)，那么计算：
        // imuShiftX、imuShiftY、imuShiftZ 里程计序列(利用速度值与加速度计算位置里程计)
        // imuVeloX、imuVeloY、imuVeloZ 速度里程计 
        // imuAngularRotationX、imuAngularRotationY、imuAngularRotationZ 角度里程计
        AccumulateIMUShiftAndRotation();
    }

    //修改点云数据的时间戳，将点云数据从ROS定义的格式转化到pcl的格式。
    //Handler: ROS提供了特殊的头类型，为tf等库设置帧ID提供了一种通用机制。
    void laserCloudHandler(const sensor_msgs::PointCloud2ConstPtr& laserCloudMsg){

        cloudHeader = laserCloudMsg->header;//记录消息的header 
        timeScanCur = cloudHeader.stamp.toSec();//记录header中的时间 
        timeNewSegmentedCloud = timeScanCur; 
        segmentedCloud->clear();
        pcl::fromROSMsg(*laserCloudMsg, *segmentedCloud);
        //接收到消息就设置为真
        newSegmentedCloud = true;
    }

    void outlierCloudHandler(const sensor_msgs::PointCloud2ConstPtr& msgIn){
        //timeNewOutlierCloud 记录header的时间
        timeNewOutlierCloud = msgIn->header.stamp.toSec();
        // outlierCloud 记录传回的点云 
        outlierCloud->clear();
        pcl::fromROSMsg(*msgIn, *outlierCloud);
        //接收到消息就设置为真
        newOutlierCloud = true;
    }

    void laserCloudInfoHandler(const cloud_msgs::cloud_infoConstPtr& msgIn)
    {
        timeNewSegmentedCloudInfo = msgIn->header.stamp.toSec(); //记录时间 
        segInfo = *msgIn;                        //segInfo 记录分割信息 
        newSegmentedCloudInfo = true;
    }

//  畸变校正:
//  这里是根据激光雷达一帧扫描期间，IMU积分得到的位姿变换，通过插值的形式，对点云畸变进行校正。
//  由于激光雷达与IMU坐标系定义不同，因此函数中先对坐标进行了交换，仅仅是为了将点云在IMU坐标系下表示。 
     // 步骤:
        //     循环点云 segmentedCloud 上的所有点
        //     1:计算每个传感器在采集每个点时的确切时间,通过确定当前点的偏航角在扫描范围内的位置实现。
        //     并更新点的 intensity 属性： intensity+(本次扫描开始到采集到该点的时间)
        //     2. 将IMU数据时间与点云数据时间对齐
        //     即确定IMU数据索引 imuPointerFront,使得该IMU采集时间与当前点的采集时间紧邻
        //     3. 通过线性插值 求解机器人取得当前点时的状态:
        //         imuRollCur,imuPitchCur,imuYawCur RPY值
        //         imuVeloXCur, imuVeloYCur, imuVeloZCur  速度值
        //         imuShiftXCur, imuShiftYCur, imuShiftZCur  位置里程计 
        //     4. 如果是一帧的第一个点,则另外记录当前点时的状态值作为当前帧的初始状态：
        //         imuRollStart, imuPitchStart, imuYawStart 起始RPY值 
        //         imuVeloXStart, imuVeloYStart, imuVeloZStart; //起始速度值
        //         imuShiftXStart, imuShiftYStart, imuShiftZStart; //起始的位置里程计
        //         imuAngularRotationXCur, imuAngularRotationYCur, imuAngularRotationZCur; // 角度里程计 
        //         cosImuRollStart, cosImuPitchStart, cosImuYawStart, sinImuRollStart, sinImuPitchStart, sinImuYawStart;//起始RPY值的正余弦
        // 并更新：
        //         imuAngularFromStartX, imuAngularFromStartY, imuAngularFromStartZ;  //   上一帧过程中 角度里程计(角度变化量)
        //         imuAngularRotationXLast, imuAngularRotationYLast, imuAngularRotationZLast; // 记录上一帧点云起始点的 角度里程计 
        //     5. 如果不是第一个点,那么计算当前扫描中,当前点的速度变化量,更新:
        //         imuVeloFromStartXCur、imuVeloFromStartYCur、imuVeloFromStartZCur 
        //     　修正因机器人的移动造成的点云失真,更新 segmentedCloud 中点的位置属性 point.x、y、z。
    void adjustDistortion()
    {
        bool halfPassed = false;
        int cloudSize = segmentedCloud->points.size();

        PointType point;

        for (int i = 0; i < cloudSize; i++) {
            point.x = segmentedCloud->points[i].y;
            point.y = segmentedCloud->points[i].z;
            point.z = segmentedCloud->points[i].x;
            // 计算点的yaw, 根据不同的偏航角，可以知道激光雷达扫过的位置有没有超过一半
            float ori = -atan2(point.x, point.z);
            if (!halfPassed) {
                if (ori < segInfo.startOrientation - M_PI / 2)
                    ori += 2 * M_PI;
                else if (ori > segInfo.startOrientation + M_PI * 3 / 2)
                    ori -= 2 * M_PI;

                if (ori - segInfo.startOrientation > M_PI)
                    halfPassed = true;
            } else {
                ori += 2 * M_PI;

                if (ori < segInfo.endOrientation - M_PI * 3 / 2)
                    ori += 2 * M_PI;
                else if (ori > segInfo.endOrientation + M_PI / 2)
                    ori -= 2 * M_PI;
            }
             float relTime = (ori - segInfo.startOrientation) / segInfo.orientationDiff;
            point.intensity = int(segmentedCloud->points[i].intensity) + scanPeriod * relTime;
            // imu与lidar时间轴对齐,这一块对IMU的处理非常巧妙，在每次处理点云时，需要遍历每一个点，把第一个点作为初始坐标，随后的点均以第一个点为参考做插值计算
            if (imuPointerLast >= 0) {
                //在这里是利用插值的方法，relTime指的是角度的变化率，乘上scanPeriod（0.1秒）便得到在一个扫描周期内的角度变化量
                float pointTime = relTime * scanPeriod;
                imuPointerFront = imuPointerLastIteration;
                //  imu数据比激光数据早,但是没有更后面的数据,不能通过插补进行优化(主要是因为imu的频率更快：100HZ)
                while (imuPointerFront != imuPointerLast) {
                    if (timeScanCur + pointTime < imuTime[imuPointerFront]) {
                        break;
                    }
                    //imuQueLength是200，imuPointerFront在0~199之间轮转
                    imuPointerFront = (imuPointerFront + 1) % imuQueLength;
                }
                // imu时间在前：要是已经是最新数据那就拿下标为imuPointerFront的数据作为Cur
                if (timeScanCur + pointTime > imuTime[imuPointerFront]) {
                    imuRollCur = imuRoll[imuPointerFront];
                    imuPitchCur = imuPitch[imuPointerFront];
                    imuYawCur = imuYaw[imuPointerFront];

                    imuVeloXCur = imuVeloX[imuPointerFront];
                    imuVeloYCur = imuVeloY[imuPointerFront];
                    imuVeloZCur = imuVeloZ[imuPointerFront];

                    imuShiftXCur = imuShiftX[imuPointerFront];
                    imuShiftYCur = imuShiftY[imuPointerFront];
                    imuShiftZCur = imuShiftZ[imuPointerFront];   
                } else {
                    //否则用插值计算当前数据［
                    // 在imu数据充足的情况下才会发生插值
                    // 当前timeScanCur + pointTime < imuTime[imuPointerFront]，
                    // imuPointerFront是最早一个时间大于timeScanCur + pointTime的imu数据指针。
                    // imuPointerBack是imuPointerFront的前一个imu数据指针
                    int imuPointerBack = (imuPointerFront + imuQueLength - 1) % imuQueLength;
                    float ratioFront = (timeScanCur + pointTime - imuTime[imuPointerBack]) 
                                                     / (imuTime[imuPointerFront] - imuTime[imuPointerBack]);
                    float ratioBack = (imuTime[imuPointerFront] - timeScanCur - pointTime) 
                                                    / (imuTime[imuPointerFront] - imuTime[imuPointerBack]);

                    // roll和pitch通常接近0，yaw变化角度较大   
                    imuRollCur = imuRoll[imuPointerFront] * ratioFront + imuRoll[imuPointerBack] * ratioBack;
                    imuPitchCur = imuPitch[imuPointerFront] * ratioFront + imuPitch[imuPointerBack] * ratioBack;
                    if (imuYaw[imuPointerFront] - imuYaw[imuPointerBack] > M_PI) {
                        imuYawCur = imuYaw[imuPointerFront] * ratioFront + (imuYaw[imuPointerBack] + 2 * M_PI) * ratioBack;
                    } else if (imuYaw[imuPointerFront] - imuYaw[imuPointerBack] < -M_PI) {
                        imuYawCur = imuYaw[imuPointerFront] * ratioFront + (imuYaw[imuPointerBack] - 2 * M_PI) * ratioBack;
                    } else {
                        imuYawCur = imuYaw[imuPointerFront] * ratioFront + imuYaw[imuPointerBack] * ratioBack;
                    }
                    // 速度与位置进行插值
                    imuVeloXCur = imuVeloX[imuPointerFront] * ratioFront + imuVeloX[imuPointerBack] * ratioBack;
                    imuVeloYCur = imuVeloY[imuPointerFront] * ratioFront + imuVeloY[imuPointerBack] * ratioBack;
                    imuVeloZCur = imuVeloZ[imuPointerFront] * ratioFront + imuVeloZ[imuPointerBack] * ratioBack;

                    imuShiftXCur = imuShiftX[imuPointerFront] * ratioFront + imuShiftX[imuPointerBack] * ratioBack;
                    imuShiftYCur = imuShiftY[imuPointerFront] * ratioFront + imuShiftY[imuPointerBack] * ratioBack;
                    imuShiftZCur = imuShiftZ[imuPointerFront] * ratioFront + imuShiftZ[imuPointerBack] * ratioBack;
                }
                
                // i = 0 时刻   如果是在点云中的第一个点，就将数据作为初始坐标系
                if (i == 0) {
                    imuRollStart = imuRollCur;
                    imuPitchStart = imuPitchCur;
                    imuYawStart = imuYawCur;

                    imuVeloXStart = imuVeloXCur;
                    imuVeloYStart = imuVeloYCur;
                    imuVeloZStart = imuVeloZCur;

                    imuShiftXStart = imuShiftXCur;
                    imuShiftYStart = imuShiftYCur;
                    imuShiftZStart = imuShiftZCur;

                    if (timeScanCur + pointTime > imuTime[imuPointerFront]) {
                        imuAngularRotationXCur = imuAngularRotationX[imuPointerFront];
                        imuAngularRotationYCur = imuAngularRotationY[imuPointerFront];
                        imuAngularRotationZCur = imuAngularRotationZ[imuPointerFront];
                    }else{
                        int imuPointerBack = (imuPointerFront + imuQueLength - 1) % imuQueLength;
                        float ratioFront = (timeScanCur + pointTime - imuTime[imuPointerBack]) 
                                                         / (imuTime[imuPointerFront] - imuTime[imuPointerBack]);
                        float ratioBack = (imuTime[imuPointerFront] - timeScanCur - pointTime) 
                                                        / (imuTime[imuPointerFront] - imuTime[imuPointerBack]);
                        imuAngularRotationXCur = imuAngularRotationX[imuPointerFront] * ratioFront + imuAngularRotationX[imuPointerBack] * ratioBack;
                        imuAngularRotationYCur = imuAngularRotationY[imuPointerFront] * ratioFront + imuAngularRotationY[imuPointerBack] * ratioBack;
                        imuAngularRotationZCur = imuAngularRotationZ[imuPointerFront] * ratioFront + imuAngularRotationZ[imuPointerBack] * ratioBack;
                    }

                    imuAngularFromStartX = imuAngularRotationXCur - imuAngularRotationXLast;
                    imuAngularFromStartY = imuAngularRotationYCur - imuAngularRotationYLast;
                    imuAngularFromStartZ = imuAngularRotationZCur - imuAngularRotationZLast;

                    imuAngularRotationXLast = imuAngularRotationXCur;
                    imuAngularRotationYLast = imuAngularRotationYCur;
                    imuAngularRotationZLast = imuAngularRotationZCur;

                    updateImuRollPitchYawStartSinCos();
                } else {
                    //如果不是第一个点    把速度与位移转换回IMU的初始坐标系下
                    VeloToStartIMU();
                    TransformToStartIMU(&point);
                }
            }
            segmentedCloud->points[i] = point;
        }

        imuPointerLastIteration = imuPointerLast;
    }


    // calculateSmoothness：根据两侧的10个点，计算曲率，这里的曲率只是一个量的大小的概念。
    // 因为曲率并不参加最终的优化，只是衡量一个点光滑与否的标志，是个相对的概念
    // 将周围十个点累加，并未开方求欧氏距离。
    // 最后在cloudSmoothness这个vector中保存了曲率与点索引对，以便后面根据曲率对点进行排序，并根据索引对直接得到点的索引顺序。
    // 赋值粗糙度序列 cloudCurvature 
    // 赋值粗糙度信息序列 cloudSmoothness,其中每一个元素<value,ind>记录某点的粗糙度以及其索引
    void calculateSmoothness()
    {
        int cloudSize = segmentedCloud->points.size();
        for (int i = 5; i < cloudSize - 5; i++) {
            //十个数的和
            float diffRange = segInfo.segmentedCloudRange[i-5] + segInfo.segmentedCloudRange[i-4]
                            + segInfo.segmentedCloudRange[i-3] + segInfo.segmentedCloudRange[i-2]
                            + segInfo.segmentedCloudRange[i-1] - segInfo.segmentedCloudRange[i] * 10
                            + segInfo.segmentedCloudRange[i+1] + segInfo.segmentedCloudRange[i+2]
                            + segInfo.segmentedCloudRange[i+3] + segInfo.segmentedCloudRange[i+4]
                            + segInfo.segmentedCloudRange[i+5];            

            cloudCurvature[i] = diffRange*diffRange;

            cloudNeighborPicked[i] = 0;
            cloudLabel[i] = 0;

            cloudSmoothness[i].value = cloudCurvature[i];
            cloudSmoothness[i].ind = i;
        }
    }

//　遮挡和噪声点标记
// 　对于深度变化比较明显的相邻两个点，远处的那一个更容易被遮挡 ,不稳定；如果某个点，和两侧的点距离差距都比较大，那么有可能是噪声等，也不稳定。
// 　在cloudNeighborPicked中标注为1,后面的特征提取中不会再考虑。
    void markOccludedPoints()
    {
        int cloudSize = segmentedCloud->points.size();
        for (int i = 5; i < cloudSize - 6; ++i){
             //depth1和depth2分别是两点的深度
            float depth1 = segInfo.segmentedCloudRange[i];
            float depth2 = segInfo.segmentedCloudRange[i+1];
            int columnDiff = std::abs(int(segInfo.segmentedCloudColInd[i+1] - segInfo.segmentedCloudColInd[i]));
            //直接采取比较点的下标，选择距离较远的那些点，并将他们标记为1,去掉其中一侧的6个点
            if (columnDiff < 10){
                if (depth1 - depth2 > 0.3){
                    cloudNeighborPicked[i - 5] = 1;
                    cloudNeighborPicked[i - 4] = 1;
                    cloudNeighborPicked[i - 3] = 1;
                    cloudNeighborPicked[i - 2] = 1;
                    cloudNeighborPicked[i - 1] = 1;
                    cloudNeighborPicked[i] = 1;
                }else if (depth2 - depth1 > 0.3){
                    cloudNeighborPicked[i + 1] = 1;
                    cloudNeighborPicked[i + 2] = 1;
                    cloudNeighborPicked[i + 3] = 1;
                    cloudNeighborPicked[i + 4] = 1;
                    cloudNeighborPicked[i + 5] = 1;
                    cloudNeighborPicked[i + 6] = 1;
                }
            }
             //diff1和diff2是当前点距离前后两个点的距离
            float diff1 = std::abs(float(segInfo.segmentedCloudRange[i-1] - segInfo.segmentedCloudRange[i]));
            float diff2 = std::abs(float(segInfo.segmentedCloudRange[i+1] - segInfo.segmentedCloudRange[i]));
           //如果当前点距离左右邻点都过远，则视其为瑕点，因为入射角可能太小导致误差较大,并将距离变化较大的点标记为1
            if (diff1 > 0.02 * segInfo.segmentedCloudRange[i] && diff2 > 0.02 * segInfo.segmentedCloudRange[i])
                cloudNeighborPicked[i] = 1;
        }
    }
//根据曲率的大小，提取四种特征点
// 将点分为多种类型，把每个线分成了6个区，在每个区内提取一定数量的角点和面点，这是为了让点分布更均匀，匹配时效果会更稳定更好。
// 然后就是在一个区的范围内，按照曲率对索引进行排序，曲率最大的20个非地面点为角点，其中最最大的2个强角点，两者是包含的关系；
// 曲率最最小的4个地面点为强面点，其余所有地面点都是一般面点，点数太多需要进行降采样，一般是五个点取一次。
// 在确定提取某个点后，其周围一定范围内的点cloudNeighborPicked会被标注为1,认为该区域的点已经被选择，不会再次被选中，这是为了防止提取出来的特征点过度聚集在某一处。
        /* 将每一条线16等分,并在每一等分上，对所有点的粗糙度排序，遴选出 
        // cornerPointsSharp        强角点
        // cornerPointsLessSharp    普通角点
        // surfPointsFlat           平面点
        // surfPointsLessFlat       除了角点的所有点(经过了降采样,且这个不属于特征点)
        // 每次选出一个特征点 ，更新该点和附近点的择选标志位 cloudNeighborPicked ,防止特征点扎堆。
        // 并更新每个点的特征标志位 cloudLabel(2:强角点;1普通角点;-1 平面点,0 其他)*/

    void extractFeatures()
    {
        cornerPointsSharp->clear();
        cornerPointsLessSharp->clear();
        surfPointsFlat->clear();
        surfPointsLessFlat->clear();
        //  提取4种特征点:根据label值，分类为sharp，lesssharp，flat，lessflat
        for (int i = 0; i < N_SCAN; i++) {
            surfPointsLessFlatScan->clear();
            for (int j = 0; j < 6; j++) {
                // sp和ep分别是这段点云的起始点与终止点
                int sp = (segInfo.startRingIndex[i] * (6 - j)    + segInfo.endRingIndex[i] * j) / 6;
                int ep = (segInfo.startRingIndex[i] * (5 - j)    + segInfo.endRingIndex[i] * (j + 1)) / 6 - 1;
                if (sp >= ep)
                    continue;
                std::sort(cloudSmoothness.begin()+sp, cloudSmoothness.begin()+ep, by_value());
                int largestPickedNum = 0;
                for (int k = ep; k >= sp; k--) {
                    int ind = cloudSmoothness[k].ind;
                    //曲率大于edgeThreshold（0.1）则视为角点，而且不能是地面点
                    if (cloudNeighborPicked[ind] == 0 && cloudCurvature[ind] > edgeThreshold && segInfo.segmentedCloudGroundFlag[ind] == false) {                    
                        largestPickedNum++;
                        if (largestPickedNum <= 2) {
                            cloudLabel[ind] = 2;
                            cornerPointsSharp->push_back(segmentedCloud->points[ind]);
                            cornerPointsLessSharp->push_back(segmentedCloud->points[ind]);
                        } else if (largestPickedNum <= 20) {
                            cloudLabel[ind] = 1;
                            cornerPointsLessSharp->push_back(segmentedCloud->points[ind]);
                        } else {
                            break;
                        }
                        //防止重复处理
                        cloudNeighborPicked[ind] = 1;
                        for (int l = 1; l <= 5; l++) {
                            int columnDiff = std::abs(int(segInfo.segmentedCloudColInd[ind + l] - segInfo.segmentedCloudColInd[ind + l - 1]));
                            if (columnDiff > 10)
                                break;
                            cloudNeighborPicked[ind + l] = 1;
                        }
                        for (int l = -1; l >= -5; l--) {
                            int columnDiff = std::abs(int(segInfo.segmentedCloudColInd[ind + l] - segInfo.segmentedCloudColInd[ind + l + 1]));
                            if (columnDiff > 10)
                                break;
                            cloudNeighborPicked[ind + l] = 1;
                        }
                    }
                }
                // 平面点只从地面点中进行选择
                int smallestPickedNum = 0;
                for (int k = sp; k <= ep; k++) {
                    int ind = cloudSmoothness[k].ind;
                    if (cloudNeighborPicked[ind] == 0 && cloudCurvature[ind] < surfThreshold && segInfo.segmentedCloudGroundFlag[ind] == true) {
                        cloudLabel[ind] = -1;
                        surfPointsFlat->push_back(segmentedCloud->points[ind]);
                        smallestPickedNum++;
                        if (smallestPickedNum >= 4) {
                            break;
                        }

                        cloudNeighborPicked[ind] = 1;
                        for (int l = 1; l <= 5; l++) {
                            int columnDiff = std::abs(int(segInfo.segmentedCloudColInd[ind + l] - segInfo.segmentedCloudColInd[ind + l - 1]));
                            if (columnDiff > 10)
                                break;
                            cloudNeighborPicked[ind + l] = 1;
                        }
                        for (int l = -1; l >= -5; l--) {
                            int columnDiff = std::abs(int(segInfo.segmentedCloudColInd[ind + l] - segInfo.segmentedCloudColInd[ind + l + 1]));
                            if (columnDiff > 10)
                                break;

                            cloudNeighborPicked[ind + l] = 1;
                        }
                    }
                }

                for (int k = sp; k <= ep; k++) {
                    if (cloudLabel[k] <= 0) {
                        surfPointsLessFlatScan->push_back(segmentedCloud->points[k]);
                    }
                }
            }
            // surfPointsLessFlatScan中有过多的点云，如果点云太多，计算量太大,就进行降采样
            surfPointsLessFlatScanDS->clear();
            downSizeFilter.setInputCloud(surfPointsLessFlatScan);
            downSizeFilter.filter(*surfPointsLessFlatScanDS);

            *surfPointsLessFlat += *surfPointsLessFlatScanDS;
        }
    }
    //publishCloud：发布4种特征点，。用来记录在一帧激光雷达数据下提取的特征点。
    void publishCloud()
    {
        sensor_msgs::PointCloud2 laserCloudOutMsg;

	    if (pubCornerPointsSharp.getNumSubscribers() != 0){
	        pcl::toROSMsg(*cornerPointsSharp, laserCloudOutMsg);
	        laserCloudOutMsg.header.stamp = cloudHeader.stamp;
	        laserCloudOutMsg.header.frame_id = "/camera";
	        pubCornerPointsSharp.publish(laserCloudOutMsg);
	    }

	    if (pubCornerPointsLessSharp.getNumSubscribers() != 0){
	        pcl::toROSMsg(*cornerPointsLessSharp, laserCloudOutMsg);
	        laserCloudOutMsg.header.stamp = cloudHeader.stamp;
	        laserCloudOutMsg.header.frame_id = "/camera";
	        pubCornerPointsLessSharp.publish(laserCloudOutMsg);
	    }

	    if (pubSurfPointsFlat.getNumSubscribers() != 0){
	        pcl::toROSMsg(*surfPointsFlat, laserCloudOutMsg);
	        laserCloudOutMsg.header.stamp = cloudHeader.stamp;
	        laserCloudOutMsg.header.frame_id = "/camera";
	        pubSurfPointsFlat.publish(laserCloudOutMsg);
	    }

	    if (pubSurfPointsLessFlat.getNumSubscribers() != 0){
	        pcl::toROSMsg(*surfPointsLessFlat, laserCloudOutMsg);
	        laserCloudOutMsg.header.stamp = cloudHeader.stamp;
	        laserCloudOutMsg.header.frame_id = "/camera";
	        pubSurfPointsLessFlat.publish(laserCloudOutMsg);
	    }
    }

    void TransformToStart(PointType const * const pi, PointType * const po)
    {
        float s = 10 * (pi->intensity - int(pi->intensity));

        float rx = s * transformCur[0];
        float ry = s * transformCur[1];
        float rz = s * transformCur[2];
        float tx = s * transformCur[3];
        float ty = s * transformCur[4];
        float tz = s * transformCur[5];

        float x1 = cos(rz) * (pi->x - tx) + sin(rz) * (pi->y - ty);
        float y1 = -sin(rz) * (pi->x - tx) + cos(rz) * (pi->y - ty);
        float z1 = (pi->z - tz);

        float x2 = x1;
        float y2 = cos(rx) * y1 + sin(rx) * z1;
        float z2 = -sin(rx) * y1 + cos(rx) * z1;

        po->x = cos(ry) * x2 - sin(ry) * z2;
        po->y = y2;
        po->z = sin(ry) * x2 + cos(ry) * z2;
        po->intensity = pi->intensity;
    }

    void TransformToEnd(PointType const * const pi, PointType * const po)
    {
        float s = 10 * (pi->intensity - int(pi->intensity));

        float rx = s * transformCur[0];
        float ry = s * transformCur[1];
        float rz = s * transformCur[2];
        float tx = s * transformCur[3];
        float ty = s * transformCur[4];
        float tz = s * transformCur[5];

        float x1 = cos(rz) * (pi->x - tx) + sin(rz) * (pi->y - ty);
        float y1 = -sin(rz) * (pi->x - tx) + cos(rz) * (pi->y - ty);
        float z1 = (pi->z - tz);

        float x2 = x1;
        float y2 = cos(rx) * y1 + sin(rx) * z1;
        float z2 = -sin(rx) * y1 + cos(rx) * z1;

        float x3 = cos(ry) * x2 - sin(ry) * z2;
        float y3 = y2;
        float z3 = sin(ry) * x2 + cos(ry) * z2;

        rx = transformCur[0];
        ry = transformCur[1];
        rz = transformCur[2];
        tx = transformCur[3];
        ty = transformCur[4];
        tz = transformCur[5];

        float x4 = cos(ry) * x3 + sin(ry) * z3;
        float y4 = y3;
        float z4 = -sin(ry) * x3 + cos(ry) * z3;

        float x5 = x4;
        float y5 = cos(rx) * y4 - sin(rx) * z4;
        float z5 = sin(rx) * y4 + cos(rx) * z4;

        float x6 = cos(rz) * x5 - sin(rz) * y5 + tx;
        float y6 = sin(rz) * x5 + cos(rz) * y5 + ty;
        float z6 = z5 + tz;

        float x7 = cosImuRollStart * (x6 - imuShiftFromStartX) 
                 - sinImuRollStart * (y6 - imuShiftFromStartY);
        float y7 = sinImuRollStart * (x6 - imuShiftFromStartX) 
                 + cosImuRollStart * (y6 - imuShiftFromStartY);
        float z7 = z6 - imuShiftFromStartZ;

        float x8 = x7;
        float y8 = cosImuPitchStart * y7 - sinImuPitchStart * z7;
        float z8 = sinImuPitchStart * y7 + cosImuPitchStart * z7;

        float x9 = cosImuYawStart * x8 + sinImuYawStart * z8;
        float y9 = y8;
        float z9 = -sinImuYawStart * x8 + cosImuYawStart * z8;

        float x10 = cos(imuYawLast) * x9 - sin(imuYawLast) * z9;
        float y10 = y9;
        float z10 = sin(imuYawLast) * x9 + cos(imuYawLast) * z9;

        float x11 = x10;
        float y11 = cos(imuPitchLast) * y10 + sin(imuPitchLast) * z10;
        float z11 = -sin(imuPitchLast) * y10 + cos(imuPitchLast) * z10;

        po->x = cos(imuRollLast) * x11 + sin(imuRollLast) * y11;
        po->y = -sin(imuRollLast) * x11 + cos(imuRollLast) * y11;
        po->z = z11;
        po->intensity = int(pi->intensity);
    }

    void PluginIMURotation(float bcx, float bcy, float bcz, float blx, float bly, float blz, 
                           float alx, float aly, float alz, float &acx, float &acy, float &acz)
    {
        float sbcx = sin(bcx);
        float cbcx = cos(bcx);
        float sbcy = sin(bcy);
        float cbcy = cos(bcy);
        float sbcz = sin(bcz);
        float cbcz = cos(bcz);

        float sblx = sin(blx);
        float cblx = cos(blx);
        float sbly = sin(bly);
        float cbly = cos(bly);
        float sblz = sin(blz);
        float cblz = cos(blz);

        float salx = sin(alx);
        float calx = cos(alx);
        float saly = sin(aly);
        float caly = cos(aly);
        float salz = sin(alz);
        float calz = cos(alz);

        float srx = -sbcx*(salx*sblx + calx*caly*cblx*cbly + calx*cblx*saly*sbly) 
                  - cbcx*cbcz*(calx*saly*(cbly*sblz - cblz*sblx*sbly) 
                  - calx*caly*(sbly*sblz + cbly*cblz*sblx) + cblx*cblz*salx) 
                  - cbcx*sbcz*(calx*caly*(cblz*sbly - cbly*sblx*sblz) 
                  - calx*saly*(cbly*cblz + sblx*sbly*sblz) + cblx*salx*sblz);
        acx = -asin(srx);

        float srycrx = (cbcy*sbcz - cbcz*sbcx*sbcy)*(calx*saly*(cbly*sblz - cblz*sblx*sbly) 
                     - calx*caly*(sbly*sblz + cbly*cblz*sblx) + cblx*cblz*salx) 
                     - (cbcy*cbcz + sbcx*sbcy*sbcz)*(calx*caly*(cblz*sbly - cbly*sblx*sblz) 
                     - calx*saly*(cbly*cblz + sblx*sbly*sblz) + cblx*salx*sblz) 
                     + cbcx*sbcy*(salx*sblx + calx*caly*cblx*cbly + calx*cblx*saly*sbly);
        float crycrx = (cbcz*sbcy - cbcy*sbcx*sbcz)*(calx*caly*(cblz*sbly - cbly*sblx*sblz) 
                     - calx*saly*(cbly*cblz + sblx*sbly*sblz) + cblx*salx*sblz) 
                     - (sbcy*sbcz + cbcy*cbcz*sbcx)*(calx*saly*(cbly*sblz - cblz*sblx*sbly) 
                     - calx*caly*(sbly*sblz + cbly*cblz*sblx) + cblx*cblz*salx) 
                     + cbcx*cbcy*(salx*sblx + calx*caly*cblx*cbly + calx*cblx*saly*sbly);
        acy = atan2(srycrx / cos(acx), crycrx / cos(acx));
        
        float srzcrx = sbcx*(cblx*cbly*(calz*saly - caly*salx*salz) 
                     - cblx*sbly*(caly*calz + salx*saly*salz) + calx*salz*sblx) 
                     - cbcx*cbcz*((caly*calz + salx*saly*salz)*(cbly*sblz - cblz*sblx*sbly) 
                     + (calz*saly - caly*salx*salz)*(sbly*sblz + cbly*cblz*sblx) 
                     - calx*cblx*cblz*salz) + cbcx*sbcz*((caly*calz + salx*saly*salz)*(cbly*cblz 
                     + sblx*sbly*sblz) + (calz*saly - caly*salx*salz)*(cblz*sbly - cbly*sblx*sblz) 
                     + calx*cblx*salz*sblz);
        float crzcrx = sbcx*(cblx*sbly*(caly*salz - calz*salx*saly) 
                     - cblx*cbly*(saly*salz + caly*calz*salx) + calx*calz*sblx) 
                     + cbcx*cbcz*((saly*salz + caly*calz*salx)*(sbly*sblz + cbly*cblz*sblx) 
                     + (caly*salz - calz*salx*saly)*(cbly*sblz - cblz*sblx*sbly) 
                     + calx*calz*cblx*cblz) - cbcx*sbcz*((saly*salz + caly*calz*salx)*(cblz*sbly 
                     - cbly*sblx*sblz) + (caly*salz - calz*salx*saly)*(cbly*cblz + sblx*sbly*sblz) 
                     - calx*calz*cblx*sblz);
        acz = atan2(srzcrx / cos(acx), crzcrx / cos(acx));
    }

    void AccumulateRotation(float cx, float cy, float cz, float lx, float ly, float lz, 
                            float &ox, float &oy, float &oz)
    {
        float srx = cos(lx)*cos(cx)*sin(ly)*sin(cz) - cos(cx)*cos(cz)*sin(lx) - cos(lx)*cos(ly)*sin(cx);
        ox = -asin(srx);

        float srycrx = sin(lx)*(cos(cy)*sin(cz) - cos(cz)*sin(cx)*sin(cy)) + cos(lx)*sin(ly)*(cos(cy)*cos(cz) 
                     + sin(cx)*sin(cy)*sin(cz)) + cos(lx)*cos(ly)*cos(cx)*sin(cy);
        float crycrx = cos(lx)*cos(ly)*cos(cx)*cos(cy) - cos(lx)*sin(ly)*(cos(cz)*sin(cy) 
                     - cos(cy)*sin(cx)*sin(cz)) - sin(lx)*(sin(cy)*sin(cz) + cos(cy)*cos(cz)*sin(cx));
        oy = atan2(srycrx / cos(ox), crycrx / cos(ox));

        float srzcrx = sin(cx)*(cos(lz)*sin(ly) - cos(ly)*sin(lx)*sin(lz)) + cos(cx)*sin(cz)*(cos(ly)*cos(lz) 
                     + sin(lx)*sin(ly)*sin(lz)) + cos(lx)*cos(cx)*cos(cz)*sin(lz);
        float crzcrx = cos(lx)*cos(lz)*cos(cx)*cos(cz) - cos(cx)*sin(cz)*(cos(ly)*sin(lz) 
                     - cos(lz)*sin(lx)*sin(ly)) - sin(cx)*(sin(ly)*sin(lz) + cos(ly)*cos(lz)*sin(lx));
        oz = atan2(srzcrx / cos(ox), crzcrx / cos(ox));
    }

    double rad2deg(double radians)
    {
        return radians * 180.0 / M_PI;
    }

    double deg2rad(double degrees)
    {
        return degrees * M_PI / 180.0;
    }
    //分别对平面特征与角点特征进行了匹配计算，首先找到各自对应的特征，下面是角部特征的函数：
    void findCorrespondingCornerFeatures(int iterCount){
        int cornerPointsSharpNum = cornerPointsSharp->points.size();
        //对角部特征点依次进行处理
        for (int i = 0; i < cornerPointsSharpNum; i++) {
            //根据IMU数据的将点云转换到上一次扫描的位置中
            TransformToStart(&cornerPointsSharp->points[i], &pointSel);
            //每五次迭代寻找一次邻域点，否则使用上次的邻域查找
            if (iterCount % 5 == 0) {
                kdtreeCornerLast->nearestKSearch(pointSel, 1, pointSearchInd, pointSearchSqDis);
                int closestPointInd = -1, minPointInd2 = -1;           
                //pointSearchSqDis是平方距离，阈值是25     
                if (pointSearchSqDis[0] < nearestFeatureSearchSqDist) {
                    closestPointInd = pointSearchInd[0];
                    //得到最近邻所在的层数
                    int closestPointScan = int(laserCloudCornerLast->points[closestPointInd].intensity);
                    float pointSqDis, minPointSqDis2 = nearestFeatureSearchSqDist;
                    for (int j = closestPointInd + 1; j < cornerPointsSharpNum; j++)
                    {
                        //最近邻需要在上下两层之内否则失败
                        if (int(laserCloudCornerLast->points[j].intensity) > closestPointScan + 2.5) {
                            break;
                        }
                        pointSqDis = (laserCloudCornerLast->points[j].x - pointSel.x) * 
                                     (laserCloudCornerLast->points[j].x - pointSel.x) + 
                                     (laserCloudCornerLast->points[j].y - pointSel.y) * 
                                     (laserCloudCornerLast->points[j].y - pointSel.y) + 
                                     (laserCloudCornerLast->points[j].z - pointSel.z) * 
                                     (laserCloudCornerLast->points[j].z - pointSel.z);
                        if (int(laserCloudCornerLast->points[j].intensity) > closestPointScan) {
                            if (pointSqDis < minPointSqDis2) {
                                minPointSqDis2 = pointSqDis;
                                minPointInd2 = j;
                            }
                        }
                    }
                    for (int j = closestPointInd - 1; j >= 0; j--) {
                        if (int(laserCloudCornerLast->points[j].intensity) < closestPointScan - 2.5) {
                            break;
                        }
                        pointSqDis = (laserCloudCornerLast->points[j].x - pointSel.x) * 
                                     (laserCloudCornerLast->points[j].x - pointSel.x) + 
                                     (laserCloudCornerLast->points[j].y - pointSel.y) * 
                                     (laserCloudCornerLast->points[j].y - pointSel.y) + 
                                     (laserCloudCornerLast->points[j].z - pointSel.z) * 
                                     (laserCloudCornerLast->points[j].z - pointSel.z);
                        if (int(laserCloudCornerLast->points[j].intensity) < closestPointScan) {
                            if (pointSqDis < minPointSqDis2) {
                                minPointSqDis2 = pointSqDis;
                                minPointInd2 = j;
                            }
                        }
                    }
                }
                pointSearchCornerInd1[i] = closestPointInd;
                pointSearchCornerInd2[i] = minPointInd2;
            }
             //这里是开始计算点到直线的距离，tripod即三角形，根据三角形余弦定理计算距离并求偏导数
            if (pointSearchCornerInd2[i] >= 0) {
                tripod1 = laserCloudCornerLast->points[pointSearchCornerInd1[i]];
                tripod2 = laserCloudCornerLast->points[pointSearchCornerInd2[i]];
                float x0 = pointSel.x;
                float y0 = pointSel.y;
                float z0 = pointSel.z;
                float x1 = tripod1.x;
                float y1 = tripod1.y;
                float z1 = tripod1.z;
                float x2 = tripod2.x;
                float y2 = tripod2.y;
                float z2 = tripod2.z;
                float m11 = ((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1));
                float m22 = ((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1));
                float m33 = ((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1));
                float a012 = sqrt(m11 * m11  + m22 * m22 + m33 * m33);
                float l12 = sqrt((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2) + (z1 - z2)*(z1 - z2));
                float la =  ((y1 - y2)*m11 + (z1 - z2)*m22) / a012 / l12;
                float lb = -((x1 - x2)*m11 - (z1 - z2)*m33) / a012 / l12;
                float lc = -((x1 - x2)*m22 + (y1 - y2)*m33) / a012 / l12;
                float ld2 = a012 / l12;
                float s = 1;
                if (iterCount >= 5) {
                    s = 1 - 1.8 * fabs(ld2);
                }
                if (s > 0.1 && ld2 != 0) {
                    coeff.x = s * la; 
                    coeff.y = s * lb;
                    coeff.z = s * lc;
                    coeff.intensity = s * ld2;                  
                    laserCloudOri->push_back(cornerPointsSharp->points[i]);
                    coeffSel->push_back(coeff);
                }
            }
        }
    }

    //平面特征匹配
    void findCorrespondingSurfFeatures(int iterCount){

        int surfPointsFlatNum = surfPointsFlat->points.size();

        for (int i = 0; i < surfPointsFlatNum; i++) {

            TransformToStart(&surfPointsFlat->points[i], &pointSel);

            if (iterCount % 5 == 0) {

                kdtreeSurfLast->nearestKSearch(pointSel, 1, pointSearchInd, pointSearchSqDis);
                int closestPointInd = -1, minPointInd2 = -1, minPointInd3 = -1;

                if (pointSearchSqDis[0] < nearestFeatureSearchSqDist) {
                    closestPointInd = pointSearchInd[0];
                    int closestPointScan = int(laserCloudSurfLast->points[closestPointInd].intensity);

                    float pointSqDis, minPointSqDis2 = nearestFeatureSearchSqDist, minPointSqDis3 = nearestFeatureSearchSqDist;
                    for (int j = closestPointInd + 1; j < surfPointsFlatNum; j++) {
                        if (int(laserCloudSurfLast->points[j].intensity) > closestPointScan + 2.5) {
                            break;
                        }

                        pointSqDis = (laserCloudSurfLast->points[j].x - pointSel.x) * 
                                     (laserCloudSurfLast->points[j].x - pointSel.x) + 
                                     (laserCloudSurfLast->points[j].y - pointSel.y) * 
                                     (laserCloudSurfLast->points[j].y - pointSel.y) + 
                                     (laserCloudSurfLast->points[j].z - pointSel.z) * 
                                     (laserCloudSurfLast->points[j].z - pointSel.z);

                        if (int(laserCloudSurfLast->points[j].intensity) <= closestPointScan) {
                            if (pointSqDis < minPointSqDis2) {
                              minPointSqDis2 = pointSqDis;
                              minPointInd2 = j;
                            }
                        } else {
                            if (pointSqDis < minPointSqDis3) {
                                minPointSqDis3 = pointSqDis;
                                minPointInd3 = j;
                            }
                        }
                    }
                    for (int j = closestPointInd - 1; j >= 0; j--) {
                        if (int(laserCloudSurfLast->points[j].intensity) < closestPointScan - 2.5) {
                            break;
                        }

                        pointSqDis = (laserCloudSurfLast->points[j].x - pointSel.x) * 
                                     (laserCloudSurfLast->points[j].x - pointSel.x) + 
                                     (laserCloudSurfLast->points[j].y - pointSel.y) * 
                                     (laserCloudSurfLast->points[j].y - pointSel.y) + 
                                     (laserCloudSurfLast->points[j].z - pointSel.z) * 
                                     (laserCloudSurfLast->points[j].z - pointSel.z);

                        if (int(laserCloudSurfLast->points[j].intensity) >= closestPointScan) {
                            if (pointSqDis < minPointSqDis2) {
                                minPointSqDis2 = pointSqDis;
                                minPointInd2 = j;
                            }
                        } else {
                            if (pointSqDis < minPointSqDis3) {
                                minPointSqDis3 = pointSqDis;
                                minPointInd3 = j;
                            }
                        }
                    }
                }

                pointSearchSurfInd1[i] = closestPointInd;
                pointSearchSurfInd2[i] = minPointInd2;
                pointSearchSurfInd3[i] = minPointInd3;
            }

            if (pointSearchSurfInd2[i] >= 0 && pointSearchSurfInd3[i] >= 0) {

                tripod1 = laserCloudSurfLast->points[pointSearchSurfInd1[i]];
                tripod2 = laserCloudSurfLast->points[pointSearchSurfInd2[i]];
                tripod3 = laserCloudSurfLast->points[pointSearchSurfInd3[i]];

                float pa = (tripod2.y - tripod1.y) * (tripod3.z - tripod1.z) 
                         - (tripod3.y - tripod1.y) * (tripod2.z - tripod1.z);
                float pb = (tripod2.z - tripod1.z) * (tripod3.x - tripod1.x) 
                         - (tripod3.z - tripod1.z) * (tripod2.x - tripod1.x);
                float pc = (tripod2.x - tripod1.x) * (tripod3.y - tripod1.y) 
                         - (tripod3.x - tripod1.x) * (tripod2.y - tripod1.y);
                float pd = -(pa * tripod1.x + pb * tripod1.y + pc * tripod1.z);

                float ps = sqrt(pa * pa + pb * pb + pc * pc);

                pa /= ps;
                pb /= ps;
                pc /= ps;
                pd /= ps;

                float pd2 = pa * pointSel.x + pb * pointSel.y + pc * pointSel.z + pd;

                float s = 1;
                if (iterCount >= 5) {
                    s = 1 - 1.8 * fabs(pd2) / sqrt(sqrt(pointSel.x * pointSel.x
                            + pointSel.y * pointSel.y + pointSel.z * pointSel.z));
                }

                if (s > 0.1 && pd2 != 0) {
                    coeff.x = s * pa;
                    coeff.y = s * pb;
                    coeff.z = s * pc;
                    coeff.intensity = s * pd2;

                    laserCloudOri->push_back(surfPointsFlat->points[i]);
                    coeffSel->push_back(coeff);
                }
            }
        }
    }

    bool calculateTransformationSurf(int iterCount){

        int pointSelNum = laserCloudOri->points.size();

        cv::Mat matA(pointSelNum, 3, CV_32F, cv::Scalar::all(0));
        cv::Mat matAt(3, pointSelNum, CV_32F, cv::Scalar::all(0));
        cv::Mat matAtA(3, 3, CV_32F, cv::Scalar::all(0));
        cv::Mat matB(pointSelNum, 1, CV_32F, cv::Scalar::all(0));
        cv::Mat matAtB(3, 1, CV_32F, cv::Scalar::all(0));
        cv::Mat matX(3, 1, CV_32F, cv::Scalar::all(0));

        float srx = sin(transformCur[0]);
        float crx = cos(transformCur[0]);
        float sry = sin(transformCur[1]);
        float cry = cos(transformCur[1]);
        float srz = sin(transformCur[2]);
        float crz = cos(transformCur[2]);
        float tx = transformCur[3];
        float ty = transformCur[4];
        float tz = transformCur[5];

        float a1 = crx*sry*srz; float a2 = crx*crz*sry; float a3 = srx*sry; float a4 = tx*a1 - ty*a2 - tz*a3;
        float a5 = srx*srz; float a6 = crz*srx; float a7 = ty*a6 - tz*crx - tx*a5;
        float a8 = crx*cry*srz; float a9 = crx*cry*crz; float a10 = cry*srx; float a11 = tz*a10 + ty*a9 - tx*a8;

        float b1 = -crz*sry - cry*srx*srz; float b2 = cry*crz*srx - sry*srz;
        float b5 = cry*crz - srx*sry*srz; float b6 = cry*srz + crz*srx*sry;

        float c1 = -b6; float c2 = b5; float c3 = tx*b6 - ty*b5; float c4 = -crx*crz; float c5 = crx*srz; float c6 = ty*c5 + tx*-c4;
        float c7 = b2; float c8 = -b1; float c9 = tx*-b2 - ty*-b1;

        for (int i = 0; i < pointSelNum; i++) {

            pointOri = laserCloudOri->points[i];
            coeff = coeffSel->points[i];

            float arx = (-a1*pointOri.x + a2*pointOri.y + a3*pointOri.z + a4) * coeff.x
                      + (a5*pointOri.x - a6*pointOri.y + crx*pointOri.z + a7) * coeff.y
                      + (a8*pointOri.x - a9*pointOri.y - a10*pointOri.z + a11) * coeff.z;

            float arz = (c1*pointOri.x + c2*pointOri.y + c3) * coeff.x
                      + (c4*pointOri.x - c5*pointOri.y + c6) * coeff.y
                      + (c7*pointOri.x + c8*pointOri.y + c9) * coeff.z;

            float aty = -b6 * coeff.x + c4 * coeff.y + b2 * coeff.z;

            float d2 = coeff.intensity;

            matA.at<float>(i, 0) = arx;
            matA.at<float>(i, 1) = arz;
            matA.at<float>(i, 2) = aty;
            matB.at<float>(i, 0) = -0.05 * d2;
        }

        cv::transpose(matA, matAt);
        matAtA = matAt * matA;
        matAtB = matAt * matB;
        cv::solve(matAtA, matAtB, matX, cv::DECOMP_QR);

        if (iterCount == 0) {
            cv::Mat matE(1, 3, CV_32F, cv::Scalar::all(0));
            cv::Mat matV(3, 3, CV_32F, cv::Scalar::all(0));
            cv::Mat matV2(3, 3, CV_32F, cv::Scalar::all(0));

            cv::eigen(matAtA, matE, matV);
            matV.copyTo(matV2);

            isDegenerate = false;
            float eignThre[3] = {10, 10, 10};
            for (int i = 2; i >= 0; i--) {
                if (matE.at<float>(0, i) < eignThre[i]) {
                    for (int j = 0; j < 3; j++) {
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
            cv::Mat matX2(3, 1, CV_32F, cv::Scalar::all(0));
            matX.copyTo(matX2);
            matX = matP * matX2;
        }

        transformCur[0] += matX.at<float>(0, 0);
        transformCur[2] += matX.at<float>(1, 0);
        transformCur[4] += matX.at<float>(2, 0);

        for(int i=0; i<6; i++){
            if(isnan(transformCur[i]))
                transformCur[i]=0;
        }

        float deltaR = sqrt(
                            pow(rad2deg(matX.at<float>(0, 0)), 2) +
                            pow(rad2deg(matX.at<float>(1, 0)), 2));
        float deltaT = sqrt(
                            pow(matX.at<float>(2, 0) * 100, 2));

        if (deltaR < 0.1 && deltaT < 0.1) {
            return false;
        }
        return true;
    }

    bool calculateTransformationCorner(int iterCount){

        int pointSelNum = laserCloudOri->points.size();

        cv::Mat matA(pointSelNum, 3, CV_32F, cv::Scalar::all(0));
        cv::Mat matAt(3, pointSelNum, CV_32F, cv::Scalar::all(0));
        cv::Mat matAtA(3, 3, CV_32F, cv::Scalar::all(0));
        cv::Mat matB(pointSelNum, 1, CV_32F, cv::Scalar::all(0));
        cv::Mat matAtB(3, 1, CV_32F, cv::Scalar::all(0));
        cv::Mat matX(3, 1, CV_32F, cv::Scalar::all(0));

        float srx = sin(transformCur[0]);
        float crx = cos(transformCur[0]);
        float sry = sin(transformCur[1]);
        float cry = cos(transformCur[1]);
        float srz = sin(transformCur[2]);
        float crz = cos(transformCur[2]);
        float tx = transformCur[3];
        float ty = transformCur[4];
        float tz = transformCur[5];

        float b1 = -crz*sry - cry*srx*srz; float b2 = cry*crz*srx - sry*srz; float b3 = crx*cry; float b4 = tx*-b1 + ty*-b2 + tz*b3;
        float b5 = cry*crz - srx*sry*srz; float b6 = cry*srz + crz*srx*sry; float b7 = crx*sry; float b8 = tz*b7 - ty*b6 - tx*b5;

        float c5 = crx*srz;

        for (int i = 0; i < pointSelNum; i++) {

            pointOri = laserCloudOri->points[i];
            coeff = coeffSel->points[i];

            float ary = (b1*pointOri.x + b2*pointOri.y - b3*pointOri.z + b4) * coeff.x
                      + (b5*pointOri.x + b6*pointOri.y - b7*pointOri.z + b8) * coeff.z;

            float atx = -b5 * coeff.x + c5 * coeff.y + b1 * coeff.z;

            float atz = b7 * coeff.x - srx * coeff.y - b3 * coeff.z;

            float d2 = coeff.intensity;

            matA.at<float>(i, 0) = ary;
            matA.at<float>(i, 1) = atx;
            matA.at<float>(i, 2) = atz;
            matB.at<float>(i, 0) = -0.05 * d2;
        }

        cv::transpose(matA, matAt);
        matAtA = matAt * matA;
        matAtB = matAt * matB;
        cv::solve(matAtA, matAtB, matX, cv::DECOMP_QR);

        if (iterCount == 0) {
            cv::Mat matE(1, 3, CV_32F, cv::Scalar::all(0));
            cv::Mat matV(3, 3, CV_32F, cv::Scalar::all(0));
            cv::Mat matV2(3, 3, CV_32F, cv::Scalar::all(0));

            cv::eigen(matAtA, matE, matV);
            matV.copyTo(matV2);

            isDegenerate = false;
            float eignThre[3] = {10, 10, 10};
            for (int i = 2; i >= 0; i--) {
                if (matE.at<float>(0, i) < eignThre[i]) {
                    for (int j = 0; j < 3; j++) {
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
            cv::Mat matX2(3, 1, CV_32F, cv::Scalar::all(0));
            matX.copyTo(matX2);
            matX = matP * matX2;
        }

        transformCur[1] += matX.at<float>(0, 0);
        transformCur[3] += matX.at<float>(1, 0);
        transformCur[5] += matX.at<float>(2, 0);

        for(int i=0; i<6; i++){
            if(isnan(transformCur[i]))
                transformCur[i]=0;
        }

        float deltaR = sqrt(
                            pow(rad2deg(matX.at<float>(0, 0)), 2));
        float deltaT = sqrt(
                            pow(matX.at<float>(1, 0) * 100, 2) +
                            pow(matX.at<float>(2, 0) * 100, 2));

        if (deltaR < 0.1 && deltaT < 0.1) {
            return false;
        }
        return true;
    }

    bool calculateTransformation(int iterCount){

        int pointSelNum = laserCloudOri->points.size();

        cv::Mat matA(pointSelNum, 6, CV_32F, cv::Scalar::all(0));
        cv::Mat matAt(6, pointSelNum, CV_32F, cv::Scalar::all(0));
        cv::Mat matAtA(6, 6, CV_32F, cv::Scalar::all(0));
        cv::Mat matB(pointSelNum, 1, CV_32F, cv::Scalar::all(0));
        cv::Mat matAtB(6, 1, CV_32F, cv::Scalar::all(0));
        cv::Mat matX(6, 1, CV_32F, cv::Scalar::all(0));

        float srx = sin(transformCur[0]);
        float crx = cos(transformCur[0]);
        float sry = sin(transformCur[1]);
        float cry = cos(transformCur[1]);
        float srz = sin(transformCur[2]);
        float crz = cos(transformCur[2]);
        float tx = transformCur[3];
        float ty = transformCur[4];
        float tz = transformCur[5];

        float a1 = crx*sry*srz; float a2 = crx*crz*sry; float a3 = srx*sry; float a4 = tx*a1 - ty*a2 - tz*a3;
        float a5 = srx*srz; float a6 = crz*srx; float a7 = ty*a6 - tz*crx - tx*a5;
        float a8 = crx*cry*srz; float a9 = crx*cry*crz; float a10 = cry*srx; float a11 = tz*a10 + ty*a9 - tx*a8;

        float b1 = -crz*sry - cry*srx*srz; float b2 = cry*crz*srx - sry*srz; float b3 = crx*cry; float b4 = tx*-b1 + ty*-b2 + tz*b3;
        float b5 = cry*crz - srx*sry*srz; float b6 = cry*srz + crz*srx*sry; float b7 = crx*sry; float b8 = tz*b7 - ty*b6 - tx*b5;

        float c1 = -b6; float c2 = b5; float c3 = tx*b6 - ty*b5; float c4 = -crx*crz; float c5 = crx*srz; float c6 = ty*c5 + tx*-c4;
        float c7 = b2; float c8 = -b1; float c9 = tx*-b2 - ty*-b1;

        for (int i = 0; i < pointSelNum; i++) {

            pointOri = laserCloudOri->points[i];
            coeff = coeffSel->points[i];

            float arx = (-a1*pointOri.x + a2*pointOri.y + a3*pointOri.z + a4) * coeff.x
                      + (a5*pointOri.x - a6*pointOri.y + crx*pointOri.z + a7) * coeff.y
                      + (a8*pointOri.x - a9*pointOri.y - a10*pointOri.z + a11) * coeff.z;

            float ary = (b1*pointOri.x + b2*pointOri.y - b3*pointOri.z + b4) * coeff.x
                      + (b5*pointOri.x + b6*pointOri.y - b7*pointOri.z + b8) * coeff.z;

            float arz = (c1*pointOri.x + c2*pointOri.y + c3) * coeff.x
                      + (c4*pointOri.x - c5*pointOri.y + c6) * coeff.y
                      + (c7*pointOri.x + c8*pointOri.y + c9) * coeff.z;

            float atx = -b5 * coeff.x + c5 * coeff.y + b1 * coeff.z;

            float aty = -b6 * coeff.x + c4 * coeff.y + b2 * coeff.z;

            float atz = b7 * coeff.x - srx * coeff.y - b3 * coeff.z;

            float d2 = coeff.intensity;

            matA.at<float>(i, 0) = arx;
            matA.at<float>(i, 1) = ary;
            matA.at<float>(i, 2) = arz;
            matA.at<float>(i, 3) = atx;
            matA.at<float>(i, 4) = aty;
            matA.at<float>(i, 5) = atz;
            matB.at<float>(i, 0) = -0.05 * d2;
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
            float eignThre[6] = {10, 10, 10, 10, 10, 10};
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

        transformCur[0] += matX.at<float>(0, 0);
        transformCur[1] += matX.at<float>(1, 0);
        transformCur[2] += matX.at<float>(2, 0);
        transformCur[3] += matX.at<float>(3, 0);
        transformCur[4] += matX.at<float>(4, 0);
        transformCur[5] += matX.at<float>(5, 0);

        for(int i=0; i<6; i++){
            if(isnan(transformCur[i]))
                transformCur[i]=0;
        }

        float deltaR = sqrt(
                            pow(rad2deg(matX.at<float>(0, 0)), 2) +
                            pow(rad2deg(matX.at<float>(1, 0)), 2) +
                            pow(rad2deg(matX.at<float>(2, 0)), 2));
        float deltaT = sqrt(
                            pow(matX.at<float>(3, 0) * 100, 2) +
                            pow(matX.at<float>(4, 0) * 100, 2) +
                            pow(matX.at<float>(5, 0) * 100, 2));

        if (deltaR < 0.1 && deltaT < 0.1) {
            return false;
        }
        return true;
    }

// 激光里程计初始化。
// 激光里程计是连续帧之间匹配的过程，需要有一个起始参考帧，如果系统没有经过初始化，那么就把当前帧作为参考帧
// 并初始化当前总的位姿transformSum（除了俯仰角和滚转角外，其余参数全部初始化为0）
    void checkSystemInitialization(){
       // 交换两团点云，并发布出去
        pcl::PointCloud<PointType>::Ptr laserCloudTemp = cornerPointsLessSharp;
        cornerPointsLessSharp = laserCloudCornerLast;
        laserCloudCornerLast = laserCloudTemp;

        laserCloudTemp = surfPointsLessFlat;
        surfPointsLessFlat = laserCloudSurfLast;
        laserCloudSurfLast = laserCloudTemp;

        kdtreeCornerLast->setInputCloud(laserCloudCornerLast);
        kdtreeSurfLast->setInputCloud(laserCloudSurfLast);

        laserCloudCornerLastNum = laserCloudCornerLast->points.size();
        laserCloudSurfLastNum = laserCloudSurfLast->points.size();

        sensor_msgs::PointCloud2 laserCloudCornerLast2;
        pcl::toROSMsg(*laserCloudCornerLast, laserCloudCornerLast2);
        laserCloudCornerLast2.header.stamp = cloudHeader.stamp;
        laserCloudCornerLast2.header.frame_id = "/camera";
        pubLaserCloudCornerLast.publish(laserCloudCornerLast2);

        sensor_msgs::PointCloud2 laserCloudSurfLast2;
        pcl::toROSMsg(*laserCloudSurfLast, laserCloudSurfLast2);
        laserCloudSurfLast2.header.stamp = cloudHeader.stamp;
        laserCloudSurfLast2.header.frame_id = "/camera";
        pubLaserCloudSurfLast.publish(laserCloudSurfLast2);

        transformSum[0] += imuPitchStart;
        transformSum[2] += imuRollStart;

        systemInitedLM = true;
    }

    //该函数根据IMU积分的结果，计算出一个初始位姿transformCur，这个位姿指的是雷达旋转一圈后发生的相对位姿变换,将当前时刻保存的IMU数据作为先验数据
    void updateInitialGuess(){
        imuPitchLast = imuPitchCur;
        imuYawLast = imuYawCur;
        imuRollLast = imuRollCur;

        imuShiftFromStartX = imuShiftFromStartXCur;
        imuShiftFromStartY = imuShiftFromStartYCur;
        imuShiftFromStartZ = imuShiftFromStartZCur;

        imuVeloFromStartX = imuVeloFromStartXCur;
        imuVeloFromStartY = imuVeloFromStartYCur;
        imuVeloFromStartZ = imuVeloFromStartZCur;

        if (imuAngularFromStartX != 0 || imuAngularFromStartY != 0 || imuAngularFromStartZ != 0){
            transformCur[0] = - imuAngularFromStartY;
            transformCur[1] = - imuAngularFromStartZ;
            transformCur[2] = - imuAngularFromStartX;
        }

        // 速度×时间
        if (imuVeloFromStartX != 0 || imuVeloFromStartY != 0 || imuVeloFromStartZ != 0){
            transformCur[3] -= imuVeloFromStartX * scanPeriod;
            transformCur[4] -= imuVeloFromStartY * scanPeriod;
            transformCur[5] -= imuVeloFromStartZ * scanPeriod;
        }
    }

//    通过约束优化相对位姿。
//    对于当前帧的每一个面点，在上一帧找到最近邻以及另外两个不共线的点组成面，该点到面的距离就是要减小的残差，通过面之间的对应关系计算出变换矩阵。
//    通过角、边特征的匹配，计算变换矩阵:找到最近邻和另外一个点构建约束，被优化的对象就是transformCur，即找到一个相对位姿变换，使得总体的残差最小。
//    然后依次使用平面点和角点,对相对位姿进行优化。
//    优化的对象transformCur特点在于,先使用平面点经历多次优化后,再使用角点(可能因为平面点更多,因而会更稳定。
// 分别对平面特征与角点特征进行了匹配计算，首先找到各自对应的特征
    void updateTransformation(){
        if (laserCloudCornerLastNum < 10 || laserCloudSurfLastNum < 100)
            return;
        for (int iterCount1 = 0; iterCount1 < 25; iterCount1++) {
            laserCloudOri->clear();
            coeffSel->clear();
            //  找到对应的特征平面
            findCorrespondingSurfFeatures(iterCount1);
            if (laserCloudOri->points.size() < 10)
                continue;
            //  面特征匹配计算变换矩阵
            if (calculateTransformationSurf(iterCount1) == false)
                break;
        }

        for (int iterCount2 = 0; iterCount2 < 25; iterCount2++) {
            laserCloudOri->clear();
            coeffSel->clear();
            //  寻找角特征
            findCorrespondingCornerFeatures(iterCount2);
            if (laserCloudOri->points.size() < 10)
                continue;
            //  当边角特征足够的时候，开始边角特征匹配
            if (calculateTransformationCorner(iterCount2) == false)
                break;
        }
    }

    //把优化得到的transformCur，将IMU信息融入到位姿更新当中，累加到transformSum中。
    void integrateTransformation(){
        //  将计算的两帧之间的位姿“累加”起来，获得相对于第一帧的旋转矩阵
        float rx, ry, rz, tx, ty, tz;
        AccumulateRotation(transformSum[0], transformSum[1], transformSum[2], 
                           -transformCur[0], -transformCur[1], -transformCur[2], rx, ry, rz);
        // 进行平移分量的更新
        float x1 = cos(rz) * (transformCur[3] - imuShiftFromStartX) 
                 - sin(rz) * (transformCur[4] - imuShiftFromStartY);
        float y1 = sin(rz) * (transformCur[3] - imuShiftFromStartX) 
                 + cos(rz) * (transformCur[4] - imuShiftFromStartY);
        float z1 = transformCur[5] - imuShiftFromStartZ;

        float x2 = x1;
        float y2 = cos(rx) * y1 - sin(rx) * z1;
        float z2 = sin(rx) * y1 + cos(rx) * z1;

        tx = transformSum[3] - (cos(ry) * x2 + sin(ry) * z2);
        ty = transformSum[4] - y2;
        tz = transformSum[5] - (-sin(ry) * x2 + cos(ry) * z2);

        // 与accumulateRotatio联合起来更新transformSum的rotation部分的工作
        // 可视为transformToEnd的下部分的逆过程
        PluginIMURotation(rx, ry, rz, imuPitchStart, imuYawStart, imuRollStart, imuPitchLast, imuYawLast, imuRollLast, rx, ry, rz);
        transformSum[0] = rx;
        transformSum[1] = ry;
        transformSum[2] = rz;
        transformSum[3] = tx;
        transformSum[4] = ty;
        transformSum[5] = tz;
    }
    
    //发布里程计位姿位姿和tf变换，这里的变换实际上是odo相对于地图原点的，存在持续的误差积累;
    void publishOdometry(){
        geometry_msgs::Quaternion geoQuat = tf::createQuaternionMsgFromRollPitchYaw(transformSum[2], -transformSum[0], -transformSum[1]);

        laserOdometry.header.stamp = cloudHeader.stamp;
        laserOdometry.pose.pose.orientation.x = -geoQuat.y;
        laserOdometry.pose.pose.orientation.y = -geoQuat.z;
        laserOdometry.pose.pose.orientation.z = geoQuat.x;
        laserOdometry.pose.pose.orientation.w = geoQuat.w;
        laserOdometry.pose.pose.position.x = transformSum[3];
        laserOdometry.pose.pose.position.y = transformSum[4];
        laserOdometry.pose.pose.position.z = transformSum[5];
        pubLaserOdometry.publish(laserOdometry);

        laserOdometryTrans.stamp_ = cloudHeader.stamp;
        laserOdometryTrans.setRotation(tf::Quaternion(-geoQuat.y, -geoQuat.z, geoQuat.x, geoQuat.w));
        laserOdometryTrans.setOrigin(tf::Vector3(transformSum[3], transformSum[4], transformSum[5]));
        tfBroadcaster.sendTransform(laserOdometryTrans);
    }

    void adjustOutlierCloud(){
        PointType point;
        int cloudSize = outlierCloud->points.size();
        for (int i = 0; i < cloudSize; ++i)
        {
            point.x = outlierCloud->points[i].y;
            point.y = outlierCloud->points[i].z;
            point.z = outlierCloud->points[i].x;
            point.intensity = outlierCloud->points[i].intensity;
            outlierCloud->points[i] = point;
        }
    }

    void publishCloudsLast(){
        updateImuRollPitchYawStartSinCos();
        int cornerPointsLessSharpNum = cornerPointsLessSharp->points.size();
        for (int i = 0; i < cornerPointsLessSharpNum; i++) {
            TransformToEnd(&cornerPointsLessSharp->points[i], &cornerPointsLessSharp->points[i]);
        }
        int surfPointsLessFlatNum = surfPointsLessFlat->points.size();
        for (int i = 0; i < surfPointsLessFlatNum; i++) {
            TransformToEnd(&surfPointsLessFlat->points[i], &surfPointsLessFlat->points[i]);
        }

        pcl::PointCloud<PointType>::Ptr laserCloudTemp = cornerPointsLessSharp;
        cornerPointsLessSharp = laserCloudCornerLast;
        laserCloudCornerLast = laserCloudTemp;

        laserCloudTemp = surfPointsLessFlat;
        surfPointsLessFlat = laserCloudSurfLast;
        laserCloudSurfLast = laserCloudTemp;

        laserCloudCornerLastNum = laserCloudCornerLast->points.size();
        laserCloudSurfLastNum = laserCloudSurfLast->points.size();

        if (laserCloudCornerLastNum > 10 && laserCloudSurfLastNum > 100) {
            kdtreeCornerLast->setInputCloud(laserCloudCornerLast);
            kdtreeSurfLast->setInputCloud(laserCloudSurfLast);
        }

        frameCount++;

        if (frameCount >= skipFrameNum + 1) {

            frameCount = 0;

            adjustOutlierCloud();
            sensor_msgs::PointCloud2 outlierCloudLast2;
            pcl::toROSMsg(*outlierCloud, outlierCloudLast2);
            outlierCloudLast2.header.stamp = cloudHeader.stamp;
            outlierCloudLast2.header.frame_id = "/camera";
            pubOutlierCloudLast.publish(outlierCloudLast2);

            sensor_msgs::PointCloud2 laserCloudCornerLast2;
            pcl::toROSMsg(*laserCloudCornerLast, laserCloudCornerLast2);
            laserCloudCornerLast2.header.stamp = cloudHeader.stamp;
            laserCloudCornerLast2.header.frame_id = "/camera";
            pubLaserCloudCornerLast.publish(laserCloudCornerLast2);

            sensor_msgs::PointCloud2 laserCloudSurfLast2;
            pcl::toROSMsg(*laserCloudSurfLast, laserCloudSurfLast2);
            laserCloudSurfLast2.header.stamp = cloudHeader.stamp;
            laserCloudSurfLast2.header.frame_id = "/camera";
            pubLaserCloudSurfLast.publish(laserCloudSurfLast2);
        }
    }

    // 入口
    void runFeatureAssociation()
    {
        // 有新数据进就重置参数变量:获得了三个点云主题的msg,且时间上一致,将标志位置设为假,判断分割点云和异常值的实时性
        if (   newSegmentedCloud && newSegmentedCloudInfo && newOutlierCloud &&
            std::abs(timeNewSegmentedCloudInfo - timeNewSegmentedCloud) < 0.05 &&
            std::abs(timeNewOutlierCloud - timeNewSegmentedCloud) < 0.05){
            //重置参数变量
            newSegmentedCloud = false;
            newSegmentedCloudInfo = false;
            newOutlierCloud = false;
        }else{
            return;
        }
        /**
        	1. Feature Extraction特征提取
        */
        //先进行形状特征的调整
        adjustDistortion();//  imu   消除因平台的移动导致的点云畸变。

        calculateSmoothness(); //  计算光滑性：附近十个点的xyz平方和

        markOccludedPoints();// 距离较大或者遮挡点标记，在后面不选择为特征点

        //提取特征
        extractFeatures();//  特征提取

        publishCloud(); // cloud for visualization
	
        /**
		2. Feature Association     在配准之前，检验LM法是否初始化，接下来就是里程计部分
        */
        // 检查系统是否进行初始化了(当前是否是第一帧)
        if (!systemInitedLM) {
            checkSystemInitialization();
            // 第一帧时，无法进行定位，为直接进入下一帧的处理进行准备,即更新
            // laserCloudCornerLast    更新上一帧的Less角点
            // laserCloudSurfLast      更新上一帧的Less平面点
            // laserCloudCornerLastNum 上一帧Less角点数量
            // laserCloudSurfLastNum   上一帧Less平面点的数量
            // 系统初始化标志位 systemInitedLM 置 true
            return;
        }
        //  更新初始位姿    提供粗配准的先验以供优化
        updateInitialGuess();
        //  一个是找特征平面，通过面之间的对应关系计算出变换矩阵。
        // 另一个部分是通过角、边特征的匹配，计算变换矩阵。
        //     位姿 transformCur[0~5] 
        // 并更新 
        //     imuPitchLast、imuYawLast、imuRollLast 上一帧RPY值
        //     imuShiftFromStartX、imuShiftFromStartY、imuShiftFromStartZ  当前帧位置变化量
        //     imuVeloFromStartX、imuVeloFromStartY、imuVeloFromStartZ  当前帧速度变化量 

        //优化并发送里程计信息
        updateTransformation();
        integrateTransformation(); //  计算旋转角的累积变化量。

        publishOdometry();//  发布里程计信息

        publishCloudsLast(); // cloud to mapOptimization
    }
};




int main(int argc, char** argv)
{
    ros::init(argc, argv, "lego_loam");

    ROS_INFO("\033[1;32m---->\033[0m Feature Association Started.");

    FeatureAssociation FA;   //调用构造函数进行初始化

    // 以200Hz的频率在主动调用FA.runFeatureAssociation()这个函数
    ros::Rate rate(150);
    while (ros::ok())// while ( 1 )
    {
        ros::spinOnce();

        FA.runFeatureAssociation();

        rate.sleep();
    }
    
    ros::spin();
    return 0;
}
