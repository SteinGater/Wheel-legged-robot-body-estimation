/*********************************************************************************************/
//MMMR 实验室
//作者： CJW
//轮腿腿式机器人机身位置速度-卡尔曼滤波
/*********************************************************************************************/
#ifndef CJW_KALMANFILTER
#define CJW_KALMANFILTER

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>

#include <math.h>
#include "CJW_Math.h"

/************************************定义常量**********************************/
//机器人机身卡尔曼滤波方式
#define EKF_IL      0   //MIT的IMU融合腿式机器人机身位置速度卡尔曼滤波
#define EKF_EIL     1   //MIT+额外位置观测的腿式机器人机身位置速度卡尔曼滤波
//支撑腿滑动阈值
#define FOOT_SLIP_MIN 1.0
//摆动腿大噪声阈值
#define SWING_NOISE   1000.0
//平地支撑腿高度直接优化
#define FOOT_PLANE   0

//轮腿估计的支撑Z置信度
#define LEG_Z_CONFIDENCE 100


/****************************************腿式机器人机身位置速度卡尔曼滤波**************************/
//构建对象：T为数据参数，如double，int等；branchn为机器人支撑腿数
//初始化对象：设置滤波方法SetType(Type,dt):Type 包含EKF_IL和EKF_EIL两种方式，Initial(dt)为采样周期
//设置噪声参数：SetPara(input)： 预测位置噪声，预测速度噪声，预测腿末端位置噪声，
//                            观测腿里程计位置噪声，观测腿里程计速度噪声，观测腿里程计高度噪声，
//                            观测外部位置噪声，观测外部速度噪声
//一般情况下设置初始的状态变量：SetBodyLegState(Bodyp,Bodyv,Footp);具体参数见不同滤波方式函数说明
/***************/
//设置观测数据：SetLegObs/SetIMUAcc/SetExtObs;根据需求设置
//运行一次卡尔曼滤波：EstimateOnce();
//获取滤波结果：GetBodyLegState(Bodyp,Bodyv,Footp);
/*********************************************************************************************/
template <typename T,int branchn>
class CJW_Leg_Body_KF
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    CJW_Leg_Body_KF(){};//构造函数
    ~CJW_Leg_Body_KF()=default;
    /*****************************************通用函数*******************************/
    //重置卡尔曼矩阵大小和数据
    virtual void KFResize(int n_x,int n_u,int n_z);
    //卡尔曼滤波计算一次
    virtual void KFonce(void);
    //设置状态估计状态值
    virtual void SetState(Eigen::Matrix<T,Eigen::Dynamic,1> input);
    virtual void SetState(T* input);
    //得到状态估计状态值
    virtual Eigen::Matrix<T,Eigen::Dynamic,1> GetState(void);
    virtual void GetState(T* input);
    //设置状态估计输入值
    virtual void SetInput(Eigen::Matrix<T,Eigen::Dynamic,1> input);
    virtual void SetInput(T* input);
    //得到状态估计输入值
    virtual Eigen::Matrix<T,Eigen::Dynamic,1> GetInput(void);
    virtual void GetInput(T* input);
    //设置状态估计观测数据
    virtual void SetObs(Eigen::Matrix<T,Eigen::Dynamic,1> input);
    virtual void SetObs(T* input);
    //得到状态估计观测数据
    virtual Eigen::Matrix<T,Eigen::Dynamic,1> GetObs(void);
    virtual void GetObs(T* input);
    //设置状态估计噪声参数
    virtual void SetPara(Eigen::Matrix<T,Eigen::Dynamic,1> input);
    virtual void SetPara(T* input);
    //得到状态估计噪声参数
    virtual Eigen::Matrix<T,Eigen::Dynamic,1> GetPara(void);
    virtual void GetPara(T* input);
    /****************************************机身状态函数*****************************/
    //设置机身的当前位置
    virtual void SetBodyp(Eigen::Matrix<T,3,1> Bodyp);
    virtual void SetBodyp(T* Bodyp);
    //设置机身的当前速度
    virtual void SetBodyv(Eigen::Matrix<T,3,1> Bodyv);
    virtual void SetBodyv(T* Bodyv);
    //得到状态估计机身的当前位置
    virtual Eigen::Matrix<T,3,1> GetBodyp(void);
    virtual void GetBodyp(T* Bodyp);
    //得到状态估计机身的当前速度
    virtual Eigen::Matrix<T,3,1> GetBodyv(void);
    virtual void GetBodyv(T* Bodyv);
    /****************************************腿式里程计*****************************/
    //设置腿式分支的当前位置
    virtual void SetLegFootp(Eigen::Matrix<T,3,branchn> Footp);
    virtual void SetLegFootp(T* Footp);
    //得到当前腿式分支位置
    virtual Eigen::Matrix<T,3,branchn> GetLegFootp(void);
    virtual void GetLegFootp(T* Footp);
    //设置状态估计机身状态位置、速度和腿式分支位置
    virtual void SetBodyLegState(Eigen::Matrix<T,3,1> Bodyp,Eigen::Matrix<T,3,1> Bodyv,
                                    Eigen::Matrix<T,3,branchn> Footp);
    virtual void SetBodyLegState(T* Bodyp,T* Bodyv, T* Footp);
    //得到状态估计机身的当前位置、当前速度和当前腿式分支位置
    virtual void GetBodyLegState(Eigen::Matrix<T,3,1> &Bodyp,Eigen::Matrix<T,3,1> &Bodyv,
                                    Eigen::Matrix<T,3,branchn> &Footp);
    virtual void GetBodyLegState(T* Bodyp,T* Bodyv, T* Footp);
    //设置腿式观测的当前状态
    virtual void SetLegObs(Eigen::Matrix<T,3,branchn> Footdp,Eigen::Matrix<T,3,branchn> Footdv,Eigen::Matrix<T,branchn,1> Contact);
    virtual void SetLegObs(T* Footdp,T* Footdv,T* Contact);
    /****************************************IMU**********************************/
    //设置IMU观测的的加速度数据
    virtual void SetIMUAcc(Eigen::Matrix<T,3,1> acc);
    virtual void SetIMUAcc(T* acc);
    /****************************************外部观测******************************/
    //设置外部观测数据的位置
    virtual void SetExtObs(Eigen::Matrix<T,3,1> p,Eigen::Matrix<T,3,1> v,Eigen::Matrix<T,3,1> a);
    virtual void SetExtObs(T* p,T* v,T* a);

    /***************************************MIT卡尔曼滤波方法*************************/
    //设置滤波方法
    virtual void SetType(int Type);
    //滤波初始化
    virtual void Initial(T dt);
    //腿式里程计加外部定位卡尔曼滤波方法滤波
    virtual void EstimateOnce(void);

protected:
    //卡尔曼滤波方法标志位及预测时间
    int KF_Type;
    T T_predict;
    int TypeN0;
    /***************************************状态参数*************************/
    //状态变量的状态ID
    T body_p_id;
    T body_v_id;
    T foot_p_id;
    /***************************************观测参数*************************/
    //腿式分支末端位置速度信息-绝对坐标系
    Eigen::Matrix<T,3,branchn> foot_dp;
    Eigen::Matrix<T,3,branchn> foot_dv;
    //腿式分支接触状态
    T foot_contact[branchn];
    Eigen::Matrix<T,3,branchn> foot_p0;
    //IMU加速度数据
    Eigen::Matrix<T,3,1> imu_acc;
    //外部观测数据
    Eigen::Matrix<T,3,1> ex_position;
    Eigen::Matrix<T,3,1> ex_velocity;
    Eigen::Matrix<T,3,1> ex_acceleration;
    /***********************************卡尔曼滤波参数*************************/
    //卡尔曼滤波尺寸大小
    int XN;
    int UN;
    int ZN;
    int ParaN;
    //卡尔曼滤波噪声系数
    Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> Para;
    //卡尔曼滤波结构参数
    Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> _x;
    Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> _u;
    Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> _z;

    Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> _A;
    Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> _B;
    Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> _C;

    Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> _P;
    Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> _Q;
    Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> _R;
    Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> _I;
};


/****************************************轮腿机器人机身位置速度卡尔曼滤波**************************/
//构建对象：T为数据参数，如double，int等；branchn为机器人支撑腿数
//初始化对象：设置滤波方法SetType(Type):Type 包含EKF_IL和EKF_EIL两种方式，Initial(dt)为采样周期
//设置噪声参数：SetPara(input)： 预测全位置噪声，预测全速度噪声，预测轮式位置噪声，预测轮式速度噪声，预测腿末端腿式位置噪声，
//                             观测腿里程计位置噪声，观测腿里程计速度噪声，观测轮里程计速度噪声，
//                             观测外部位置噪声，观测外部速度噪声，
//一般情况下设置初始的状态变量：SetBodyWheelLegState(Bodyp,Bodyv,Bodyp_w,Bodyv_w,Footp_l);
/***************/
//设置观测数据：SetLegObs/SetIMUAcc/SetExtObs/SetWheelObs;根据需求设置
//运行一次卡尔曼滤波：EstimateOnce();
//获取滤波结果：GetBodyLegState(Bodyp,Bodyv,Footp);
/*********************************************************************************************/
template <typename T,int branchn>
class CJW_WheelLeg_Body_KF: public CJW_Leg_Body_KF<T,branchn>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    CJW_WheelLeg_Body_KF():CJW_Leg_Body_KF<T,branchn>() {};
    ~CJW_WheelLeg_Body_KF(){};

    /****************************************机身状态函数*****************************/
    //设置轮式里程计机身的当前位置
    virtual void SetBodyp_w(Eigen::Matrix<T,3,1> Bodyp);
    virtual void SetBodyp_w(T* Bodyp);
    //设置轮式里程计机身的当前速度
    virtual void SetBodyv_w(Eigen::Matrix<T,3,1> Bodyv);
    virtual void SetBodyv_w(T* Bodyv);
    //得到轮式里程计机身的当前位置
    virtual Eigen::Matrix<T,3,1> GetBodyp_w(void);
    virtual void GetBodyp_w(T* Bodyp);
    //得到轮式里程计机身的当前速度
    virtual Eigen::Matrix<T,3,1> GetBodyv_w(void);
    virtual void GetBodyv_w(T* Bodyv);
    /****************************************轮式里程计*****************************/
    //设置轮式观测的当前状态
    virtual void SetWheelObs(Eigen::Matrix<T,3,branchn> Wheeldv);
    virtual void SetWheelObs(T* Wheeldv);
    //设置状态估计机身的轮腿累加当前位置速度，机身的轮里程计当前位置速度，分支腿里程计末端位置
    virtual void SetBodyWheelLegState(  Eigen::Matrix<T,3,1> Bodyp,Eigen::Matrix<T,3,1> Bodyv,
                                        Eigen::Matrix<T,3,1> Bodyp_w,Eigen::Matrix<T,3,1> Bodyv_w,
                                        Eigen::Matrix<T,3,branchn> Footp_l);
    virtual void SetBodyWheelLegState(T* Bodyp,T* Bodyv,T* Bodyp_w,T* Bodyv_w,T* Footp_l);
    //得到状态估计机身的轮腿累加当前位置速度，机身的轮里程计当前位置速度，分支腿里程计末端位置
    virtual void GetBodyWheelLegState(  Eigen::Matrix<T,3,1> &Bodyp,Eigen::Matrix<T,3,1> &Bodyv,
                                        Eigen::Matrix<T,3,1> &Bodyp_w,Eigen::Matrix<T,3,1> &Bodyv_w,
                                        Eigen::Matrix<T,3,branchn> &Footp_l);
    virtual void GetBodyWheelLegState(T* Bodyp,T* Bodyv,T* Bodyp_w,T* Bodyv_w,T* Footp_l);
    /***************************************轮腿里程计加外部定位卡尔曼滤波方法*************************/
    //设置滤波方法
    virtual void SetType(int Type);
    //滤波初始化
    virtual void Initial(T dt);
    //腿式里程计加外部定位卡尔曼滤波方法滤波
    virtual void EstimateOnce(void);

protected:
    /***************************************状态参数*************************/
    //增加轮式里程计的状态变量
    T body_p_w_id;
    T body_v_w_id;
    T body_p_l_id;
    T body_v_l_id;
    T foot_body_p_id;
    T foot_body_v_id;
    T wheel_body_v_id;
    /***************************************观测参数*************************/
    //轮式分支末端速度信息-绝对坐标系
    Eigen::Matrix<T,3,branchn> wheel_dv;
    

};



#endif
