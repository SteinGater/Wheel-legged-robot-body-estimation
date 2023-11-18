#include "CJW_KalmanFilter.h"

/****************************************腿式机器人机身位置速度卡尔曼滤波**************************/
/********************卡尔曼滤波基础运算*****************************/
// 重置卡尔曼矩阵大小和数据
template <typename T, int branchn>
void CJW_Leg_Body_KF<T, branchn>::KFResize(int n_x, int n_u, int n_z)
{
  _x.resize(n_x, 1);
  _u.resize(n_u, 1);
  _z.resize(n_z, 1);
  _A.resize(n_x, n_x);
  _B.resize(n_x, n_u);
  _C.resize(n_z, n_x);
  _P.resize(n_x, n_x);
  _Q.resize(n_x, n_x);
  _R.resize(n_z, n_z);
  _I.resize(n_x, n_x);
  _x.setZero();
  _u.setZero();
  _z.setZero();
  _A.setIdentity();
  _B.setZero();
  _C.setZero();
  _P.setIdentity();
  _Q.setIdentity();
  _R.setIdentity();
  _I.setIdentity();
}
// 卡尔曼滤波计算一次
template <typename T, int branchn>
void CJW_Leg_Body_KF<T, branchn>::KFonce(void)
{
  _x = _A * _x + _B * _u;
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> At = _A.transpose();
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> Pm = _A * _P * At + _Q;
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> Ct = _C.transpose();
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> zModel = _C * _x;
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> ey = _z - zModel;
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> S = _C * Pm * Ct + _R;

  // 利用LU分解求逆矩阵
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> S_ey = S.lu().solve(ey);
  _x += Pm * Ct * S_ey;

  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> S_C = S.lu().solve(_C);
  _P = (_I - Pm * Ct * S_C) * Pm;

  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> Pt = _P.transpose();
  _P = (_P + Pt) / T(2);
}
// 设置状态估计状态值
template <typename T, int branchn>
void CJW_Leg_Body_KF<T, branchn>::SetState(Eigen::Matrix<T, Eigen::Dynamic, 1> input)
{
  _x = input;
}
template <typename T, int branchn>
void CJW_Leg_Body_KF<T, branchn>::SetState(T *input)
{
  for (int i = 0; i < XN; i++)
  {
    _x(i, 0) = input[i];
  }
}
// 得到状态估计状态值
template <typename T, int branchn>
Eigen::Matrix<T, Eigen::Dynamic, 1> CJW_Leg_Body_KF<T, branchn>::GetState(void)
{
  return _x;
}
template <typename T, int branchn>
void CJW_Leg_Body_KF<T, branchn>::GetState(T *input)
{
  for (int i = 0; i < XN; i++)
  {
    input[i] = _x(i, 0);
  }
}
// 设置状态估计输入值
template <typename T, int branchn>
void CJW_Leg_Body_KF<T, branchn>::SetInput(Eigen::Matrix<T, Eigen::Dynamic, 1> input)
{
  _u = input;
}
template <typename T, int branchn>
void CJW_Leg_Body_KF<T, branchn>::SetInput(T *input)
{
  for (int i = 0; i < UN; i++)
  {
    _u(i, 0) = input[i];
  }
}
// 得到状态估计输入值
template <typename T, int branchn>
Eigen::Matrix<T, Eigen::Dynamic, 1> CJW_Leg_Body_KF<T, branchn>::GetInput(void)
{
  return _u;
}
template <typename T, int branchn>
void CJW_Leg_Body_KF<T, branchn>::GetInput(T *input)
{
  for (int i = 0; i < UN; i++)
  {
    input[i] = _u(i, 0);
  }
}
// 设置状态估计观测数据
template <typename T, int branchn>
void CJW_Leg_Body_KF<T, branchn>::SetObs(Eigen::Matrix<T, Eigen::Dynamic, 1> input)
{
  _z = input;
}
template <typename T, int branchn>
void CJW_Leg_Body_KF<T, branchn>::SetObs(T *input)
{
  for (int i = 0; i < ZN; i++)
  {
    _z(i, 0) = input[i];
  }
}
// 得到状态估计状态值
template <typename T, int branchn>
Eigen::Matrix<T, Eigen::Dynamic, 1> CJW_Leg_Body_KF<T, branchn>::GetObs(void)
{
  return _z;
}
template <typename T, int branchn>
void CJW_Leg_Body_KF<T, branchn>::GetObs(T *input)
{
  for (int i = 0; i < ZN; i++)
  {
    input[i] = _z(i, 0);
  }
}
// 设置状态估计噪声参数
template <typename T, int branchn>
void CJW_Leg_Body_KF<T, branchn>::SetPara(Eigen::Matrix<T, Eigen::Dynamic, 1> input)
{
  Para = input;
}
template <typename T, int branchn>
void CJW_Leg_Body_KF<T, branchn>::SetPara(T *input)
{
  for (int i = 0; i < ParaN; i++)
  {
    Para(i, 0) = input[i];
  }
}
// 得到状态估计状态值
template <typename T, int branchn>
Eigen::Matrix<T, Eigen::Dynamic, 1> CJW_Leg_Body_KF<T, branchn>::GetPara(void)
{
  return Para;
}
template <typename T, int branchn>
void CJW_Leg_Body_KF<T, branchn>::GetPara(T *input)
{
  for (int i = 0; i < ParaN; i++)
  {
    input[i] = Para(i, 0);
  }
}
/****************************************机身状态函数*****************************/
// 设置状态估计机身的当前位置
template <typename T, int branchn>
void CJW_Leg_Body_KF<T, branchn>::SetBodyp(Eigen::Matrix<T, 3, 1> input)
{
  _x.block(body_p_id, 0, 3, 1) = input;
}
template <typename T, int branchn>
void CJW_Leg_Body_KF<T, branchn>::SetBodyp(T *input)
{
  for (int i = 0; i < 3; i++)
  {
    _x(body_p_id + i, 0) = input[i];
  }
}
// 设置状态估计机身的当前速度
template <typename T, int branchn>
void CJW_Leg_Body_KF<T, branchn>::SetBodyv(Eigen::Matrix<T, 3, 1> input)
{
  _x.block(body_v_id, 0, 3, 1) = input;
}
template <typename T, int branchn>
void CJW_Leg_Body_KF<T, branchn>::SetBodyv(T *input)
{
  for (int i = 0; i < 3; i++)
  {
    _x(body_v_id + i, 0) = input[i];
  }
}
// 得到状态估计机身的当前位置
template <typename T, int branchn>
Eigen::Matrix<T, 3, 1> CJW_Leg_Body_KF<T, branchn>::GetBodyp(void)
{
  return _x.block(body_p_id, 0, 3, 1);
}
template <typename T, int branchn>
void CJW_Leg_Body_KF<T, branchn>::GetBodyp(T *Bodyp)
{
  for (int i = 0; i < 3; i++)
  {
    Bodyp[i] = _x(body_p_id + i, 0);
  }
}
// 得到状态估计机身的当前速度
template <typename T, int branchn>
Eigen::Matrix<T, 3, 1> CJW_Leg_Body_KF<T, branchn>::GetBodyv(void)
{
  return _x.block(body_v_id, 0, 3, 1);
}
template <typename T, int branchn>
void CJW_Leg_Body_KF<T, branchn>::GetBodyv(T *Bodyv)
{
  for (int i = 0; i < 3; i++)
  {
    Bodyv[i] = _x(body_v_id + i, 0);
  }
}
/****************************************腿式里程计*****************************/
// 设置腿式分支的当前位置
template <typename T, int branchn>
void CJW_Leg_Body_KF<T, branchn>::SetLegFootp(Eigen::Matrix<T, 3, branchn> input)
{
  for (int i = 0; i < branchn; i++)
  {
    _x.block(foot_p_id + 3 * i, 0, 3, 1) = input.block(0, i, 3, 1);
  }
}
template <typename T, int branchn>
void CJW_Leg_Body_KF<T, branchn>::SetLegFootp(T *input)
{
  for (int i = 0; i < branchn; i++)
  {
    for (int j = 0; j < 3; j++)
    {
      _x(foot_p_id + 3 * i + j, 0) = input[3 * i + j];
    }
  }
}
// 得到当前腿式分支位置
template <typename T, int branchn>
Eigen::Matrix<T, 3, branchn> CJW_Leg_Body_KF<T, branchn>::GetLegFootp(void)
{
  Eigen::Matrix<T, 3, branchn> legp;
  for (int i = 0; i < branchn; i++)
  {
    legp.block(0, i, 3, 1) = _x.block(foot_p_id + 3 * i, 0, 3, 1);
  }
  return legp;
}
template <typename T, int branchn>
void CJW_Leg_Body_KF<T, branchn>::GetLegFootp(T *Footp)
{
  for (int i = 0; i < branchn; i++)
  {
    for (int j = 0; j < 3; j++)
    {
      Footp[3 * i + j] = _x(foot_p_id + 3 * i + j, 0);
    }
  }
}
// 设置状态估计机身状态位置、速度和腿式分支位置
template <typename T, int branchn>
void CJW_Leg_Body_KF<T, branchn>::SetBodyLegState(Eigen::Matrix<T, 3, 1> Bodyp, Eigen::Matrix<T, 3, 1> Bodyv,
                                                  Eigen::Matrix<T, 3, branchn> Footp)
{
  _x.block(body_p_id, 0, 3, 1) = Bodyp;
  _x.block(body_v_id, 0, 3, 1) = Bodyv;
  for (int i = 0; i < branchn; i++)
  {
    _x.block(foot_p_id + 3 * i, 0, 3, 1) = Footp.block(0, i, 3, 1);
  }
}
template <typename T, int branchn>
void CJW_Leg_Body_KF<T, branchn>::SetBodyLegState(T *Bodyp, T *Bodyv, T *Footp)
{
  for (int i = 0; i < 3; i++)
  {
    _x(body_p_id + i, 0) = Bodyp[i];
    _x(body_v_id + i, 0) = Bodyv[i];
  }
  for (int i = 0; i < branchn; i++)
  {
    for (int j = 0; j < 3; j++)
    {
      _x(foot_p_id + 3 * i + j, 0) = Footp[3 * i + j];
    }
  }
}
// 得到状态估计机身的当前位置、当前速度和当前腿式分支位置
template <typename T, int branchn>
void CJW_Leg_Body_KF<T, branchn>::GetBodyLegState(Eigen::Matrix<T, 3, 1> &Bodyp, Eigen::Matrix<T, 3, 1> &Bodyv,
                                                  Eigen::Matrix<T, 3, branchn> &Footp)
{
  Bodyp = _x.block(body_p_id, 0, 3, 1);
  Bodyv = _x.block(body_v_id, 0, 3, 1);
  for (int i = 0; i < branchn; i++)
  {
    Footp.block(0, i, 3, 1) = _x.block(foot_p_id + 3 * i, 0, 3, 1);
  }
}
template <typename T, int branchn>
void CJW_Leg_Body_KF<T, branchn>::GetBodyLegState(T *Bodyp, T *Bodyv, T *Footp)
{
  for (int i = 0; i < 3; i++)
  {
    Bodyp[i] = _x(body_p_id + i, 0);
    Bodyv[i] = _x(body_v_id + i, 0);
  }
  for (int i = 0; i < branchn; i++)
  {
    for (int j = 0; j < 3; j++)
    {
      Footp[3 * i + j] = _x(foot_p_id + 3 * i + j, 0);
    }
  }
}
// 设置腿式观测的当前状态
template <typename T, int branchn>
void CJW_Leg_Body_KF<T, branchn>::SetLegObs(Eigen::Matrix<T, 3, branchn> Footdp, Eigen::Matrix<T, 3, branchn> Footdv, Eigen::Matrix<T, branchn, 1> Contact)
{
  foot_dp = Footdp;
  foot_dv = Footdv;
  for (int i = 0; i < branchn; i++)
  {
    // 支撑腿前后运动的置信空间，设置不变高度
    int flag = 0;
    if (Contact(i, 0) >= 1)
    {
      Eigen::Matrix<T,3,1> the_dp=foot_p0.block(0,i,3,1)-_x.block(foot_p_id+3 * i, 0,3,1);
      if (foot_contact[i] < 0) // 落地必有一次触发
      {
        flag = 1;
      }
      else if (the_dp.norm() > FOOT_SLIP_MIN) // 滑动触发
      {
        flag = 1;
      }
    }
    if (flag)
    {
      foot_p0.block(0,i,3,1) = _x.block(foot_p_id+3 * i, 0,3,1);
    }
    foot_contact[i] = Contact(i, 0);
  }
}
template <typename T, int branchn>
void CJW_Leg_Body_KF<T, branchn>::SetLegObs(T *Footdp, T *Footdv, T *Contact)
{
  for (int i = 0; i < branchn; i++)
  {
    for (int j = 0; j < 3; j++)
    {
      foot_dp(j, i) = Footdp[i * 3 + j];
      foot_dv(j, i) = Footdv[i * 3 + j];
    }
    // 支撑腿前后运动的置信空间，设置不变高度
    int flag = 0;
    if (Contact[i] >= 1)
    {
      Eigen::Matrix<T,3,1> the_dp=foot_p0.block(0,i,3,1)-_x.block(foot_p_id+3 * i, 0,3,1);
      if (foot_contact[i] < 1) // 落地必有一次触发
      {
        flag = 1;
      }
      else if (the_dp.norm() > FOOT_SLIP_MIN) // 滑动触发
      {
        flag = 1;
      }
    }
    if (flag)
    {
      for(int j=0;j<3;j++)
      {
        foot_p0(j,i) = _x(foot_p_id+3 * i+j, 0);
      }
    }
    foot_contact[i] = Contact[i];
  }
}

/****************************************IMU**********************************/
// 设置IMU观测的的加速度数据
template <typename T, int branchn>
void CJW_Leg_Body_KF<T, branchn>::SetIMUAcc(Eigen::Matrix<T, 3, 1> Acc)
{
  imu_acc.block(0, 0, 3, 1) = Acc;
}
template <typename T, int branchn>
void CJW_Leg_Body_KF<T, branchn>::SetIMUAcc(T *Acc)
{
  for (int i = 0; i < 3; i++)
  {
    imu_acc(i, 0) = Acc[i];
  }
}

/****************************************外部观测******************************/
// 设置外部观测数据的位置
template <typename T, int branchn>
void CJW_Leg_Body_KF<T, branchn>::SetExtObs(Eigen::Matrix<T, 3, 1> p, Eigen::Matrix<T, 3, 1> v, Eigen::Matrix<T, 3, 1> a)
{
  ex_position = p;
  ex_velocity = v;
  ex_acceleration = a;
}
template <typename T, int branchn>
void CJW_Leg_Body_KF<T, branchn>::SetExtObs(T *p, T *v, T *a)
{
  for (int i = 0; i < 3; i++)
  {
    ex_position(i, 0) = p[i];
    ex_velocity(i, 0) = v[i];
    ex_acceleration(i, 0) = a[i];
  }
}

/***************************************MIT卡尔曼滤波方法*************************/
// 初始化状态估计结构参数
template <typename T, int branchn>
void CJW_Leg_Body_KF<T, branchn>::SetType(int Type)
{
  KF_Type = Type;
  T_predict = T(0);
  // 初始化卡尔曼滤波结构体大小：MIT算法
  XN = 3 * branchn + 6;
  UN = 3;
  if (KF_Type == EKF_EIL)
  {
    ZN = 7 * branchn + 6;
  }
  else
  {
    ZN = 7 * branchn;
  }
  ParaN = 8;
  // 重置卡尔曼矩阵大小和数据
  KFResize(XN, UN, ZN);
  Para.resize(ParaN, 1);
  Para.setZero();
  // 状态变量id设置
  body_p_id = 0;
  body_v_id = 3;
  foot_p_id = 6;
}
// 滤波初始化
template <typename T, int branchn>
void CJW_Leg_Body_KF<T, branchn>::Initial(T dt)
{
  T_predict = dt;
  // 初始化状态方程和协方差
  _A.setIdentity();
  _A.block(0, 3, 3, 3) = T_predict * Eigen::Matrix<T, 3, 3>::Identity();
  _B.setZero();
  _B.block(0, 0, 3, 3) = 0.5 * T_predict * T_predict * Eigen::Matrix<T, 3, 3>::Identity();
  _B.block(3, 0, 3, 3) = T_predict * Eigen::Matrix<T, 3, 3>::Identity();
  _C.setZero();
  for (int i = 0; i < branchn; i++)
  {
    _C.block(3 * i, 0, 3, 3) = Eigen::Matrix<T, 3, 3>::Identity();
    _C.block(3 * i, 6 + 3 * i, 3, 3) = T(-1) * Eigen::Matrix<T, 3, 3>::Identity();
    _C.block(3 * (branchn + i), 3, 3, 3) = Eigen::Matrix<T, 3, 3>::Identity();
    _C(6 * branchn + i, 8 + 3 * i) = T(1);
  }
  if (KF_Type == EKF_EIL)
  {
    _C.block(7 * branchn, 0, 3, 3) = Eigen::Matrix<T, 3, 3>::Identity();
    _C.block(7 * branchn + 3, 3, 3, 3) = Eigen::Matrix<T, 3, 3>::Identity();
  }
  _P.setIdentity();
  _P = T(100) * _P;
  _Q.setIdentity();
  _R.setIdentity();
  // 其他参数
  foot_p0.setZero();
}
// 腿式里程计加外部定位卡尔曼滤波方法滤波
template <typename T, int branchn>
void CJW_Leg_Body_KF<T, branchn>::EstimateOnce(void)
{
  // 载入噪声参数
  T Kpb = Para(0);
  T Kvb = Para(1);
  T Kpf = Para(2);
  T Kpbf = Para(3);
  T Kpfv = Para(4);
  T Kfz = Para(5);
  T Kpex = Para(6);
  T Kvex = Para(7);
  // 基准噪声
  _Q.setIdentity();
  _Q.block(0, 0, 3, 3) = (T_predict / 20.f) * Kpb * Eigen::Matrix<T, 3, 3>::Identity();
  _Q.block(3, 3, 3, 3) = (T_predict * 9.8f / 20.f) * Kvb * Eigen::Matrix<T, 3, 3>::Identity();
  _Q.block(6, 6, 3 * branchn, 3 * branchn) = T_predict * Kpf * Eigen::Matrix<T, 3 * branchn, 3 * branchn>::Identity();
  _R.setIdentity();
  _R.block(0, 0, 3 * branchn, 3 * branchn) = Kpbf * Eigen::Matrix<T, 3 * branchn, 3 * branchn>::Identity();
  _R.block(3 * branchn, 3 * branchn, 3 * branchn, 3 * branchn) = Kpfv * Eigen::Matrix<T, 3 * branchn, 3 * branchn>::Identity();
  _R.block(6 * branchn, 6 * branchn, branchn, branchn) = Kfz * Eigen::Matrix<T, branchn, branchn>::Identity();
  if (KF_Type == EKF_EIL)
  {
    _R.block(7 * branchn, 7 * branchn, 3, 3) = Kpex * Eigen::Matrix<T, 3, 3>::Identity();
    _R.block(7 * branchn + 3, 7 * branchn + 3, 3, 3) = Kvex * Eigen::Matrix<T, 3, 3>::Identity();
  }
  // 机身状态缓存
  Eigen::Matrix<T, 3, 1> p0 = _x.block(0, 0, 3, 1);
  Eigen::Matrix<T, 3, 1> v0 = _x.block(3, 0, 3, 1);
  // 载入IMU
  _u = imu_acc;
  if (KF_Type == EKF_EIL)
  {
    _z.block(7 * branchn, 0, 3, 1) = ex_position;
    _z.block(7 * branchn + 3, 0, 3, 1) = ex_velocity;
  }
  // 分支状态观测数值优化
  for (int i = 0; i < branchn; i++)
  {
    // 摆动支撑置信区间
    T trust = fmin(foot_contact[i], T(1));
    T thek0 = 1 + SWING_NOISE * (1 - trust);
    // 腿分支计算缓存数据
    int i1 = 3 * i;
    int qindex = 6 + i1;
    int rindex1 = i1;
    int rindex2 = 3 * branchn + i1;
    int rindex3 = 6 * branchn + i;
    // 优化噪声
    _Q.block(qindex, qindex, 3, 3) = thek0 * _Q.block(qindex, qindex, 3, 3);
    _R.block(rindex1, rindex1, 3, 3) = _R.block(rindex1, rindex1, 3, 3);
    _R.block(rindex2, rindex2, 3, 3) = thek0 * _R.block(rindex2, rindex2, 3, 3);
    _R(rindex3, rindex3) = thek0 * _R(rindex3, rindex3);
    // 优化观测值
    _z.block(rindex1, 0, 3, 1) = -foot_dp.block(0, i, 3, 1);
    _z.block(rindex2, 0, 3, 1) = (1.0f - trust) * v0 + trust * (-foot_dv.block(0, i, 3, 1));
    _z(rindex3, 0) = (1.0f - trust) * (p0(2) + foot_dp(2, i)) + FOOT_PLANE * trust * foot_p0(2, i);
  }
  // 标准卡尔曼滤波
  KFonce();
  // 防止计算结果出现不可逆的情况
  if (_P.block(0, 0, 2, 2).determinant() > T(0.000001))
  {
    _P.block(0, 2, 2, 4 + 3 * branchn).setZero();
    _P.block(2, 0, 4 + 3 * branchn, 2).setZero();
    _P.block(0, 0, 2, 2) /= T(10);
  }
}

/***********************************************************************************/
template class CJW_Leg_Body_KF<double, 4>;
/***********************************************************************************/





/****************************************轮腿机器人机身位置速度卡尔曼滤波**************************/
/*******************************机身状态函数***********************/
// 设置轮式里程计机身的当前位置
template <typename T, int branchn>
void CJW_WheelLeg_Body_KF<T, branchn>::SetBodyp_w(Eigen::Matrix<T, 3, 1> Bodyp)
{
  this->_x.block(body_p_w_id, 0, 3, 1) = Bodyp;
}
template <typename T, int branchn>
void CJW_WheelLeg_Body_KF<T, branchn>::SetBodyp_w(T *Bodyp)
{
  for (int i = 0; i < 3; i++)
  {
    this->_x(body_p_w_id + i, 0) = Bodyp[i];
  }
}
// 设置轮式里程计机身的当前速度
template <typename T, int branchn>
void CJW_WheelLeg_Body_KF<T, branchn>::SetBodyv_w(Eigen::Matrix<T, 3, 1> Bodyv)
{
  this->_x.block(body_v_w_id, 0, 3, 1) = Bodyv;
}
template <typename T, int branchn>
void CJW_WheelLeg_Body_KF<T, branchn>::SetBodyv_w(T *Bodyv)
{
  for (int i = 0; i < 3; i++)
  {
    this->_x(body_v_w_id + i, 0) = Bodyv[i];
  }
}
// 得到轮式里程计机身的当前位置
template <typename T, int branchn>
Eigen::Matrix<T, 3, 1> CJW_WheelLeg_Body_KF<T, branchn>::GetBodyp_w()
{
  return this->_x.block(body_p_w_id, 0, 3, 1);
}
template <typename T, int branchn>
void CJW_WheelLeg_Body_KF<T, branchn>::GetBodyp_w(T *Bodyp)
{
  for (int i = 0; i < 3; i++)
  {
    Bodyp[i] = this->_x(body_p_w_id + i, 0);
  }
}
// 得到轮式里程计机身的当前速度
template <typename T, int branchn>
Eigen::Matrix<T, 3, 1> CJW_WheelLeg_Body_KF<T, branchn>::GetBodyv_w()
{
  return this->_x.block(body_v_w_id, 0, 3, 1);
}
template <typename T, int branchn>
void CJW_WheelLeg_Body_KF<T, branchn>::GetBodyv_w(T *Bodyv)
{
  for (int i = 0; i < 3; i++)
  {
    Bodyv[i] = this->_x(body_v_w_id + i, 0);
  }
}
/****************************************轮式里程计*****************************/
// 设置轮式观测的当前状态
template <typename T, int branchn>
void CJW_WheelLeg_Body_KF<T, branchn>::SetWheelObs(Eigen::Matrix<T, 3, branchn> Wheeldv)
{
  wheel_dv = Wheeldv;
}
template <typename T, int branchn>
void CJW_WheelLeg_Body_KF<T, branchn>::SetWheelObs(T *Wheeldv)
{
  for (int i = 0; i < branchn; i++)
  {
    for (int j = 0; j < 3; j++)
    {
      wheel_dv(j, i) = Wheeldv[i * 3 + j];
    }
  }
}
// 设置状态估计机身的轮腿累加当前位置速度，机身的轮里程计当前位置速度，分支腿里程计末端位置
template <typename T, int branchn>
void CJW_WheelLeg_Body_KF<T, branchn>::SetBodyWheelLegState(Eigen::Matrix<T, 3, 1> Bodyp, Eigen::Matrix<T, 3, 1> Bodyv,
                                                            Eigen::Matrix<T, 3, 1> Bodyp_w, Eigen::Matrix<T, 3, 1> Bodyv_w,
                                                            Eigen::Matrix<T, 3, branchn> Footp_l)
{
  this->_x.block(this->body_p_id, 0, 3, 1) = Bodyp;
  this->_x.block(this->body_v_id, 0, 3, 1) = Bodyv;
  this->_x.block(body_p_w_id, 0, 3, 1) = Bodyp_w;
  this->_x.block(body_v_w_id, 0, 3, 1) = Bodyv_w;
  this->_x.block(this->foot_p_id, 0, 3, branchn) = Footp_l;
}
template <typename T, int branchn>
void CJW_WheelLeg_Body_KF<T, branchn>::SetBodyWheelLegState(T *Bodyp, T *Bodyv, T *Bodyp_w, T *Bodyv_w, T *Footp_l)
{
  for (int i = 0; i < 3; i++)
  {
    this->_x(this->body_p_id + i, 0) = Bodyp[i];
    this->_x(this->body_v_id + i, 0) = Bodyv[i];
    this->_x(body_p_w_id + i, 0) = Bodyp_w[i];
    this->_x(body_v_w_id + i, 0) = Bodyv_w[i];
  }
  for (int i = 0; i < branchn; i++)
  {
    for (int j = 0; j < 3; j++)
    {
      this->_x(this->foot_p_id + j + 3 * i, 0) = Footp_l[i * 3 + j];
    }
  }
}
// 得到状态估计机身的轮腿累加当前位置速度，机身的轮里程计当前位置速度，分支腿里程计末端位置
template <typename T, int branchn>
void CJW_WheelLeg_Body_KF<T, branchn>::GetBodyWheelLegState(Eigen::Matrix<T, 3, 1> &Bodyp, Eigen::Matrix<T, 3, 1> &Bodyv,
                                                            Eigen::Matrix<T, 3, 1> &Bodyp_w, Eigen::Matrix<T, 3, 1> &Bodyv_w,
                                                            Eigen::Matrix<T, 3, branchn> &Footp_l)
{
  Bodyp = this->_x.block(this->body_p_id, 0, 3, 1);
  Bodyv = this->_x.block(this->body_v_id, 0, 3, 1);
  Bodyp_w = this->_x.block(body_p_w_id, 0, 3, 1);
  Bodyv_w = this->_x.block(body_v_w_id, 0, 3, 1);
  Footp_l = this->_x.block(this->foot_p_id, 0, 3, branchn);
}
template <typename T, int branchn>
void CJW_WheelLeg_Body_KF<T, branchn>::GetBodyWheelLegState(T *Bodyp, T *Bodyv, T *Bodyp_w, T *Bodyv_w, T *Footp_l)
{
  for (int i = 0; i < 3; i++)
  {
    Bodyp[i] = this->_x(this->body_p_id + i, 0);
    Bodyv[i] = this->_x(this->body_v_id + i, 0);
    Bodyp_w[i] = this->_x(body_p_w_id + i, 0);
    Bodyv_w[i] = this->_x(body_v_w_id + i, 0);
  }
  for (int i = 0; i < branchn; i++)
  {
    for (int j = 0; j < 3; j++)
    {
      Footp_l[i * 3 + j] = this->_x(this->foot_p_id + j + 3 * i, 0);
    }
  }
}
/***************************************轮腿里程计加外部定位卡尔曼滤波方法*************************/
// 设置滤波方法
template <typename T, int branchn>
void CJW_WheelLeg_Body_KF<T, branchn>::SetType(int Type)
{
  this->KF_Type = Type;
  this->T_predict = T(0);
  // 初始化卡尔曼滤波结构体大小
  this->XN = 3 * branchn + 12;
  this->UN = 3;
  if (this->KF_Type == EKF_EIL)
  {
    this->ZN = 6 + 9 * branchn;
  }
  else
  {
    this->ZN = 9 * branchn;
  }
  this->ParaN = 10;
  // 重置卡尔曼矩阵大小和数据
  this->KFResize(this->XN, this->UN, this->ZN);
  this->Para.resize(this->ParaN, 1);
  this->Para.setZero();
  // 状态变量id设置
  this->body_p_id = 0;
  this->body_v_id = 3;
  body_p_w_id = 6;
  body_v_w_id = 9;
  this->foot_p_id = 9;
  //观测变量ID
  foot_body_p_id = 0;
  foot_body_v_id = 3*branchn;
  wheel_body_v_id = 6*branchn;
}
// 滤波初始化
template <typename T, int branchn>
void CJW_WheelLeg_Body_KF<T, branchn>::Initial(T dt)
{
  this->T_predict = dt;
  Eigen::Matrix<T, 3, 3> I0 = Eigen::Matrix<T, 3, 3>::Identity();
  // 初始化状态方程和协方差
  this->_A.setIdentity();
  this->_A.block(0, 3, 3, 3) = dt * I0;
  this->_B.setZero();
  this->_B.block(0, 0, 3, 3) = 0.5 * dt * dt * I0;
  this->_B.block(3, 0, 3, 3) = dt * I0;
  this->_C.setZero();
  for (int ii = 0; ii < branchn; ii++)
  {
    int ii3=3*ii;
    this->_C.block(ii3, this->body_p_id, 3, 3) = I0;
    //this->_C.block(ii3, body_p_w_id, 3, 3) = -1.0*I0;
    this->_C.block(ii3, this->foot_p_id + ii3, 3, 3) = -1.0*I0;
    this->_C.block(foot_body_v_id+ii3, this->body_v_id, 3, 3) = I0;
    this->_C.block(foot_body_v_id+ii3, body_v_w_id, 3, 3) = -1.0*I0;
    this->_C.block(wheel_body_v_id+ii3, body_v_w_id, 3, 3) = I0;
  }
  if (this->KF_Type == EKF_EIL)
  {
    this->_C.block(this->ZN-6, 0, 3, 3) = I0;
    this->_C.block(this->ZN-3, 3, 3, 3) = I0;
  }
  this->_P.setIdentity();
  this->_P = T(100.0) * this->_P;
  this->_Q.setIdentity();
  this->_R.setIdentity();
}
//腿式里程计加外部定位卡尔曼滤波方法滤波
template <typename T, int branchn>
void CJW_WheelLeg_Body_KF<T, branchn>::EstimateOnce(void)
{
  // 载入噪声参数
  T Kpb = this->Para(0);
  T Kvb = this->Para(1);
  T Kpbw = this->Para(2);
  T Kvbw = this->Para(3);
  T Kpf = this->Para(4);
  T Kpbf = this->Para(5);
  T Kvbfl = this->Para(6);
  T Kvbfw = this->Para(7);
  T Kpex = this->Para(8);
  T Kvex = this->Para(9);
  //基准噪声
  Eigen::Matrix<T, 3, 3> I0 = Eigen::Matrix<T, 3, 3>::Identity();
  this->_Q.setIdentity();
  this->_Q.block(this->body_p_id, this->body_p_id, 3, 3) = this->T_predict*Kpb * I0;
  this->_Q.block(this->body_v_id, this->body_v_id, 3, 3) = this->T_predict*Kvb * I0;
  this->_Q.block(body_p_w_id, body_p_w_id, 3, 3) = this->T_predict*Kpbw * I0;
  this->_Q.block(body_v_w_id, body_v_w_id, 3, 3) = this->T_predict*Kvbw * I0;
  this->_Q.block(this->foot_p_id, this->foot_p_id, 3 * branchn, 3 * branchn) = this->T_predict*Kpf * Eigen::Matrix<T, 3 * branchn, 3 * branchn>::Identity();
  this->_R.setIdentity();
  this->_R.block(foot_body_p_id, foot_body_p_id, 3 * branchn, 3 * branchn) = Kpbf * Eigen::Matrix<T, 3 * branchn, 3 * branchn>::Identity();
  this->_R.block(foot_body_v_id, foot_body_v_id, 3 * branchn, 3 * branchn) = Kvbfl * Eigen::Matrix<T, 3 * branchn, 3 * branchn>::Identity();
  this->_R.block(wheel_body_v_id, wheel_body_v_id, 3 * branchn, 3 * branchn) = Kvbfw * Eigen::Matrix<T, 3 * branchn, 3 * branchn>::Identity();
  if (this->KF_Type == EKF_EIL)
  {
    this->_R.block(this->ZN-6, this->ZN-6, 3, 3) = Kpex * I0;
    this->_R.block(this->ZN-3, this->ZN-3, 3, 3) = Kvex * I0;
  }
  // 机身状态缓存
  Eigen::Matrix<T, 3, 1> p0 = this->_x.block(this->body_p_id, 0, 3, 1);
  Eigen::Matrix<T, 3, 1> v0 = this->_x.block(this->body_v_id, 0, 3, 1);
  Eigen::Matrix<T, 3, 1> p0w = this->_x.block(body_p_w_id, 0, 3, 1);
  Eigen::Matrix<T, 3, 1> v0w = this->_x.block(body_v_w_id, 0, 3, 1);
  // 载入IMU
  this->_u = this->imu_acc;
  // 分支状态观测数值优化
  for (int i = 0; i < branchn; i++)
  {
    // 摆动支撑置信区间
    T trust = fmin(this->foot_contact[i], T(1));
    T foot_dz=this->_x(this->foot_p_id+3*i+2,0)-this->foot_p0(2,i)*FOOT_PLANE;
    T thek0 = 1 + SWING_NOISE * (1 - trust);
    T thek1 = 1 + SWING_NOISE * (1 - trust*(1-std::exp(-LEG_Z_CONFIDENCE*foot_dz*foot_dz)));
    Eigen::Matrix<T, 3, 3> theIk=thek0*I0;
    theIk(2,2)= thek1;
    // 腿分支计算缓存数据
    int i3 = 3 * i;
    int q_f = this->foot_p_id   + i3;
    int r_fp = foot_body_p_id   + i3;
    int r_fv = foot_body_v_id   + i3;
    int r_wv = wheel_body_v_id  + i3;
    // 优化噪声
    this->_Q.block(q_f, q_f, 3, 3) = theIk * this->_Q.block(q_f, q_f, 3, 3);
    this->_R.block(r_fp, r_fp, 3, 3) = this->_R.block(r_fp, r_fp, 3, 3);
    this->_R.block(r_fv, r_fv, 3, 3) = thek0 * this->_R.block(r_fv, r_fv, 3, 3);
    this->_R.block(r_wv, r_wv, 3, 3) = thek0 * this->_R.block(r_wv, r_wv, 3, 3);
    // 优化观测值
    this->_z.block(r_fp, 0, 3, 1) = -this->foot_dp.block(0, i, 3, 1);
    this->_z.block(r_fv, 0, 3, 1) = (1.0f - trust) * (v0 - v0w) + trust * (-this->foot_dv.block(0, i, 3, 1));
    this->_z.block(r_wv, 0, 3, 1) = (1.0f - trust) * v0w + trust * (-wheel_dv.block(0, i, 3, 1));
  }
  if (this->KF_Type == EKF_EIL)
  {
    this->_z.block(this->ZN-6, 0, 3, 1) = this->ex_position;
    this->_z.block(this->ZN-3, 0, 3, 1) = this->ex_velocity;
  }
  // 标准卡尔曼滤波
  this->KFonce();
  // 防止计算结果出现大数据的情况
  /*if ((this->_P.block(this->body_p_id, this->body_p_id, 2, 2).determinant() > T(0.000001))
      ||(this->_P.block(body_p_w_id, body_p_w_id, 2, 2).determinant() > T(0.000001)))
  {
    this->_P.block(this->body_p_id, this->body_p_id+2, 2, this->XN-2-this->body_p_id).setZero();
    this->_P.block(this->body_p_id+2, this->body_p_id, this->XN-2-this->body_p_id, 2).setZero();
    this->_P.block(body_p_w_id, body_p_w_id+2, 2, this->XN-2-body_p_w_id).setZero();
    this->_P.block(body_p_w_id+2, body_p_w_id, this->XN-2-body_p_w_id, 2).setZero();
    this->_P.block(this->body_p_id, this->body_p_id, 2, 2) /= T(10);
    this->_P.block(body_p_w_id, body_p_w_id, 2, 2) /= T(10);
  }*/
  int NSS=2;
  if ((this->_P.block(this->body_p_id, this->body_p_id, NSS, NSS).determinant() > T(0.000001))
      ||(this->_P.block(body_p_w_id, body_p_w_id, NSS, NSS).determinant() > T(0.000001)))
  {
    this->_P.block(this->body_p_id, this->body_p_id+NSS, NSS, this->XN-NSS-this->body_p_id).setZero();
    this->_P.block(this->body_p_id+NSS, this->body_p_id, this->XN-NSS-this->body_p_id, NSS).setZero();
    this->_P.block(body_p_w_id, body_p_w_id+NSS, NSS, this->XN-NSS-body_p_w_id).setZero();
    this->_P.block(body_p_w_id+NSS, body_p_w_id, this->XN-NSS-body_p_w_id, NSS).setZero();
    this->_P.block(this->body_p_id, this->body_p_id, NSS, NSS) /= T(10);
    this->_P.block(body_p_w_id, body_p_w_id, NSS, NSS) /= T(10);
  }
}

/***********************************************************************************/
template class CJW_WheelLeg_Body_KF<double, 4>;
/***********************************************************************************/