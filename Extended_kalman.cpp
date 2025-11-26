#define _CRT_SECURE_NO_WARNINGS
float H_MAG[18] = { 1,0,0,0,0,0, 0,1,0,0,0,0, 0,0,1,0,0,0 };
#include "Extended_kalman.h"
#include "stdio.h"
#include "math.h"
#include "string.h"

#ifndef PI
#define PI 3.14159265358979f
#endif

// 传感器数量
const static char sensors = 1;

/*========================== 全局变量定义 ==============================*/
// EKF状态变量
static float P_mag_ekf_temp[sensors][16], P_EKF[sensors][16], Q_GYR[sensors][16];
static float Racc_static[sensors][9], Racc_use[sensors][9], Rmag_static[sensors][9], Rmag_use[sensors][9];
static float k_first[sensors] = { 0 };

static int InitNumber[sensors] = { 1 };
static float MeanACC[sensors][3] = { {0, 0, 0} };
static float MeanMAG[sensors][3] = { {0, 0, 0} };
static float MAG_InitialModule[sensors] = { 0 };

// 磁力计相关状态变量
static float MAG_Z0[sensors] = { 0 };
static float MAG_Zm[sensors] = { 0 };
static float MAG_Fa0[sensors][3] = { {0, 0, 0} };
static float MAG_Fam[sensors][3] = { {0, 0, 0} };
static float MAG_MO0[sensors][3] = { {0, 0, 0} };
static float MAG_MOm[sensors][3] = { {0, 0, 0} };
static float MAG_OMOm[sensors][3] = { {0, 0, 0} };
static float MAG_OFam[sensors][3] = { {0, 0, 0} };
static char MAG_LastTrue[sensors] = { 0 };
static char MAG_Change[sensors] = { 0 };

// 改进的磁力计融合新增变量
static float MAG_MOm_xy[sensors][2] = { {0, 0} }; // 假北状态下的水平分量基准
static float MAG_ReferenceStrength[sensors] = { 0 }; // 磁场强度基准值
static float mag_declination_reference[sensors] = { 0 }; // 动态磁偏角基准
static float MAG_InitialModule_m[sensors] = { 0 }; // 假北状态下的磁场模值基准

// 固定阈值参数结构
typedef struct {
    float jiaom;        // 姿态角度差阈值
    float jiaom_mag;    // 磁场方向差阈值  
    float bx;           // X轴分量差阈值
    float by;           // Y轴分量差阈值
    float bz;           // Z轴分量差阈值
    float module;       // 模值差异阈值
    float strength;     // 磁场强度偏差阈值
    int trust_count;    // 连续可信帧数要求
} FixedThreshold;

static FixedThreshold fixed_threshold[sensors] = {
    {0.12f, 0.035f, 0.07f, 0.07f, 0.085f, 0.095f, 0.15f, 5}
};

static int trust_frame_count[sensors] = { 0 };
static int true_north_condition_count[sensors] = { 0 };
static char current_state[sensors][16] = { "true_north" }; // 当前状态
static float learning_rate[sensors] = { 0.01f }; // 学习率
static float declination_threshold[sensors] = { 0.035f }; // 磁偏角阈值
static float strength_threshold[sensors] = { 0.15f }; // 磁场强度阈值

// ACC EKF变量
static float X_ACC_EKF[sensors][3] = { {0, 0, 0} };
static float P_ACC_EKF[sensors][9] = { {0} };
static float Q_ACC_EKF[sensors][9] = { {0} };
static float R_ACC_EKF[sensors][9] = { {0} };
static float ERR_ACC_EKF[sensors] = { 0 };

// MAG EKF变量
static float X_MAG_EKF[sensors][6] = { {0} };
static float P_MAG_EKF[sensors][36] = { {0} };
static float Q_MAG_EKF[sensors][36] = { {0} };
static float R_MAG_EKF[sensors][9] = { {0} };
static float A_MAG_EKF[sensors][36] = { {0} };
static float H_MAG_EKF[sensors][18] = { {0} };

// 静态检测和运动检测
static int gyrStatic[sensors] = { 0 };
static int accStatic[sensors] = { 0 };
static float accLast[sensors][3] = { {0, 0, 0} };
static float QuitReturnCtrlNumber[sensors] = { 0 };
static int QuickRecover[sensors] = { 0 };

// 单位矩阵
static float eye3[9] = { 0 };
static float eye4[16] = { 0 };
static float eye6[36] = { 0 };
static float jiaomThresold[sensors] = { 0 };
static float jiaommagThresold[sensors] = { 0 };

// 环境状态
static float EnvironmentMagStatus[sensors] = { 0 };
static float RecoverSignalEKF[sensors] = { 2.1f };
static char sensorQuitReturnCtrl[sensors] = { 0 };
static float sensorQuitReturnState[sensors] = { -2.1f };
static bool sensorQuitTimeReturnCtrl[sensors] = { false };

static int MagModelStatus = 1;
static int SpecialModelStatus = 1;

// 时间变量
static float t_a[sensors] = { 0 };
static float t_m[sensors] = { 0 };
static float t_quit[sensors] = { 0 };
static float quat_EKF[sensors][4] = { {0} };
static float quat_Filter[sensors][4] = { {0} };
static int gyr_vec_flag[sensors] = { 0 };
static int count[sensors] = { 0 };

/*========================== 工具函数实现 ==============================*/

// 限制函数
static float clamp(float val, float min_val, float max_val) {
    if (val < min_val) return min_val;
    if (val > max_val) return max_val;
    return val;
}

// 矩阵转置
static void matrix_trans(float* mata, char r, char c, float* matb) {
    char i, j;
    for (i = 0; i < r; i++) {
        for (j = 0; j < c; j++) {
            matb[j * r + i] = mata[i * c + j];
        }
    }
}

// 矩阵加法
static void matrix_add(float* mata, float* matb, char r, char c, float* matc) {
    char i, j;
    for (i = 0; i < r; i++) {
        for (j = 0; j < c; j++) {
            matc[i * c + j] = mata[i * c + j] + matb[i * c + j];
        }
    }
}

// 矩阵减法
static void matrix_sub(float* mata, float* matb, char r, char c, float* matc) {
    char i, j;
    for (i = 0; i < r; i++) {
        for (j = 0; j < c; j++) {
            matc[i * c + j] = mata[i * c + j] - matb[i * c + j];
        }
    }
}

// 矩阵乘法
static void matrix_mul(float* mat1, char row1, char columns1, float* mat2, char columns2, float* mat) {
    char i, j, k;
    for (i = 0; i < row1; i++) {
        for (j = 0; j < columns2; j++) {
            mat[i * columns2 + j] = 0.0;
            for (k = 0; k < columns1; k++) {
                mat[i * columns2 + j] += mat1[i * columns1 + k] * mat2[k * columns2 + j];
            }
        }
    }
}

// 4x4矩阵乘法
static void matrix_4_mul(float* mat1, float* mat2, float* mat) {
    char i, j, k;
    for (i = 0; i < 4; i++) {
        for (j = 0; j < 4; j++) {
            mat[i * 4 + j] = 0.0;
            for (k = 0; k < 4; k++) {
                mat[i * 4 + j] += mat1[i * 4 + k] * mat2[k * 4 + j];
            }
        }
    }
}

// 3x3矩阵求逆
static void matrix_3_inv(float* mata, float* mata_inv) {
    float mata_det;
    mata_det = mata[0] * mata[4] * mata[8] + mata[1] * mata[5] * mata[6] + mata[2] * mata[3] * mata[7]
        - mata[2] * mata[4] * mata[6] - mata[5] * mata[7] * mata[0] - mata[8] * mata[1] * mata[3];

    if (fabsf(mata_det) < 1e-10f) {
        // 行列式接近0，返回单位矩阵
        mata_inv[0] = 1.0f; mata_inv[1] = 0.0f; mata_inv[2] = 0.0f;
        mata_inv[3] = 0.0f; mata_inv[4] = 1.0f; mata_inv[5] = 0.0f;
        mata_inv[6] = 0.0f; mata_inv[7] = 0.0f; mata_inv[8] = 1.0f;
        return;
    }

    mata_inv[0] = (mata[4] * mata[8] - mata[5] * mata[7]) / mata_det;
    mata_inv[1] = (mata[2] * mata[7] - mata[1] * mata[8]) / mata_det;
    mata_inv[2] = (mata[1] * mata[5] - mata[2] * mata[4]) / mata_det;
    mata_inv[3] = (mata[5] * mata[6] - mata[3] * mata[8]) / mata_det;
    mata_inv[4] = (mata[0] * mata[8] - mata[2] * mata[6]) / mata_det;
    mata_inv[5] = (mata[2] * mata[3] - mata[0] * mata[5]) / mata_det;
    mata_inv[6] = (mata[3] * mata[7] - mata[4] * mata[6]) / mata_det;
    mata_inv[7] = (mata[1] * mata[6] - mata[0] * mata[7]) / mata_det;
    mata_inv[8] = (mata[0] * mata[4] - mata[1] * mata[3]) / mata_det;
}

// 四元数转方向余弦矩阵
static void matrix_Cbn(float* quat, float* Cbn) {
    Cbn[0] = quat[0] * quat[0] + quat[1] * quat[1] - quat[2] * quat[2] - quat[3] * quat[3];
    Cbn[1] = 2 * (quat[1] * quat[2] - quat[0] * quat[3]);
    Cbn[2] = 2 * (quat[1] * quat[3] + quat[0] * quat[2]);
    Cbn[3] = 2 * (quat[1] * quat[2] + quat[0] * quat[3]);
    Cbn[4] = quat[0] * quat[0] - quat[1] * quat[1] + quat[2] * quat[2] - quat[3] * quat[3];
    Cbn[5] = 2 * (quat[2] * quat[3] - quat[0] * quat[1]);
    Cbn[6] = 2 * (quat[1] * quat[3] - quat[0] * quat[2]);
    Cbn[7] = 2 * (quat[2] * quat[3] + quat[0] * quat[1]);
    Cbn[8] = quat[0] * quat[0] - quat[1] * quat[1] - quat[2] * quat[2] + quat[3] * quat[3];
}

// 平方和
static float SquaresSum(float* vec, unsigned int len) {
    float q_len = 0;
    for (unsigned int i = 0; i < len; i++) {
        q_len += vec[i] * vec[i];
    }
    return q_len;
}

// 模值
static float Module(float* vec, unsigned int len) {
    return sqrtf(SquaresSum(vec, len));
}

// 归一化
static char Norm(float* vec, unsigned int len) {
    float q_len = Module(vec, len);
    if (q_len < 1e-10f) {
        return 0;
    }
    for (unsigned int i = 0; i < len; i++) {
        vec[i] /= q_len;
    }
    return 1;
}

// 安全反余弦
static float ACos(float value) {
    if (value >= 1.0f) return 0.0f;
    else if (value <= -1.0f) return PI;
    else {
        return acosf(value);
    }
}

// 数组复制
static char Copy(float* out, float* in, unsigned int len) {
    for (unsigned int i = 0; i < len; i++) {
        out[i] = in[i];
    }
    return 1;
}

/*========================== 改进的磁力计融合函数 ==============================*/

// 计算磁偏角
static float calculate_mag_declination(char sensor, float* acc_T, float* m_n) {
    (void)acc_T; // 未使用参数
    (void)sensor; // 未使用参数

    // 地理北向在水平面上的投影
    float north_vec[2] = { 1.0f, 0.0f };

    // 磁场向量在水平面上的投影
    float mag_horizontal[2] = { m_n[0], m_n[1] };
    float mag_horizontal_norm = sqrtf(mag_horizontal[0] * mag_horizontal[0] +
        mag_horizontal[1] * mag_horizontal[1]);

    if (mag_horizontal_norm < 1e-6f) {
        return PI; // 磁场水平分量为0时，默认磁偏角超阈值
    }

    // 归一化磁场水平分量
    mag_horizontal[0] /= mag_horizontal_norm;
    mag_horizontal[1] /= mag_horizontal_norm;

    // 计算磁偏角
    float dot_product = north_vec[0] * mag_horizontal[0] + north_vec[1] * mag_horizontal[1];
    float cross_product = north_vec[0] * mag_horizontal[1] - north_vec[1] * mag_horizontal[0];

    float mag_declination = acosf(clamp(dot_product, -1.0f, 1.0f));

    // 修正角度方向
    if (cross_product < 0) {
        mag_declination = -mag_declination;
    }

    return mag_declination;
}

// 计算磁场强度偏差
static float calculate_mag_strength_bias(char sensor, float current_strength) {
    if (MAG_ReferenceStrength[sensor] < 1e-6f) {
        return 1.0f;
    }

    float relative_bias = fabsf(current_strength - MAG_ReferenceStrength[sensor]) /
        MAG_ReferenceStrength[sensor];
    return fminf(relative_bias, 1.0f);
}

// 固定阈值可信度判断
static char fixed_threshold_check(char sensor, float jiaom, float jiaom_mag,
    float bx_diff, float by_diff, float bz_diff,
    float module_diff, float strength_bias) {
    int passed_conditions = 0;
    int total_conditions = 0;

    FixedThreshold* thresh = &fixed_threshold[sensor];

    if (!isnan(jiaom)) {
        total_conditions++;
        if (jiaom <= thresh->jiaom) {
            passed_conditions++;
        }
    }

    if (!isnan(jiaom_mag)) {
        total_conditions++;
        if (jiaom_mag <= thresh->jiaom_mag) {
            passed_conditions++;
        }
    }

    if (!isnan(bx_diff)) {
        total_conditions++;
        if (bx_diff <= thresh->bx) {
            passed_conditions++;
        }
    }

    if (!isnan(by_diff)) {
        total_conditions++;
        if (by_diff <= thresh->by) {
            passed_conditions++;
        }
    }

    if (!isnan(bz_diff)) {
        total_conditions++;
        if (bz_diff <= thresh->bz) {
            passed_conditions++;
        }
    }

    if (!isnan(module_diff)) {
        total_conditions++;
        if (module_diff <= thresh->module) {
            passed_conditions++;
        }
    }

    if (!isnan(strength_bias)) {
        total_conditions++;
        if (strength_bias <= thresh->strength) {
            passed_conditions++;
        }
    }

    char key_conditions_ok = 1;
    // 关键条件检查
    if (!isnan(bx_diff) && bx_diff > thresh->bx) {
        key_conditions_ok = 0;
    }
    if (!isnan(by_diff) && by_diff > thresh->by) {
        key_conditions_ok = 0;
    }
    if (!isnan(jiaom_mag) && jiaom_mag > thresh->jiaom_mag) {
        key_conditions_ok = 0;
    }

    char trust_flag = 0;
    if (total_conditions > 0) {
        float pass_ratio = (float)passed_conditions / total_conditions;
        trust_flag = (pass_ratio >= 0.6f) && key_conditions_ok;
    }

    if (trust_flag) {
        trust_frame_count[sensor]++;
    }
    else {
        trust_frame_count[sensor] = 0;
    }

    return trust_flag;
}

// 真北->假北切换检测
static char check_switch_to_fake_north(char sensor, float jiao0_2, float jiao_mag_2,
    float bz2, float mag0_ss) {
    char switch_to_fake_north = 0;

    // 如果当前是真北状态但条件不再满足，考虑切换到假北
    if (strcmp(current_state[sensor], "true_north") == 0) {
        char true_north_conditions[2] = {
            (jiao0_2 < 0.15f) && (jiao_mag_2 < 0.08f) &&
            (fabsf(bz2 - MAG_Z0[sensor]) < 0.0725f) &&
            (fabsf(mag0_ss - MAG_InitialModule[sensor]) < 0.07f),

            (jiao0_2 < 0.1f) && (fabsf(bz2 - MAG_Z0[sensor]) < 0.05f) &&
            (fabsf(mag0_ss - MAG_InitialModule[sensor]) < 0.05f)
        };

        // 如果所有真北条件都不满足
        if (!true_north_conditions[0] && !true_north_conditions[1]) {
            switch_to_fake_north = 1;
        }
    }

    return switch_to_fake_north;
}

// 假北->真北切换检测  
static char check_switch_to_true_north(char sensor, float jiao0_2, float jiao_mag_2,
    float bz2, float mag0_ss, int MagModelStatus) {
    char switch_to_true_north = 0;

    // 如果当前是假北状态但真北条件重新满足
    if (strcmp(current_state[sensor], "fake_north") == 0) {
        char true_north_conditions =
            (((jiao0_2 < 0.18f) && (jiao_mag_2 < 0.15f) &&
                (fabsf(bz2 - MAG_Z0[sensor]) < 0.08f) &&
                (fabsf(mag0_ss - MAG_InitialModule[sensor]) < 0.08f)) ||
                ((jiao0_2 < 0.1f) && (fabsf(bz2 - MAG_Z0[sensor]) < 0.05f) &&
                    (fabsf(mag0_ss - MAG_InitialModule[sensor]) < 0.05f))) &&
            (mag0_ss < 0.5f) && (mag0_ss > 0.15f) && MagModelStatus == 1;

        // 额外要求：连续多帧满足真北条件
        if (true_north_conditions) {
            true_north_condition_count[sensor]++;
            if (true_north_condition_count[sensor] >= fixed_threshold[sensor].trust_count) {
                switch_to_true_north = 1;
                true_north_condition_count[sensor] = 0;
            }
        }
        else {
            true_north_condition_count[sensor] = 0;
        }
    }

    return switch_to_true_north;
}

/*========================== EKF初始化函数 ==============================*/

static void ACCEKF_init(char sensor) {
    for (int i = 0; i < 9; i++) {
        P_ACC_EKF[sensor][i] = 0;
        Q_ACC_EKF[sensor][i] = 0;
        R_ACC_EKF[sensor][i] = 0;
    }
    ERR_ACC_EKF[sensor] = 0;
    P_ACC_EKF[sensor][0] = 1e-3f; P_ACC_EKF[sensor][4] = 1e-3f; P_ACC_EKF[sensor][8] = 1e-3f;
    Q_ACC_EKF[sensor][0] = 3e-1f; Q_ACC_EKF[sensor][4] = 3e-1f; Q_ACC_EKF[sensor][8] = 3e-1f;
    R_ACC_EKF[sensor][0] = 1e+3f; R_ACC_EKF[sensor][4] = 1e+3f; R_ACC_EKF[sensor][8] = 1e+3f;
}

static void MAGEKF_init(char sensor) {
    for (int i = 0; i < 36; i++) {
        A_MAG_EKF[sensor][i] = 0;
        P_MAG_EKF[sensor][i] = 0;
        Q_MAG_EKF[sensor][i] = 0;
        if (i < 9) {
            R_MAG_EKF[sensor][i] = 0;
        }
        if (i < 18) {
            H_MAG_EKF[sensor][i] = 0;
        }
    }
    A_MAG_EKF[sensor][0] = 1; A_MAG_EKF[sensor][7] = 1; A_MAG_EKF[sensor][14] = 1;
    A_MAG_EKF[sensor][21] = 1; A_MAG_EKF[sensor][28] = 1; A_MAG_EKF[sensor][35] = 1;
    P_MAG_EKF[sensor][0] = 1e-3f; P_MAG_EKF[sensor][7] = 1e-3f; P_MAG_EKF[sensor][14] = 1e-3f;
    P_MAG_EKF[sensor][21] = 1e-3f; P_MAG_EKF[sensor][28] = 1e-3f; P_MAG_EKF[sensor][35] = 1e-3f;
    Q_MAG_EKF[sensor][0] = 1e-2f; Q_MAG_EKF[sensor][7] = 1e-2f; Q_MAG_EKF[sensor][14] = 1e-2f;
    Q_MAG_EKF[sensor][21] = 1e-2f; Q_MAG_EKF[sensor][28] = 1e-2f; Q_MAG_EKF[sensor][35] = 1e-2f;
    R_MAG_EKF[sensor][0] = 1e+2f; R_MAG_EKF[sensor][4] = 1e+2f; R_MAG_EKF[sensor][8] = 1e+2f;
    H_MAG_EKF[sensor][0] = 1; H_MAG_EKF[sensor][4] = 1; H_MAG_EKF[sensor][8] = 1;
    H_MAG_EKF[sensor][9] = 1; H_MAG_EKF[sensor][13] = 1; H_MAG_EKF[sensor][17] = 1;
}

static void EYE_init(void) {
    for (int i = 0; i < 36; i++) {
        eye6[i] = 0;
        if (i < 9) {
            eye3[i] = 0;
        }
        if (i < 16) {
            eye4[i] = 0;
        }
    }
    eye3[0] = 1; eye3[4] = 1; eye3[8] = 1;
    eye4[0] = 1; eye4[5] = 1; eye4[10] = 1; eye4[15] = 1;
    eye6[0] = 1; eye6[7] = 1; eye6[14] = 1;
    eye6[21] = 1; eye6[28] = 1; eye6[35] = 1;
}

static void ekf_init(char sensor) {
    for (int i = 0; i < 16; i++) {
        P_EKF[sensor][i] = 0;
        Q_GYR[sensor][i] = 0;
        if (i < 9) {
            Racc_static[sensor][i] = 0;
            Racc_use[sensor][i] = 0;
            Rmag_static[sensor][i] = 0;
            Rmag_use[sensor][i] = 0;
        }
    }
    P_EKF[sensor][0] = 1e-1f; P_EKF[sensor][5] = 1e-1f; P_EKF[sensor][10] = 1e-1f; P_EKF[sensor][15] = 1e-1f;
    Q_GYR[sensor][0] = 1e-2f; Q_GYR[sensor][5] = 1e-2f; Q_GYR[sensor][10] = 1e-2f; Q_GYR[sensor][15] = 1e-2f;
    Racc_static[sensor][0] = 1e+2f; Racc_static[sensor][4] = 1e+2f; Racc_static[sensor][8] = 1e+2f;
    Rmag_static[sensor][0] = 8e+2f; Rmag_static[sensor][4] = 8e+2f; Rmag_static[sensor][8] = 8e+2f;

    ACCEKF_init(sensor);
    MAGEKF_init(sensor);
    EYE_init();

    // 初始化阈值
    jiaomThresold[sensor] = 0.15f;
    jiaommagThresold[sensor] = 0.08f;
}

/*========================== 四元数运算函数 ==============================*/

// 四元数矩阵
static void w_matrix(float* quat, float* q_matrix, int mode, int matT) {
    if (mode == 1) {
        if (matT == 1) {
            q_matrix[0] = quat[0]; q_matrix[1] = quat[1]; q_matrix[2] = quat[2]; q_matrix[3] = quat[3];
            q_matrix[4] = -quat[1]; q_matrix[5] = quat[0]; q_matrix[6] = -quat[3]; q_matrix[7] = quat[2];
            q_matrix[8] = -quat[2]; q_matrix[9] = quat[3]; q_matrix[10] = quat[0]; q_matrix[11] = -quat[1];
            q_matrix[12] = -quat[3]; q_matrix[13] = -quat[2]; q_matrix[14] = quat[1]; q_matrix[15] = quat[0];
        }
        else {
            q_matrix[0] = quat[0]; q_matrix[1] = -quat[1]; q_matrix[2] = -quat[2]; q_matrix[3] = -quat[3];
            q_matrix[4] = quat[1]; q_matrix[5] = quat[0]; q_matrix[6] = quat[3]; q_matrix[7] = -quat[2];
            q_matrix[8] = quat[2]; q_matrix[9] = -quat[3]; q_matrix[10] = quat[0]; q_matrix[11] = quat[1];
            q_matrix[12] = quat[3]; q_matrix[13] = quat[2]; q_matrix[14] = -quat[1]; q_matrix[15] = quat[0];
        }
    }
    else {
        if (matT == 1) {
            q_matrix[0] = quat[0]; q_matrix[1] = quat[1]; q_matrix[2] = quat[2]; q_matrix[3] = quat[3];
            q_matrix[4] = -quat[1]; q_matrix[5] = quat[0]; q_matrix[6] = quat[3]; q_matrix[7] = -quat[2];
            q_matrix[8] = -quat[2]; q_matrix[9] = -quat[3]; q_matrix[10] = quat[0]; q_matrix[11] = quat[1];
            q_matrix[12] = -quat[3]; q_matrix[13] = quat[2]; q_matrix[14] = -quat[1]; q_matrix[15] = quat[0];
        }
        else {
            q_matrix[0] = quat[0]; q_matrix[1] = -quat[1]; q_matrix[2] = -quat[2]; q_matrix[3] = -quat[3];
            q_matrix[4] = quat[1]; q_matrix[5] = quat[0]; q_matrix[6] = -quat[3]; q_matrix[7] = quat[2];
            q_matrix[8] = quat[2]; q_matrix[9] = quat[3]; q_matrix[10] = quat[0]; q_matrix[11] = -quat[1];
            q_matrix[12] = quat[3]; q_matrix[13] = -quat[2]; q_matrix[14] = quat[1]; q_matrix[15] = quat[0];
        }
    }
}

/*========================== 外部接口函数 ==============================*/

void EKF_EnableSensorQuitReturn(void) {
    for (char i = 0; i < sensors; i++) {
        sensorQuitReturnCtrl[i] = 0x01;
    }
}

void EKF_ChangeMagModel(int modelsign) {
    if ((modelsign > 0) && (modelsign <= 3)) {
        MagModelStatus = modelsign;
        if (modelsign == 2) {
            for (char i = 0; i < sensors; i++) {
                jiaomThresold[i] = 0.15f;
                jiaommagThresold[i] = 0.08f;
            }
        }
        else {
            for (char i = 0; i < sensors; i++) {
                jiaomThresold[i] = 0.15f;
                jiaommagThresold[i] = 0.08f;
            }
        }
    }
    else {
        MagModelStatus = 1;
        for (char i = 0; i < sensors; i++) {
            jiaomThresold[i] = 0.15f;
            jiaommagThresold[i] = 0.08f;
        }
    }
}

void EKF_ChangeSpecialModel(int modelsign) {
    if ((modelsign > 0) && (modelsign <= 2)) {
        SpecialModelStatus = modelsign;
    }
    else {
        SpecialModelStatus = 1;
    }
}

/*========================== 主Kalman滤波函数 ==============================*/

uint8_t KalmanFilter(float* acc_mcu, float* gyr_mcu, float* mag_mcu, float dtTime, float* q_out) {
    char sensor = 0;
    float acc[3] = { acc_mcu[0], acc_mcu[1], acc_mcu[2] };
    float gyr[3] = { gyr_mcu[0], gyr_mcu[1], gyr_mcu[2] };
    float mag[3] = { mag_mcu[0], mag_mcu[1], mag_mcu[2] };

    float acc_EKF[3], mag_EKF[3];
    float accErr, gyrErr, g = 9.8f;
    float q_temp[4], q_temp2[4], acc_T[3], P_temp[16], X_temp[3];
    float k_a, k_m, float_temp, jiao_acc, jiaom_mag, jiaom, Ryu, bx, by, bz;
    float dist_ACC1, dist_ACC2, dist_ACC3;
    float mid1[36], mid2[36], mid3[36], mid4[36], mid5[36], PRY[3], A[16], H1[12], K1[12], H2[12], K2[12];
    float A_ACC[9];
    float Cbn[9], m_n[3], m_n2[3];
    char acc_update = 0;

    // 新增变量声明
    float jiao0_2, jiao_mag_2, jiao0, jiao_mag;

    if (dtTime <= 0.0f || dtTime > 0.005f)
    {
        dtTime = 0.005f;
        // （可选）可以在这里加入调试信息，提示 dtTime 异常
        // printf("Warning: Abnormal dtTime, reset to default: %f\n", DT_DEFAULT);
    }

    if ((acc[0] == 0 && acc[1] == 0 && acc[2] == 0) || (mag[0] == 0 && mag[1] == 0 && mag[2] == 0)) {
        return 0;
    }

    float mag0_s, mag0_ss, acc0_s, acc0_ss, gyr0_s, mag0_s_EKF, mag0_ss_EKF;
    mag0_s = mag[0] * mag[0] + mag[1] * mag[1] + mag[2] * mag[2];
    mag0_ss = sqrtf(mag0_s);
    acc0_s = acc[0] * acc[0] + acc[1] * acc[1] + acc[2] * acc[2];
    acc0_ss = sqrtf(acc0_s);
    gyr0_s = gyr[0] * gyr[0] + gyr[1] * gyr[1] + gyr[2] * gyr[2];
    gyrErr = sqrtf(gyr0_s);
    accErr = fabsf(acc0_ss - g);

    if (acc0_ss != 0) {
        acc[0] = acc[0] / acc0_ss;
        acc[1] = acc[1] / acc0_ss;
        acc[2] = acc[2] / acc0_ss;
    }

    if (mag0_s != 0) {
        mag[0] = mag[0] / mag0_ss;
        mag[1] = mag[1] / mag0_ss;
        mag[2] = mag[2] / mag0_ss;
    }

    // 初始化阶段
    if (k_first[sensor] <= 0.1f) {
        if ((acc[0] == 0 && acc[1] == 0 && acc[2] == 0) || (mag[0] == 0 && mag[1] == 0 && mag[2] == 0)) {
            return 0;
        }

        if (accErr < 0.5f && gyrErr < 0.5f) {
            k_first[sensor] += dtTime;
        }
        else {
            k_first[sensor] = 0;
            InitNumber[sensor] = 1;
            MAG_Fa0[sensor][0] = 0; MAG_Fa0[sensor][1] = 0; MAG_Fa0[sensor][2] = 0;
            MeanACC[sensor][0] = 0; MeanACC[sensor][1] = 0; MeanACC[sensor][2] = 0;
            MeanMAG[sensor][0] = 0; MeanMAG[sensor][1] = 0; MeanMAG[sensor][2] = 0;
            MAG_InitialModule[sensor] = 0;
        }

        ekf_init(sensor);

        for (int i = 0; i < 3; i++) {
            MeanACC[sensor][i] = ((InitNumber[sensor] - 1) * MeanACC[sensor][i] + acc[i]) / InitNumber[sensor];
            MeanMAG[sensor][i] = ((InitNumber[sensor] - 1) * MeanMAG[sensor][i] + mag[i]) / InitNumber[sensor];
            MAG_InitialModule[sensor] = ((InitNumber[sensor] - 1) * MAG_InitialModule[sensor] + mag0_ss) / InitNumber[sensor];
        }

        Copy(X_ACC_EKF[sensor], acc_mcu, 3);
        X_MAG_EKF[sensor][0] = mag_mcu[0];
        X_MAG_EKF[sensor][1] = mag_mcu[1];
        X_MAG_EKF[sensor][2] = mag_mcu[2];
        X_MAG_EKF[sensor][3] = 0;
        X_MAG_EKF[sensor][4] = 0;
        X_MAG_EKF[sensor][5] = 0;

        PRY[0] = atan2f(MeanACC[sensor][1], MeanACC[sensor][2]);
        PRY[1] = atan2f(-MeanACC[sensor][0], sqrtf(MeanACC[sensor][1] * MeanACC[sensor][1] + MeanACC[sensor][2] * MeanACC[sensor][2]));
        PRY[2] = atan2f((MeanMAG[sensor][2] * sinf(PRY[0]) - MeanMAG[sensor][1] * cosf(PRY[0])),
            (MeanMAG[sensor][0] * cosf(PRY[1]) + (MeanMAG[sensor][1] * sinf(PRY[0]) + MeanMAG[sensor][2] * cosf(PRY[0])) * sinf(PRY[1])));

        quat_EKF[sensor][0] = cosf(0.5f * PRY[0]) * cosf(0.5f * PRY[1]) * cosf(0.5f * PRY[2]) + sinf(0.5f * PRY[0]) * sinf(0.5f * PRY[1]) * sinf(0.5f * PRY[2]);
        quat_EKF[sensor][1] = sinf(0.5f * PRY[0]) * cosf(0.5f * PRY[1]) * cosf(0.5f * PRY[2]) - cosf(0.5f * PRY[0]) * sinf(0.5f * PRY[1]) * sinf(0.5f * PRY[2]);
        quat_EKF[sensor][2] = cosf(0.5f * PRY[0]) * sinf(0.5f * PRY[1]) * cosf(0.5f * PRY[2]) + sinf(0.5f * PRY[0]) * cosf(0.5f * PRY[1]) * sinf(0.5f * PRY[2]);
        quat_EKF[sensor][3] = cosf(0.5f * PRY[0]) * cosf(0.5f * PRY[1]) * sinf(0.5f * PRY[2]) - sinf(0.5f * PRY[0]) * sinf(0.5f * PRY[1]) * cosf(0.5f * PRY[2]);

        matrix_Cbn(quat_EKF[sensor], Cbn);
        matrix_mul(Cbn, 3, 3, MeanMAG[sensor], 1, m_n);
        bx = sqrtf(m_n[0] * m_n[0] + m_n[1] * m_n[1]);
        by = 0;
        bz = m_n[2];

        MAG_Fa0[sensor][0] = bx;
        MAG_Fa0[sensor][1] = by;
        MAG_Fa0[sensor][2] = bz;

        float_temp = Module(MAG_Fa0[sensor], 3);
        for (int i = 0; i < 3; i++) {
            MAG_Fa0[sensor][i] /= float_temp;
            MAG_MO0[sensor][i] = MAG_Fa0[sensor][i];
            MAG_Fam[sensor][i] = MAG_Fa0[sensor][i];
            MAG_MOm[sensor][i] = m_n[i];
        }
        MAG_Z0[sensor] = bz;
        MAG_LastTrue[sensor] = 1;

        Copy(quat_Filter[sensor], quat_EKF[sensor], 4);
        Copy(q_out, quat_EKF[sensor], 4);

        t_a[sensor] = 0.005f * sensor;
        t_m[sensor] = 0.005f * sensor + 0.015f;

        InitNumber[sensor]++;
        EnvironmentMagStatus[sensor] = 2.1f;
        Copy(accLast[sensor], acc_mcu, 3);
        QuitReturnCtrlNumber[sensor] = 0;
        QuickRecover[sensor] = 0;

        q_out[0] += EnvironmentMagStatus[sensor];
        return 1;
    }

    // 陀螺仪积分预测
    if (gyrErr > 0.05f && dtTime < 0.01f) {
        // ACC EKF预测
        A_ACC[0] = 1; A_ACC[1] = -gyr[2] * dtTime; A_ACC[2] = gyr[1] * dtTime;
        A_ACC[3] = gyr[2] * dtTime; A_ACC[4] = 1; A_ACC[5] = -gyr[0] * dtTime;
        A_ACC[6] = -gyr[1] * dtTime; A_ACC[7] = gyr[0] * dtTime; A_ACC[8] = 1;

        matrix_mul(X_ACC_EKF[sensor], 1, 3, A_ACC, 3, X_temp);
        Copy(X_ACC_EKF[sensor], X_temp, 3);

        matrix_trans(A_ACC, 3, 3, mid1);
        matrix_mul(mid1, 3, 3, P_ACC_EKF[sensor], 3, mid2);
        matrix_mul(mid2, 3, 3, A_ACC, 3, mid3);
        matrix_add(mid3, Q_ACC_EKF[sensor], 3, 3, P_ACC_EKF[sensor]);

        // MAG EKF预测
        gyr_vec_flag[sensor] = 0;
        A_MAG_EKF[sensor][0] = A_ACC[0];
        A_MAG_EKF[sensor][1] = A_ACC[1];
        A_MAG_EKF[sensor][2] = A_ACC[2];
        A_MAG_EKF[sensor][6] = A_ACC[3];
        A_MAG_EKF[sensor][7] = A_ACC[4];
        A_MAG_EKF[sensor][8] = A_ACC[5];
        A_MAG_EKF[sensor][12] = A_ACC[6];
        A_MAG_EKF[sensor][13] = A_ACC[7];
        A_MAG_EKF[sensor][14] = A_ACC[8];

        matrix_mul(X_MAG_EKF[sensor], 1, 3, A_ACC, 3, X_temp);
        Copy(X_MAG_EKF[sensor], X_temp, 3);

        matrix_trans(A_MAG_EKF[sensor], 6, 6, mid1);
        matrix_mul(mid1, 6, 6, P_MAG_EKF[sensor], 6, mid2);
        matrix_mul(mid2, 6, 6, A_MAG_EKF[sensor], 6, mid3);
        matrix_add(mid3, Q_MAG_EKF[sensor], 6, 6, P_MAG_EKF[sensor]);

        // 四元数预测
        A[0] = 1; A[1] = -0.5f * gyr[0] * dtTime; A[2] = -0.5f * gyr[1] * dtTime; A[3] = -0.5f * gyr[2] * dtTime;
        A[4] = 0.5f * gyr[0] * dtTime; A[5] = 1; A[6] = 0.5f * gyr[2] * dtTime; A[7] = -0.5f * gyr[1] * dtTime;
        A[8] = 0.5f * gyr[1] * dtTime; A[9] = -0.5f * gyr[2] * dtTime; A[10] = 1; A[11] = 0.5f * gyr[0] * dtTime;
        A[12] = 0.5f * gyr[2] * dtTime; A[13] = 0.5f * gyr[1] * dtTime; A[14] = -0.5f * gyr[0] * dtTime; A[15] = 1;

        matrix_mul(A, 4, 4, quat_EKF[sensor], 1, q_temp);
        Copy(quat_EKF[sensor], q_temp, 4);
        if (Norm(quat_EKF[sensor], 4) == 0) { return 0; }

        matrix_mul(A, 4, 4, quat_Filter[sensor], 1, q_temp);
        Copy(quat_Filter[sensor], q_temp, 4);
        if (Norm(quat_Filter[sensor], 4) == 0) { return 0; }

        matrix_4_mul(A, P_EKF[sensor], mid1);
        matrix_trans(A, 4, 4, mid2);
        matrix_4_mul(mid1, mid2, mid3);
        matrix_add(mid3, Q_GYR[sensor], 4, 4, P_EKF[sensor]);

        // 限制协方差矩阵
        float_temp = 100.0f;
        for (char i = 0; i < 16; i++) {
            if (P_EKF[sensor][i] > float_temp) {
                P_EKF[sensor][i] = float_temp;
            }
            else if (P_EKF[sensor][i] < -float_temp) {
                P_EKF[sensor][i] = -float_temp;
            }
        }
    }

    // 加速度计更新
    t_a[sensor] = t_a[sensor] + dtTime;
    float maxErr;
    float K_ACC[9];

    if (t_a[sensor] > 0.03f) {
        t_a[sensor] = t_a[sensor] - 0.03f;
        acc_update = 1;
        Copy(accLast[sensor], acc_mcu, 3);

        // ACC EKF更新
        if (accErr < 2.0f) {
            float_temp = gyrErr * accErr + 1e-4f;
            maxErr = fmaxf(float_temp, ERR_ACC_EKF[sensor]);
            Copy(mid1, R_ACC_EKF[sensor], 9);
            mid1[0] *= maxErr;
            mid1[4] *= maxErr;
            mid1[8] *= maxErr;

            matrix_add(P_ACC_EKF[sensor], mid1, 3, 3, mid2);
            matrix_3_inv(mid2, mid3);
            matrix_mul(P_ACC_EKF[sensor], 3, 3, mid3, 3, K_ACC);
            ERR_ACC_EKF[sensor] = float_temp;

            mid1[0] = acc_mcu[0] - X_ACC_EKF[sensor][0];
            mid1[1] = acc_mcu[1] - X_ACC_EKF[sensor][1];
            mid1[2] = acc_mcu[2] - X_ACC_EKF[sensor][2];

            matrix_mul(K_ACC, 3, 3, mid1, 1, mid2);
            matrix_add(X_ACC_EKF[sensor], mid2, 3, 1, mid3);
            Copy(X_ACC_EKF[sensor], mid3, 3);

            matrix_sub(eye3, K_ACC, 3, 3, mid1);
            matrix_mul(mid1, 3, 3, P_ACC_EKF[sensor], 3, P_temp);
            Copy(P_ACC_EKF[sensor], P_temp, 9);
        }

        Copy(acc_EKF, X_ACC_EKF[sensor], 3);
        Norm(acc_EKF, 3);
    }

    // 计算加速度误差
    acc_T[0] = 2 * (quat_EKF[sensor][1] * quat_EKF[sensor][3] - quat_EKF[sensor][0] * quat_EKF[sensor][2]);
    acc_T[1] = 2 * (quat_EKF[sensor][2] * quat_EKF[sensor][3] + quat_EKF[sensor][0] * quat_EKF[sensor][1]);
    acc_T[2] = quat_EKF[sensor][0] * quat_EKF[sensor][0] - quat_EKF[sensor][1] * quat_EKF[sensor][1] - quat_EKF[sensor][2] * quat_EKF[sensor][2] + quat_EKF[sensor][3] * quat_EKF[sensor][3];

    if (acc0_ss > 0) {
        jiao_acc = ACos(acc_EKF[0] * acc_T[0] + acc_EKF[1] * acc_T[1] + acc_EKF[2] * acc_T[2]);
    }
    else {
        jiao_acc = 100.0f;
    }

    // 快速回归检测
    if ((acc_update == 1) || (sensorQuitReturnCtrl[sensor] == 1)) {
        if (accErr > 10.0f && gyrErr > 10.0f) {
            QuitReturnCtrlNumber[sensor] += 0.1f;
        }
        else if (accErr < 0.8f && gyrErr < 0.5f) {
            if (jiao_acc > 0.4f) {
                QuitReturnCtrlNumber[sensor] += 0.1f;
            }
            else {
                QuitReturnCtrlNumber[sensor] -= 0.02f;
            }
        }
        else if (accErr < 2.0f && gyrErr < 1.0f) {
            QuitReturnCtrlNumber[sensor] -= 0.02f;
        }

        if (QuitReturnCtrlNumber[sensor] < 0) {
            QuitReturnCtrlNumber[sensor] = 0;
        }

        if ((QuitReturnCtrlNumber[sensor] > 10.0f || sensorQuitReturnCtrl[sensor] == 1) && accErr < 0.8f && gyrErr < 0.5f) {
            QuickRecover[sensor] = 1;
        }
    }

    if (acc_update)
    {
        if (((accErr < 1.0f) && (gyrErr < 1.0f) && (jiao_acc < 0.5f)) || ((accErr < 0.5f) && (gyrErr < 0.2f))) {
            //Copy(acc_EKF, acc, 3);

            if ((accErr < 0.2f) && (gyrErr < 0.2f)) {
                k_a = 1 + 10 * fabsf(acc0_s - g * g);
            }
            else if ((accErr < 0.5f) && (gyrErr < 0.5f)) {
                k_a = 1 + 50 * fabsf(acc0_s - g * g);
            }
            else {
                k_a = 1 + 100 * fabsf(acc0_s - g * g) * (1 + jiao_acc * 20);
            }
            //			if ((accErr < 0.2f) && (gyrErr < 0.2f)) {
            //				k_a = 0.1+10 * fabsf(acc0_s - g * g);
            //			}
            //			else if ((accErr < 0.5f) && (gyrErr < 0.5f)) {
            //				k_a = 0.5 + 50 * fabsf(acc0_s - g * g);
            //			}
            //			else {
            //				k_a = 1 + 100 * fabsf(acc0_s - g * g) * (1 + jiao_acc * 20);
            //			}
            Racc_use[sensor][0] = k_a * Racc_static[sensor][0];
            Racc_use[sensor][4] = k_a * Racc_static[sensor][4];
            Racc_use[sensor][8] = k_a * Racc_static[sensor][8];

            H1[0] = -quat_EKF[sensor][2] * 2; H1[1] = quat_EKF[sensor][3] * 2; H1[2] = -quat_EKF[sensor][0] * 2; H1[3] = quat_EKF[sensor][1] * 2;
            H1[4] = quat_EKF[sensor][1] * 2; H1[5] = quat_EKF[sensor][0] * 2; H1[6] = quat_EKF[sensor][3] * 2; H1[7] = quat_EKF[sensor][2] * 2;
            H1[8] = quat_EKF[sensor][0] * 2; H1[9] = -quat_EKF[sensor][1] * 2; H1[10] = -quat_EKF[sensor][2] * 2; H1[11] = quat_EKF[sensor][3] * 2;
            matrix_mul(H1, 3, 4, P_EKF[sensor], 4, mid1);
            matrix_trans(H1, 3, 4, mid2);
            matrix_mul(mid1, 3, 4, mid2, 3, mid3);
            matrix_add(mid3, Racc_use[sensor], 3, 3, mid1);
            matrix_3_inv(mid1, mid3);
            matrix_mul(P_EKF[sensor], 4, 4, mid2, 3, mid1);
            matrix_mul(mid1, 4, 3, mid3, 3, K1);   //K1(4,3,k)=P_EKF(4,4,k)*H1(3,4,k)'/(H1(3,4,k)*P_EKF(4,4,k)*H1(3,4,k)'+Racc_use(3,3,k));

            mid1[0] = acc_EKF[0] - acc_T[0];
            mid1[1] = acc_EKF[1] - acc_T[1];
            mid1[2] = acc_EKF[2] - acc_T[2];
            dist_ACC1 = mid1[0] * mid1[0] + mid1[1] * mid1[1] + mid1[2] * mid1[2];
            float_temp = sqrtf(mid1[0] * mid1[0] + mid1[1] * mid1[1] + mid1[2] * mid1[2]);//修复力度关系 float_temp=2Rsin(Q_GYR/2),1对应每帧5.8度的修复力
            Ryu = 0.08;
            if (float_temp > Ryu) {
                mid1[0] = mid1[0] * Ryu / float_temp;
                mid1[1] = mid1[1] * Ryu / float_temp;
                mid1[2] = mid1[2] * Ryu / float_temp;
            }
            matrix_mul(K1, 4, 3, mid1, 1, mid2);
            matrix_add(quat_EKF[sensor], mid2, 4, 1, q_temp);     //quat_EKF(4,1,k)=quat_EKF(4,1,k)+K1(4,3,k)*(a(3,1,k)-Ha(3,1,k));
            if (Norm(q_temp, 4) == 0) { return 0; }
            acc_T[0] = 2 * (q_temp[1] * q_temp[3] - q_temp[0] * q_temp[2]);
            acc_T[1] = 2 * (q_temp[2] * q_temp[3] + q_temp[0] * q_temp[1]);
            acc_T[2] = q_temp[0] * q_temp[0] - q_temp[1] * q_temp[1] - q_temp[2] * q_temp[2] + q_temp[3] * q_temp[3];
            mid1[0] = acc_EKF[0] - acc_T[0];
            mid1[1] = acc_EKF[1] - acc_T[1];
            mid1[2] = acc_EKF[2] - acc_T[2];
            dist_ACC2 = mid1[0] * mid1[0] + mid1[1] * mid1[1] + mid1[2] * mid1[2];

            matrix_sub(quat_EKF[sensor], mid2, 4, 1, q_temp2);     //quat_EKF(4,1,k)=quat_EKF(4,1,k)-K1(4,3,k)*(a(3,1,k)-Ha(3,1,k));
            if (Norm(q_temp2, 4) == 0) { return 0; }
            acc_T[0] = 2 * (q_temp2[1] * q_temp2[3] - q_temp2[0] * q_temp2[2]);
            acc_T[1] = 2 * (q_temp2[2] * q_temp2[3] + q_temp2[0] * q_temp2[1]);
            acc_T[2] = q_temp2[0] * q_temp2[0] - q_temp2[1] * q_temp2[1] - q_temp2[2] * q_temp2[2] + q_temp2[3] * q_temp2[3];
            mid1[0] = acc_EKF[0] - acc_T[0];
            mid1[1] = acc_EKF[1] - acc_T[1];
            mid1[2] = acc_EKF[2] - acc_T[2];
            dist_ACC3 = mid1[0] * mid1[0] + mid1[1] * mid1[1] + mid1[2] * mid1[2];
            if (dist_ACC2 <= dist_ACC3) {
                if (dist_ACC1 > dist_ACC2) {
                    Copy(quat_EKF[sensor], q_temp, 4);
                    matrix_mul(K1, 4, 3, H1, 4, mid1);
                    matrix_sub(eye4, mid1, 4, 4, mid2);
                    matrix_4_mul(mid2, P_EKF[sensor], P_temp);  //P2(4,4,k)=(eye(4)-K1(4,3,k)*H1(3,4,k))*P_EKF(4,4,k);	
                    Copy(P_EKF[sensor], P_temp, 16);
                }
            }
            else {
                if (dist_ACC1 > dist_ACC3) {
                    Copy(quat_EKF[sensor], q_temp2, 4);
                    matrix_mul(K1, 4, 3, H1, 4, mid1);
                    matrix_add(eye4, mid1, 4, 4, mid2);
                    matrix_4_mul(mid2, P_EKF[sensor], P_temp);  //P2(4,4,k)=(eye(4)+K1(4,3,k)*H1(3,4,k))*P_EKF(4,4,k);	
                    Copy(P_EKF[sensor], P_temp, 16);
                }

            }
        }
    }

    // ========== 改进的磁力计融合部分 ==========
    t_m[sensor] = t_m[sensor] + dtTime;
    int flagg = 1;
    if (MagModelStatus == 2) { flagg = 0; }

    if (t_m[sensor] > 0.03f && flagg) {
        if (gyrErr > 0.05f) {
            // 计算磁场在导航系的投影
            matrix_Cbn(quat_EKF[sensor], Cbn);
            matrix_mul(Cbn, 3, 3, mag, 1, m_n2);
            float bx2 = sqrtf(m_n2[0] * m_n2[0] + m_n2[1] * m_n2[1]);
            float bz2 = m_n2[2];
            float Fam2[3] = { bx2, 0, bz2 };
            float Fam2_norm = Module(Fam2, 3);
            Fam2[0] /= Fam2_norm; Fam2[1] /= Fam2_norm; Fam2[2] /= Fam2_norm;

            jiao_mag_2 = fabsf(atan2f(m_n2[1], m_n2[0]));
            jiao0_2 = ACos(Fam2[0] * MAG_Fa0[sensor][0] + Fam2[2] * MAG_Fa0[sensor][2]);

            k_m = 1.0f;
            t_m[sensor] = t_m[sensor] - 0.03f;

            // MAG EKF更新 - 与MATLAB代码保持一致
            gyr_vec_flag[sensor] += 1;
            if (gyr_vec_flag[sensor] <= 2) {
                Rmag_use[sensor][0] = k_m * R_MAG_EKF[sensor][0];
                Rmag_use[sensor][4] = k_m * R_MAG_EKF[sensor][4];
                Rmag_use[sensor][8] = k_m * R_MAG_EKF[sensor][8];

                // H_MAG_EKF = [eye(3);eye(3)]
                float H_MAG_EKF_local[18] = { 1,0,0,0,0,0, 0,1,0,0,0,0, 0,0,1,0,0,0 };

                float H_MAG_EKF_T[18];
                matrix_trans(H_MAG_EKF_local, 6, 3, H_MAG_EKF_T);

                // K_MAG_EKF = self.P_MAG_EKF*H_MAG_EKF/(H_MAG_EKF'*self.P_MAG_EKF*H_MAG_EKF+self.Rmag_use);
                matrix_mul(H_MAG_EKF_T, 3, 6, P_MAG_EKF[sensor], 6, mid1); // H'*P
                matrix_mul(mid1, 3, 6, H_MAG_EKF_local, 3, mid2); // H'*P*H
                matrix_add(mid2, Rmag_use[sensor], 3, 3, mid3); // H'*P*H + R
                matrix_3_inv(mid3, mid4); // inv(H'*P*H + R)
                matrix_mul(P_MAG_EKF[sensor], 6, 6, H_MAG_EKF_local, 3, mid1); // P*H
                matrix_mul(mid1, 6, 3, mid4, 3, mid2); // K = P*H*inv(H'*P*H + R)

                // mid1 = self.X_MAG_EKF*H_MAG_EKF
                float mid5[3];
                matrix_mul(X_MAG_EKF[sensor], 1, 6, H_MAG_EKF_local, 3, mid5);

                // mid = mag_mcu-mid1
                float mid6[3] = {
                    mag_mcu[0] - mid5[0],
                    mag_mcu[1] - mid5[1],
                    mag_mcu[2] - mid5[2]
                };

                // self.X_MAG_EKF = self.X_MAG_EKF+mid*K_MAG_EKF'
                float mid7[6];
                matrix_mul(mid2, 6, 3, mid6, 1, mid7);
                for (int i = 0; i < 6; i++) {
                    X_MAG_EKF[sensor][i] += mid7[i];
                }

                // self.P_MAG_EKF = (eye(6)-K_MAG_EKF*H_MAG_EKF')*self.P_MAG_EKF
                matrix_mul(mid2, 6, 3, H_MAG_EKF_T, 6, mid1); // K*H'
                matrix_sub(eye6, mid1, 6, 6, mid3); // I - K*H'
                matrix_mul(mid3, 6, 6, P_MAG_EKF[sensor], 6, mid4);
                Copy(P_MAG_EKF[sensor], mid4, 36);
            }

            Copy(mag_EKF, X_MAG_EKF[sensor], 3);
            mag0_s_EKF = mag_EKF[0] * mag_EKF[0] + mag_EKF[1] * mag_EKF[1] + mag_EKF[2] * mag_EKF[2];
            mag0_ss_EKF = sqrtf(mag0_s_EKF);
            Norm(mag_EKF, 3);

            // 计算EKF滤波后的磁场在导航系的投影
            matrix_mul(Cbn, 3, 3, mag_EKF, 1, m_n);
            bx = sqrtf(m_n[0] * m_n[0] + m_n[1] * m_n[1]);
            bz = m_n[2];
            float Fam[3] = { bx, 0, bz };
            float Fam_norm = Module(Fam, 3);
            Fam[0] /= Fam_norm; Fam[1] /= Fam_norm; Fam[2] /= Fam_norm;

            jiao0 = ACos(Fam[0] * MAG_Fa0[sensor][0] + Fam[2] * MAG_Fa0[sensor][2]);
            jiao_mag = fabsf(atan2f(m_n[1], m_n[0]));
            char FixMAG = 0;
            char local_trust_flag = 0;

            // 环境状态判断
            if ((mag0_ss < 0.50f) && (mag0_ss > 0.15f)) {
                EnvironmentMagStatus[sensor] = 0; // 安全
            }
            else {
                EnvironmentMagStatus[sensor] = -2.1f; // 磁场异常
            }

            // 状态切换检测
            char switch_to_fake_north_flag = check_switch_to_fake_north(sensor, jiao0_2, jiao_mag_2, bz2, mag0_ss);
            char switch_to_true_north_flag = check_switch_to_true_north(sensor, jiao0_2, jiao_mag_2, bz2, mag0_ss, MagModelStatus);

            if (switch_to_fake_north_flag) {
                strcpy(current_state[sensor], "fake_north");
                MAG_LastTrue[sensor] = 0;
                Copy(MAG_MOm[sensor], m_n, 3);
                Copy(MAG_Fam[sensor], Fam, 3);
                Copy(MAG_OMOm[sensor], m_n2, 3);
                Copy(MAG_OFam[sensor], Fam2, 3);
                MAG_Zm[sensor] = bz2;
                X_MAG_EKF[sensor][0] = mag_mcu[0];
                X_MAG_EKF[sensor][1] = mag_mcu[1];
                X_MAG_EKF[sensor][2] = mag_mcu[2];
                X_MAG_EKF[sensor][3] = 0;
                X_MAG_EKF[sensor][4] = 0;
                X_MAG_EKF[sensor][5] = 0;

                for (int i = 0; i < 36; i++) {
                    P_MAG_EKF[sensor][i] = 0;
                }
                P_MAG_EKF[sensor][0] = 1e-3f; P_MAG_EKF[sensor][7] = 1e-3f; P_MAG_EKF[sensor][14] = 1e-3f;
                P_MAG_EKF[sensor][21] = 1e-3f; P_MAG_EKF[sensor][28] = 1e-3f; P_MAG_EKF[sensor][35] = 1e-3f;

                MAG_InitialModule_m[sensor] = mag0_ss_EKF;
                MAG_MOm_xy[sensor][0] = m_n[0];
                MAG_MOm_xy[sensor][1] = m_n[1];
                true_north_condition_count[sensor] = 0;

            }
            else if (switch_to_true_north_flag) {
                strcpy(current_state[sensor], "true_north");
                MAG_LastTrue[sensor] = 1;
                trust_frame_count[sensor] = 0;
                MAG_MOm[sensor][0] = m_n[0]; MAG_MOm[sensor][1] = m_n[1]; MAG_MOm[sensor][2] = m_n[2];
                MAG_InitialModule_m[sensor] = mag0_ss_EKF;
            }

            // 根据当前状态进行处理
            if (strcmp(current_state[sensor], "true_north") == 0) {
                FixMAG = 1;
                float MAG_MO[3];
                Copy(MAG_MO, MAG_MO0[sensor], 3);
                k_m = 1.0f + 50.0f * fabsf(mag0_s - MAG_InitialModule[sensor] * MAG_InitialModule[sensor]);

                // 真北状态下更新基准
                MAG_ReferenceStrength[sensor] = 0.98f * MAG_ReferenceStrength[sensor] + 0.02f * mag0_ss_EKF;
                float mag_declination = calculate_mag_declination(sensor, acc_T, m_n);
                if (!isnan(mag_declination) && !isinf(mag_declination)) {
                    mag_declination_reference[sensor] = 0.98f * mag_declination_reference[sensor] + 0.02f * mag_declination;
                }

            }
            else if (strcmp(current_state[sensor], "fake_north") == 0) {
                // 计算各种差异
                jiaom = ACos(Fam[0] * MAG_Fam[sensor][0] + Fam[2] * MAG_Fam[sensor][2]);
                float denom = Fam[0] * sqrtf(MAG_MOm[sensor][0] * MAG_MOm[sensor][0] + MAG_MOm[sensor][1] * MAG_MOm[sensor][1]) + 1e-6f;
                jiaom_mag = ACos((m_n[0] * MAG_MOm[sensor][0] + m_n[1] * MAG_MOm[sensor][1]) / denom);

                float bx_diff = fabsf(m_n[0] - MAG_MOm_xy[sensor][0]);
                float by_diff = fabsf(m_n[1] - MAG_MOm_xy[sensor][1]);
                float bz_diff = fabsf(bz2 - MAG_Zm[sensor]);
                float module_diff = fabsf(mag0_ss_EKF - MAG_InitialModule_m[sensor]);
                float strength_bias = calculate_mag_strength_bias(sensor, mag0_ss_EKF);
                float mag_declination = calculate_mag_declination(sensor, acc_T, m_n);
                float declination_abs = fabsf(mag_declination - mag_declination_reference[sensor]);

                // 使用固定阈值判断当前帧是否可信
                local_trust_flag = fixed_threshold_check(sensor, jiaom, jiaom_mag, bx_diff, by_diff, bz_diff, module_diff, strength_bias);

                // 双条件判断 - 与MATLAB代码完全一致
                if ((declination_abs <= declination_threshold[sensor] && strength_bias <= strength_threshold[sensor]) ||
                    (local_trust_flag && trust_frame_count[sensor] >= fixed_threshold[sensor].trust_count)) {

                    FixMAG = 1;
                    float MAG_MO[3];
                    Copy(MAG_MO, MAG_MOm[sensor], 3);
                    k_m = 1.0f + 50.0f * fabsf(mag0_s_EKF - MAG_InitialModule_m[sensor] * MAG_InitialModule_m[sensor]);

                    // 稳定假北状态下缓慢更新基准
                    MAG_ReferenceStrength[sensor] = (1.0f - learning_rate[sensor]) * MAG_ReferenceStrength[sensor] + learning_rate[sensor] * mag0_ss_EKF;
                    if (!isnan(mag_declination) && !isinf(mag_declination)) {
                        mag_declination_reference[sensor] = (1.0f - learning_rate[sensor]) * mag_declination_reference[sensor] + learning_rate[sensor] * mag_declination;
                    }

                    // 平滑更新假北基准
                    float smooth_factor = 0.0015f;
                    for (int i = 0; i < 3; i++) {
                        MAG_MOm[sensor][i] = (1.0f - smooth_factor) * MAG_MOm[sensor][i] + smooth_factor * m_n[i];
                        MAG_Fam[sensor][i] = (1.0f - smooth_factor) * MAG_Fam[sensor][i] + smooth_factor * Fam[i];
                        MAG_OMOm[sensor][i] = (1.0f - smooth_factor) * MAG_OMOm[sensor][i] + smooth_factor * m_n2[i];
                        MAG_OFam[sensor][i] = (1.0f - smooth_factor) * MAG_OFam[sensor][i] + smooth_factor * Fam2[i];
                    }
                    MAG_MOm_xy[sensor][0] = (1.0f - smooth_factor) * MAG_MOm_xy[sensor][0] + smooth_factor * m_n[0];
                    MAG_MOm_xy[sensor][1] = (1.0f - smooth_factor) * MAG_MOm_xy[sensor][1] + smooth_factor * m_n[1];

                    Norm(MAG_MOm[sensor], 3);
                    Norm(MAG_Fam[sensor], 3);
                    Norm(MAG_OMOm[sensor], 3);
                    Norm(MAG_OFam[sensor], 3);

                    MAG_Zm[sensor] = MAG_MOm[sensor][2];
                    MAG_InitialModule_m[sensor] = (1.0f - smooth_factor) * MAG_InitialModule_m[sensor] + smooth_factor * mag0_ss_EKF;
                }
                else {
                    // 不可信，重置假北相关状态
                    Copy(MAG_MOm[sensor], m_n, 3);
                    Copy(MAG_Fam[sensor], Fam, 3);
                    Copy(MAG_OMOm[sensor], m_n2, 3);
                    Copy(MAG_OFam[sensor], Fam2, 3);
                    MAG_Zm[sensor] = bz2;
                    X_MAG_EKF[sensor][0] = mag_mcu[0];
                    X_MAG_EKF[sensor][1] = mag_mcu[1];
                    X_MAG_EKF[sensor][2] = mag_mcu[2];
                    X_MAG_EKF[sensor][3] = 0;
                    X_MAG_EKF[sensor][4] = 0;
                    X_MAG_EKF[sensor][5] = 0;

                    for (int i = 0; i < 36; i++) {
                        P_MAG_EKF[sensor][i] = 0;
                    }
                    P_MAG_EKF[sensor][0] = 1e-3f; P_MAG_EKF[sensor][7] = 1e-3f; P_MAG_EKF[sensor][14] = 1e-3f;
                    P_MAG_EKF[sensor][21] = 1e-3f; P_MAG_EKF[sensor][28] = 1e-3f; P_MAG_EKF[sensor][35] = 1e-3f;

                    MAG_InitialModule_m[sensor] = mag0_ss_EKF;
                    MAG_MOm_xy[sensor][0] = m_n[0];
                    MAG_MOm_xy[sensor][1] = m_n[1];
                    trust_frame_count[sensor] = 0;
                }
            }

            // 磁力计修正
            if (FixMAG == 1 && (MagModelStatus == 1 || MagModelStatus == 2)) {
                float MAG_MO[3];
                if (strcmp(current_state[sensor], "true_north") == 0) {
                    Copy(MAG_MO, MAG_MO0[sensor], 3);
                }
                else {
                    Copy(MAG_MO, MAG_MOm[sensor], 3);
                }

                Rmag_use[sensor][0] = k_m * Rmag_static[sensor][0];
                Rmag_use[sensor][4] = k_m * Rmag_static[sensor][4];
                Rmag_use[sensor][8] = k_m * Rmag_static[sensor][8];

                // H2矩阵计算 - 与MATLAB代码一致
                H2[0] = 2 * (MAG_MO[0] * quat_EKF[sensor][0] + MAG_MO[1] * quat_EKF[sensor][3] - MAG_MO[2] * quat_EKF[sensor][2]);
                H2[1] = 2 * (MAG_MO[0] * quat_EKF[sensor][1] + MAG_MO[1] * quat_EKF[sensor][2] + MAG_MO[2] * quat_EKF[sensor][3]);
                H2[2] = 2 * (-MAG_MO[0] * quat_EKF[sensor][2] + MAG_MO[1] * quat_EKF[sensor][1] - MAG_MO[2] * quat_EKF[sensor][0]);
                H2[3] = 2 * (-MAG_MO[0] * quat_EKF[sensor][3] + MAG_MO[1] * quat_EKF[sensor][0] + MAG_MO[2] * quat_EKF[sensor][1]);

                H2[4] = 2 * (-MAG_MO[0] * quat_EKF[sensor][3] + MAG_MO[1] * quat_EKF[sensor][0] + MAG_MO[2] * quat_EKF[sensor][1]);
                H2[5] = 2 * (MAG_MO[0] * quat_EKF[sensor][2] - MAG_MO[1] * quat_EKF[sensor][1] + MAG_MO[2] * quat_EKF[sensor][0]);
                H2[6] = 2 * (MAG_MO[0] * quat_EKF[sensor][1] + MAG_MO[1] * quat_EKF[sensor][2] + MAG_MO[2] * quat_EKF[sensor][3]);
                H2[7] = 2 * (-MAG_MO[0] * quat_EKF[sensor][0] - MAG_MO[1] * quat_EKF[sensor][3] + MAG_MO[2] * quat_EKF[sensor][2]);

                H2[8] = 2 * (MAG_MO[0] * quat_EKF[sensor][2] - MAG_MO[1] * quat_EKF[sensor][1] + MAG_MO[2] * quat_EKF[sensor][0]);
                H2[9] = 2 * (MAG_MO[0] * quat_EKF[sensor][3] - MAG_MO[1] * quat_EKF[sensor][0] - MAG_MO[2] * quat_EKF[sensor][1]);
                H2[10] = 2 * (MAG_MO[0] * quat_EKF[sensor][0] + MAG_MO[1] * quat_EKF[sensor][3] - MAG_MO[2] * quat_EKF[sensor][2]);
                H2[11] = 2 * (MAG_MO[0] * quat_EKF[sensor][1] + MAG_MO[1] * quat_EKF[sensor][2] + MAG_MO[2] * quat_EKF[sensor][3]);

                // K2=self.P_EKF*H2'/(H2*self.P_EKF*H2'+self.Rmag_use)
                matrix_mul(H2, 3, 4, P_EKF[sensor], 4, mid1); // H2*P
                matrix_trans(H2, 3, 4, mid2); // H2'
                matrix_mul(mid1, 3, 4, mid2, 3, mid3); // H2*P*H2'
                matrix_add(mid3, Rmag_use[sensor], 3, 3, mid1); // H2*P*H2' + R
                matrix_3_inv(mid1, mid3); // inv(H2*P*H2' + R)
                matrix_mul(P_EKF[sensor], 4, 4, mid2, 3, mid1); // P*H2'
                matrix_mul(mid1, 4, 3, mid3, 3, K2); // K2 = P*H2'*inv(H2*P*H2' + R)

                // self.P_EKF=(eye4-K2*H2)*self.P_EKF
                matrix_mul(K2, 4, 3, H2, 4, mid1); // K2*H2
                matrix_sub(eye4, mid1, 4, 4, mid2); // I - K2*H2
                matrix_4_mul(mid2, P_EKF[sensor], P_temp);
                Copy(P_EKF[sensor], P_temp, 16);

                // mid1=self.MAG_MO*T_SY(self.quat_EKF)
                float mid8[3];
                matrix_mul(MAG_MO, 1, 3, Cbn, 3, mid8); // 相当于MATLAB中的 T_SY

                // mid2=self.mag_EKF-mid1
                float mid9[3] = {
                    mag_EKF[0] - mid8[0],
                    mag_EKF[1] - mid8[1],
                    mag_EKF[2] - mid8[2]
                };

                // MAG解耦算法 - 与MATLAB代码完全一致
                // q_temp=self.quat_EKF+mid2*K2'
                float mid10[4];
                matrix_mul(K2, 4, 3, mid9, 1, mid10);
                for (int i = 0; i < 4; i++) {
                    q_temp[i] = quat_EKF[sensor][i] + mid10[i];
                }
                Norm(q_temp, 4);

                // mid = w_matrix(self.quat_EKF,1)
                float mid11[16];
                w_matrix(quat_EKF[sensor], mid11, 1, 0); // mode=1, matT=0 对应MATLAB的w_matrix(quat,1)

                // q_temp2 = q_temp*mid
                matrix_mul(q_temp, 1, 4, mid11, 4, q_temp2);
                q_temp2[1] = 0; q_temp2[2] = 0; // 解耦：只保留x和w分量
                Norm(q_temp2, 4);

                // self.quat_EKF=q_temp2*mid'
                w_matrix(quat_EKF[sensor], mid11, 1, 1); // mode=1, matT=1 对应转置
                matrix_mul(q_temp2, 1, 4, mid11, 4, q_temp);
                Copy(quat_EKF[sensor], q_temp, 4);
                if (Norm(quat_EKF[sensor], 4) == 0) { return 0; }
            }
        }
        else {
            t_m[sensor] = t_m[sensor] - 0.03f;
        }
    }

    // 快速回归处理（与原始代码相同）
    if (sensorQuitTimeReturnCtrl[sensor]) {
        t_quit[sensor] = t_quit[sensor] + dtTime;
        sensorQuitReturnCtrl[sensor] = 0;
        QuickRecover[sensor] = 0;
        if (t_quit[sensor] > 0.5f) {
            sensorQuitTimeReturnCtrl[sensor] = false;
            t_quit[sensor] = 0;
        }
    }

    if (QuickRecover[sensor] || sensorQuitReturnCtrl[sensor]) {
        if (!sensorQuitTimeReturnCtrl[sensor]) {
            QuickRecover[sensor] = 0;
            QuitReturnCtrlNumber[sensor] = 0;

            Copy(X_ACC_EKF[sensor], acc_mcu, 3);
            X_MAG_EKF[sensor][0] = mag_mcu[0];
            X_MAG_EKF[sensor][1] = mag_mcu[1];
            X_MAG_EKF[sensor][2] = mag_mcu[2];
            X_MAG_EKF[sensor][3] = 0;
            X_MAG_EKF[sensor][4] = 0;
            X_MAG_EKF[sensor][5] = 0;

            ekf_init(sensor);

            PRY[0] = atan2f(acc[1], acc[2]);
            PRY[1] = atan2f(-acc[0], sqrtf(acc[1] * acc[1] + acc[2] * acc[2]));
            PRY[2] = atan2f((mag[2] * sinf(PRY[0]) - mag[1] * cosf(PRY[0])),
                (mag[0] * cosf(PRY[1]) + (mag[1] * sinf(PRY[0]) + mag[2] * cosf(PRY[0])) * sinf(PRY[1])));

            q_temp[0] = cosf(0.5f * PRY[0]) * cosf(0.5f * PRY[1]) * cosf(0.5f * PRY[2]) + sinf(0.5f * PRY[0]) * sinf(0.5f * PRY[1]) * sinf(0.5f * PRY[2]);
            q_temp[1] = sinf(0.5f * PRY[0]) * cosf(0.5f * PRY[1]) * cosf(0.5f * PRY[2]) - cosf(0.5f * PRY[0]) * sinf(0.5f * PRY[1]) * sinf(0.5f * PRY[2]);
            q_temp[2] = cosf(0.5f * PRY[0]) * sinf(0.5f * PRY[1]) * cosf(0.5f * PRY[2]) + sinf(0.5f * PRY[0]) * cosf(0.5f * PRY[1]) * sinf(0.5f * PRY[2]);
            q_temp[3] = cosf(0.5f * PRY[0]) * cosf(0.5f * PRY[1]) * sinf(0.5f * PRY[2]) - sinf(0.5f * PRY[0]) * sinf(0.5f * PRY[1]) * cosf(0.5f * PRY[2]);

            float cosom = quat_EKF[sensor][0] * q_temp[0] + quat_EKF[sensor][1] * q_temp[1] + quat_EKF[sensor][2] * q_temp[2] + quat_EKF[sensor][3] * q_temp[3];
            if (cosom < 0) {
                for (int i = 0; i < 4; i++) {
                    q_temp[i] = -q_temp[i];
                }
            }
            Copy(quat_EKF[sensor], q_temp, 4);

            if (sensorQuitReturnCtrl[sensor]) {
                matrix_Cbn(quat_EKF[sensor], Cbn);
                matrix_mul(Cbn, 3, 3, mag, 1, m_n);
                bx = sqrtf(m_n[0] * m_n[0] + m_n[1] * m_n[1]);
                bz = m_n[2];

                MAG_Fa0[sensor][0] = bx;
                MAG_Fa0[sensor][1] = 0;
                MAG_Fa0[sensor][2] = bz;

                float_temp = Module(MAG_Fa0[sensor], 3);
                for (int i = 0; i < 3; i++) {
                    MAG_Fa0[sensor][i] /= float_temp;
                    MAG_MO0[sensor][i] = MAG_Fa0[sensor][i];
                    MAG_Fam[sensor][i] = MAG_Fa0[sensor][i];
                    MAG_MOm[sensor][i] = m_n[i];
                }
                MAG_Z0[sensor] = bz;
                MAG_LastTrue[sensor] = 1;
                MAG_InitialModule[sensor] = mag0_ss;
            }

            if (sensorQuitReturnState[sensor] < -2.0f) {
                sensorQuitReturnState[sensor] = 2.1f;
            }
            else if (sensorQuitReturnState[sensor] > 2.0f) {
                sensorQuitReturnState[sensor] = -2.1f;
            }

            sensorQuitReturnCtrl[sensor] = 0;
            sensorQuitTimeReturnCtrl[sensor] = true;
        }
    }

    // 四元数滤波 - 与MATLAB代码完全一致
    float_temp = quat_EKF[sensor][0] * quat_Filter[sensor][0] + quat_EKF[sensor][1] * quat_Filter[sensor][1] +
        quat_EKF[sensor][2] * quat_Filter[sensor][2] + quat_EKF[sensor][3] * quat_Filter[sensor][3];
    if (float_temp < 0) {
        for (int i = 0; i < 4; i++) {
            quat_EKF[sensor][i] = -quat_EKF[sensor][i];
        }
        float_temp = -float_temp;
    }
    if (float_temp > 1.0f) float_temp = 1.0f;

    float Angle = acosf(float_temp);
    float sinAngle = sinf(Angle);

    if (sinAngle < 1e-5f) {
        Copy(quat_Filter[sensor], quat_EKF[sensor], 4);
    }
    else {
        float t = (1e-5f + (sinAngle - 1e-5f) * 0.01f) / sinAngle;
        float InvSin = 1.0f / sinAngle;
        float fCoeff0 = sinf((1.0f - t) * Angle);
        float fCoeff1 = sinf(t * Angle);

        quat_Filter[sensor][0] = (quat_Filter[sensor][0] * fCoeff0 + quat_EKF[sensor][0] * fCoeff1) * InvSin;
        quat_Filter[sensor][1] = (quat_Filter[sensor][1] * fCoeff0 + quat_EKF[sensor][1] * fCoeff1) * InvSin;
        quat_Filter[sensor][2] = (quat_Filter[sensor][2] * fCoeff0 + quat_EKF[sensor][2] * fCoeff1) * InvSin;
        quat_Filter[sensor][3] = (quat_Filter[sensor][3] * fCoeff0 + quat_EKF[sensor][3] * fCoeff1) * InvSin;
        Norm(quat_Filter[sensor], 4);
    }

    Copy(q_out, quat_Filter[sensor], 4);
    q_out[0] += EnvironmentMagStatus[sensor];
    q_out[1] += sensorQuitReturnState[sensor];

    return 1;
}

// 调试打印函数
static void Print(float* mat, char r, char c) {
    printf("=[ ");
    char i, j;
    for (i = 0; i < r; i++) {
        for (j = 0; j < c; j++) {
            printf("%f ", mat[i * c + j]);
        }
        printf(";");
    }
    printf("]\n");
}