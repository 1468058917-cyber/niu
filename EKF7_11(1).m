%% 卡尔曼滤波 EKF-WJ-B-3.2.2.10 (改进版)
%% 卡尔曼滤波
classdef EKF7_11 < handle
    properties(Access = public)
        %预设变量
        sensors = 1;
        P_EKF=zeros(4);Q=zeros(4);Ra=zeros(3);
        Rm=zeros(3);R1=zeros(3);R2=zeros(3);
        fa=zeros(1,3);fa_num=0;
        
        k_first=0;%校准时间计数
        q_k2;
        
        magXZS; P_k1; P_k2; quat_k1; quat_k2;quat_k3;
        MAG_Fa0; MAG_MO0;MAG_Fam;MAG_MOm;fa1; t_a; t_m; pass_r=0; MAG_MO; MAG_Z0;
        MAG_OMOm;MAG_OFam;MAG_Zm;
        EnvironmentMagStatus;
        RecoverSignalEKF;
        InitNumber;MeanACC;MeanMAG;MAG_InitialModule;
        X_ACC_EKF=[0,0,0];X_MAG_EKF;MAG_LastTrue;gyrstatic;accstatic;
        accLast;FastMovementTime;QuickRecover;quat_Filter;QuitReturnCtrlNumber;
        sensorQuitReturnCtrl;
        
        jiao_acc_see=0;
        accErr_see=0;
        gyrErr_see=0;
        RecoverSignalEKF_see=0;

        P_ACC_EKF;Q_ACC_EKF;ERR_ACC_EKF;R_ACC_EKF;acc_EKF;
        P_MAG_EKF;Q_MAG_EKF;R_MAG_EKF;mag_EKF;

        quat_EKF;Racc_use;Racc_static;MAG_InitialModule_m;
        Rmag_use;Rmag_static;
        gyr_vec_flag=0;
        refer_dtime = 0;
        refer_dtimeTh = 1;
        
        quat_GYR=[1,0,0,0];
        
        % 从EKF10_13_8_1移植的磁力计相关属性
        MAG_MOm_xy; % 假北状态下的水平分量基准(x,y)
        MAG_ReferenceStrength = 0; % 磁场强度基准值
        strength_threshold = 0.15; % 磁场强度归一化偏差阈值
        mag_declination_reference = 0; % 动态磁偏角基准
        declination_threshold = deg2rad(3); % 磁偏角阈值
        
        % 固定阈值参数
        fixed_threshold = struct(...
            'jiaom', 0.12, ...           % 姿态角度差阈值
            'jiaom_mag', 0.035, ...      % 磁场方向差阈值
            'bx', 0.07, ...              % X轴分量差阈值
            'by', 0.07, ...              % Y轴分量差阈值
            'bz', 0.085, ...             % Z轴分量差阈值
            'module', 0.095, ...         % 模值差异阈值
            'strength', 0.15, ...        % 磁场强度偏差阈值
            'trust_count', 5 ...         % 连续可信帧数要求
        );
        
        trust_frame_count = 0; % 连续可信帧计数
        true_north_condition_count = 0; % 真北条件连续满足计数
        current_state = 'true_north'; % 当前状态
        
        % MODIFIED: 增加基准更新学习率，用于稳定假北状态下的缓慢更新
        learning_rate = 0.01; % 学习率，控制基准更新的速度
        
    end
    
    methods(Access = public)
        function self=EKF7_11() %初始化实验数据
            self.ekf_init();
            self.k_first = 0;
            self.fa = zeros(1,3);
            self.fa_num = 0;
            self.InitNumber = 1;
            self.MAG_Fa0 = zeros(1,3);
            self.MeanACC = zeros(1,3);
            self.MeanMAG = zeros(1,3);
            self.MAG_InitialModule = 0;
            self.MAG_ReferenceStrength = 0;
            self.MAG_MOm_xy = zeros(1,2);
            self.true_north_condition_count = 0;
            self.mag_declination_reference = 0; % 显式初始化
        end
        
        function ekf_init(self)
            self.P_EKF=eye(4)*1e-1;%quaternion variance when in calculate
            self.Q=eye(4)*0.4e-2;%gyroscope variance，没有更新，直接用
            self.Racc_static=eye(3)*1e+2;
            self.Rmag_static=eye(3)*8e+2;
            self.R1=zeros(3); %accelerometer variance
            self.R2=zeros(3); %magetomeeter variance
            self.ACCEKF_init();
            self.MAGEKF_init();

            self.sensorQuitReturnCtrl = 0;
        end
        
        function ACCEKF_init(self)
            self.ERR_ACC_EKF=0;
            self.P_ACC_EKF = eye(3)*1e-3;
            self.Q_ACC_EKF = eye(3)*3e-1;
            self.R_ACC_EKF = eye(3)*1e+3;
        end
        
        function MAGEKF_init(self)
            self.P_MAG_EKF = eye(6)*1e-3;
            self.Q_MAG_EKF = eye(6)*1e-2;
            self.R_MAG_EKF = eye(3)*5e+2;
        end
        
        % 从EKF10_13_8_1移植的辅助函数
        function out = clamp(~, val, min_val, max_val)
            out = max(min(val, max_val), min_val);
        end
        
        % NEW: 磁偏角计算函数（基于重力方向）
        function mag_declination = calculate_mag_declination(self, acc_T, m_n)
            % acc_T：载体坐标系下的重力方向（由加速度计EKF输出修正）
            % m_n：导航坐标系下的磁场向量
            % 计算磁偏角（磁场方向与地理北向的夹角，范围[-pi, pi]）
            
            % 地理北向在水平面上的投影（2维向量）
            north_vec = [1, 0];  % 与mag_horizontal维度匹配
            
            % 磁场向量在水平面上的投影（去除z轴分量，2维向量）
            mag_horizontal = m_n(1:2);
            mag_horizontal_norm = module(mag_horizontal);
            
            if mag_horizontal_norm < 1e-6
                mag_declination = pi; % 磁场水平分量为0时，默认磁偏角超阈值
                return;
            end
            
            % 归一化磁场水平分量
            mag_horizontal = mag_horizontal / mag_horizontal_norm;
            
            % 计算磁偏角（点积求夹角，叉积判断方向）
            dot_product = dot(north_vec, mag_horizontal);
            cross_product = north_vec(1)*mag_horizontal(2) - north_vec(2)*mag_horizontal(1);
            mag_declination = acos(self.clamp(dot_product, -1, 1));
            
            % 修正角度方向（叉积符号决定正负）
            if cross_product < 0
                mag_declination = -mag_declination;
            end
        end
        
        function strength_bias = calculate_mag_strength_bias(self, current_strength)
            % 计算磁场强度偏差
            if self.MAG_ReferenceStrength < 1e-6
                strength_bias = 1;
                return;
            end
            
            relative_bias = abs(current_strength - self.MAG_ReferenceStrength) / self.MAG_ReferenceStrength;
            strength_bias = min(relative_bias, 1.0);
        end
        
        function [trust_flag, passed_conditions] = fixed_threshold_check(self, jiaom, jiaom_mag, bx_diff, by_diff, bz_diff, module_diff, strength_bias)
            % 固定阈值可信度判断
            passed_conditions = 0;
            total_conditions = 0;
            
            if ~isnan(jiaom)
                total_conditions = total_conditions + 1;
                if jiaom <= self.fixed_threshold.jiaom
                    passed_conditions = passed_conditions + 1;
                end
            end
            
            if ~isnan(jiaom_mag)
                total_conditions = total_conditions + 1;
                if jiaom_mag <= self.fixed_threshold.jiaom_mag
                    passed_conditions = passed_conditions + 1;
                end
            end
            
            if ~isnan(bx_diff)
                total_conditions = total_conditions + 1;
                if bx_diff <= self.fixed_threshold.bx
                    passed_conditions = passed_conditions + 1;
                end
            end
            
            if ~isnan(by_diff)
                total_conditions = total_conditions + 1;
                if by_diff <= self.fixed_threshold.by
                    passed_conditions = passed_conditions + 1;
                end
            end
            
            if ~isnan(bz_diff)
                total_conditions = total_conditions + 1;
                if bz_diff <= self.fixed_threshold.bz
                    passed_conditions = passed_conditions + 1;
                end
            end
            
            if ~isnan(module_diff)
                total_conditions = total_conditions + 1;
                if module_diff <= self.fixed_threshold.module
                    passed_conditions = passed_conditions + 1;
                end
            end
            
            if ~isnan(strength_bias)
                total_conditions = total_conditions + 1;
                if strength_bias <= self.fixed_threshold.strength
                    passed_conditions = passed_conditions + 1;
                end
            end
            
            if total_conditions > 0
                pass_ratio = passed_conditions / total_conditions;
                key_conditions_ok = true;
                
                % 关键条件检查
                if ~isnan(bx_diff) && bx_diff > self.fixed_threshold.bx
                    key_conditions_ok = false;
                end
                if ~isnan(by_diff) && by_diff > self.fixed_threshold.by
                    key_conditions_ok = false;
                end
                if ~isnan(jiaom_mag) && jiaom_mag > self.fixed_threshold.jiaom_mag
                    key_conditions_ok = false;
                end
                
                trust_flag = (pass_ratio >= 0.6) && key_conditions_ok;
            else
                trust_flag = false;
            end
            
            if trust_flag
                self.trust_frame_count = self.trust_frame_count + 1;
            else
                self.trust_frame_count = 0;
            end
        end
        
        function switch_to_fake_north = check_switch_to_fake_north(self, jiao0_2, jiao_mag_2, bz2, mag0_ss)
            % 真北->假北切换检测
            switch_to_fake_north = false;
            
            % 如果当前是真北状态但条件不再满足，考虑切换到假北
            if strcmp(self.current_state, 'true_north')
                true_north_conditions = [
                    (jiao0_2 < 0.15) && (jiao_mag_2 < 0.08) && (abs(bz2 - self.MAG_Z0) < 0.0725) && (abs(mag0_ss - self.MAG_InitialModule) < 0.07), ...
                    (jiao0_2 < 0.1) && (abs(bz2 - self.MAG_Z0) < 0.05) && (abs(mag0_ss - self.MAG_InitialModule) < 0.05)
                ];
                
                % 如果所有真北条件都不满足
                if ~any(true_north_conditions)
                    switch_to_fake_north = true;
                    % fprintf('真北条件不满足，准备切换到假北状态\n');
                end
            end
        end
        
        function switch_to_true_north = check_switch_to_true_north(self, jiao0_2, jiao_mag_2, bz2, mag0_ss, MagModelStatus)
            % 假北->真北切换检测
            switch_to_true_north = false;
            
            % 如果当前是假北状态但真北条件重新满足
            if strcmp(self.current_state, 'fake_north')
                % 使用新的条件：假北转真北的条件更宽松
                true_north_conditions = ...
                    (((jiao0_2 < 0.18) && (jiao_mag_2 < 0.15) && (abs(bz2 - self.MAG_Z0) < 0.08) && (abs(mag0_ss - self.MAG_InitialModule) < 0.08)) ...
                    || ((jiao0_2 < 0.1) && (abs(bz2 - self.MAG_Z0) < 0.05) && (abs(mag0_ss - self.MAG_InitialModule) < 0.05))) ...
                    && (mag0_ss < 0.5) && (mag0_ss > 0.15) && MagModelStatus == 1;
                
                % 额外要求：连续5帧满足真北条件
                if true_north_conditions
                    self.true_north_condition_count = self.true_north_condition_count + 1;
                    if self.true_north_condition_count >= self.fixed_threshold.trust_count
                        switch_to_true_north = true;
                        % fprintf('真北条件重新满足，准备切换回真北状态\n');
                        self.true_north_condition_count = 0; % 重置计数
                    end
                else
                    self.true_north_condition_count = 0;
                end
            end
        end
        
        % 改进的磁力计融合主函数
        function q_out=KalmanFilter(self, acc_mcu, gyr_mcu, mag_mcu, dtTime)
            MagModelStatus=1;
            g=9.8;
            eye4=eye(4);
            acc=acc_mcu;
            gyr=gyr_mcu;
            mag=mag_mcu;
            
            if(all(acc==0)||all(mag==0))
                % disp('acc || mag 都为0');
                q_out=zeros(1,4);
                return;
            end
            
            acc_update = 0;
            mag0_s=sum(mag.*mag);
            mag0_ss=sqrt(mag0_s); 
            acc0_s=sum(acc.*acc);
            acc0_ss=sqrt(acc0_s);
            gyr0_s=sum(gyr.*gyr);
            gyrErr = sqrt(gyr0_s);
            accErr = abs(acc0_ss - g);
            
            if acc0_ss ~=0
                acc=acc/acc0_ss;
            end
            if mag0_s>0
                mag=mag / mag0_ss;
            end
            
            % 初始化阶段
            if (self.k_first <= 0.1)
                if (accErr < 0.5 && gyrErr < 0.5)
                    self.k_first = self.k_first+dtTime;
                else
                    self.k_first = 0;
                    self.InitNumber = 1;
                    self.MAG_Fa0 = zeros(1,3);
                    self.MeanACC = zeros(1,3);
                    self.MeanMAG = zeros(1,3);
                    self.MAG_InitialModule = 0;
                    self.MAG_ReferenceStrength = 0;
                    self.mag_declination_reference = 0;
                end
                self.ekf_init();
                
                for ii=1:3
                    self.MeanACC(ii)=((self.InitNumber-1)*self.MeanACC(ii)+acc(ii)) /self.InitNumber;
                    self.MeanMAG(ii)=((self.InitNumber-1)*self.MeanMAG(ii)+mag(ii)) /self.InitNumber;
                    self.MAG_InitialModule = ((self.InitNumber - 1)*self.MAG_InitialModule + mag0_ss) / self.InitNumber;
                end

                self.X_ACC_EKF = acc_mcu;
                self.X_MAG_EKF = [mag,zeros(1,3)];

                PRY(1) = atan2(self.MeanACC(2), self.MeanACC(3));
                PRY(2) = atan2(-self.MeanACC(1), sqrt(self.MeanACC(2) * self.MeanACC(2) + self.MeanACC(3) * self.MeanACC(3)));
                PRY(3) = atan2((self.MeanMAG(3) * sin(PRY(1)) - self.MeanMAG(2) * cos(PRY(1))),(self.MeanMAG(1) * cos(PRY(2)) + (self.MeanMAG(2) * sin(PRY(1)) + self.MeanMAG(3) * cos(PRY(1))) * sin(PRY(2))));
                self.quat_EKF(1) = cos(0.5 * PRY(1)) * cos(0.5 * PRY(2)) * cos(0.5 * PRY(3)) + sin(0.5 * PRY(1)) * sin(0.5 * PRY(2)) * sin(0.5 * PRY(3));
                self.quat_EKF(2) = sin(0.5 * PRY(1)) * cos(0.5 * PRY(2)) * cos(0.5 * PRY(3)) - cos(0.5 * PRY(1)) * sin(0.5 * PRY(2)) * sin(0.5 * PRY(3));
                self.quat_EKF(3) = cos(0.5 * PRY(1)) * sin(0.5 * PRY(2)) * cos(0.5 * PRY(3)) + sin(0.5 * PRY(1)) * cos(0.5 * PRY(2)) * sin(0.5 * PRY(3));
                self.quat_EKF(4) = cos(0.5 * PRY(1)) * cos(0.5 * PRY(2)) * sin(0.5 * PRY(3)) - sin(0.5 * PRY(1)) * sin(0.5 * PRY(2)) * cos(0.5 * PRY(3));

                m_n=self.MeanMAG*T_SY(self.quat_EKF)';
                bx=sqrt(m_n(1)^2+m_n(2)^2);
                by=0;
                bz=m_n(3);
                self.MAG_Fa0=[bx,by,bz];
                float_temp = module(self.MAG_Fa0);
                self.MAG_Fa0 = self.MAG_Fa0/float_temp;
                self.MAG_MO0=self.MAG_Fa0;
                self.MAG_Fam = self.MAG_Fa0;
                self.MAG_MOm = m_n;
                self.MAG_Z0 = bz;
                self.MAG_LastTrue=1;
                self.quat_Filter = self.quat_EKF;
                self.quat_GYR = self.quat_EKF;
                q_out=self.quat_EKF;
                self.t_a=0;
                self.t_m=0.015;
                self.InitNumber = self.InitNumber+1;
                self.EnvironmentMagStatus = 2.1;
                self.accLast = acc_mcu;
                self.QuitReturnCtrlNumber = 0;
                self.QuickRecover = 0;
                self.gyr_vec_flag=0;
                
                self.acc_EKF = acc_mcu;
                self.MAG_InitialModule_m = self.MAG_InitialModule;
                self.MAG_ReferenceStrength = self.MAG_InitialModule; % 初始化磁场强度基准
                self.MAG_MOm_xy = m_n(1:2);
                self.current_state = 'true_north';
                self.true_north_condition_count = 5; % 初始化为5，直接进入真北状态
                return;
            end

            % 预测部分
            if (gyrErr > 0.05&&dtTime<0.1)
                % ACC预测
                A_ACC = [1,-gyr(3)*dtTime,gyr(2)*dtTime;...
                gyr(3)*dtTime,1,-gyr(1)*dtTime;...
                -gyr(2)*dtTime,gyr(1)*dtTime,1];
                self.X_ACC_EKF = self.X_ACC_EKF*A_ACC;
                self.P_ACC_EKF = A_ACC'*self.P_ACC_EKF*A_ACC+self.Q_ACC_EKF;
                
                % MAG预测
                self.gyr_vec_flag=0;
                A_MAG_EKF=eye(6);
                A_MAG_EKF(1:3,1:3)=eye(3)+[0,-gyr(3),gyr(2);gyr(3),0,-gyr(1);-gyr(2),gyr(1),0]*dtTime;
                self.X_MAG_EKF(1:3) = self.X_MAG_EKF(1:3)*A_ACC;
                self.P_MAG_EKF = A_MAG_EKF'*self.P_MAG_EKF*A_MAG_EKF+self.Q_MAG_EKF;
                
                % 四元数预测
                A=w_matrix([1,gyr*dtTime*0.5],1);
                self.quat_EKF=self.quat_EKF*A';
                self.quat_EKF=self.Norm(self.quat_EKF);

                self.quat_Filter=self.quat_Filter*A';
                self.quat_Filter=self.Norm(self.quat_Filter);
                
                self.quat_GYR=self.quat_GYR*A';
                self.quat_GYR=self.Norm(self.quat_GYR);

                self.P_EKF=A*self.P_EKF*A'+self.Q;
                temp = 100;
                self.P_EKF(self.P_EKF>temp)=temp;
                self.P_EKF(self.P_EKF<-temp)=-temp;
            end

            % 加速度计更新
            self.t_a=self.t_a+dtTime;
            if ((self.t_a > 0.03)&&accErr<0.5&&1)
                self.t_a=self.t_a-0.03;
                acc_update = 1;
                self.accLast = acc;
                if (accErr<2)
                    float_temp = gyrErr*accErr+1e-4;
                    maxerr = max(float_temp,self.ERR_ACC_EKF);
                    mid = self.R_ACC_EKF*maxerr;
                    K_ACC = self.P_ACC_EKF/(self.P_ACC_EKF+mid);
                    self.ERR_ACC_EKF = float_temp;
                    mid = acc_mcu-self.X_ACC_EKF;
                    self.X_ACC_EKF = self.X_ACC_EKF+mid*K_ACC';
                    self.P_ACC_EKF = (eye(3)-K_ACC)*self.P_ACC_EKF;
                end
                self.acc_EKF = self.X_ACC_EKF;
                self.acc_EKF= self.Norm(self.acc_EKF);
            end

            % 加速度计修正四元数
            acc_T(1) = 2 * (self.quat_EKF(2) * self.quat_EKF(4) - self.quat_EKF(1) * self.quat_EKF(3));
            acc_T(2) = 2 * (self.quat_EKF(3) * self.quat_EKF(4) + self.quat_EKF(1) * self.quat_EKF(2));
            acc_T(3) = self.quat_EKF(1) * self.quat_EKF(1) - self.quat_EKF(2) * self.quat_EKF(2) - self.quat_EKF(3) * self.quat_EKF(3) + self.quat_EKF(4) * self.quat_EKF(4);
            
            if (acc0_ss > 0)
                jiao_acc = self.ACos(self.acc_EKF(1) * acc_T(1) + self.acc_EKF(2) * acc_T(2) + self.acc_EKF(3) * acc_T(3));
            else
                jiao_acc = 100;
            end
            
            if (acc_update == 1|| self.sensorQuitReturnCtrl==1)
                if (accErr > 10 && gyrErr > 10)
                    self.QuitReturnCtrlNumber = self.QuitReturnCtrlNumber+0.1;
                elseif (accErr < 0.8 && gyrErr < 0.5)
                    if (jiao_acc > 0.4)
                        self.QuitReturnCtrlNumber = self.QuitReturnCtrlNumber+ 0.1;
                    else
                        self.QuitReturnCtrlNumber = self.QuitReturnCtrlNumber- 0.02;
                    end
                elseif (accErr < 2 && gyrErr < 1)
                    self.QuitReturnCtrlNumber = self.QuitReturnCtrlNumber- 0.02;
                end
                if (self.QuitReturnCtrlNumber < 0)
                    self.QuitReturnCtrlNumber = 0;
                end
                if((self.QuitReturnCtrlNumber > 10 || self.sensorQuitReturnCtrl==1)&& accErr < 0.8 && gyrErr < 0.5)
                    self.QuickRecover = 1;
                end
            end
            
            if(acc_update==1)
                 if (((accErr < 1)&&(gyrErr<1)&&(jiao_acc<0.5))||((accErr<0.5)&&(gyrErr<0.2)))
                        if((accErr < 0.2)&&(gyrErr<0.2))
                            k_a = 1 + 10 * abs(acc0_s - g * g);
                        elseif((accErr < 0.5)&&(gyrErr<0.5))
                            k_a = 1 + 50 * abs(acc0_s - g * g);
                        else
                            k_a = 1 + 100 * abs(acc0_s - g * g)*(1+jiao_acc*20);
                        end
                        self.Racc_use = k_a * self.Racc_static;

                        H1 = 2*[- self.quat_EKF(3),  self.quat_EKF(4), - self.quat_EKF(1), self.quat_EKF(2);...
                              self.quat_EKF(2),   self.quat_EKF(1),  self.quat_EKF(4), self.quat_EKF(3);...
                              self.quat_EKF(1), -  self.quat_EKF(2), - self.quat_EKF(3),  self.quat_EKF(4)];
                        K1=self.P_EKF*H1'/(H1*self.P_EKF*H1'+self.Racc_use);

                        mid1(1) = self.acc_EKF(1) -  acc_T(1);
                        mid1(2) = self.acc_EKF(2) -  acc_T(2);
                        mid1(3) = self.acc_EKF(3) -  acc_T(3);
                        dist_ACC1 = mid1(1) * mid1(1) + mid1(2) * mid1(2) + mid1(3) * mid1(3);
                        temp = sqrt(dist_ACC1);
                        Ryu = 0.08;
                        if (temp > Ryu)
                            mid1(1) = mid1(1) * Ryu / temp;
                            mid1(2) = mid1(2) * Ryu / temp;
                            mid1(3) = mid1(3) * Ryu / temp;
                        end

                        mid2=mid1*K1';
                    qtemp=self.quat_EKF+mid2;
                    qtemp=self.Norm(qtemp);
                    acc_T(1) = 2 * (qtemp(2) * qtemp(4) - qtemp(1) * qtemp(3));
                    acc_T(2) = 2 * (qtemp(3) * qtemp(4) + qtemp(1) * qtemp(2));
                    acc_T(3) = 1 - 2 * qtemp(2) * qtemp(2) - 2 * qtemp(3) * qtemp(3);
                    mid1(1) = self.acc_EKF(1) -  acc_T(1);
                    mid1(2) = self.acc_EKF(2) -  acc_T(2);
                    mid1(3) = self.acc_EKF(3) -  acc_T(3);
                    dist_ACC2 = mid1(1) * mid1(1) + mid1(2) * mid1(2) + mid1(3) * mid1(3);
                    
                    qtemp2=self.quat_EKF-mid2;
                    qtemp2=self.Norm(qtemp2);
                    acc_T(1) = 2 * (qtemp2(2) * qtemp2(4) - qtemp2(1) * qtemp2(3));
                    acc_T(2) = 2 * (qtemp2(3) * qtemp2(4) + qtemp2(1) * qtemp2(2));
                    acc_T(3) = 1 - 2 * qtemp2(2) * qtemp2(2) - 2 * qtemp2(3) * qtemp2(3);
                    mid1(1) = self.acc_EKF(1) -  acc_T(1);
                    mid1(2) = self.acc_EKF(2) -  acc_T(2);
                    mid1(3) = self.acc_EKF(3) -  acc_T(3);
                    dist_ACC3 = mid1(1) * mid1(1) + mid1(2) * mid1(2) + mid1(3) * mid1(3);
                    if dist_ACC2<dist_ACC3
                        if dist_ACC1>dist_ACC2
                        self.quat_EKF=qtemp;
                        self.P_EKF=(eye4-K1*H1)*self.P_EKF;
                        end
                    else
                        if dist_ACC1>dist_ACC3
                        self.quat_EKF=qtemp2;
                        self.P_EKF=(eye4+K1*H1)*self.P_EKF;
                        end
                    end
                 end
            end

            % ========== 改进的磁力计融合部分 ==========
            self.t_m = self.t_m + dtTime;
            if (self.t_m > 0.03&&1)
                if gyrErr >0.05
                    m_n2=mag*T_SY(self.quat_EKF)';
                    bx2 = sqrt(m_n2(1)^2+m_n2(2)^2);
                    bz2 = m_n2(3);
                    Fam2 = [bx2,0,bz2];
                    float_temp = module(Fam2);
                    Fam2 = Fam2/float_temp;
                    jiao_mag_2 = abs(atan2(m_n2(2),m_n2(1)));
                    jiao0_2 = self.ACos(Fam2(1) * self.MAG_Fa0(1) + Fam2(3) * self.MAG_Fa0(3));
                    
                    k_m = 1;
                    self.t_m = self.t_m - 0.03;
                    self.gyr_vec_flag = self.gyr_vec_flag+1;
                    
                    if (self.gyr_vec_flag<=2)
                        self.Rmag_use = k_m * self.R_MAG_EKF;
                        H_MAG_EKF = [eye(3);eye(3)];
                        K_MAG_EKF = self.P_MAG_EKF*H_MAG_EKF/(H_MAG_EKF'*self.P_MAG_EKF*H_MAG_EKF+self.Rmag_use);
                        mid1 = self.X_MAG_EKF*H_MAG_EKF;
                        mid = mag_mcu-mid1;
                        self.X_MAG_EKF = self.X_MAG_EKF+mid*K_MAG_EKF';
                        self.P_MAG_EKF = (eye(6)-K_MAG_EKF*H_MAG_EKF')*self.P_MAG_EKF;
                    end
    
                    self.mag_EKF = self.X_MAG_EKF(1:3);
                    mag0_s_EKF=sum(self.mag_EKF.*self.mag_EKF);
                    mag0_ss_EKF = sqrt(mag0_s_EKF);
                    self.mag_EKF=self.Norm(self.mag_EKF);
    
                    m_n=self.mag_EKF*T_SY(self.quat_EKF)';
                    bx=sqrt(m_n(1)^2+m_n(2)^2);
                    by=0;
                    bz=m_n(3);
                    Fam = [bx,by,bz];
                    float_temp = module(Fam);
                    Fam = Fam/float_temp;
                    
                    FixMAG = 0;
                    local_trust_flag = false; % 用于标记当前帧是否可信
                    
                    % ========== 状态判断逻辑 ==========
                    % 1. 状态切换检测
                    switch_to_fake_north = self.check_switch_to_fake_north(jiao0_2, jiao_mag_2, bz2, mag0_ss);
                    switch_to_true_north = self.check_switch_to_true_north(jiao0_2, jiao_mag_2, bz2, mag0_ss, MagModelStatus);

                    if switch_to_fake_north
                        self.current_state = 'fake_north';
                        self.MAG_LastTrue = 0;
                        self.MAG_MOm = m_n;
                        self.MAG_Fam = Fam;
                        self.MAG_OMOm = m_n2;
                        self.MAG_OFam = Fam2;
                        self.MAG_Zm = bz2;
                        self.X_MAG_EKF(1:3) = mag_mcu;
                        self.X_MAG_EKF(4:6) = zeros(1,3);
                        self.P_MAG_EKF = eye(6) * 1e-3;
                        self.MAG_InitialModule_m = mag0_ss_EKF;
                        self.MAG_MOm_xy = m_n(1:2);
                        self.true_north_condition_count = 0;
                    elseif switch_to_true_north
                        self.current_state = 'true_north';
                        self.MAG_LastTrue = 1;
                        self.trust_frame_count = 0; % 重置可信帧计数
                        % 切换回真北时，可考虑重新初始化假北基准
                        self.MAG_MOm = m_n; 
                        self.MAG_InitialModule_m = mag0_ss_EKF;
                    end
                    
                    % 2. 根据当前状态进行处理
                    if strcmp(self.current_state, 'true_north')
                        FixMAG = 1;
                        self.MAG_MO = self.MAG_MO0;
                        k_m = 1 + 50 * abs(mag0_s - self.MAG_InitialModule^2);
                        
                        % MODIFIED: 真北状态下，使用高信任度更新基准
                        self.MAG_ReferenceStrength = 0.98 * self.MAG_ReferenceStrength + 0.02 * mag0_ss_EKF;
                        mag_declination = self.calculate_mag_declination(self.acc_EKF, m_n);
                        if ~isnan(mag_declination) && ~isinf(mag_declination)
                            self.mag_declination_reference = 0.98 * self.mag_declination_reference + 0.02 * mag_declination;
                        end

                    elseif strcmp(self.current_state, 'fake_north')
                        % 计算固定阈值参数
                        jiaom = self.ACos(Fam(1) * self.MAG_Fam(1) + Fam(3) * self.MAG_Fam(3));
                        jiaom_mag = self.ACos((m_n(1) * self.MAG_MOm(1) + m_n(2) * self.MAG_MOm(2))/ (Fam(1)*sqrt(self.MAG_MOm(1)^2 + self.MAG_MOm(2)^2) + eps));
                        bx_diff = abs(m_n(1) - self.MAG_MOm_xy(1));
                        by_diff = abs(m_n(2) - self.MAG_MOm_xy(2));
                        bz_diff = abs(bz2 - self.MAG_Zm);
                        module_diff = abs(mag0_ss_EKF - self.MAG_InitialModule_m);
                        
                        % 计算磁场强度偏差和磁偏角
                        strength_bias = self.calculate_mag_strength_bias(mag0_ss_EKF);
                        mag_declination = self.calculate_mag_declination(self.acc_EKF, m_n);
                        declination_abs = abs(mag_declination - self.mag_declination_reference);
                        
                        % 使用固定阈值判断当前帧是否可信
                        [local_trust_flag, ~] = self.fixed_threshold_check(jiaom, jiaom_mag, bx_diff, by_diff, bz_diff, module_diff, strength_bias);

                        % 双条件判断
                        if (declination_abs <= self.declination_threshold && strength_bias <= self.strength_threshold) || ...
                           (local_trust_flag && self.trust_frame_count >= self.fixed_threshold.trust_count)
                            
                            FixMAG = 1;
                            self.MAG_MO = self.MAG_MOm;
                            k_m = 1 + 50 * abs(mag0_s_EKF - self.MAG_InitialModule_m^2);
                            
                            % MODIFIED: 稳定假北状态下，使用小学习率缓慢更新基准
                            self.MAG_ReferenceStrength = (1 - self.learning_rate) * self.MAG_ReferenceStrength + self.learning_rate * mag0_ss_EKF;
                            if ~isnan(mag_declination) && ~isinf(mag_declination)
                                self.mag_declination_reference = (1 - self.learning_rate) * self.mag_declination_reference + self.learning_rate * mag_declination;
                            end
                            
                            % 平滑更新假北基准
                            smooth_factor = 0.0015;
                            self.MAG_MOm = (1-smooth_factor)*self.MAG_MOm + smooth_factor*m_n;
                            self.MAG_Fam = (1-smooth_factor)*self.MAG_Fam + smooth_factor*Fam;
                            self.MAG_OMOm = (1-smooth_factor)*self.MAG_OMOm + smooth_factor*m_n2;
                            self.MAG_OFam = (1-smooth_factor)*self.MAG_OFam + smooth_factor*Fam2;
                            self.MAG_MOm_xy = (1-smooth_factor)*self.MAG_MOm_xy + smooth_factor*m_n(1:2);
                            self.MAG_MOm = self.Norm(self.MAG_MOm);
                            self.MAG_Fam = self.Norm(self.MAG_Fam);
                            self.MAG_OMOm = self.Norm(self.MAG_OMOm);
                            self.MAG_OFam = self.Norm(self.MAG_OFam);
                            self.MAG_Zm = self.MAG_MOm(3);
                            self.MAG_InitialModule_m = (1-smooth_factor)*self.MAG_InitialModule_m + smooth_factor*mag0_ss_EKF;
                        else
                            % 不可信，重置假北相关状态
                            self.MAG_MOm = m_n;
                            self.MAG_Fam = Fam;
                            self.MAG_OMOm = m_n2;
                            self.MAG_OFam = Fam2;
                            self.MAG_Zm = bz2;
                            self.X_MAG_EKF(1:3)= mag_mcu;
                            self.X_MAG_EKF(4:6) = zeros(1,3);
                            self.P_MAG_EKF = eye(6)*1e-3;
                            self.MAG_InitialModule_m = mag0_ss_EKF;
                            self.MAG_MOm_xy = m_n(1:2);
                            self.trust_frame_count = 0; % 重置可信帧计数
                        end
                    end
                
                    if FixMAG == 1 && (MagModelStatus == 1 || MagModelStatus == 2)
                        self.Rmag_use = k_m * self.Rmag_static;
                        H2 = 2*[self.MAG_MO(1) * self.quat_EKF(1) + self.MAG_MO(2) * self.quat_EKF(4) - self.MAG_MO(3) * self.quat_EKF(3),self.MAG_MO(1) * self.quat_EKF(2) + self.MAG_MO(2) * self.quat_EKF(3) + self.MAG_MO(3) * self.quat_EKF(4),-self.MAG_MO(1) * self.quat_EKF(3) + self.MAG_MO(2) * self.quat_EKF(2) - self.MAG_MO(3) * self.quat_EKF(1), -self.MAG_MO(1) * self.quat_EKF(4) + self.MAG_MO(2) * self.quat_EKF(1) + self.MAG_MO(3) * self.quat_EKF(2);...
                            -self.MAG_MO(1) * self.quat_EKF(4) + self.MAG_MO(2) * self.quat_EKF(1) + self.MAG_MO(3) * self.quat_EKF(2),self.MAG_MO(1) * self.quat_EKF(3) - self.MAG_MO(2) * self.quat_EKF(2) + self.MAG_MO(3) * self.quat_EKF(1), self.MAG_MO(1) * self.quat_EKF(2) + self.MAG_MO(2) * self.quat_EKF(3) + self.MAG_MO(3) * self.quat_EKF(4),-self.MAG_MO(1) * self.quat_EKF(1) - self.MAG_MO(2) * self.quat_EKF(4) + self.MAG_MO(3) * self.quat_EKF(3);...
                            self.MAG_MO(1) * self.quat_EKF(3) - self.MAG_MO(2) * self.quat_EKF(2) + self.MAG_MO(3) * self.quat_EKF(1),self.MAG_MO(1) * self.quat_EKF(4) - self.MAG_MO(2) * self.quat_EKF(1) - self.MAG_MO(3) * self.quat_EKF(2),self.MAG_MO(1) * self.quat_EKF(1) + self.MAG_MO(2) * self.quat_EKF(4) - self.MAG_MO(3) * self.quat_EKF(3), self.MAG_MO(1) * self.quat_EKF(2) + self.MAG_MO(2) * self.quat_EKF(3) + self.MAG_MO(3) * self.quat_EKF(4)];
                        K2=self.P_EKF*H2'/(H2*self.P_EKF*H2'+self.Rmag_use);
                        self.P_EKF=(eye4-K2*H2)*self.P_EKF;
                        mid1=self.MAG_MO*T_SY(self.quat_EKF);
                        mid2=self.mag_EKF-mid1;
                        
                    % MAG去耦合算法    
                        q_temp=self.quat_EKF+mid2*K2';
                        q_temp=self.Norm(q_temp);
                        mid = w_matrix(self.quat_EKF,1);
                        q_temp2 = q_temp*mid;
                        q_temp2(2) = 0; q_temp2(3)=0;
                        q_temp2 = self.Norm(q_temp2);
                        self.quat_EKF=q_temp2*mid';
                        self.quat_EKF = self.Norm(self.quat_EKF);
                    end
                else
                    self.t_m = self.t_m - 0.03;
                end
            end
            
            % 四元数平滑
            float_temp = sum(self.quat_EKF.*self.quat_Filter);
            if (float_temp < 0)
                self.quat_EKF = -self.quat_EKF;
                float_temp = -float_temp;
            end
            if (float_temp > 1)
                float_temp = 1;
            end
            Angle = self.ACos(float_temp);
            sinAngle = sin(Angle);
            if (sinAngle < 1e-5)
                self.quat_Filter = self.quat_EKF;
            else
                t = (1e-5 + (sinAngle - 1e-5) * 0.01) / sinAngle;
                InvSin = 1.0 / sinAngle;
                fCoeff0 = sin((1 - t) * Angle);
                fCoeff1 = sin(t * Angle);
                self.quat_Filter = (self.quat_Filter * fCoeff0 + self.quat_EKF * fCoeff1) * InvSin;
                self.quat_Filter = self.Norm(self.quat_Filter);
            end
            
            q_out = self.quat_Filter;
        end
        
        function out=Norm(self, vec)
            out=vec/module(vec);
        end
        
        function out=ASin(self, value)
            if (value >= 1)
                out= pi/2;
            elseif (value <= -1)
                out= -pi/2;
            else
                out= asin(value);
            end
        end
        
        function out=ACos(self, value)
            if (value >= 1)
                out= 0;
            elseif (value <= -1)
                out= pi;
            else
                out= acos(value);
            end
        end
        
        function out=antisymmetry(self,vect)
            out = [0,-vect(3),vect(2);...
                vect(3),0,-vect(1);...
                -vect(2),vect(1),0];
        end
    end
end










