%% 将本段嵌入你的主循环前后，解决：姿态跳变、抖动、精度不足
cfg = mag_pose_stabilize_utils('default_config');
st = mag_pose_stabilize_utils('init_state', cfg);

% ===== 你的循环内（替换原来 u 的计算和姿态/位置直接使用） =====
% [u, A, phase_deg] = mag_pose_stabilize_utils('lockin', win, Fs, f0, cfg);
% U(K,i) = u;

% 在得到 ET1.q3 与位置 [x y z] 后：
% dt = N / Fs;
% [qf, eul_f, p_f, st] = mag_pose_stabilize_utils('stabilize', ET1.q3', [x y z], dt, st, cfg);
% Q3(K,:) = qf;
% P(K,:)  = p_f;
% deg(K,:)= eul_f;

% ===== 推荐再加两个离线后处理 =====
% deg(:,1) = smoothdata(deg(:,1), 'movmedian', 7);
% deg(:,2) = smoothdata(deg(:,2), 'movmedian', 7);
% deg(:,3) = smoothdata(deg(:,3), 'movmedian', 7);
% P = smoothdata(P, 1, 'sgolay', 9);
