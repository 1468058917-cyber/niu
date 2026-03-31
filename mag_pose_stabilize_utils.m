function varargout = mag_pose_stabilize_utils(mode, varargin)
%MAG_POSE_STABILIZE_UTILS
% 针对磁定位姿态/位置跳变、抖动和精度问题的工具集合。
%
% 用法：
%   cfg = mag_pose_stabilize_utils('default_config');
%   st  = mag_pose_stabilize_utils('init_state', cfg);
%   [u, amp, ph] = mag_pose_stabilize_utils('lockin', win, fs, f0, cfg);
%   [qf, ef, pf, st] = mag_pose_stabilize_utils('stabilize', q_raw, p_raw, dt, st, cfg);
%
% 说明：
% 1) lockin 提取替代 trapz(v_fit)（后者对相位和窗边界敏感，易导致符号翻转）。
% 2) stabilize 内置四元数符号连续化 + 欧拉角unwrap + 姿态/位置低通 + 异常值门限。

switch lower(mode)
    case 'default_config'
        varargout{1} = default_config();
    case 'init_state'
        cfg = varargin{1};
        varargout{1} = init_state(cfg);
    case 'lockin'
        [win, fs, f0, cfg] = deal(varargin{:});
        [u, amp, ph] = lockin_extract(win, fs, f0, cfg);
        varargout = {u, amp, ph};
    case 'stabilize'
        [q_raw, p_raw, dt, st, cfg] = deal(varargin{:});
        [qf, ef, pf, st] = stabilize_pose(q_raw, p_raw, dt, st, cfg);
        varargout = {qf, ef, pf, st};
    otherwise
        error('Unknown mode: %s', mode);
end

end

function cfg = default_config()
cfg = struct();

% lock-in 解调参数
cfg.lockin_band = [20, 60];      % 与你当前激励频率30Hz相匹配
cfg.lockin_order = 4;

% 姿态稳定参数
cfg.max_angle_step_deg = 12;     % 帧间欧拉角最大可信变化（超出判为跳变）
cfg.ori_slerp_alpha = 0.22;      % 四元数平滑系数（越小越稳，延迟越大）

% 位置稳定参数
cfg.max_pos_step = 0.015;        % 单位按你的坐标而定（若是m，可设1.5cm）
cfg.pos_alpha = 0.20;

% 幅值质量控制
cfg.min_amp = 1e-5;              % 幅值过低时拒绝更新，避免噪声主导相位
end

function st = init_state(cfg)
st = struct();
st.q = [1, 0, 0, 0];
st.euler_deg = [0, 0, 0];
st.p = [0, 0, 0];
st.initialized = false;
st.cfg = cfg;
end

function [u, amp, phase_deg] = lockin_extract(win, fs, f0, cfg)
win = double(win(:));
N = numel(win);
t = (0:N-1)'/fs;

% 预处理：去直流 + 带通
win = win - mean(win);
[b, a] = butter(cfg.lockin_order, cfg.lockin_band/(fs/2), 'bandpass');
win_f = filtfilt(b, a, win);

% 同步解调（I/Q）
ref_c = cos(2*pi*f0*t);
ref_s = sin(2*pi*f0*t);
I = (2/N) * sum(win_f .* ref_c);
Q = (2/N) * sum(win_f .* ref_s);

amp = hypot(I, Q);
phase_deg = atan2(Q, I) * 180/pi;

% 有符号幅值（替代 sign(trapz(...))）
if amp < cfg.min_amp
    u = 0;
else
    u = I; % 对应cos通道的有符号投影，连续性更好
end
end

function [qf, ef_deg, pf, st] = stabilize_pose(q_raw, p_raw, dt, st, cfg)
q_raw = normalize_quat(row_vec(q_raw));
p_raw = row_vec(p_raw);

if ~st.initialized
    st.q = q_raw;
    st.euler_deg = quat2euler_zyx_deg(q_raw);
    st.p = p_raw;
    st.initialized = true;
    qf = st.q;
    ef_deg = st.euler_deg;
    pf = st.p;
    return;
end

% 1) 四元数符号连续化：避免 q 和 -q 引起欧拉角180°翻转
if dot(st.q, q_raw) < 0
    q_raw = -q_raw;
end

% 2) 四元数球面插值低通（姿态平滑）
q_pred = slerp(st.q, q_raw, cfg.ori_slerp_alpha);
q_pred = normalize_quat(q_pred);

% 3) 欧拉角 unwrap + 跳变门限
e_prev = st.euler_deg;
e_new = quat2euler_zyx_deg(q_pred);
e_new = unwrap_euler_deg(e_prev, e_new);

d_ang = abs(e_new - e_prev);
if any(d_ang > cfg.max_angle_step_deg)
    blend = max(0.05, cfg.max_angle_step_deg ./ max(d_ang, 1e-9));
    blend = min(blend);
    q_pred = slerp(st.q, q_raw, blend);
    q_pred = normalize_quat(q_pred);
    e_new = quat2euler_zyx_deg(q_pred);
    e_new = unwrap_euler_deg(e_prev, e_new);
end

% 4) 位置跳变门限 + 一阶低通
dp = p_raw - st.p;
step_norm = norm(dp);
if step_norm > cfg.max_pos_step
    p_raw = st.p + dp * (cfg.max_pos_step / max(step_norm, 1e-12));
end
pf = st.p * (1 - cfg.pos_alpha) + p_raw * cfg.pos_alpha;

% 更新状态
st.q = q_pred;
st.euler_deg = e_new;
st.p = pf;

qf = st.q;
ef_deg = st.euler_deg;

% 避免dt未使用告警（未来可扩展到速度模型）
if dt < 0 %#ok<UNRCH>
    error('dt must be non-negative');
end
end

function v = row_vec(v)
v = reshape(v, 1, []);
end

function q = normalize_quat(q)
q = q / max(norm(q), 1e-12);
end

function q = slerp(q1, q2, t)
q1 = normalize_quat(q1);
q2 = normalize_quat(q2);
cth = max(min(dot(q1, q2), 1), -1);

if cth < 0
    q2 = -q2;
    cth = -cth;
end

if cth > 0.9995
    q = normalize_quat((1 - t) * q1 + t * q2);
    return;
end

th = acos(cth);
s1 = sin((1 - t) * th) / sin(th);
s2 = sin(t * th) / sin(th);
q = normalize_quat(s1 * q1 + s2 * q2);
end

function e = quat2euler_zyx_deg(q)
% q = [w x y z]
w = q(1); x = q(2); y = q(3); z = q(4);

sinr_cosp = 2*(w*x + y*z);
cosr_cosp = 1 - 2*(x*x + y*y);
roll = atan2(sinr_cosp, cosr_cosp);

sinp = 2*(w*y - z*x);
sinp = max(min(sinp, 1), -1);
pitch = asin(sinp);

siny_cosp = 2*(w*z + x*y);
cosy_cosp = 1 - 2*(y*y + z*z);
yaw = atan2(siny_cosp, cosy_cosp);

e = [roll, pitch, yaw] * 180/pi;
end

function e = unwrap_euler_deg(e_prev, e_now)
e = e_now;
for k = 1:3
    d = e_now(k) - e_prev(k);
    d = mod(d + 180, 360) - 180;
    e(k) = e_prev(k) + d;
end
end
