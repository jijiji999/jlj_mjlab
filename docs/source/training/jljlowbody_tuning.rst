.. _jljlowbody-tuning:

JLJLowBody 调参入口
==================

这一页记录 JLJLowBody 速度训练里最常改的配置入口：PD 随机化、
action scale、关节命令延迟、脚底接触材料、地形和相关奖励。
下面的路径都相对于仓库根目录。


任务入口
--------

JLJLowBody 的环境配置在
``src/mjlab/tasks/velocity/config/jljlowbody/env_cfgs.py``。主要 factory 是：

- ``jljlowbody_flat_env_cfg``：平地训练。
- ``jljlowbody_rough_env_cfg``：标准粗糙地形训练。
- ``jljlowbody_blind_rough_env_cfg``：无 ``height_scan`` 的轻量
  粗糙地形训练。
- ``jljlowbody_capsule_*_env_cfg``：使用脚底 capsule 碰撞体的对应版本。

这些 factory 共同支持下面几个开关：

- ``use_fixed_action_scale``：是否使用统一固定 action scale。
- ``randomize_pd_gains``：是否开启 PD 增益随机化。
- ``randomize_ankle_encoder_bias``：是否在 capsule flat 训练任务里额外开启
  踝关节 encoder bias 随机化。
- ``use_actuator_delay``：是否开启执行器命令延迟。
- ``use_standing_command_curriculum``：是否开启后加入的低速稳定性课程，
  默认是 ``False``。
- ``include_actor_base_lin_vel``：actor 观测里是否加入 base 线速度。
- ``play``：回放模式，会关闭训练用扰动、课程等部分配置。

注册到命令行任务名的位置是
``src/mjlab/tasks/velocity/config/jljlowbody/__init__.py``。如果要改变某个
已注册任务的默认开关，需要改这里调用 factory 的参数，
或者新增一个 task registration。命令行可以覆盖已经构建出的
dataclass 字段，但这些 factory 入参本身不是普通 CLI flag。


PD 与 PD 随机化
---------------

名义 PD 参数在
``src/mjlab/asset_zoo/robots/jljbot/jljlowbody_constants.py`` 中逐关节配置，
常量名是 ``JLJLOWBODY_ACTUATOR_*``。每个关节独立设置：

- ``stiffness``：位置误差的比例增益，相当于 kp。
- ``damping``：速度误差的阻尼增益，相当于 kd。
- ``effort_limit`` 和 ``saturation_effort``：力矩上限。
- ``armature``：电机侧等效转动惯量。
- ``velocity_limit``：速度相关的力矩饱和上限。

训练时的 PD 随机化在
``src/mjlab/tasks/velocity/config/jljlowbody/randomization.py`` 中配置：

.. code-block:: python

   JLJLOWBODY_PD_RANDOMIZATION_KP_RANGE = (0.8, 1.2)
   JLJLOWBODY_PD_RANDOMIZATION_KD_RANGE = (0.9, 1.1)

开关在各个 env factory 的 ``randomize_pd_gains`` 参数上，默认是 ``True``。
只有训练模式启用，``play=True`` 时不会添加 ``pd_gains`` 随机化事件。

随机化事件使用 ``dr.pd_gains``，``operation="scale"``，所以物理意义是：

- 每个环境启动时，把当前 actuator 的 kp 乘以
  ``JLJLOWBODY_PD_RANDOMIZATION_KP_RANGE`` 内采样到的比例。
- 把 kd 乘以 ``JLJLOWBODY_PD_RANDOMIZATION_KD_RANGE`` 内采样到的比例。
- 例如 kp range ``(0.8, 1.2)`` 表示名义 kp 的 80% 到 120%。

关闭示例：

.. code-block:: python

   cfg = jljlowbody_rough_env_cfg(randomize_pd_gains=False)


踝关节 Encoder Bias
-------------------

``jljlowbody_capsule_flat_env_cfg`` 额外支持踝关节 encoder bias 随机化，
用于覆盖脚踝装配、标定或建模偏差。默认训练模式开启，``play=True`` 时关闭。

.. code-block:: python

   JLJLOWBODY_ANKLE_ENCODER_BIAS_RANGE = (-0.05, 0.05)
   JLJLOWBODY_ANKLE_JOINT_NAMES = (".*_ankle_.*_joint",)

基础速度环境已经有全关节 ``encoder_bias``，范围是 ``(-0.02, 0.02)``。
capsule flat 任务会再添加 ``ankle_encoder_bias`` 事件，只匹配左右脚踝
pitch 和 roll 关节，并把这些关节的 bias 改为上面的更大范围。

关闭示例：

.. code-block:: python

   cfg = jljlowbody_capsule_flat_env_cfg(randomize_ankle_encoder_bias=False)


Action Scale
------------

action scale 由 ``use_fixed_action_scale`` 控制。默认是 ``True``，使用
``env_cfgs.py`` 里的统一固定值：

.. code-block:: python

   JLJLOWBODY_FIXED_ACTION_SCALE = 0.5

如果把 ``use_fixed_action_scale=False``，则使用
``jljlowbody_constants.py`` 里按 nominal PD 自动计算的逐关节配置：

.. code-block:: python

   scale = 0.25 * effort_limit / stiffness

这里的数值是策略 action 到关节位置 target 的比例。数值越大，
同样大小的 policy action 会给更大的关节位置目标变化；数值越小，
动作更保守。如果修改 ``JLJLOWBODY_ACTUATORS`` 中的 ``stiffness`` 或
``effort_limit``，``JLJLOWBODY_ACTION_SCALE`` 会随之更新。训练中的
``pd_gains`` domain randomization 不会逐环境动态改变 action scale；scale
对应的是机器人配置里的 nominal PD。


关节命令延迟
------------

延迟的名义范围在 ``jljlowbody_constants.py``：

.. code-block:: python

   JLJLOWBODY_DELAY_MIN_LAG = 0
   JLJLOWBODY_DELAY_MAX_LAG = 2

这些值会写入每个 ``DcMotorActuatorCfg`` 的 ``delay_min_lag`` 和
``delay_max_lag``。单位是 physics step。当前基础速度环境的仿真步长在
``src/mjlab/tasks/velocity/velocity_env_cfg.py``：

.. code-block:: python

   timestep = 0.002
   decimation = 10

因此 ``0`` 到 ``2`` 个 physics step 对应 ``0`` 到 ``4 ms`` 的执行器命令
延迟；policy 控制周期是 ``0.002 * 10 = 0.02 s``。

开关在各个 env factory 的 ``use_actuator_delay`` 参数上，默认是 ``True``。
关闭时，机器人 articulation 会被复制一份，并把所有 actuator 的
``delay_min_lag`` 和 ``delay_max_lag`` 都置为 ``0``。

关闭示例：

.. code-block:: python

   cfg = jljlowbody_flat_env_cfg(use_actuator_delay=False)


脚底接触材料
------------

脚底碰撞体材料相关参数在 ``jljlowbody_constants.py``。
当前脚底 collision 匹配规则是：

.. code-block:: python

   FOOT_COLLISION_REGEX = r"^(left|right)_foot[1-5]_collision$"

脚底接触参数是：

.. code-block:: python

   JLJLOWBODY_FOOT_FRICTION = (0.6,)
   JLJLOWBODY_FOOT_SOLREF = (0.01, 1.0)
   JLJLOWBODY_FOOT_SOLIMP = (0.9, 0.95, 0.001, 0.5, 2.0)

物理意义：

- ``friction``：脚底接触摩擦参数。训练中还会通过
  ``foot_friction`` 事件在 ``(0.2, 1.3)`` 内随机化脚底摩擦。
- ``solref``：MuJoCo 接触约束参考参数。第一个值主要对应接触恢复
  时间尺度，越小接触越硬；第二个值对应阻尼比，
  接近 ``1.0`` 通常接近临界阻尼。
- ``solimp``：MuJoCo 接触阻抗曲线参数，控制接触约束从软到硬的
  变化方式。当前配置偏硬，适合比较硬的脚底。

训练时脚底摩擦随机化来自基础速度环境
``src/mjlab/tasks/velocity/velocity_env_cfg.py`` 的 ``foot_friction`` 事件。
JLJLowBody 会在 ``env_cfgs.py`` 里把这个事件的 ``geom_names`` 改为自己的
脚底碰撞体名字。


地形配置
--------

标准 rough 地形来自基础速度环境
``src/mjlab/tasks/velocity/velocity_env_cfg.py`` 的 ``ROUGH_TERRAINS_CFG``。
``jljlowbody_rough_env_cfg`` 会沿用这个地形，并把 terrain curriculum 打开。

flat 地形在 ``jljlowbody_flat_env_cfg`` 中设置：

.. code-block:: python

   cfg.scene.terrain.terrain_type = "plane"
   cfg.scene.terrain.terrain_generator = None

同时会移除 ``height_scan`` 观测、``out_of_terrain_bounds`` termination 和
``terrain_levels`` curriculum。

blind rough 地形在 ``env_cfgs.py`` 的
``_make_jljlowbody_blind_rough_terrain_cfg`` 中设置。关键参数是：

.. code-block:: python

   JLJLOWBODY_BLIND_ROUGH_TERRAIN_SIZE = (6.0, 6.0)
   JLJLOWBODY_BLIND_ROUGH_NUM_ROWS = 6
   JLJLOWBODY_BLIND_ROUGH_TERRAIN_TYPES = (
     "flat",
     "random_rough_low",
   )

``random_rough_low`` 当前使用轻量随机粗糙地形：

.. code-block:: python

   noise_range = (0.01, 0.08)
   noise_step = 0.01
   horizontal_scale = 0.2
   downsampled_scale = 0.2
   scale_with_difficulty = True

blind rough 的难度按训练 step 推进，配置在
``JLJLOWBODY_BLIND_ROUGH_TERRAIN_STAGES``。该任务会移除 ``height_scan``，
所以策略不能直接看到地形高度。


其他相关配置
------------

JLJLowBody 还有几个常用调参入口都在 ``env_cfgs.py``：

- ``_LINK_MASS_SCALE_RANGE``：link 质量和转动惯量的伪惯量随机化范围。
- ``JLJLOWBODY_ACTOR_NOISE_RANGES``：actor 观测噪声范围。
- ``JLJLOWBODY_ANKLE_ENCODER_BIAS_RANGE``：capsule flat 训练任务里的
  踝关节 encoder bias 随机化范围。
- ``JLJLOWBODY_FOOT_SWING_HEIGHT_PARAMS``：摆脚高度奖励参数。
- ``JLJLOWBODY_ACTION_ACC_WEIGHT``：二阶 action 突变惩罚权重。
- ``JLJLOWBODY_AIR_TIME_COMMAND_THRESHOLD``：air time 奖励生效的命令阈值。
- ``JLJLOWBODY_STANDING_COMMAND_STAGES``：可选的站立和低速命令课程。
  当前默认关闭，可通过 ``use_standing_command_curriculum=True`` 开启。

基础速度环境里已有一阶动作变化惩罚 ``action_rate_l2``，
JLJLowBody 额外加入 ``action_acc_l2``，用于惩罚二阶动作突变，
减少 action jitter。
