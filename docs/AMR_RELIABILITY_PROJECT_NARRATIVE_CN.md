# 面向仓储 AMR 的仿真感知 Policy 与恢复路由研究原型

## 一句话概括

这个项目做的是一个仓储移动机器人可靠性研究原型：先在 Gazebo/Nav2 仿真环境里生成机器人导航数据，用激光雷达和深度图训练一个监督学习 policy，让它学习“当前观测下下一步该往哪走”；然后分析这个 policy 在外部堵路、传感器退化等情况下为什么会错；最后根据不同错误机制建立不同恢复路由，而不是所有错误都走同一个 fallback。

## 项目动机

仓储 AMR 出错的原因并不一样。有时候是前方真的堵了，有时候是传感器退化，有时候是定位漂移，有时候只是 policy 在方向边界上判断不稳定。

如果把这些错误都当成同一种“失败”，恢复机制就会很粗糙。这个项目的核心问题是：

> 能不能先训练一个导航 policy，再分析它自己的错误机制，然后根据这些机制建立不同 recovery route？

这套思路是从 ECG 可靠性项目迁移过来的：

```text
先训练模型
-> 再分析模型错误
-> 找到错误机制
-> 根据机制建立恢复路由
```

## 当前系统做了什么

现在系统已经完成了一条完整链路：

```text
Gazebo/Nav2 仿真
-> 生成 scan/depth 感知数据
-> Nav2 /plan 生成专家动作标签
-> 训练监督学习 navigation policy
-> 分析高置信错误
-> 建立 recovery route 原型
-> 做 scan/depth/fusion ablation
```

policy 学的是离散导航动作：

```text
NORTH, SOUTH, EAST, WEST
```

`STAY` 目前没有作为普通动作标签，因为 Nav2 `/plan` 给的是移动方向，不是安全停止动作。后面 `STAY` 更适合作为安全监督器或 recovery action。

## 为什么用仿真数据是合理的

现在没有真实机器人数据，所以我们用 Gazebo 合成数据。这不是现实验证，但它适合做研究原型，因为它可以控制变量：

- 可以明确设置外部堵路；
- 可以明确设置传感器退化；
- 可以固定 seed；
- 可以同步记录 scan、depth、policy label；
- 可以用 Nav2 plan 自动生成专家动作标签。

因此现在的 claim 应该很稳：

> 这是一个基于仿真的 AMR policy reliability pipeline，不是真实机器人部署验证。

## 监督学习 Policy 是怎么来的

它不是 RL，也不是机器人自己试错学出来的。它是监督学习：

| 输入 | 标签 |
|---|---|
| 激光雷达 scan + 目标信息 | Nav2 专家下一步方向 |
| 深度图 depth + 目标信息 | Nav2 专家下一步方向 |
| scan + depth + 目标信息 | Nav2 专家下一步方向 |

也就是说，模型学的是：

```text
当前传感器观测 + 目标在哪里 -> 下一步往哪个方向走
```

## 实验设计

正式实验矩阵：

| 项目 | 设置 |
|---|---|
| 仿真软件 | Gazebo + Nav2 |
| episode 数 | 36 / 36 成功 |
| seed | 10, 16, 18 |
| 数据划分 | seed 10 train, seed 16 val, seed 18 test |
| 目标方向 | east_south, west_near, north_near, south_axis |
| 场景 | nominal, external_path_blockage, perception_degradation |
| 感知输入 | scan, depth, scan+depth fusion |

这个设计让机器人在不同目标方向和不同扰动下产生专家动作标签，再训练和评估 policy。

## 核心 Ablation 结果

held-out test 结果：

| 模型 | accuracy | macro F1 | weighted F1 | ECE | 高置信错误 |
|---|---:|---:|---:|---:|---:|
| depth baseline | 0.8773 | 0.6666 | 0.8819 | 0.0553 | 62 |
| scan baseline | 0.8694 | 0.6741 | 0.8776 | 0.0778 | 78 |
| scan+depth baseline | 0.8641 | 0.6559 | 0.8673 | 0.0721 | 64 |
| scan focal | 0.8076 | 0.6343 | 0.8235 | 0.0617 | 51 |
| scan+depth focal | 0.8111 | 0.6318 | 0.8265 | 0.0549 | 52 |

结论：

- Depth 不是装饰，它在 test 上是当前最稳的 baseline。
- Scan 在 macro F1 上略好，说明它对少数方向类别仍有价值。
- 简单把 scan 和 depth 拼接起来的 fusion 没有变好，所以不能说 fusion 已经成功。
- Focal loss 能减少高置信错误，但 accuracy 下降明显，所以它只是候选升级，不是最终升级。

## Policy 主要错在哪里

最重要的发现是：policy 的错误不是随机散开的，而是集中在几个机制上。

高置信错误的主要模式：

| 场景 | 错误 | 机制 | 路由 |
|---|---|---|---|
| perception_degradation | SOUTH -> EAST | 感知退化导致方向轴混淆 | CAUTIOUS_REPLAN |
| external_path_blockage | EAST -> SOUTH | 堵路导致高置信方向错 | REPLAN |
| external_path_blockage | WEST -> SOUTH | 堵路导致高置信方向错 | REPLAN |
| nominal 边界情况 | NORTH <-> WEST | 方向边界不确定 | CAUTIOUS_MODE |

这说明不同错误来源确实应该走不同恢复机制。

## Recovery Route 原型

现在建立的恢复路由是：

| 错误机制 | 恢复路由 |
|---|---|
| perception_axis_confusion | CAUTIOUS_REPLAN |
| perception_lateral_depth_confusion | CAUTIOUS_REPLAN |
| blocked_path_high_conf_direction_error | REPLAN |
| blocked_path_direction_error | REPLAN |
| localization_state_error | RELOCALIZE |
| boundary_direction_confusion | CAUTIOUS_MODE |
| geometric_policy_residual | HUMAN_REVIEW |

高置信错误覆盖情况：

| 模型 | test 高置信错误 | 可操作路由覆盖率 |
|---|---:|---:|
| depth baseline | 62 | 1.000 |
| fusion baseline | 64 | 0.984 |
| fusion focal | 52 | 1.000 |
| scan baseline | 78 | 1.000 |
| scan focal | 51 | 1.000 |

这不是说 recovery 已经闭环成功，而是说：

> policy 的高置信残余错误现在可以被分配到有意义的恢复路线。

## 这个项目现在的贡献

目前项目已经不是一个简单 demo，而是一个完整研究原型：

1. 有 Gazebo/Nav2 仿真环境；
2. 有 scan 和 depth 感知数据；
3. 有 Nav2 专家标签；
4. 有监督学习 policy；
5. 有 scan/depth/fusion ablation；
6. 有高置信错误分析；
7. 有基于错误机制的 recovery route 原型。

它的研究价值在于：

> 从 policy 自己的错误机制出发建立恢复路由，而不是拍脑袋手写规则。

## 目前能证明什么

可以证明：

- 仿真环境能自动生成专家标注数据；
- scan/depth 可以训练 navigation policy；
- depth 在当前实验里确实提升了部分可靠性指标；
- policy 的高置信错误有清晰机制；
- 感知退化和外部堵路会导致不同类型错误；
- recovery route 可以从这些错误机制中建立出来。

## 目前还不能证明什么

不能 claim：

- 真实机器人已经可靠；
- recovery route 闭环执行已经成功；
- fusion 已经解决；
- 多 seed 统计显著性已经充分；
- STAY 已经作为普通 policy 动作学会；
- relocalization 已经有充分证据。

## 下一步最合理的实验

下一步应该做 route selector 的闭环前评估：

1. 加入 localization drift 场景，补齐 `RELOCALIZE` 证据；
2. 实现一个 route selector；
3. 评估它能捕获多少高置信错误；
4. 同时评估它会不会过度拦截正确动作；
5. 比较 rule-based route、learned route、uncertainty route、gated fusion route。

## 可以给老师讲的版本

可以这样介绍：

> 我做了一个面向仓储 AMR 的仿真可靠性项目。机器人在 Gazebo/Nav2 中运行，我记录它的激光雷达和深度图观测，并用 Nav2 plan 自动生成专家动作标签。然后我训练一个监督学习导航 policy，分析它在外部堵路和传感器退化下的高置信错误。结果显示，policy 的错误不是随机的，而是呈现出不同机制，例如感知退化导致 SOUTH -> EAST，堵路导致方向误判。最后我根据这些机制建立不同 recovery route，例如 cautious replan、replan、relocalize 和 cautious mode。

一句更短的版本：

> 这个项目展示了如何从 AMR policy 的残余错误机制出发，建立证据驱动的恢复路由。

## 关键文件

英文叙事版：

- `docs/AMR_RELIABILITY_PROJECT_NARRATIVE.md`

中文叙事版：

- `docs/AMR_RELIABILITY_PROJECT_NARRATIVE_CN.md`

核心证据文档：

- `docs/GAZEBO_DEPTH_FUSION_FORMAL_V1_RESULTS.md`
- `docs/GAZEBO_POLICY_RESIDUAL_ROUTE_ABLATION_RESULTS.md`
- `docs/GAZEBO_SCAN_POLICY_FORMAL_V1_RESULTS.md`

核心输出目录：

- `outputs/gazebo_depth_policy_formal_v1`
- `outputs/gazebo_scan_policy_depth_matrix_train_v2`
- `outputs/gazebo_depth_policy_formal_train_v2`
- `outputs/gazebo_fusion_policy_formal_train_v2`
- `outputs/gazebo_policy_residual_routes_v1`

## 最终总结

这个项目现在已经形成了一个比较完整的博士申请型研究原型。它不是简单做了一个导航模型，而是完成了：

```text
仿真数据生成
-> 感知 policy 学习
-> 错误机制分析
-> ablation 证据
-> recovery route 原型
```

最强的项目亮点是：它把“模型错误分析”和“恢复机制设计”连在了一起。这一点和 ECG 可靠性项目的思想是一致的，也比单纯展示一个机器人导航 demo 更有研究味道。
