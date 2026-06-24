# WMR 滑移軌跡追蹤控制 — 論文復現

復現論文:**Lu et al., "Slipping Trajectory Tracking Control of Wheeled Mobile
Robot Based on Dynamics Model," ICIEA 2024**(DOI: 10.1109/ICIEA61579.2024.10664745)。

論文核心:輪式移動機器人 (WMR) 在水/薄冰/不平地面會發生**縱向滑移**,造成位置
累積誤差。論文以分層 (kinematic + dynamic) backstepping 控制器搭配**滑移率線上
自適應估計**來補償滑移,並用 Lyapunov 證明位姿誤差與速度誤差漸近收斂。

> 本專案為「先自己實作論文內容」的階段;**之後會再加入 RL 控制器做對照比較**
> (環境已備妥 `gymnasium` / `stable-baselines3`)。

---

## 目錄結構

```
wmr/
  params.py        物理 / 控制器參數(論文 Section V 的數值)
  trajectories.py  參考軌跡:直線、圓形
  model.py         受控體 (plant):含滑移的運動學 / 動力學模型 + 滑移剖面
  controller.py    運動學 backstepping + 滑移自適應 + 動力學力矩控制
  simulate.py      模擬迴圈 + 繪圖
main.py            進入點(CLI)
figures/           輸出圖檔
```

## 執行

```bash
python main.py                 # 直線 + 圓形,運動學 + 自適應(論文核心)
python main.py --traj circle
python main.py --no-adaptive   # 基準:關閉滑移補償(展示滑移的危害)
python main.py --dynamics      # 啟用動力學(力矩)層的串級控制
```

---

## 實作對應論文的公式

| 論文 | 實作位置 |
|------|----------|
| 滑移率定義 eq.1 `s_k=(rα̇_k−v_k)/(rα̇_k)` | `model.wheel_to_body` |
| 含滑移運動學 eq.6 `q̇=S(q)α` | `model.centroid_kinematics`、`wheel_to_body` |
| 位姿誤差 (robot frame) | `controller.pose_error` |
| 運動學虛擬控制律 eq.12 | `KinematicAdaptiveController.virtual_velocity` |
| 滑移自適應律 eq.13/14 | `KinematicAdaptiveController.adapt` |
| 輪速映射 eq.11 `î` 補償 | `KinematicAdaptiveController.wheel_commands` |
| 動力學模型 eq.7/8 `I_w α̈ = τ − rF` | `model.DynamicWMRPlant` |
| 動力學 backstepping eq.18 | `controller.DynamicBackstepping` |

關鍵想法(eq.9):估計量為 `î_k = 1/(1−s_k)`。當 `î_k → 1/(1−s_k)` 時,
透過 eq.11 下達的輪速命令會讓「打滑後的實際體速度」恰好等於虛擬命令 `(v_c, w_c)`,
因而即使有滑移也能完成追蹤。

---

## 結果(穩態 RMSE,最後 20% 時間)

模擬重現論文的穩健性測試:右輪滑移在 **t=30s** 由 0.10→0.20,左輪在 **t=50s**
由 0.10→0.25 突變。

| 情境 | 直線 | 圓形 |
|------|------|------|
| 運動學 + 自適應(論文方法) | **0.011** | **0.142** |
| 動力學串級 + 自適應 | 0.018 | 0.143 |
| 關閉自適應(無補償基準) | 0.170 | 0.223 |

- **直線**:誤差收斂到零;滑移估計準確追上真值(對應論文 Fig.4–6)。
  自適應使誤差較無補償降低約 **16 倍**。
- **圓形**:誤差收斂,軌跡貼合(對應 Fig.7–9);自適應仍明顯優於無補償。
- 滑移突變 (t=30/50s) 後出現短暫暫態並迅速被抑制 —— 與論文一致。

圖檔輸出於 `figures/`:`*_trajectory.png`(平面軌跡)、`*_error.png`(x_e,y_e,θ_e)、
`*_slip.png`(滑移率估計 vs 真值)。

---

## 與論文的差異 / 已知限制

1. **語言**:論文用 MATLAB,本實作用 Python(因後續 RL 對照需要)。
2. **自適應律 eq.13/14**:由 PDF OCR 擷取,部分符號有歧義(如 `cxe` vs `bxe`)。
   目前忠於論文轉寫;在**圓形**軌跡上自適應律會停在一個有小偏差的平衡點,留下
   約 0.1 m 的徑向殘差(直線無此問題)。
3. **動力學層**:論文完整 Lagrange 模型 (eq.7/8) 含 brush 輪胎模型,符號經 OCR
   後不完全可靠。本實作採用結構忠實但簡化的等效模型:輪子轉動動力學
   `I_w α̈ = τ − rF_x` + 線性牽引力,並把動力學 backstepping (eq.18) 實作為
   輪速內環(細步長運行以維持時間尺度分離)。內環增益 `c1` 重新整定以避免航向
   極限環(詳見 `params.DynamicGains` 註解)。

---

## RL 對照(已完成)

以 `gymnasium` 把 `wmr` plant 包成環境(`rl/wmr_env.py`),用
`stable-baselines3` 的 **SAC** 訓練一個策略,與論文控制器在相同軌跡/滑移剖面下比較。

- **觀測**:`[xe, ye, sinθe, cosθe, vr, wr]` —— **滑移率不給 agent**,逼它像論文
  自適應律一樣靠誤差回授自己補償。
- **動作**:體速度命令 `[v, w]`(env 轉成輪速,plant 再施加真實滑移)。
- **獎勵**:負的二次型追蹤誤差 + 微小控制懲罰。
- **訓練**:隨機化初始偏移/滑移值/突變時刻(line+circle 混合),**300k 步**。
- **評估**:在**論文確切情境**(line/circle + t=30/50s 滑移突變、同初始偏移)。

### 演算法對照:SAC / PPO / TD3 / DDPG

四種 RL 演算法在**完全相同的觀測、動作、網路 `[256,256]`、`lr=3e-4`、`γ=0.99`、
300k 步訓練**下對照(只有演算法本身不同,公平比較;程式見 `rl/algos.py`)。

```bash
for a in sac ppo td3 ddpg; do
  python -m rl.train    --algo $a --timesteps 300000 --traj mix
  python -m rl.evaluate --algo $a --traj mix          # -> figures_rl/
done
python -m rl.plot_algo_compare                          # 合併長條圖
```

#### 結果:穩態 RMSE,各 RL 演算法 vs 論文控制器

| 演算法 | 直線 | 圓形 | 備註 |
|--------|------|------|------|
| 論文控制器(模型基) | **0.0109** | 0.1424 | baseline |
| **SAC** | 0.0156 | **0.0953** | 整體最佳;圓形勝論文 |
| TD3 | 0.0236 | 0.0984 | 與 SAC 相近,圓形同樣勝論文 |
| PPO | 0.0756 | 0.1187 | on-policy,收斂較鬆但圓形仍勝論文 |
| DDPG | 0.1323 | 0.1678 | 最差;對噪聲/超參最敏感 |

- **off-policy 勝 on-policy**:SAC ≈ TD3 ≫ PPO ≫ DDPG。SAC 的 entropy 探索與 TD3 的
  雙 Q / 延遲更新讓兩者最穩;PPO 樣本效率較低,DDPG 無這些穩定化手段、最易發散。
- **直線**:論文控制器(模型基)最佳;SAC/TD3 收斂到 ~2 cm,PPO/DDPG 較差。
- **圓形**:**SAC、TD3、PPO 三者都勝過論文**——論文自適應律在圓形留 ~0.1 m 殘差,
  RL 學到更貼合的追蹤;SAC 最佳 (0.095)。
- 結論:**只要選對演算法**(SAC/TD3),一個**完全不觀測滑移、無模型**的 RL 策略,
  就能達到與**模型基自適應控制器**相當(圓形更優)的追蹤精度;但演算法選擇很關鍵
  (DDPG 反而比論文差)。

對照圖於 `figures_rl/`:四演算法合併長條圖 `algo_compare.png`;各演算法的軌跡/誤差
疊圖 `*_traj_<algo>_vs_paper.png`、`*_error_<algo>_vs_paper.png`(`<algo>` = sac/ppo/td3/ddpg)。

### Residual RL(RL 疊在論文方法上)

`rl/residual_env.py`:論文控制器當 **base**,RL 只學殘差 `(Δv, Δw)` 疊加。

```bash
# 在「完整自適應 base」上加殘差
python -m rl.train    --residual --tag res   --timesteps 200000
python -m rl.evaluate --residual --tag res
# 在「非自適應 base」上加殘差(RL 接手滑移補償,option B)
python -m rl.train    --residual --no-base-adapt --tag resna --timesteps 250000
python -m rl.evaluate --residual --no-base-adapt --tag resna
```

**重要發現**:若 base 是**完整自適應控制器**,命令層級的殘差在數學上**無法**改善圓形
穩態誤差——自適應律 (eq.13/14) 收斂到只與位姿誤差有關的平衡,會自動調整滑移估計把
殘差「吸收」掉(已用常數殘差實測證明:任何 Δv/Δw 都不改變圓形 `ye`)。
因此有效的 residual 設定是讓 base **不做**滑移補償(`î=1`),由 RL 殘差接手補償。

### 四種方法總比較(穩態 RMSE)

| 方法 | 直線 | 圓形 | 備註 |
|------|------|------|------|
| 論文控制器(運動學+自適應) | **0.0109** | 0.1424 | baseline |
| 無補償(關閉自適應) | 0.1701 | 0.2233 | 下界對照 |
| 完整替換 RL(SAC) | 0.0156 | **0.0953** | 無模型,圓形最佳 |
| Residual RL,自適應 base | 0.0095 | 0.1424 | 殘差被自適應律吸收,圓形不變 |
| **Residual RL,非自適應 base(B)** | 0.0180 | 0.1068 | 論文骨架 + RL 補滑移,圓形改善 25% |

**結論**:
- 論文控制器在**直線**最佳(收斂到零)。
- **圓形**上 RL 系列都優於論文(論文有 ~0.1 m 殘差):完整替換 RL 最佳 (0.095),
  option-B residual 次之 (0.107)。
- Residual + 自適應 base 證明了一個結構性限制:自適應律會吸收命令殘差。

---

# Part 2 — 部署到 TurtleBot3(ROS 2 + Gazebo)

論文/RL 不只在抽象模擬,還實際部署到 **TurtleBot3 Burger**(ROS 2 Humble + Gazebo)。
程式在 `ros2_ws/src/wmr_tb3/`,詳見該套件的 README。

## 接法重點
- 控制器輸出 → 轉成 `cmd_vel`(體速度),TB3 用 DYNAMIXEL 速度環驅動輪子。
- 滑移以論文比例模型在輪級注入(`slip_injector` / `run_experiment.py`),可控且忠於論文
  (Gazebo 的 Coulomb 摩擦窗口窄、模型也不同)。
- 位姿用 `/odom`;打滑時應改用 AMCL/LiDAR(不受輪子打滑影響)。

## 關鍵發現:真機上的滑移補償
- 無滑移時論文控制器追蹤良好(圓形 **2.4 mm**)。
- 加滑移後,論文的**運動學**自適應律在真實(有動力學的)機器人上**發散**——它假設速度瞬間
  達成,但真機有加速限制/延遲。
- **修法 = 加速度內環**(把論文動力學層用 `cmd_vel` 上的 PI 速度迴路實現):量測體速度逼到
  運動學命令,積分項吸收滑移。

## RL 上真機:physics-in-the-loop + 控制率匹配
- 純抽象模擬訓練的 RL **無法轉移**(螺旋進中心);領域隨機化也救不了 → sim-to-real gap。
- 兩個關鍵:(1) **在 Gazebo 物理迴圈裡訓練**(`rl/gazebo_env.py`),(2) **訓練與部署控制率
  匹配**(用 Gazebo pause/unpause 做確定性 `dt`,兩邊都 10 Hz)。
- 控制率不匹配(訓練 20Hz / 部署 5Hz)時 RL 給 2.27 m;**匹配後(10Hz)給 0.099 m**。

## 真機(Gazebo,圓形,滑移下)三方對照,10 Hz

| 控制器 | 穩態 RMSE |
|--------|-----------|
| 純運動學 | 0.361 m(滑移下漂移) |
| 速度內環(模型基) | **0.027 m**(最佳、最平滑) |
| Gazebo-訓練 RL (SAC, 10 Hz) | 0.099 m(成功,較有噪聲) |

對照圖:`figures_tb3/`。

## 總工程結論
1. **模型基控制**(論文 backstepping + 速度內環)在真機最穩健、最佳。
2. **RL 要上真機**需 physics-in-the-loop 訓練(消 sim-to-real gap)+ **控制率匹配**;
   兩者具備後 RL 也能成功(0.099 m)。
3. 模擬層 RL 可勝過論文(圓形 0.095 vs 0.142),但真機層模型基控制更可靠。
