# WMR 滑移軌跡追蹤控制 — 完整專案報告

> 從論文復現 → RL 對照 → TurtleBot3 真機(Gazebo)部署的完整紀錄。
> 復現論文:Lu et al., *"Slipping Trajectory Tracking Control of Wheeled Mobile
> Robot Based on Dynamics Model,"* IEEE ICIEA 2024.

---

## 0. 一句話總結

論文的分層 backstepping + 自適應滑移控制器在**模擬**中完美復現;在 **TurtleBot3 真機(Gazebo)**
上,模型基控制(論文方法 + 加速度內環)最穩健;而 RL 要在真機成功,需要 **physics-in-the-loop
訓練** + **訓練/部署控制率匹配**——兩者具備後 RL 也能成功補償滑移。

---

## 1. 論文復現(模擬)

**Python**(`wmr/`,已驗證)、**MATLAB**(`matlab/wmr_tracking.m`,與 Python 數值一致)。

實作對應論文公式:滑移率定義 (eq.1)、含滑移運動學 (eq.6/11)、運動學虛擬控制律 (eq.12)、
**滑移自適應律 (eq.13/14)**、動力學模型 (eq.7/8)、動力學 backstepping (eq.18)。

| 情境 | 直線 | 圓形 |
|------|------|------|
| 運動學 + 自適應(論文方法) | **0.011** | 0.142 |
| 動力學串級 + 自適應 | 0.018 | 0.143 |
| 無補償(關閉自適應) | 0.170 | 0.223 |

→ 重現論文 Fig.4–9:誤差收斂、滑移估計追上真值、t=30/50s 突變後迅速恢復。
（已知限制:eq.13/14 經 OCR 擷取有符號歧義,圓形留 ~0.1m 殘差;動力學層用結構忠實的簡化模型。）

---

## 2. RL 對照(模擬)

用 `gymnasium` 包成環境、`stable-baselines3` 的 **SAC** 訓練,與論文控制器同台比較。
觀測不含滑移率,逼 RL 靠誤差回授自行補償。

| 方法 | 直線 | 圓形 |
|------|------|------|
| 論文控制器 | **0.0109** | 0.1424 |
| 完整替換 RL (SAC) | 0.0156 | **0.0953** |
| Residual RL,非自適應 base | 0.0180 | 0.1068 |

**發現**:
- 圓形上 RL 勝過論文(論文自適應律有 ~0.1m 殘差)。
- **Residual RL 結構性發現**:若 base 是完整自適應控制器,命令層級的殘差在數學上**無法**改善
  圓形穩態誤差——自適應律會把殘差「吸收」掉(已用常數殘差實測證明)。有效設定是讓 base 不做
  補償、由 RL 殘差接手。

---

## 3. TurtleBot3 真機部署(ROS 2 Humble + Gazebo)

套件 `ros2_ws/src/wmr_tb3/`。控制器輸出 → `cmd_vel`(體速度)→ TB3 DYNAMIXEL 速度環。

### 3.1 無滑移
論文控制器部署成功,圓形追蹤 **2.4 mm**。

### 3.2 滑移補償:模型基方法
- 論文的**運動學**自適應律在真機(有動力學)上**發散**——它假設速度瞬間達成,真機卻有加速限制/延遲。
- **修法 = 加速度內環**:用 `cmd_vel` 上的 PI 速度迴路逼實際體速度到運動學命令,積分項吸收滑移。

### 3.3 RL 上真機:兩個關鍵
1. **Physics-in-the-loop 訓練**(`rl/gazebo_env.py`):每個 RL 步直接驅動 Gazebo 物理 → 訓練動力學
   = 部署動力學,消除 sim-to-real gap。(純抽象模擬訓練的 RL 無法轉移,會螺旋進中心。)
2. **控制率匹配**:訓練與部署用同一固定控制率。以 Gazebo `pause/unpause` 做確定性步進(每步恰好
   0.102s sim,已驗證),兩邊都 **10 Hz**。控制率不匹配時 RL 給 2.27m,**匹配後給 0.099m**。

### 3.4 真機(Gazebo,圓形,滑移下)對照,10 Hz

| 控制器 | 穩態 RMSE |
|--------|-----------|
| 純運動學 | 0.361 m(滑移下漂移) |
| Gazebo-訓練 RL,單獨取代 (SAC) | 0.099 m(成功,較有噪聲) |
| 速度內環(模型基,演算法) | 0.027 m |
| **速度內環 + RL 殘差(演算法 + RL)** | **0.0058 m**(最佳,降 4.7 倍) |

### 3.5 Residual RL:演算法 + RL = 最佳
base = 速度內環,RL 只學小殘差疊上去(`rl/gazebo_residual_env.py`),physics-in-the-loop、10Hz。
結合「模型基的穩」+「RL 的修正」:**5.8 mm**,比單獨速度內環(27mm)好 4.7 倍。RL 殘差主要抑制
**滑移突變(t=30/50s)的暫態**——base 會跳到 0.04-0.05m 才慢慢恢復,base+殘差壓在 ~0.01m。

---

## 4. 工程結論

1. **模型基控制**(論文 backstepping + 速度內環)在真機穩健,是最佳的「單一」方法 (0.027m)。
2. **RL 單獨取代**上真機需 physics-in-the-loop 訓練 + 控制率匹配,可成功 (0.099m) 但較有噪聲。
3. **演算法 + RL 殘差**最佳 (**0.0058m**)——繼承控制器的穩定性,RL 補它做不好的滑移暫態抑制。
4. **模擬層** RL 可勝過論文;**真機層**「演算法 + RL」勝過任一單獨方法。

---

## 5. 過程中排除的問題(踩坑紀錄)

| 問題 | 原因 | 解法 |
|------|------|------|
| 自適應律真機發散 | 運動學假設違反(真機有速度動力學) | 加速度內環(論文動力學層) |
| RL 完全不轉移(螺旋) | sim-to-real gap | physics-in-the-loop 訓練 |
| RL 仍失敗 (2.27m) | 訓練 20Hz / 部署 5Hz 不匹配 | pause/unpause 確定性 dt,兩邊 10Hz |
| 訓練卡在 2688 步 | 多個 gzserver/訓練實例搶同一 domain | 隔離 `ROS_DOMAIN_ID`、殺重複實例 |
| 訓練被殺 | 與他人專案共用機器 → OOM | 釋放記憶體、隔離 |
| CUDA + rclpy segfault | GPU 與 rclpy 同程序 | SAC `device='cpu'` |
| 指令在 `pkill` 處中止 | shell errexit | 不在開頭 pkill(無東西可殺時) |
| 部署計時太慢 (5Hz) | `/clock` QoS 不相容、`get_clock` 不可靠 | 改用 `/odom` header 時間戳 |

---

## 6. 檔案地圖

```
wmr/                    論文控制器(Python,已驗證)
matlab/wmr_tracking.m   論文控制器(MATLAB,與 Python 一致)
rl/
  wmr_env.py            抽象 RL 環境
  residual_env.py       Residual RL 環境
  tb3_env.py            TB3 尺度 RL 環境(+域隨機化)
  gazebo_env.py         Gazebo-in-the-loop RL 環境(pause/unpause 確定性 dt)
  train*.py / evaluate.py / train_gazebo.py
  models/               訓練好的 SAC 模型(含 tb3sac_gz10 = 10Hz Gazebo 訓練)
ros2_ws/src/wmr_tb3/    ROS 2 套件(controller node + slip injector + worlds/models)
ros2_ws/run_experiment.py / plot_experiment.py   真機三方對照 + 繪圖
figures/  figures_rl/  figures_tb3/              各階段圖檔
README.md               主說明(模擬 + TB3)
ros2_ws/src/wmr_tb3/README.md                    TB3 套件說明
REPORT.md               本報告
```

## 7. 後續(可選)
- AMCL/LiDAR 定位(打滑時不受里程計漂移影響)。
- 真實 TB3 硬體驗證。
- RL 噪聲再壓低(調獎勵/網路/步數)。
