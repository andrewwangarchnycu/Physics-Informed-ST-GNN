# Building Morphology and Urban Open Space Optimization: A Physics-Informed Graph Neural Network Approach

## 2. METHODOLOGY

### 2.1 Framework Overview

本研究的方法論建立在一套由資料生成、圖結構建模、物理導向時空預測，以及設計優化所串接而成的完整流程。依據程式實作，整體系統可分為五個相互銜接的模組：(1) 以基地法規為邊界的參數化都市型態採樣；(2) 結合 EPW 氣象強迫、模擬輸出與實測校準的高解析度熱環境資料集建置；(3) 將建築與開放空間感測點轉換為異質圖；(4) 以 Physics-Informed Spatio-Temporal Graph Neural Network, PIN-ST-GNN 進行 UTCI 預測；以及 (5) 將已訓練模型嵌入多目標搜尋，以探索兼顧熱舒適與開放空間綠覆率的設計方案。

與傳統以規則網格直接輸入卷積模型的方式相比，本研究更重視「形態語意」與「微氣候時序」的耦合。程式中，建築量體以 object nodes 表示，開放空間中的評估位置則以 air nodes 表示；環境氣候序列另以全域向量方式輸入模型。此設計使得局部遮蔭、天空可視因子、鄰近建築高度與逐時外部氣象，可在同一預測框架中共同作用。換言之，本研究不是單純以深度學習擬合 UTCI，而是以結構化都市幾何、顯式氣候條件與物理約束共同構成預測基礎。

### 2.2 Calibrated Synthetic Dataset Generation

依據 `site_constraints.yaml` 與資料生成腳本，本研究首先在 80 m x 80 m 的基地中建立參數化情境。基地位置對應新竹附近區域座標，法規條件設定為 FAR = 2.5、BCR = 0.60，並採用 3 m 最小退縮、3 至 12 層樓高、最小建築面積 80 m2 的限制。幾何採樣器 (`02_geometry_sampler.py`) 會在每一個情境中隨機生成 2 至 5 棟建築，建築平面可為矩形或 L 型，其中 L 型量體在滿足面積門檻時以 30% 機率被引入，以增加形態多樣性。建築高度由樓層數乘以 3.6 m 層高取得，並同步記錄 footprint、coverage、gross floor area (GFA)、centroid 與 open space 幾何。為了保留都市開放空間中的綠化干預潛力，腳本亦於可用開放空間中額外配置 2 至 5 株樹木，樹高介於 4 至 12 m，樹冠半徑按高度比例估計。

氣候邊界條件並非直接以單一固定數值指定，而是由 EPW 解析程序自 EnergyPlus weather file 建立逐時強迫。程式會讀入 EPW 並擷取 7 月 hottest typical day 的 8:00 至 18:00 共 11 個時段，形成後續模擬與訓練一致的時間窗。除 EPW 外，本研究的 v2 管線還引入中央氣象署 (CWB) 與環境部 IoT 感測資料進行校準。`04_sensing_calibration.py` 顯示校準參數包含粗糙度長度 `roughness_length`、道路反照率 `albedo_road` 與氣溫偏移 `ta_bias_offset`，並以差分演化或 L-BFGS-B 最小化代理損失：

$$
\mathcal{L}_{calib}=2.0\cdot RMSE(T_a^{pred},T_a^{obs})+0.5\cdot RMSE(MRT^{proxy}),
$$

其中代理氣溫由 EPW 氣溫加上粗糙度與偏移修正得到，代理 MRT 則由短波、長波與地表反照近似計算。此步驟的目的不是取代高保真模擬，而是讓後續批次生成的熱場更接近實際夏季城市微氣候。

資料集建置上，本研究採用兩階段策略。v1 階段先生成原始情境與模擬結果，v2 階段再進行 2 倍空間解析度提升與實測氣候校正。根據 `dataset_summary_v2.json` 與 `TRAINING_PROGRESS_v2.txt`，最終共建立 300 個 v2 情境，所有情境皆重採樣至 1.0 m 網格，固定包含 6,241 個 air nodes 與 11 個時間步。所有情境被整併為 `ground_truth_v2.h5`，並切分為 205 筆訓練、41 筆驗證與 54 筆測試資料。對應的全資料標準化統計為：氣溫 mean/std = 30.8255/1.1458、MRT = 55.1912/11.7281、風速 = 4.1134/0.7666、相對濕度 = 64.9091/2.1086、UTCI = 35.6146/2.9841。這些統計量被直接寫入 HDF5 正規化群組，供資料讀取與部署階段一致使用。

為便於論文表述，資料集核心特徵可整理如表 1。

| 項目 | 程式實作內容 |
| --- | --- |
| 基地尺寸 | 80 m x 80 m |
| 法規限制 | FAR 2.5, BCR 0.60, setback 3 m |
| 建築數量 | 每情境 2-5 棟 |
| 樹木數量 | 每情境 2-5 株 |
| 時間解析度 | 8:00-18:00，共 11 個時段 |
| 空間解析度 | v2 為 1.0 m grid |
| 節點數 | 每情境 6,241 air nodes |
| 資料切分 | train/val/test = 205/41/54 |

### 2.3 Heterogeneous Graph Representation of Urban Morphology and Open Space

在圖建模階段，本研究將都市熱環境重新表示為異質圖，而非將場域壓平成單一規則矩陣。依 `dataset.py`，圖中至少包含兩類節點。第一類為 object nodes，用以表徵建築物，其特徵維度為 7，包含標準化高度、樓層數、footprint 面積、重心座標、GFA，以及 L 型平面指標。第二類為 air nodes，用以表徵基地內的開放空間感測點與熱舒適估測位置，其 v2 特徵維度為 9，依序為標準化氣溫、MRT、風速、相對濕度、SVF、逐時遮蔭狀態、最近建築高度、最近樹高，以及新增的表面溫度特徵。

此一表示法的關鍵在於，air node 不只是幾何位置點，而是同時攜帶局部微氣候、幾何遮擋與近鄰形態訊息。SVF 由多方向遮蔽關係估算；遮蔭狀態則根據太陽高度角與方位角，結合建築高度投影求得；最近建築與樹木高度用來反映周邊立體環境的調節作用。當模型處理開放空間熱場時，實際上是在處理「被形態條件所制約的環境點」。

邊的定義則分為靜態與保留動態兩類。就目前資料集實作而言，靜態邊包含：(1) `semantic`，即 object-to-object 全連接邊，用以建模不同建築量體之間的語意與幾何關聯；(2) `contiguity`，即 air-to-air 的 KNN 邊，預設 `k=8`，用以刻畫開放空間中近鄰熱場的連續性。程式架構另保留 `dynamic_edges` 介面，使關係型別可在未來擴充為遮蔭、對流或植栽蒸散等逐時關係；然而在目前訓練資料流程中，`dynamic_edges` 為空字典列表，因此本研究實際落地的異質圖核心仍是建築語意關係與 air node 的空間鄰接。

形式上，可將場域圖寫為：

$$
G=(V_{obj}\cup V_{air}, E_{semantic}\cup E_{contiguity}),
$$

其中 $V_{obj}$ 為建築節點集合，$V_{air}$ 為開放空間節點集合；$E_{semantic}$ 反映建築彼此關係，$E_{contiguity}$ 則反映感測點之間的局部傳播關係。此表示法使都市形態與開放空間不再是分離描述，而是被組織成可供圖神經網路直接聚合的關聯系統。

### 2.4 PIN-ST-GNN Architecture

模型主體實作於 `urbangraph.py`，其結構可概括為 Input MLP -> RGCN x 3 -> Global Context Fusion -> Temporal LSTM -> Output MLP。首先，object node 與 air node 分別通過兩組輸入編碼器，映射至共同的 128 維隱表示空間。之後在每個時間步，模型將 object 與該時刻 air embeddings 串接成單一節點張量，並輸入三層 RGCN 進行關係式訊息傳遞。`RGCNBlock` 對每一種 relation type 各自配置一組線性轉換矩陣，再以 destination node 的度數做正規化聚合，因此單層更新可表示為：

$$
h_i^{(l+1)}=\sigma \left( W_{self}h_i^{(l)} + \sum_{r\in \mathcal{R}} \frac{1}{c_{i,r}} \sum_{j\in \mathcal{N}_i^r} W_r h_j^{(l)} \right),
$$

其中 $\mathcal{R}$ 為關係集合，$c_{i,r}$ 為第 $r$ 類關係下的鄰居數。程式中同時加入 residual connection、LayerNorm、PReLU 與 dropout，以穩定深層關係聚合。

空間編碼之後，模型並未直接逐時輸出 UTCI，而是再將每個 air node 的空間表徵與全域氣候上下文結合。`build_env_time_seq()` 會從 EPW hottest typical day 建立 7 維環境向量 `[Ta, RH, WS, WDsin, WDcos, GHI, SolAlt]`，以及 2 維時間編碼 `[sin(hour), cos(hour)]`。`GlobalContextMLP` 將兩者投影後與空間 hidden states 串接，再經過 fusion MLP 轉換為 LSTM 輸入。

時序模組採用單層 LSTM，hidden size 為 256。特別的是，程式並非以零向量初始化 LSTM，而是使用第一個時間步經 RGCN 處理後的 air node 表徵，經 `h0_proj` 與 warm-up 線性層生成初始 hidden state 與 cell state。此設計能讓時序模型從具備空間語意的狀態出發，而非從無資訊起點預測。最後，模型僅取 LSTM 最後時刻 hidden state，再由二層 MLP 一次性解碼出 11 個時段的 UTCI 序列，輸出形狀為 $(N_{air}, T)$。這代表模型學習的是「單一節點的日間熱舒適軌跡」，而非獨立逐時回歸。

### 2.5 Physics-Informed Objective Function

本研究的訓練目標由資料誤差、物理懲罰與可選的感測監督所構成。依 `compute_loss()`，總損失寫為：

$$
\mathcal{L}_{total}=\mathcal{L}_{data}+\mathcal{L}_{physics}+\lambda_{sense}\mathcal{L}_{sensor},
$$

其中 `L_data` 採 MSE，直接比較預測與 HDF5 中標準化 UTCI 真值；`L_sensor` 為選用項，當感測節點 UTCI 可用時，以遮罩方式額外監督特定節點與時段。核心特色在於 `L_physics`，其又由三個部分構成：

$$
\mathcal{L}_{physics}=L_{rad}+L_{temp}+L_{wind}.
$$

第一，`radiation_penalty` 要求日照條件下、SVF 較高且未遮蔭的節點，其 UTCI 平均值應高於遮蔭節點；若出現「遮蔭區比曝曬區更熱」且超過 margin 的情形，則施以二次懲罰。第二，`temporal_smoothness_penalty` 限制相鄰時間步的 UTCI 變化量，程式預設以標準化尺度下的 `max_delta = 0.625` 作為閾值，對應註解中的每小時約 5°C 上限，用以避免日內序列產生不合理跳動。第三，`wind_obstruction_penalty` 針對鄰近高建築的節點施加上界約束；當這些節點的 UTCI 高於全場平均加上 1.5 倍標準差時，即給予額外懲罰，以反映高量體背風面熱負荷不應無限制偏離整體場域。

值得指出的是，這些 physics penalties 並非從完整控制方程式直接離散而來，而是由都市戶外熱環境的物理常識與形態效應轉化為可微分的 soft constraints。其作用在於縮小純資料驅動模型的可行解空間，使預測結果在統計上貼近資料、在行為上也更符合遮蔭、熱惰性與風阻的基本規律。此一設定特別適合本研究所面對的 surrogate modeling 任務，亦即在維持推論速度的同時，降低不合理預測對設計決策的干擾。

### 2.6 Deployment-Oriented Inference and Urban Open Space Optimization

本研究的方法不以離線預測為終點，而是進一步將模型部署於設計工作流。`geometry_converter.py` 會將 Rhino/Grasshopper 端輸入的基地邊界、建築 footprint、樹木配置與感測解析度，轉換成與訓練階段一致的 `sensor_pts`、`obj_feat`、`air_feat` 與 `static_edges`。這意味著研究中的 surrogate model 可以直接接受設計草案作為輸入，並於部署端快速回傳基地內的 UTCI 空間分布與統計值。

更進一步地，`07_optimization` 模組將已訓練的 PIN-ST-GNN 嵌入 NSGA-II。染色體以 `[0,1]^n` 連續變數表示，每棟建築以 6 個基因編碼其中心位置、寬度、深度、旋轉與樓層數，每株樹則以 4 個基因編碼其位置、半徑與高度。設計解經 decode 後，先由 `ConstraintChecker` 驗證五類限制：FAR、BCR、setback、基地內含，以及建築重疊；再由 `FitnessEvaluator` 呼叫 GNN 估計平均 UTCI，並計算樹冠覆蓋率作為 green ratio。其雙目標可寫為：

$$
\min f_1=\overline{UTCI}, \qquad \max f_2=Green\ Ratio.
$$

程式內實際以前者最小化、後者取負值後一併最小化。NSGA-II 採 feasibility-first 非支配排序、crowding distance、SBX crossover 與 polynomial mutation，於每一代保留兼具熱舒適與開放空間綠化潛力的 Pareto 解。換言之，本研究的方法論最終形成一個由「形態生成 -> 微氣候代理預測 -> 約束式多目標優化」所構成的閉環架構，使 Building Morphology 與 Urban Open Space 的設計不再仰賴單次模擬比對，而能在物理導向代理模型支持下進行系統化搜尋。
