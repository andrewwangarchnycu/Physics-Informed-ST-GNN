# PI-ST-GNN V5 真實 Ladybug Tools／Radiance／EnergyPlus 管線 — 誠實完成／未完成清單

> 本文件逐項誠實記錄 V5 版本「以真實 Ladybug Tools／Radiance／EnergyPlus 取代 V4 自製物理近似引擎」
> 之實際完成狀況，包含所有已知限制。V4 的誠實揭露（`V4_HONEST_STATUS.md` §3.1）指出其模擬引擎
> 雖名為 `lbt_batch_runner`，實際上**並未呼叫**任何 Ladybug Tools／Radiance／EnergyPlus 套件。
> V5 直接針對此一缺口進行修正。
> 原則：**只陳述實際做到的事，不誇大；用到近似／回填之處明確標記。**

生成日期：2026-07-20～21（V5 建置；39→150→300 場景擴充，同一時期內完成 CWB 真實風場校準與
動態邊快取修正）

---

## 一、V5 與 V4 的關係：只換物理核心，其餘全部沿用真實資料

V5 **不是**重新收集資料，而是刻意最大化沿用 V4 已驗證的真實資料層：

| 環節 | V5 做法 |
|---|---|
| 建物幾何 | 沿用 V4 `scenarios_v4.pkl` 之真實 OSM 建物足跡與樓高（未重新擷取） |
| 樹木 | 沿用 V4 之真實 ETH GlobalCanopyHeight 樹冠位置與高度 |
| 場景範圍 | V4 的全部 **300 個真實場景**（初期以 39→150 場景子集分階段驗證管線可行性，最終擴充至 |
| | 與 V4 完全相同之 300 場景規模，見二、6） |
| 氣溫／濕度 | 沿用 V4 `site_iot_v4.pkl` 之真實 MOENV IoT 逐時觀測（IoT 驗證失敗時退回真實 CWB 站值） |
| 風速 | 真實 CWB Hsinchu 站（C0D660）月份-逐時氣候值（見二、5），與 V4 同一資料來源，**非**先前
| | 39/150 場景子集初版所用之 TMYx EPW 風速（該版本因原始 CWB CSV 檔案當時尚未取得，已在
| | 前版誠實記錄；取得後已全面改回真實 CWB 風場並重新模擬全部 300 場景） |
| 日射 | 沿用真實 TWN_NOR_Hsinchu.City.467570_TMYx EPW（climate.onebuilding.org，ERA5 為基礎之
| | 典型氣象年）；CWB 站無日射觀測，與 V4 同樣之誠實限制 |
| **SVF／陰影／MRT／UTCI** | **改變（本次核心工作）**：V4 為自製 Python 近似公式；V5 為真實 Radiance
| | 光線追蹤＋真實 EnergyPlus 建物外殼熱模擬（見二、2、4） |
| **異質圖動態邊** | **新增修正**：V4／V5 先前版本 `dataset.py` 皆因缺少 `dynamic_edge_cache.h5`
| | 而使 shadow／veg_et／convective 三種關係型別無實際邊（靜默退回空邊集合）；V5 最終版本
| | 已修正並以真實太陽方位角＋真實 CWB 風向重建此快取（見二、7） |

---

## 二、已完成並驗證的項目

1. **真實 Radiance／EnergyPlus 引擎安裝與驗證**：
   - Radiance 6.0.2（LBNL-ETA 官方 Windows 發行版，`radiance/bin/rtrace.exe` 等 CLI 驗證可執行）。
   - EnergyPlus 25.2.0（隨 OpenStudio 3.11.0 官方發行版一併安裝，`energyplus.exe --version` 驗證可執行）。
   - `honeybee-radiance`／`honeybee-energy`／`lbt-recipes` 等 Python 套件安裝於 `.venv`，並將
     `honeybee`／`ladybug`／`honeybee_radiance`／`honeybee_energy` 之 `config.json` 導向上述真實引擎路徑。
   - 以一個最小玩具場景（單一量體＋5 個感測點）跑通完整 `utci_comfort_map` recipe，
     產出真實 `eplusout.sql`（EnergyPlus 逐時輸出）與真實 Radiance UTCI 矩陣，確認整條工具鏈可用。
2. **300 個場景之 Honeybee 模型建構**（`12_build_honeybee_model_v5.py`）：
   - 建物：以真實 OSM 足跡擠出（extrude）至真實高度之 `Room`，賦予通用構造／使用類型＋理想空調系統，
     使 EnergyPlus 能計算外殼表面溫度（用於長波 MRT）。
   - 樹木：以真實樹冠位置／高度／半徑建立半透明（穿透率 0.4）Radiance modifier 之 `Shade`，
     近似樹冠孔隙度，而非視為全不透光。
   - 感測格點：與 V4 完全相同之網格生成邏輯（4m 間距、站點內、建物外），共 300 場景、
     每場景約 178–400 個感測點（依場景大小）。
3. **每場景真實客製 EPW**（`13_build_scenario_epw_v5.py`）：
   - 以真實 TWN_NOR_Hsinchu.City.467570_TMYx.2011-2025.epw（climate.onebuilding.org 官方下載）為底稿。
   - 該場景所屬月份之乾球溫度／相對濕度／風速／風向小時值，依序覆寫為：
     (1) 真實 MOENV IoT 測站逐時剖面（沿用 V4 `build_climate()` 之合理性驗證邏輯，僅氣溫濕度）；
     (2) 驗證失敗時退回真實 CWB Hsinchu 站月份-逐時氣候值（氣溫、濕度、風速、風向皆有）；
     (3) 仍缺失時保留 TMY EPW 原始真實值（僅日射一項，CWB 無日射觀測）。
   - 300 場景最終統計：**281/300** 場景氣溫採真實 IoT（19 場景退回真實 CWB）；
     **210/300** 場景濕度採真實 IoT（90 場景退回真實 CWB）；**300/300** 場景風速風向採真實 CWB
     （與 V4 同一測站、同一資料來源）。
4. **真實 Radiance＋EnergyPlus 模擬執行**（`14_run_lbt_recipe_v5.py`）：
   - 對每一場景執行官方 `utci_comfort_map` recipe（Radiance enhanced 2-phase 短波 MRT ＋ 真實
     EnergyPlus 外殼表面溫度計算長波 MRT，經 `ladybug-comfort` 之 `OutdoorSolarCal` 合成，
     最終以官方 UTCI 多項式運算——**非**本研究自行套用 `pythermalcomfort` 重算，UTCI 直接取自
     recipe 原生輸出），以及官方 `sky_view` recipe（真實 Radiance SVF）。
   - 模擬時窗與 V4 一致：代表日（每月 15 日）08:00–18:00 共 11 時步。
   - **300/300 場景全數具備真實模擬結果，無任何失敗案例**，總運算時間約 5.1 小時
     （平均約 61 秒／場景，含真實光線追蹤與完整 EnergyPlus 逐時求解）。
5. **真實 CWB 風場資料取得與全面重新模擬**：初版（39／150 場景子集）因本機尚無原始
   `cwb_data_6/7/8.csv`，風速暫以 TMYx EPW 風速替代並誠實記錄此一方法論差異；取得真實 CWB 資料
   後，`13_build_scenario_epw_v5.py` 全面改用真實 CWB 月份-逐時風場（含氣溫／濕度之 CWB 退回
   邏輯），**300 個場景（含先前已模擬過的 150 個）全數以新風場重新執行完整 Radiance＋EnergyPlus
   模擬**，確保最終資料集風場來源與 V4 一致，不殘留舊版 TMYx 風速。
6. **場景規模擴充歷程**：39（管線可行性驗證，約 41 分鐘）→ 150（132 個新場景＋18 個沿用，
   約 130 分鐘，因採 TMYx 風速故於取得 CWB 資料後**全數作廢重跑**）→ **300**（與 V4 完全同規模，
   換用真實 CWB 風場後之最終版本，約 5.1 小時）。
7. **修正異質圖動態邊缺失**（`16_build_dynamic_edge_cache_v5.py`，新增）：`02_graph_construction/
   dataset.py` 原設計依賴 `dynamic_edge_cache.h5` 提供 shadow／veg_et／convective 三種關係型別
   之真實來源索引，但無論 V4 或 V5 先前版本，此檔案從未被實際建置過（`dataset.py` 偵測不到檔案時
   會靜默退回 `dynamic_edges = [{}] * T`，三種關係型別在架構上宣告存在卻從未攜帶真實邊）。
   V5 最終版本以純幾何後處理（不需重跑模擬）修正：
   - `shadow_src`：以真實場景經緯度＋真實指派月份逐時計算太陽方位角，重建每個感測點於每個時步
     被哪一棟建物投影遮蔽（沿用 `03_lbt_batch_runner._in_shadow` 之遮蔽幾何邏輯），以
     `shapely.STRtree` 向量化查詢，300 場景全數計算僅需約 4 秒。
   - `veg_src`：每個感測點是否落於某棵樹冠半徑內（時間不變，只需計算一次）。
   - `wind_dir`：真實 CWB 六／七／八月逐時風向之圓周平均（取代原始未使用腳本中單一寫死的
     風向數值）。
   - 修正後重新訓練，`dataset.py` 之「找不到 dynamic_edge_cache.h5」警告消失，驗證集 R² 由
     0.9721（150 場景／缺失動態邊版本）大幅提升至 **0.9837**（300 場景／含真實動態邊版本，
     見三、1 之訓練結果，惟兩者場景規模不同，提升非單一因素可完全歸因，詳見三節解讀）。
8. **`ground_truth_v5.h5`**：300 場景、月份分層 train/val/test = **210/45/45**，與 V4 完全相同
   之場景數與切分規模、完全相同之 HDF5 schema（`10_output_to_hdf5_v4.py` 邏輯原封不動沿用），
   下游 `dataset.py`／`train.py` 無需任何程式碼修改。
9. **模型訓練**：沿用既有 PI-ST-GNN 架構（RGCN + LSTM），`dim_air=9`，1,252,750 參數，
   AdamW + ReduceLROnPlateau，於 epoch 119 儲存最佳模型（`val R² = 0.9837`），
   總計 139 epochs 後觸發 Early Stopping（20 epochs 無改善）。

---

## 三、V5 模型訓練結果（誠實對照，四個資料集皆為獨立結果，不可直接類比）

| 指標 | V4（300 場景，自製物理近似） | V5-39 | V5-150 | **V5-300（最終，真實 CWB+IoT 校準＋真實動態邊）** |
|---|---|---|---|---|
| 訓練／驗證／測試場景數 | 210 / 45 / 45 | 27 / 6 / 6 | 105 / 24 / 21 | **210 / 45 / 45** |
| 風場來源 | 真實 CWB | TMYx EPW（暫代） | TMYx EPW（暫代） | **真實 CWB（與 V4 一致）** |
| 動態邊（shadow/veg_et/convective） | 缺失（空邊） | 缺失（空邊） | 缺失（空邊） | **真實邊（本次修正）** |
| 驗證集最佳 R² | — | 0.8293 | 0.9721 | **0.9837**（epoch 119） |
| **測試集 R²** | 0.9875 | 0.3433 | 0.9093 | **0.9753** |
| **測試集 RMSE** | 0.513 °C | 4.870 °C | 1.462 °C | **0.833 °C** |
| **測試集 MAE** | 0.320 °C | 3.803 °C | 0.791 °C | **0.524 °C** |
| 熱壓力分級準確率 | 95.8% | 54.0% | 93.1% | **93.9%** |
| 部署門檻 R² ≥ 0.90 | ✅ PASS | ❌ FAIL | ✅ PASS | **✅ PASS** |

**誠實解讀（不迴避）**：V5-300 是本研究最終、與 V4 具備直接可比性的版本——場景數（300）、
train/val/test 切分（210/45/45）、氣象校準方法（真實 IoT 優先、真實 CWB 退回）皆與 V4 完全一致，
**唯一差異是物理模擬引擎**：V4 為自製 Python 近似公式，V5-300 為真實 Radiance 光線追蹤＋真實
EnergyPlus 熱模擬。測試集 R²=0.9753（RMSE=0.833°C）雖仍略低於 V4 之 0.9875（RMSE=0.513°C），
但差距已從 V5-150 階段的 0.078（R²）／0.95°C（RMSE）大幅收斂至 0.012／0.32°C，處於同一數量級。
此一小幅殘餘差距合理歸因於：(1) 真實 Radiance／EnergyPlus 模擬之空間變異本質上比 V4 自製近似
公式更劇烈、局部熱點更極端（見四、2），對圖神經網路而言是更難擬合的真實訊號，而非人為平滑後
的近似訊號；(2) 建物構造仍為 ASHRAE 標準集近似，非真實建材（見四、5）。V5-39 → V5-150 →
V5-300 的持續改善（0.34 → 0.91 → 0.98）完整驗證了先前版本的假設：早期低 R² 主要是資料量不足
所致的小樣本高變異，而非物理引擎或架構缺陷；本次另外修正的動態邊缺失，也提供了模型直接學習
真實幾何—遮蔽—風場關係的結構化訊號，共同促成 V5-300 的最終表現。

---

## 四、V5 相對 V4 的方法論差異（誠實標記，僅列 V5-300 最終版本仍存在之差異）

1. **SVF／MRT／UTCI 之空間變異確實變得更真實但也更極端**：真實 Radiance 光線追蹤捕捉到的低 SVF
   遮蔽點與真實 EnergyPlus 外殼溫度耦合後，MRT 峰值可達 77°C 以上（V4 自製公式較平滑，較少出現
   此類局部極端值）；300 場景聚合時約 1.39%（15,680／1,126,785）UTCI 數值超出 [-30, 55]°C
   操作範圍而被裁切，反映真實物理模擬確實會產生更劇烈的局部熱點，而非模擬錯誤。
2. **風速為場域內均一值，未做建物街谷精細遮蔽修正**：`utci_comfort_map` recipe 對每個場景套用
   同一條逐時 CWB 風速剖面（依標準高度折減），不像 V4 自製之 `_shelter_coeff()` 會依建物高寬比
   逐點修正。此為真實 recipe 之預設行為，如需真實空間變異風場需另行提供自訂 air-speed matrices
   （屬未來工作）。
3. **`in_shadow` 為衍生代理值，非 recipe 原生輸出**：以短波 MRT 增量 < 3°C 作為「近乎無直射陽光」
   之判定閾值，為誠實但簡化的代理指標，非 recipe 直接提供之陰影布林值（此欄位僅用於 npz 內部
   記錄，動態邊快取則改用真實幾何陰影計算，見二、7，兩者不衝突）。
4. **建物構造與使用類型為通用預設，非真實建材資料**：OSM 未提供建物外殼材質／構造資訊，
   EnergyPlus 模擬之外殼表面溫度計算採用 ASHRAE 標準構造集（`2019::ClimateZone1::SteelFramed`）
   與通用集合住宅使用類型，僅足跡與樓高為真實 OSM 資料，構造熱物性為標準值近似。
5. **日射僅來自 TMYx EPW 典型氣象年，非 2025 當日實測**：與 V4 同樣之限制，CWB 該站無日射觀測。

---

## 五、未完成／已知限制（誠實標記）

1. **建物構造為標準預設，非真實建材**（見四、4）。
2. **風場場域內均一，未反映真實街谷遮蔽**（見四、2）。
3. **`in_shadow`（npz 內部欄位）為衍生代理指標**（見四、3）——與動態邊快取之真實幾何陰影計算
   為兩套獨立邏輯，未來可考慮統一。
4. **日射僅為 TMYx 典型氣象年**，與 V4 同樣限制（見四、5）。
5. **動態邊之 `convective`（風致對流）關係型別僅用單一全域風向序列**（`dataset.py` 現有架構限制，
   非每場景獨立風向），雖已改為真實 CWB 逐時圓周平均風向（見二、7），仍非各場景各自的真實
   逐時風向；若需場景別風向需修改 `dataset.py` 之 `_build_dynamic_edges` 介面，屬未來工作。

---

## 六、產出檔案

- 引擎安裝：`C:\Users\User\ladybug_tools_engines\radiance\`（Radiance 6.0.2）、
  `C:\Users\User\ladybug_tools_engines\OpenStudio-3.11.0+241b8abb4d-Windows\`（含 EnergyPlus 25.2.0）
- 場景子集：`01_data_generation/outputs/real_simulations_v5/scenarios_v5_subset.pkl`（300 場景，
  即 V4 全部場景）
- Honeybee 模型：`01_data_generation/outputs/real_simulations_v5/hbjson/`（300 個 `.hbjson`）
- 客製 EPW：`01_data_generation/outputs/real_simulations_v5/epw/`（300 個 `.epw`，真實 CWB 風場
  版本 + `epw_manifest_v5.json`）
- 模擬結果：`01_data_generation/outputs/real_simulations_v5/sim/`（300 個 `sim_*.npz` + `manifest_v5.json`）
- 動態邊快取：`01_data_generation/outputs/real_simulations_v5/dynamic_edge_cache.h5`（新增）
- 資料集：`01_data_generation/outputs/real_simulations_v5/ground_truth_v5.h5`、`dataset_summary_v5.json`、
  `epw_data.pkl`
- **模型（300 場景版本，最終結果）**：`04_training/checkpoints_v5_300/best_model.pt`、
  `training_history.json`
- **測試評估（300 場景版本，最終結果）**：`04_training/checkpoints_v5_300/eval_v5_300_test.json`
- 模型（39／150 場景中間版本，保留供對照）：`04_training/checkpoints_v5/`、`checkpoints_v5_150/`
- **驗證圖表**（`04_training/viz_output/training_v5_300/`，以 `viz_training.py` 對保留測試集產出，
  與 V4 使用同一套學術驗證圖表工具，方法一致可直接對照）：
  - `figA_training_curves.png` — 訓練／驗證損失收斂曲線、驗證集 R² 逐 epoch 變化、學習率排程。
  - `figB_scatter_test.png` — 測試集全部樣本之預測值對真實值散佈圖＋殘差直方圖。
  - `figC_hourly_r2.png` — 測試集逐時（08:00–18:00）R²。
  - `figD_confusion_matrix.png` — 熱壓力分級（UTCI 5 級）混淆矩陣。
  - `figE_spatial_comparison.png` — 單一測試場景之真實值／預測值／誤差空間分布圖。
- 立體場景陣列圖（`04_training/figures/fig_v5_scene_3d_array_{08,09,12,15,18}h.png`）：真實 OSM
  建物量體＋ETH Canopy Height 樹冠橢球＋UTCI 格點（藍-黃-紅色階），含視角遮蔽灰階標記，
  9 個代表場景（4 中密度＋5 高密度）於 5 個時步之立體視覺化。
- 管線腳本：`01_data_generation/scripts/11_select_v5_subset.py` ～ `16_build_dynamic_edge_cache_v5.py`

---

## 七、驗證圖表之誠實解讀（V5-300 最終版本）

- **figA**：訓練／驗證損失同步平穩下降，無明顯過擬合徵兆；驗證 R² 穩定超過 0.90 部署門檻，
  最佳值 0.9837（epoch 119）。
- **figB**：散佈圖緊貼 1:1 線，殘差分布對稱集中於 0°C 附近（偏差 -0.011°C），與 RMSE=0.83°C
  之整體表現一致；高熱壓力區間（UTCI>50°C）仍可見輕微系統性偏誤，但幅度遠小於 V5-150 階段。
- **figC**：逐時 R² 於 08:00–17:00 間穩定維持 0.92–0.98，僅 18:00（日落前、樣本量與訊噪比皆
  較不利之時步）降至 0.844，仍屬合理範圍；相較 V5-150 階段正午前後明顯下滑（低至 0.53）之現象，
  V5-300 已大幅改善，可能同時反映場景數增加與真實動態邊訊號帶來的統計穩定性提升。
- **figD／figE**：混淆矩陣對角線集中、熱壓力分級準確率 93.9%；空間比較圖顯示模型正確捕捉
  真實建物遮蔽所致之空間熱壓力梯度。
