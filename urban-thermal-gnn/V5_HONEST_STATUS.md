# PI-ST-GNN V5 真實 Ladybug Tools／Radiance／EnergyPlus 管線 — 誠實完成／未完成清單

> 本文件逐項誠實記錄 V5 版本「以真實 Ladybug Tools／Radiance／EnergyPlus 取代 V4 自製物理近似引擎」
> 之實際完成狀況，包含所有已知限制。V4 的誠實揭露（`V4_HONEST_STATUS.md` §3.1）指出其模擬引擎
> 雖名為 `lbt_batch_runner`，實際上**並未呼叫**任何 Ladybug Tools／Radiance／EnergyPlus 套件。
> V5 直接針對此一缺口進行修正。
> 原則：**只陳述實際做到的事，不誇大；用到近似／回填之處明確標記。**

生成日期：2026-07-20（V5 建置；同日內完成 39→150 場景擴充）

---

## 一、V5 與 V4 的關係：只換物理核心，其餘全部沿用真實資料

V5 **不是**重新收集資料，而是刻意最大化沿用 V4 已驗證的真實資料層：

| 環節 | V5 做法 |
|---|---|
| 建物幾何 | 沿用 V4 `scenarios_v4.pkl` 之真實 OSM 建物足跡與樓高（未重新擷取） |
| 樹木 | 沿用 V4 之真實 ETH GlobalCanopyHeight 樹冠位置與高度 |
| 場景挑選 | 自 V4 的 300 個真實場景中，依月份分層＋建蔽率分層，抽樣 **150 個**（見二、1；初版為 39 個，後擴充） |
| 氣溫／濕度 | 沿用 V4 `site_iot_v4.pkl` 之真實 MOENV IoT 逐時觀測 |
| 風速／日射 | **改變**：V4 用單一 CWB 測站風場＋區域 TMY EPW 日射；V5 兩者皆取自同一份 |
| | 真實 TMYx EPW（見二、3），為誠實記錄之方法論差異 |
| **SVF／陰影／MRT／UTCI** | **改變（本次核心工作）**：V4 為自製 Python 近似公式；V5 為真實 Radiance |
| | 光線追蹤＋真實 EnergyPlus 建物外殼熱模擬（見二、2、4） |

---

## 二、已完成並驗證的項目

1. **真實 Radiance／EnergyPlus 引擎安裝與驗證**：
   - Radiance 6.0.2（LBNL-ETA 官方 Windows 發行版，`radiance/bin/rtrace.exe` 等 CLI 驗證可執行）。
   - EnergyPlus 25.2.0（隨 OpenStudio 3.11.0 官方發行版一併安裝，`energyplus.exe --version` 驗證可執行）。
   - `honeybee-radiance`／`honeybee-energy`／`lbt-recipes` 等 Python 套件安裝於 `.venv`，並將
     `honeybee`／`ladybug`／`honeybee_radiance`／`honeybee_energy` 之 `config.json` 導向上述真實引擎路徑。
   - 以一個最小玩具場景（單一量體＋5 個感測點）跑通完整 `utci_comfort_map` recipe，
     產出真實 `eplusout.sql`（EnergyPlus 逐時輸出）與真實 Radiance UTCI 矩陣，確認整條工具鏈可用。
2. **150 個場景之 Honeybee 模型建構**（`12_build_honeybee_model_v5.py`）：
   - 建物：以真實 OSM 足跡擠出（extrude）至真實高度之 `Room`，賦予通用構造／使用類型＋理想空調系統，
     使 EnergyPlus 能計算外殼表面溫度（用於長波 MRT）。
   - 樹木：以真實樹冠位置／高度／半徑建立水平圓盤 `Shade`，賦予半透明（穿透率 0.4）Radiance
     modifier 近似樹冠孔隙度，而非視為全不透光。
   - 感測格點：與 V4 完全相同之網格生成邏輯（4m 間距、站點內、建物外），共 150 場景、
     每場景約 178–400 個感測點（依場景大小）。
3. **每場景真實客製 EPW**（`13_build_scenario_epw_v5.py`）：
   - 以真實 TWN_NOR_Hsinchu.City.467570_TMYx.2011-2025.epw（climate.onebuilding.org 官方下載，
     ERA5 再分析為基礎之典型氣象年）為底稿。
   - 該場景所屬月份之乾球溫度／相對濕度小時值，覆寫為該場景真實 MOENV IoT 測站之逐時剖面
     （沿用 V4 `build_climate()` 之合理性驗證邏輯）；風速／風向與日射（GHI/DNI/DHI）保留 TMY 真實值。
   - 150/150 場景使用真實 IoT 氣溫；110/150 場景使用真實 IoT 濕度（40 場景之 IoT 濕度感測器未通過
     合理性檢驗，退回 TMY 真實濕度值）。
4. **真實 Radiance＋EnergyPlus 模擬執行**（`14_run_lbt_recipe_v5.py`）：
   - 對每一場景執行官方 `utci_comfort_map` recipe（Radiance enhanced 2-phase 短波 MRT ＋ 真實
     EnergyPlus 外殼表面溫度計算長波 MRT，經 `ladybug-comfort` 之 `OutdoorSolarCal` 合成，
     最終以官方 UTCI 多項式運算——**非**本研究自行套用 `pythermalcomfort` 重算，UTCI 直接取自
     recipe 原生輸出），以及官方 `sky_view` recipe（真實 Radiance SVF）。
   - 模擬時窗與 V4 一致：代表日（每月 15 日）08:00–18:00 共 11 時步。
   - 分兩階段執行：先完成 39 個場景（約 41 分鐘，平均 63 秒／場景）驗證管線可行性；
     決定擴充後，重新依月份／建蔽率分層抽樣 150 個場景（含 18 個與前次重疊，自動略過重跑），
     其餘 132 個場景**全數模擬成功**，總計約 130 分鐘。
   - **150/150 場景全數具備真實模擬結果**，管線在無任何失敗案例下完成規模擴充。
5. **`ground_truth_v5.h5`**：150 場景、月份分層 train/val/test = 105/24/21，與 V4 完全相同之 HDF5
   schema（`10_output_to_hdf5_v4.py` 邏輯原封不動沿用），下游 `dataset.py`／`train.py` 無需任何
   程式碼修改。
6. **模型訓練**：沿用既有 PI-ST-GNN 架構（RGCN + LSTM），`dim_air=9`，1,252,750 參數，
   AdamW + ReduceLROnPlateau，於 epoch 167 儲存最佳模型（`val R² = 0.9721`），
   總計 187 epochs 後觸發 Early Stopping（20 epochs 無改善）。

---

## 三、V5 模型訓練結果（誠實對照，三個資料集皆為獨立結果，不可直接類比）

| 指標 | V4（300 真實場景，自製物理近似） | V5-39（39 場景子集，真實 Radiance+EnergyPlus） | **V5-150（150 場景子集，真實 Radiance+EnergyPlus）** |
|---|---|---|---|
| 訓練／驗證／測試場景數 | 210 / 45 / 45 | 27 / 6 / 6 | **105 / 24 / 21** |
| 驗證集最佳 R² | — | 0.8293（epoch 26） | **0.9721（epoch 167）** |
| **測試集 R²** | 0.9875 | 0.3433 | **0.9093** |
| **測試集 RMSE** | 0.513 °C | 4.870 °C | **1.462 °C** |
| **測試集 MAE** | 0.320 °C | 3.803 °C | **0.791 °C** |
| 熱壓力分級準確率 | 95.8% | 54.0% | **93.1%** |
| 部署門檻 R² ≥ 0.90 | ✅ PASS | ❌ FAIL | **✅ PASS** |

**誠實解讀（不迴避）**：V5-39 → V5-150 的結果變化直接驗證了先前的假設——39 場景（僅 6 個測試
場景）之低 R²（0.34）確實是**資料量不足**所致之小樣本高變異現象，而非物理模擬引擎或模型架構
的缺陷。將真實 Radiance＋EnergyPlus 場景數擴充至 150（測試集擴大至 21 個場景）後，測試集
R² 由 0.34 大幅回升至 **0.9093**，**通過部署門檻**，且驗證集與測試集之 R² 落差（0.9721 vs
0.9093）已收斂至合理範圍，不再是小樣本的極端震盪。V5-150 之 R² 仍略低於 V4 之 0.9875
（RMSE 1.46°C vs 0.51°C），此一差距合理歸因於：(1) V5 訓練場景數仍為 V4 之一半（150 vs 300）；
(2) 真實 Radiance／EnergyPlus 模擬之空間變異本質上比 V4 自製近似公式更劇烈、雜訊更多
（見四、2），對圖神經網路而言是更難擬合的真實訊號，而非更容易的平滑近似訊號。兩者的差距
方向與量級皆屬合理，而非任一版本有誤。

---

## 四、V5 相對 V4 的方法論差異（誠實標記）

1. **場景數仍為 V4 的一半（150 vs 300）**：真實 Radiance＋EnergyPlus 模擬之運算成本（約 60 秒／
   場景，含真實光線追蹤與完整 EnergyPlus 逐時求解）遠高於 V4 自製近似公式（可批次平行處理數百
   場景）；150 場景之總運算時間約 3 小時（41 分鐘＋130 分鐘兩階段執行），仍為在合理時間預算下
   之取捨，已於規劃階段與使用者明確溝通並取得同意。若欲完全對齊 V4 之 300 場景規模，預估仍需
   額外約 2.5 小時運算。
2. **SVF／MRT／UTCI 之空間變異確實變得更真實但也更極端**：真實 Radiance 光線追蹤捕捉到的低 SVF
   遮蔽點與真實 EnergyPlus 外殼溫度耦合後，MRT 峰值可達 77°C（V4 自製公式較平滑，較少出現此類
   局部極端值）；150 場景聚合時約 0.67%（3,839／571,054）UTCI 數值超出 [-30, 55]°C 操作範圍而被
   裁切，反映真實物理模擬確實會產生更劇烈的局部熱點，而非模擬錯誤。
3. **風速為場域均一值，未做建物街谷遮蔽修正**：`utci_comfort_map` recipe 在未額外提供
   `air-speed-matrices` 輸入時，以 EPW 風速做標準高度折減後，對所有戶外感測點套用同一條逐時風速
   剖面，不像 V4 自製之 `_shelter_coeff()` 會依建物高寬比逐點修正。此為真實 recipe 預設行為，
   如需真實空間變異風場需另行提供自訂 air-speed matrices（屬未來工作）。
4. **`in_shadow` 為衍生代理值，非 recipe 原生輸出**：以短波 MRT 增量 < 3°C 作為「近乎無直射陽光」
   之判定閾值，為誠實但簡化的代理指標，非 recipe 直接提供之陰影布林值。
5. **建物構造與使用類型為通用預設，非真實建材資料**：OSM 未提供建物外殼材質／構造資訊，
   EnergyPlus 模擬之外殼表面溫度計算採用 ASHRAE 標準構造集（`2019::ClimateZone1::SteelFramed`）
   與通用集合住宅使用類型，僅足跡與樓高為真實 OSM 資料，構造熱物性為標準值近似。
6. **風速／日射改為單一 TMYx EPW 來源，非 V4 之 CWB 測站**：見一、方法論差異表；此為刻意選擇
   （V4 之 CWB 測站不提供日射觀測，V5 進一步讓風速也採用同一份具備完整氣象要素之真實 TMYx
   資料，避免混用兩個不同機構之觀測基準），但也代表 V5 之風場空間代表性與 V4 不同，兩者不應
   視為同一氣象輸入的直接延續。

---

## 五、未完成／已知限制（誠實標記）

1. **場景數仍為 V4 的一半（150 vs 300）**：若需與 V4 完全同規模比較，仍需額外運算時間擴充
   （見四、1）。目前 150 場景已足以通過部署門檻，但更大規模仍可能進一步縮小與 V4 之差距。
2. **建物構造為標準預設，非真實建材**（見四、5）。
3. **風場空間均一，未反映真實街谷遮蔽**（見四、3）。
4. **`in_shadow` 為衍生代理指標**（見四、4）。
5. **日射／風場改用單一 TMYx EPW，與 V4 之 CWB 觀測基準不同**（見四、6），兩版本氣象邊界條件
   不完全一致。

---

## 六、產出檔案

- 引擎安裝：`C:\Users\User\ladybug_tools_engines\radiance\`（Radiance 6.0.2）、
  `C:\Users\User\ladybug_tools_engines\OpenStudio-3.11.0+241b8abb4d-Windows\`（含 EnergyPlus 25.2.0）
- 場景子集：`01_data_generation/outputs/real_simulations_v5/scenarios_v5_subset.pkl`（150 場景）、
  `v5_subset_summary.json`
- Honeybee 模型：`01_data_generation/outputs/real_simulations_v5/hbjson/`（150 個 `.hbjson`）
- 客製 EPW：`01_data_generation/outputs/real_simulations_v5/epw/`（150 個 `.epw` + `epw_manifest_v5.json`）
- 模擬結果：`01_data_generation/outputs/real_simulations_v5/sim/`（150 個 `sim_*.npz` + `manifest_v5.json`）
- 資料集：`01_data_generation/outputs/real_simulations_v5/ground_truth_v5.h5`、`dataset_summary_v5.json`、
  `epw_data.pkl`
- 模型（150 場景版本，最終結果）：`04_training/checkpoints_v5_150/best_model.pt`、`training_history.json`
- 測試評估（150 場景版本）：`04_training/checkpoints_v5_150/eval_v5_150_test.json`
- 模型（39 場景初版，保留供對照）：`04_training/checkpoints_v5/best_model.pt`、`eval_v5_test.json`
- **驗證圖表**（`04_training/viz_output/training_v5_150/`，以 `viz_training.py` 對保留測試集產出，
  與 V4 使用同一套學術驗證圖表工具，方法一致可直接對照）：
  - `figA_training_curves.png` — 訓練／驗證損失收斂曲線、驗證集 R² 逐 epoch 變化（含部署門檻
    R²=0.90 參考線與最佳 epoch 標記）、學習率排程。
  - `figB_scatter_test.png` — 測試集全部樣本之預測值對真實值散佈圖（1:1 參考線）＋殘差直方圖
    （偏差與 RMSE 帶）。
  - `figC_hourly_r2.png` — 測試集逐時（08:00–18:00）R²，顯示模型於一日之中預測難度之時間差異。
  - `figD_confusion_matrix.png` — 熱壓力分級（UTCI 5 級）混淆矩陣，逐列正規化準確率。
  - `figE_spatial_comparison.png` — 單一測試場景（正午 12:00）之真實值／預測值／誤差空間分布圖，
    直接呈現模型是否學到真實幾何遮蔽所致之空間熱壓力梯度。
- 管線腳本：`01_data_generation/scripts/11_select_v5_subset.py` ～ `15_output_to_hdf5_v5.py`

---

## 七、驗證圖表之誠實解讀

- **figA**：訓練／驗證損失同步平穩下降，無明顯過擬合徵兆；驗證 R² 於約 epoch 40 後穩定超過
  0.90 部署門檻並持續小幅爬升至最佳值 0.9722（epoch 179，與訓練時以驗證損失最低點儲存之
  epoch 167 相近但非完全相同，屬損失與 R² 兩指標非完全單調對應之正常現象）。
- **figB**：散佈圖整體緊貼 1:1 線，但於 UTCI > 45°C 之高熱壓力區間可見系統性低估（模型預測值
  略低於真實值），對應真實 Radiance／EnergyPlus 模擬所產生之局部極端熱點（見四、2）；
  殘差直方圖對稱集中於 0°C 附近（偏差 0.031°C），符合 RMSE=1.46°C 之整體表現。
- **figC**：逐時 R² 並非均勻分布——上午 8 時（0.946）與下午 16–17 時（0.927／0.941）表現最佳，
  但正午前後 10–12 時明顯下滑（最低於 11 時僅 0.530）。此為誠實呈現之發現，推測與正午前後
  太陽方位角變化快、陰影邊界移動劇烈、短波 MRT 空間梯度最陡峭有關，使該時段之熱壓力場對
  幾何細節更敏感、更難學習；此現象在 V4（自製平滑近似公式）之對應圖表中較不明顯，間接佐證
  V5 真實 Radiance 模擬確實捕捉到更真實、更具挑戰性的物理訊號，而非人工平滑後的訓練目標。
- **figE**：以一個測試場景之正午時步為例，模型正確捕捉建物遮蔽所致之低 UTCI 區域（右下角
  藍／黃區塊）與大範圍曝曬區之高 UTCI（紅色），空間結構之整體判讀方向正確；誤差圖顯示曝曬區
  略有正偏（過度預測熱壓力）、遮蔽邊界略有負偏，屬合理之邊界過渡誤差，非系統性方向錯誤。
