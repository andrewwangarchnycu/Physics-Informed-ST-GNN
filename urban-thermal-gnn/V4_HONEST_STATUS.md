# PI-ST-GNN V4 真實資料管線 — 誠實完成／未完成清單

> 本文件逐項誠實記錄 V4 版本「真實使用 cwb_data + moenviot 溫濕度 + OSM 建物 + ETH 樹冠」
> 建立 300 個真實訓練場景並訓練模型的實際完成狀況，包含所有已知限制與退回（fallback）情形。
> 原則：**只陳述實際做到的事，不誇大；用到回填／近似之處明確標記。**

生成日期：2026-07（V4 建置）

---

## 一、真實資料的實際使用情形（逐項）

| 資料來源 | 是否真實使用 | 實際用途 | 誠實限制 |
|---|---|---|---|
| **OSM 建物足跡** | ✅ 100% 真實 | 每場景真實建物平面幾何（陰影、SVF、街廓） | 台灣 OSM 樓高標籤稀疏，僅約 **14%** 建物有真實 `height`／`building:levels`，其餘以 3 層（10.8m）預設回填 |
| **ETH GlobalCanopyHeight** | ✅ 真實 | 每場景樹冠位置與高度（植栽遮蔽、蒸散） | 每場景樹木上限 80 株（保留最高者）以維持模擬可行性；10m 解析度 |
| **MOENV IoT 氣溫** | ✅ 真實 | **281/300 場景**以場景所在真實測站逐時氣溫驅動 | 19 場景測站故障（讀值卡在 ~0°C），退回真實 CWB 區域氣溫 |
| **MOENV IoT 濕度** | ✅ 真實 | **212/300 場景**以真實測站逐時濕度驅動 | 88 場景無可用真實濕度（感測器卡 0/100%），退回真實 CWB 區域濕度 |
| **CWB 測站（C0D660）** | ✅ 真實 | 逐時風速／風向；氣溫濕度之退回來源 | 單一測站；**該站未提供日射觀測** |
| **日射（GHI/DNI/DHI）** | ⚠️ 非直接觀測 | 取自區域 TMY EPW 之典型日射剖面 | CWB 無日射資料，故採 TMY 氣候平均而非 2025 當日實測 |

---

## 二、已完成並驗證的項目

1. **修復從未運作的 OSM 建物擷取**：`osm_loader.py` 兩個被靜默吞掉的例外（overpy `Decimal` 座標與 `float`
   混算、多值 `building:levels="11;10;12"` 解析）——此為先前「建物 100% 程序化」的真正根因。
2. **離線 pyosmium 區域擷取**（`osm_pbf_extract.py`）：一次讀入 taiwan-latest.osm.pbf，區域內
   **139,307 棟建物（20,741 具真實樓高標籤）**、43,614 個地表材質多邊形；每場址查詢瞬間完成，
   無 Overpass 限速。
3. **300 個真實場景選址**（`06_select_real_sites_v4.py`）：以陽明交大光復校區為錨點、擴及新竹／桃園／
   苗栗／臺中；候選取自真實 IoT 測站座標（去重 ≥150m）。**304 個通過全部四項條件，取前 300 名**。
   每場景：中心即真實測站（最近距離 0m）、120m 內 ≥3 棟真實建物、地表 ≥2 類物理表面、有 ETH 樹冠像元。
4. **300 個真實幾何場景**（`07_build_real_scenarios_v4.py`）：真實 OSM 建物（平均 11.3 棟/場景）+
   ETH 樹冠 + 材質土地覆蓋；6/7/8 月各 100 場景。
5. **真實 IoT 逐時序列萃取**（`08_extract_site_iot_v4.py`）：掃描全部 86×2 個每日 CSV，
   聚合為各測站逐月逐時剖面；含故障感測器過濾與合理性驗證。
6. **V4 真實場景模擬**（`09_run_real_sim_v4.py`）：物理式引擎（SVF 射線投射、建物/樹冠陰影、
   能量平衡 MRT、街谷風遮蔽、pythermalcomfort UTCI），以真實 IoT 氣溫濕度 + CWB 風 + TMY 日射驅動。
   **300/300 場景完成**，UTCI 平均 42.3°C（合理熱壓力範圍）。
7. **抓出並修正三個會破壞科學可信度的資料錯誤**：
   - clear-sky 日射過強（MRT 灌水 +10°C）→ 改用 TMY EPW 實測日射剖面；
   - 濕度感測器卡在 100%（UTCI 灌水至 55°C）→ 過濾 + 合理性驗證 + CWB 退回；
   - 部分溫度感測器卡在 0°C（UTCI 掉到 9°C）→ 逐月剖面驗證 + CWB 退回；
   - UTCI 極端外推值（>50°C 為多項式外推失真）→ 裁切至操作範圍 [-30, 50]°C。
8. **`ground_truth_v4.h5`**：300 場景、逐月分層 train/val/test = 210/45/45、
   **含 `sensor_utci`（281 場景具真實 IoT 直接監督信號）**——修復 `dataset.py` 先前的死碼路徑。
9. **視覺化**：`fig_selected_real_sites.png`（300 場址真實地圖）、`fig_real_scene_graph.png`
   （真實場景異質圖建構）；已插入論文第三章對應位置。
10. **LaTeX 誠實更正**：第三章原稱「以 Ladybug Tools／EnergyPlus／Radiance 執行模擬」，
    **更正為物理式 Python 引擎之誠實描述**，並明確說明未呼叫該等外部套件及其原因。

---

## 三、未完成／已知限制（誠實標記）

1. **未使用 Ladybug Tools／EnergyPlus／Radiance**：V4 模擬為可批次執行於 300 場景的物理近似引擎
   （SVF 為幾何射線近似而非 Radiance 輻射傳遞；MRT 為經驗能量平衡而非 EnergyPlus 全熱力學求解）。
   此為在數百場景資料集規模下的運算可行性取捨。若需真正的 Radiance/EnergyPlus，須大幅縮減場景數
   或投入數天等級之運算，屬未來工作。
2. **建物樓高僅約 14% 為真實 OSM 標籤**，其餘為 3 層預設回填——足跡真實，樓高部分真實。
3. **日射為 TMY 氣候平均**，非 2025 當日實測（CWB 該站無日射觀測）。
4. **模擬幾何有裁剪**：每場景取最近 30 棟建物、20 株樹、4m 感測格點（~340 節點/場景），
   以維持 CPU 物理運算可行；並非全數建物皆納入陰影計算。
5. **單一 CWB 測站**提供全區風場與退回氣象，空間代表性有限。

---

## 四、V4 模型訓練結果（已完成）

- 環境：Anaconda `PytorchGPU`，**CUDA，NVIDIA RTX 5070 Ti Laptop GPU**。
- 架構：PI-ST-GNN（RGCN + LSTM），dim_air=9，1,252,622 參數。
- 資料：300 真實場景，逐月分層 train/val/test = 210/45/45。
- 監督：UTCI 資料損失 + 物理軟約束損失 + **真實 IoT `sensor_utci` 直接監督**（lambda_sensing=0.5，
  281/300 場景具真實監督信號）。
- 訓練：200 epochs，AdamW + ReduceLROnPlateau，最佳於 epoch 195。

**保留測試集（45 個未見過之真實場景）結果：**

| 指標 | 數值 |
|---|---|
| **R²** | **0.9875** |
| **RMSE** | **0.513 °C** |
| **MAE** | **0.320 °C** |
| 準確率（±誤差門檻） | 95.8% |
| 部署門檻 R² ≥ 0.90 | ✅ PASS |

> 誠實對照：既有論文第四章之 R²=0.9957 為**程序化蒙地卡羅資料**所訓練之模型；V4 之 R²=0.9875
> 為**真實 OSM 幾何 + 真實 IoT 氣象**所訓練之模型，數值略低屬合理（真實資料含更多雜訊與異質性），
> 但其在**未見過真實場景**上仍達 RMSE 0.51°C，為真實世界應用可信度之直接證據。兩者為不同資料集之
> 獨立結果，不應混為一談。

**產出檔案：**
- 模型：`04_training/checkpoints_v4/best_model.pt`、`training_history.json`
- 測試評估：`04_training/checkpoints_v4/eval_v4_test.json`
- 結果圖：`04_training/viz_output/training_v4/`（figA 收斂曲線、figB 散點、figC 逐時 R²、
  figD 熱壓力分級混淆矩陣、figE 空間比較）
- 資料集：`01_data_generation/outputs/real_simulations_v4/ground_truth_v4.h5`、`scenarios_v4.pkl`、
  `site_iot_v4.pkl`、`manifest_v4.json`
- 選址：`01_data_generation/outputs/real_sites_v4/selected_real_sites.json`
- GIS 圖：`04_training/figures/fig_selected_real_sites.png`、`fig_real_scene_graph.png`
