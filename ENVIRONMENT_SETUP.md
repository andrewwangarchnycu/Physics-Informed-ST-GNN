# 跨電腦環境建立指南（urban-thermal-gnn ML 訓練）

本文件說明如何在**另一部電腦**上，透過本專案已連線的 GitHub repo
(`https://github.com/andrewwangarchnycu/Physics-Informed-ST-GNN`)，建立一個
可重現的標準環境並執行 `urban-thermal-gnn/04_training` 之機器學習訓練。

**最重要的前提認知**：`git clone` **不會**帶來訓練所需的大型資料檔與模型權重——
`.gitignore` 已將 `*.h5`、`*.pkl`、`*.pt`、`*.epw` 等副檔名全數排除（見下方
「未受 git 追蹤、須手動搬移的檔案」）。只做 `git pull` 不做檔案搬移，訓練腳本
會直接因檔案不存在而報錯。

---

## 1. 系統需求

| 項目 | 需求 |
|---|---|
| 作業系統 | Windows 10/11（與開發機一致；程式碼未特別測試 Linux/Mac 路徑分隔字元） |
| Python | **3.11.x**（開發機為 3.11.0；避免用 3.13/3.14，部分套件尚未提供對應 wheel） |
| GPU（選用但強烈建議） | NVIDIA GPU + 對應驅動；無 GPU 亦可跑，`device` 會自動 fallback 至 CPU，但單場景推論延遲會從約 60ms 上升至數秒 |
| 硬碟空間 | 至少 **5 GB** 可用空間（模擬資料 + 多組 checkpoint，詳見第 4 節） |
| Git | 已安裝並已能存取本 repo（依題述「已連線」，此步驟可略過） |

---

## 2. 建立 Python 虛擬環境

```powershell
# 於 repo 根目錄
python -m venv .venv
.venv\Scripts\activate

python -m pip install --upgrade pip
```

> 開發機目前**沒有**使用虛擬環境（全域安裝），這是應避免的做法——新電腦請務必
> 用 venv 隔離，避免與系統既有 Python 套件衝突。

---

## 3. 安裝依賴套件

### 3.1 先裝 PyTorch（務必先於 requirements.txt，且需對應目標機器的 GPU）

`requirements.txt` **刻意不鎖定** torch 版本，因為 torch 的正確安裝指令因 GPU
世代與 CUDA 驅動版本而異，锁定版本反而在跨機器時最容易裝錯。

於 <https://pytorch.org/get-started/locally/> 依目標電腦之作業系統／GPU／CUDA
版本產生安裝指令，例如：

```powershell
# 範例：CUDA 13.0，適用於 RTX 50 系列（Blackwell 架構）等新顯卡
pip install torch --index-url https://download.pytorch.org/whl/cu130

# 較舊顯卡（如 RTX 30/40 系列）通常 CUDA 12.4/12.6 穩定版即已足夠：
pip install torch --index-url https://download.pytorch.org/whl/cu124

# 若目標機器無 NVIDIA GPU，改裝 CPU 版：
pip install torch
```

> **開發機上曾發現並已修復的真實問題**：開發機雖配備 NVIDIA GeForce RTX
> 5070 Ti Laptop GPU，但先前作用中環境安裝的是 `torch==2.11.0+cpu`
> （CPU-only 版本，`torch.cuda.is_available()` 回傳 `False`），代表先前
> 訓練/推論實際上都在 CPU 上跑。已改用上方 `cu130` 指令重新安裝為
> `torch==2.13.0+cu130`，重新驗證 `torch.cuda.is_available()` 已回傳
> `True`，並以真實訓練好的 checkpoint 在 GPU 上完成一次完整 NSGA-II
> 最佳化任務（318.9 秒，相較修復前 CPU 版本之 953.7 秒約快 3 倍）確認
> GPU 管線可正常運作。新電腦安裝時同樣務必於裝完 torch 後立即執行下方
> 第 5 節的驗證腳本，確認 `torch.cuda.is_available() == True`，避免重蹈覆轍。
>
> RTX 50 系列（Blackwell 架構，如 RTX 5070 Ti，compute capability 12.0）
> 需要 CUDA 12.8 以上（建議直接用 cu130，對應 driver 支援之 CUDA 13.1）
> 才有完整原生核心支援；若新電腦是更舊的 GPU（如 RTX 30/40 系列），
> CUDA 12.4/12.6 的穩定版通常已足夠，且該版本線提供的 torch 版本較新
> （最高到 2.13.0）。

### 3.2 安裝其餘依賴

```powershell
pip install -r urban-thermal-gnn/requirements.txt
```

`requirements.txt` 已依實際 import 分組並附註何時才需要（核心訓練 / GIS 資料
生成 / 熱舒適計算 / 部署伺服器 / 超參數搜尋），若只需執行
`04_training/train.py`（使用既有 `.h5` 資料，不重新跑 GIS 擷取或 EnergyPlus
模擬），只需核心分組即可，可省略 GIS／熱舒適／部署三組。

---

## 4. 未受 git 追蹤、須手動搬移的檔案

以下檔案因 `.gitignore` 排除（`*.h5`、`*.pkl`、`*.pt`、`*.epw`），`git clone`
後**不會存在**於新電腦，必須另行搬移（USB 隨身碟、區網共享、雲端硬碟皆可，
純檔案複製即可，無需透過 git）：

### 4.1 訓練資料（`01_data_generation/outputs/raw_simulations/`）

| 檔案 | 大小 | 用途 |
|---|---|---|
| `ground_truth.h5` | 27 MB | 初版訓練資料（`dim_air=8`），對應 `checkpoints_v2` |
| `ground_truth_v2.h5` | 633 MB | 訓練資料 v2（`dim_air=9`，含地表溫度特徵），對應 `checkpoints_v2_fixed`（**論文正文所報告之 R²=0.9965 模型**） |
| `ground_truth_v3.h5` | 633 MB | 訓練資料 v3（`dim_air=10`，另含街道峽谷 H/W 特徵），對應 `checkpoints_v3` |
| `epw_data.pkl` | 944 KB | EnergyPlus 氣象檔前處理快取（`FitnessEvaluator`／`app.py` 皆須） |
| `scenarios.pkl` | 392 KB | 隨機生成場景之幾何定義 |

> 只需搬移實際會用到的那一組 `.h5`。若只延續論文正文結果，僅需
> `ground_truth_v2.h5` + `epw_data.pkl` + `scenarios.pkl`（共約 634 MB）。

### 4.2 模型權重（`urban-thermal-gnn/checkpoints_*/` 資料夾，**注意不在 repo 根目錄，而在 `urban-thermal-gnn/` 之下**）

| 資料夾 | `best_model.pt` 大小 | epoch / val R² / dim_air |
|---|---|---|
| `urban-thermal-gnn/checkpoints_v2/` | 4.9 MB | epoch=90, R²=0.9968, dim_air=8 |
| `urban-thermal-gnn/checkpoints_v2_fixed/` | 4.9 MB | epoch=129, R²=0.9965, dim_air=9（**論文正文使用之權重**） |
| `urban-thermal-gnn/checkpoints_v3/` | 4.9 MB | epoch=98, R²=0.9965, dim_air=10 |
| `urban-thermal-gnn/04_training/ablation_ckpts/{V0,V3,V4,V5,V6}/` | 各 0.4--4.9 MB | 第四章消融實驗各變體權重，僅需重現消融結果時才需要 |

### 4.3 建議搬移方式

以下指令皆假設在 **`urban-thermal-gnn/` 目錄內**執行（而非 repo 根目錄）：

```powershell
# 於開發機（urban-thermal-gnn/ 內），將必要檔案打包（以 v2_fixed 這條「論文正文」路徑為例）
robocopy "01_data_generation\outputs\raw_simulations" "<搬移目的地>\raw_simulations" ground_truth_v2.h5 epw_data.pkl scenarios.pkl
robocopy "checkpoints_v2_fixed" "<搬移目的地>\checkpoints_v2_fixed" best_model.pt

# 於新電腦（同樣在 urban-thermal-gnn/ 內），複製回完全相同的相對路徑
robocopy "<搬移來源>\raw_simulations" "01_data_generation\outputs\raw_simulations" ground_truth_v2.h5 epw_data.pkl scenarios.pkl
robocopy "<搬移來源>\checkpoints_v2_fixed" "checkpoints_v2_fixed" best_model.pt
```

路徑必須與開發機**完全一致**（相對於 `urban-thermal-gnn/` 目錄），因為 `train.py`／
`app.py`／`run_real_nsga2.py` 皆以寫死的相對路徑讀取這些檔案（見
`urban-thermal-gnn/04_training/train.py:310-312`、
`urban-thermal-gnn/06_deployment/app.py:56-58`）。

---

## 5. 環境驗證腳本

新電腦裝完套件、搬完資料後，於 repo 根目錄執行：

```powershell
python urban-thermal-gnn\check_environment.py
```

此腳本會檢查：Python 版本、所有必要套件是否安裝、torch 是否偵測到 GPU、
以及第 4 節列出的關鍵資料/權重檔案是否存在於正確路徑，並輸出一份
PASS／FAIL 清單。全部 PASS 後才建議啟動實際訓練。

---

## 6. 啟動訓練

驗證通過後，於 `urban-thermal-gnn/04_training/` 執行：

```powershell
python train.py --h5 ../01_data_generation/outputs/raw_simulations/ground_truth_v2.h5 ^
                 --scenarios ../01_data_generation/outputs/raw_simulations/scenarios.pkl ^
                 --epw ../01_data_generation/outputs/raw_simulations/epw_data.pkl
```

（預設參數即指向上述路徑，若第 4 節之搬移路徑完全一致，可省略上述參數直接
執行 `python train.py`。）
