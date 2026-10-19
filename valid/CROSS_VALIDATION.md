# GPU 十折交叉驗證

在專案根目錄、已安裝 CUDA PyTorch 的環境執行。訓練只支援 NVIDIA CUDA，沒有 CPU fallback。
舊訓練與推論程式不變；新增的設定引用 `constants.TRAINING_CONFIG`。

```powershell
python -m valid.train_ntu_cross_validation
```

預設讀取 constants 中的 `data_root`，依既有設定訓練 200 epochs / fold、batch size 32。
也可指定骨架資料夾（不是 RGB 影片資料夾）：

```powershell
python -m valid.train_ntu_cross_validation --data-root "D:\NTU60\nturgb+d_yolo_skeletons" --gpu 0 --batch-size 32
```

先產生分組與樣本統計，不訓練（這項檔案檢查不需要 GPU）：

```powershell
python -m valid.train_ntu_cross_validation --data-root "D:\NTU60\nturgb+d_yolo_skeletons" --prepare-only
```

先測單折流程，之後正式實驗需重新執行全部十折：

```powershell
python -m valid.train_ntu_cross_validation --folds 1 --epochs 1 --batch-size 8
```

## 實驗規則

- `cross_validation_config.py` 固定十組人物；第 k 組測試、下一組驗證（第十折回到第一組），其餘八組訓練。
- 訓練與驗證只使用 constants 的已知類別；測試使用全部可用類別。未知類別不參與選模型或調門檻。
- 解析資料夾第一層的 `.npy`，支援 `S001C001P001R001A001.npy` 與 `_rgb.npy`；拒絕同一樣本的重複版本。
- 要求 P001–P040 都有資料，且所選各折 train/val/test 都包含所有已知類別；測試另須包含未知類別。
- manifest 提供各人物、各動作樣本數；不假定實際轉換後數量等於官方原始影片數。prepare-only 只檢查清單；陣列形狀、NaN/Inf、全零在讀取時檢查。
- 每折從頭建立模型、中心、optimizer；不讀舊 checkpoint，不跨折續訓。`last.pth` 保存狀態供檢查，目前沒有自動 resume 功能。
- 保留現有四項 loss（分類、重建、中心、稀疏記憶權重）；訓練用 AMP，校準與測試用 GPU FP32。
- 新實驗逐筆正規化，避免 batch 組合影響結果；這與舊版 batch 標準差正規化不同。固定 seed 和關閉 cuDNN benchmark 不保證跨硬體逐位一致。
- 最佳 checkpoint 依已知驗證 Accuracy 選擇；平手保留較早 epoch。
- 類別中心只由訓練資料建立；距離與 log1p(MSE) 縮放範圍只由已知驗證資料建立。
- 預設融合分數為 0.6 × 距離 + 0.4 × 重建誤差；測試不重新縮放、不截斷分數。
- 以已知驗證分數第 95 百分位（higher）為門檻，嚴格 `score > threshold` 判未知。驗證集經驗誤報率至多 5%，不保證測試集也是 5%。
- 這是自訂人物互斥十折，不是 NTU 官方 Cross-Subject protocol。測試結果不能用於回頭調超參數。

## 輸出

預設位置：`output/cross_validation/<時間戳>/`。`--output` 可指定新目錄；已存在會拒絕覆寫。

- `experiment.json`：分組、實際資料統計、有效設定、GPU / PyTorch 版本。
- `fold_01/` 至 `fold_10/`：manifest、history.jsonl、best_val.pth、last.pth、calibration.json、radar_meta_params.pth、metrics.json、test_predictions.npz。
- `test_predictions.npz` 包含檔案路徑、已知標籤索引（未知為 -1）、未知旗標、分類預測、距離、MSE、融合分數。
- `summary.csv`：各折指標；`summary.json`：各折指標及不加權平均、樣本標準差（ddof=1）。指標均採 0–1，不是百分數。
- 僅完成部分折時 `complete_10_fold=false`；單折標準差為 null。每完成一折更新總表。
- known_accuracy / known_macro_f1 是拒絕判定前的已知分類表現；未知偵測正類為 unknown=1，混淆矩陣順序為 known、unknown。

每折的雷達參數只供這次實驗使用；沒有覆寫正式推論的模型或 radar_meta_params.pth。
由於正規化與比較符號不同，不應直接把新權重及參數替換進舊推論流程。

## 驗證程式

```powershell
python -m unittest valid.test_cross_validation -v
$env:RUN_CV_GPU_TEST = "1"
python -m unittest valid.test_cross_validation -v
Remove-Item Env:RUN_CV_GPU_TEST
```

GPU smoke test 在暫存資料上執行真實單 epoch 訓練、校準、測試及輸出驗證，不能視為 NTU60 模型效能。
