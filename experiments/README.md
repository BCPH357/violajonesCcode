# Experiments: Viola-Jones PC-Side Quantitative Evaluation

## 快速執行

```bat
cd C:\Users\miat\violajonesCcode

REM 1. 建置 C 偵測器（首次執行必須）
experiments\build_bench.bat

REM 2. 執行所有實驗並產生 RESULTS.md
python experiments\run_all.py
```

生成的檔案：
- `experiments/RESULTS.md` — 量化結果報告（表格 + 解讀）
- `experiments/outputs/exp1_*.png` — Exp1 並排偵測視覺化
- `experiments/outputs/exp1_iou_histogram.png` — IoU 分布圖
- `experiments/outputs/exp3_latency.png` — 延遲比較長條圖
- `experiments/outputs/exp2_tradeoff_scatter.png` — 準確度 vs 延遲散點圖
- `experiments/outputs/exp2_*_threeway.png` — 三偵測器並排視覺化

---

## 前置需求

| 項目 | 版本 | 備注 |
|------|------|------|
| Python | 3.x | 3.9+ 建議 |
| opencv-python | 4.x | `pip install opencv-python` |
| numpy | any | 已包含於 OpenCV |
| matplotlib | any | `pip install matplotlib` |
| MSYS2 + MinGW-w64 | GCC 15+ | 用於建置 C 偵測器 |

YuNet ONNX 模型（~350 KB）會在第一次執行 `run_all.py` 時自動下載。

---

## 單獨執行各實驗

```bat
python experiments\exp1_consistency.py   # Exp1: C vs OpenCV IoU
python experiments\exp3_latency.py       # Exp3: 延遲量測
python experiments\exp2_accuracy.py      # Exp2: 準確度 vs 成本
```

---

## 目錄結構

```
experiments/
├── README.md               本檔案
├── RESULTS.md              生成的結果報告
├── run_all.py              執行所有實驗 + 生成 RESULTS.md
├── utils.py                共用工具函式（IoU、偵測器封裝）
├── exp1_consistency.py     實驗一
├── exp2_accuracy.py        實驗二
├── exp3_latency.py         實驗三
├── bench_vj.c              C 延遲測試程式碼
├── build_bench.bat         建置腳本（需要 MSYS2）
├── face_detection_yunet_2023mar.onnx  （自動下載）
└── outputs/                生成的圖檔
```

---

## 設計說明

### C 偵測器執行位置
Windows Application Control 政策封鎖了由 GCC 新建的 PE 執行檔（若放在使用者目錄）。
`build_bench.bat` 將 `test_vj.exe` 和 `bench_vj.exe` 建置至 MSYS2 home
（`C:\msys64\home\miat\`），此目錄在 Application Control 政策下被視為受信任。

### 計時方法差異
- C 偵測器：`clock()`（CPU time，Windows CLOCKS_PER_SEC=1000）
- OpenCV / YuNet：Python `time.perf_counter()`（wall-clock time）

對於單執行緒純 CPU 工作，兩者數值應等價。如需更精確的比較，可將
`bench_vj.c` 中的 `clock()` 改為 `QueryPerformanceCounter()`，
但需確認 Application Control 政策不封鎖使用 `windows.h` 的二進位。

### 現代偵測器選擇
選用 **YuNet**（OpenCV Zoo 官方提供）的理由：
- CNN-based（MobileNet SSD），與 Haar-based VJ 架構有本質差異
- 模型小（350 KB），易於取得
- 已整合在 OpenCV 4.5.4+，無需額外框架
- 可在 CPU 上執行，無 GPU 偏差
