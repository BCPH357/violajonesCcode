# CLAUDE.md — Viola-Jones 專案完整指引

## 專案目的

這是一個純 C 的 Viola-Jones 人臉偵測器，目標是把單窗口 cascade 評估函式
`vj_evaluate_window_fixed()` 硬體化到 FPGA（PYNQ-Z2 / Zynq-7020）上，
作為 rPPG 訊號擷取晶片的 ROI 偵測前端。

**只有單窗口評估器要上 FPGA；多尺度掃描迴圈、grouping、best_face 留在 PS 軟體。**

---

## 目前進度快照

| 階段 | 狀態 |
|------|------|
| Step 0：數值範圍調查 | ✅ 完成（見 docs/QUANT_ANALYSIS.md） |
| Step 1：定點化框架 + 吸收 inv_area | ✅ 完成 |
| Step 2：weight 乘法 → 移位（shift/negate） | ✅ 完成，100% 一致 |
| Step 3：座標縮放 → 預算查表（Route A） | ✅ 完成，100% 一致 |
| Step 4：消除 sqrtf（四分支平方比較） | ✅ 完成，100% 一致 |
| Step 5：全回歸驗證（四圖 0 翻面，IoU=1.000） | ✅ 完成，PASS |
| HLS wrapper 準備（vj_cascade_top.cpp） | ✅ 完成，PC 預驗 PASS |
| **Vitis HLS C-synthesis** | ⬜ **下一步** |

---

## 檔案結構

```
violajonesCcode/
├── haarcascade_frontalface_default.xml  ← OpenCV 訓練好的 cascade 參數
├── src/
│   ├── vj_types.h          ← 共用型別 (vj_cascade_t, VJ_MAX_RECTS=3, ...)
│   ├── vj_integral.h/c     ← 積分圖計算；vj_rect_sum() / vj_rect_sum_sq() 為 static inline
│   ├── vj_cascade_data.h/c ← 從 XML 載入浮點 cascade 參數（PS 端用）
│   ├── vj_detect.h/c       ← 浮點版偵測器（golden reference）
│   ├── vj_fixed.h          ← 定點框架型別定義 + HLS entry point 宣告
│   └── vj_fixed.c          ← 定點實作（演算法黃金參考，修改前請看注意事項）
├── hls/
│   ├── vj_cascade_top.cpp  ← Vitis HLS top function（AXI-Lite wrapper）
│   ├── hls_test_data.h     ← 測試資料（II 陣列、兩個 golden 向量、cascade 表）
│   ├── ap_int.h            ← 本地 ap_int<N> stub（語法驗證用，非 Vitis HLS 實作）
│   └── test_ap_int_syntax.cpp ← __SYNTHESIS__ + ap_int<82> 路徑語法驗證
├── docs/
│   ├── HW_OPTIMIZATION_ROADMAP.md ← Steps 0-5 的設計思路與驗證方法
│   └── QUANT_ANALYSIS.md          ← Step 0 量化分析（Q 格式依據）
├── experiments/
│   ├── step1_verify.py ~ step5_verify.py ← 各步驟回歸驗證腳本
│   ├── gen_hls_data.py    ← 產生 hls/hls_test_data.h
│   └── ...                ← 其他實驗腳本（exp1/2/3）
└── test/
    ├── testt.jpg, lena.jpg, test.jpg, test_image.png ← 測試影像
    └── test_main.c
```

---

## 演算法核心：vj_evaluate_window_fixed()

**入口**（HLS synthesis target）：
```c
int vj_evaluate_window_fixed(
    const vj_cascade_fixed_t  *fc,           // 定點 cascade 參數表
    const vj_scaled_feature_t *scaled_feats, // 當前 scale 的預算座標（ROM 查表）
    const uint32_t *ii,                       // 積分圖
    const uint64_t *sii,                      // 平方積分圖
    int ii_stride,
    int win_x, int win_y, int win_w, int win_h
);
```

**Call graph（全部 inline / static）：**
```
vj_evaluate_window_fixed()
  ├── vj_rect_sum()        ← 4 array reads + 3 adds（inline，vj_integral.h）
  ├── vj_rect_sum_sq()     ← 同上，對 sii
  ├── vj_apply_weight()    ← switch on 2-bit weight_code（static inline，vj_fixed.h）
  └── cmp_lhs_lt_rhs()     ← 四分支符號安全平方比較（static inline，vj_fixed.c）
```

**浮點操作消除紀錄：**
| 操作 | 消除方式 | Step |
|------|---------|------|
| `inv_area` 除法 | 兩邊同乘 area（`feat_val_raw` 不除） | 1 |
| `weight * rect_sum` 乘法 | shift/negate（weight 只有 {-1,+2,+3}） | 2 |
| `rc->x * scale_x` 座標浮點乘 | 預算 vj_scaled_feature_t 查表（Route A） | 3 |
| `sqrtf(variance)` | 四分支 `wide_t` 平方比較 | 4 |

---

## 定點 Q 格式

```
threshold_q15     = round(threshold   * 2^15)   int32_t   Q1.15
left_val_q10      = round(left_val    * 2^10)   int32_t   Q6.10
right_val_q10     = round(right_val   * 2^10)   int32_t   Q6.10
stage_sum                                        int32_t   Q6.10  （與 left/right_val 同格式）
stage_threshold_q10 = round(stage_thr * 2^10)   int32_t   Q6.10
feat_val_raw                                     int32_t   整數（積分圖和）
va_sq = sqsum*area - sum^2                       int64_t   = variance × area²
```

---

## wide_t（Step 4 平方比較）

`cmp_lhs_lt_rhs()` 中需要比較 `L²` 和 `T²×va_sq`，數值最大約 1.44×10²⁴，超出 int64。
用條件編譯：

```c
/* src/vj_fixed.c */
#ifdef __SYNTHESIS__
#  include "ap_int.h"
   typedef ap_int<82> wide_t;   /* Vitis HLS：N=82（理論上界，保守值） */
#else
   typedef __int128 wide_t;     /* PC golden：__int128（GCC 擴充） */
#endif
```

- 理論上界：`ap_uint<81>` / `ap_int<82>`（見 hls_test_data.h `HLS_APINT_N_SIGNED=82`）
- 實測上界（testt.jpg）：`ap_uint<73>` / `ap_int<74>`
- **進 HLS 後依 synthesis report 再縮窄，現在先用 82**

---

## 資料結構

### vj_cascade_fixed_t（cascade 主結構）
```c
typedef struct {
    int num_stages;           // 25
    int num_weak_total;       // 2913
    int num_features;         // 2913
    int window_w, window_h;  // 24, 24（base window）
    const vj_stage_fixed_t   *stages;           // [25]
    const vj_wc_fixed_t      *weak_classifiers; // [2913]
    const vj_feature_fixed_t *features_fixed;   // [2913]
} vj_cascade_fixed_t;
```

### vj_scaled_feature_t（每 scale 一份，Route A 查表）
```c
typedef struct {
    vj_scaled_rect_t rects[VJ_MAX_RECTS];  // [3]，dx/dy/rw/rh 為 int16_t
    /* num_rects 從 features_fixed 讀，不重複存 */
} vj_scaled_feature_t;
/* 容量：~68 KB / scale × ~20 scales ≈ 1.33 MB total */
```

---

## HLS Wrapper（hls/vj_cascade_top.cpp）

```cpp
int vj_cascade_top(int win_x, int win_y, int win_w, int win_h)
{
#pragma HLS INTERFACE s_axilite port=win_x   bundle=CTRL
#pragma HLS INTERFACE s_axilite port=win_y   bundle=CTRL
#pragma HLS INTERFACE s_axilite port=win_w   bundle=CTRL
#pragma HLS INTERFACE s_axilite port=win_h   bundle=CTRL
#pragma HLS INTERFACE s_axilite port=return  bundle=CTRL
    return vj_evaluate_window_fixed(&g_cascade, g_scaled,
                                    g_test_ii, g_test_sii, TEST_II_STRIDE,
                                    win_x, win_y, win_w, win_h);
}
```

**hls_test_data.h 重要符號：**
- `g_test_ii[3969]` / `g_test_sii[3969]`：testt.jpg 的 62×62 裁切 II（stride=63）
- `g_scaled_feats[2913]` / `#define g_scaled g_scaled_feats`：win=50×50 的查表
- `g_stages[25]`, `g_weak_classifiers[2913]`, `g_features_fixed[2913]`
- `g_cascade`：`vj_cascade_fixed_t` 完整組裝好的結構
- `TEST_II_STRIDE = HLS_TEST_II_STRIDE = 63`
- Pass vector：`(6,6,50,50)` → 期望 1；Reject vector：`(0,0,50,50)` → 期望 0

---

## 編譯驗證指令（Windows MSYS2 環境）

### PC 黃金驗證（C 模式）
```bash
gcc -std=c11 -x c -Isrc -Ihls -o hls/test_top.exe hls/vj_cascade_top.cpp src/vj_fixed.c
./hls/test_top.exe
# pass (6,6,50,50) -> 1  PASS
# reject (0,0,50,50) -> 0  PASS
```

### g++ C++14 模式（模擬 Vitis HLS 前端）
```bash
# 分開編：vj_fixed.c 用 gcc（C 模式），wrapper 用 g++（C++ 模式）
gcc -std=c11 -Isrc -c src/vj_fixed.c -o build_vj_fixed.o
g++ -std=c++14 -Isrc -Ihls -c hls/vj_cascade_top.cpp -o build_gpp_top.o
g++ -static-libstdc++ -static-libgcc -o hls/test_top_cpp_s.exe build_gpp_top.o build_vj_fixed.o
./hls/test_top_cpp_s.exe   # 需用 -static 才能在本機 AppLocker 政策下執行
```

### __SYNTHESIS__ + ap_int<82> 語法驗證
```bash
g++ -std=c++14 -Ihls -static-libstdc++ -static-libgcc \
    -o hls/test_ap_int_syntax.exe hls/test_ap_int_syntax.cpp
./hls/test_ap_int_syntax.exe
# ap_int<82> synthesis path: all 5 branches PASS
```

### vj_fixed.c 以 C++ 模式完整編譯（模擬 Vitis HLS csim 路徑）
```bash
g++ -std=c++14 -D__SYNTHESIS__ -Isrc -Ihls -x c++ -c src/vj_fixed.c -o build_vj_synth.o
# 應 exit 0（零 error，#pragma HLS 有 unknown-pragma warnings 是正常的）
```

**注意**：本機 Windows AppLocker 政策會阻擋 g++ 動態連結的 exe，
加 `-static-libstdc++ -static-libgcc` 即可執行。gcc 產生的 exe 不受此限。

---

## HLS 階段待解項（演算法已驗證，硬體可行性待 synthesis report）

### 1. 座標查表 1.33 MB > PYNQ-Z2 BRAM 約 630 KB

`vj_scaled_feature_t` 每 scale 68 KB × 約 20 scales = 1.33 MB，放不進 BRAM。
**拿到 C-synthesis 報告後**再決定：
- 縮減 scale 層數（限制掃描範圍）
- 收窄座標位寬（int16 → int8）
- 改放 DDR（AXI master 存取）
- 退回 Route B（scale_x 定點乘法，不查表）

### 2. `wide_t = ap_int<82>` 電路面積

理論上界 82-bit 乘法器面積可能過大。
**拿到 synthesis 資源報告後**依實際數值範圍縮窄：
- 實測上界：`ap_int<74>`（testt.jpg 全掃描）
- 理論上界：`ap_int<82>`（保守值，現用）

**現在不要提前優化這兩項。等 synthesis 數據。**

---

## HLS 第一步

在 Vitis HLS 建專案，加入：
- `src/vj_fixed.h`, `src/vj_fixed.c`
- `src/vj_types.h`, `src/vj_integral.h`, `src/vj_integral.c`
- `hls/vj_cascade_top.cpp`（top function = `vj_cascade_top`）
- `hls/hls_test_data.h`（C simulation testbench 資料）
- `hls/ap_int.h`（Vitis HLS 自己的 ap_int.h 會覆蓋此 stub）

跑 C simulation 確認 pass/reject 向量正確，再跑 C synthesis 拿 resource / timing 報告。

---

## 修改限制

### vj_fixed.c — 演算法黃金參考

此檔案的 `vj_evaluate_window_fixed()` 已與浮點版逐窗口等價（Steps 0-5 全通過，四圖 0 翻面）。

**允許改動**：
- 純機械性的 C++ 相容修正（malloc cast 等），不影響演算法邏輯

**禁止改動**：
- Q 格式、比較邏輯、wide_t 的分支結構
- 「在沒有 synthesis 數據前提前優化 HLS 問題」

### hls_test_data.h — 自動產生，不要手改

若需要更新，重跑：
```bash
cd violajonesCcode
python experiments/gen_hls_data.py
```
腳本會重新輸出 Section 1-6（包含 g_cascade、g_scaled、TEST_II_STRIDE）。

---

## cascade 基本數字

| 項目 | 值 |
|------|-----|
| Stages | 25 |
| Weak classifiers（總計） | 2913 |
| Features（總計） | 2913 |
| Base window | 24 × 24 px |
| Weight 取值 | {-1, +2, +3}（100% 可移位） |
| Rects per feature | 2 或 3（最多 VJ_MAX_RECTS=3） |

---

## 環境注意事項

- **建置工具**：MSYS2 / MinGW64（gcc + g++ 皆可用）
- **Python**：用於驗證腳本和 gen_hls_data.py（需 opencv-python, numpy）
- **AppLocker 限制**：g++ 動態連結 exe 被阻擋，加 `-static-libstdc++ -static-libgcc`
- **PowerShell 建議**：長時間 Python 腳本（step5_verify.py）用 PowerShell tool，
  Bash tool 有時會 segfault；step5 掃描四張圖很耗時，可能需要 3-5 分鐘
- **ap_int.h stub**：本地 `hls/ap_int.h` 以 `__int128` 模擬語法，
  Vitis HLS 的 `<ap_int.h>` 會自動覆蓋（HLS include path 優先）
