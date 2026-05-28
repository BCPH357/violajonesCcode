# Step 0 量化分析報告（HLS 前置定點化基礎數據）

**對象**：`vj_evaluate_window` 的定點化規劃  
**測試影像**：testt.jpg (267×400)、lena.jpg (512×512)、test.jpg (1200×902)  
**方法**：靜態分析 `haarcascade_frontalface_default.xml` + Python 複刻插樁（全圖所有窗口，含 cascade early-exit）

---

## 1. 靜態分析：cascade 參數取值

### 1-A. Weight 取值集合（核心結論）

| 值 | 出現次數 | 移位表達 |
|----|---------|---------|
| **-1.0** | 2913 | `-x`（取負） |
| **+2.0** | 2012 | `x << 1` |
| **+3.0** | 1458 | `(x << 1) + x` |

**結論：weight 全為小整數，100% 可用移位加減取代，無乘法。** Step 2 可直接依此實作，無例外情況需要特別處理。

每個 feature 恰好有 2 或 3 個 rect，weight 分配慣例是：
- 2-rect feature：[-1, +2] 或 [-1, +3]
- 3-rect feature：[-1, +2, -1] 或類似

### 1-B. Weak classifier threshold

| 統計量 | 值 |
|-------|-----|
| min | -0.4022 |
| max | +0.6606 |
| mean | 0.0032 |
| std | 0.0631 |

**Q 格式建議**：`Q1.15`（1 整數位 + 符號 + 15 小數位 → int16_t）  
精度：2⁻¹⁵ ≈ 3.05×10⁻⁵，threshold 最小有效差異 ~0.001，15 小數位綽綽有餘。  
或使用 int32_t `Q1.23` 獲得更多餘裕（視後續乘法需求決定）。

### 1-C. left_val / right_val（weak classifier 投票值）

| 統計量 | left_val | right_val |
|-------|---------|----------|
| min | -30.0 | -30.0 |
| max | +4.651 | +10.0 |
| mean | -0.147 | -0.081 |
| std | 0.913 | 1.060 |

**注意**：min = -30.0 是極端值，但出現機率低（尾端）；大多數值在 ±2 以內（p1~p99 對應 stage_sum 結果）。  
**Q 格式建議**：`Q5.10`（5 整數位 + 符號 + 10 小數位 → int16_t 剛好）  
精度：2⁻¹⁰ ≈ 9.77×10⁻⁴，對 -30~+10 的投票值足夠。  
若擔心累積誤差可升至 `Q5.18`（int32_t）。

### 1-D. stage_threshold

| 統計量 | 值 |
|-------|-----|
| min | -5.0426 |
| max | -2.9928 |
| 全部負數 | ✓ |

**全部為負**，stage_sum 必須超過（大於）負數才通過。  
**Q 格式建議**：與 stage_sum 相同格式（`Q5.10` 或 `Q5.18`）。

### 1-E. num_weak per stage

```
Stage  0:   9  Stage  1:  16  Stage  2:  27  Stage  3:  32  Stage  4:  52
Stage  5:  53  Stage  6:  62  Stage  7:  72  Stage  8:  83  Stage  9:  91
Stage 10:  99  Stage 11: 115  Stage 12: 127  Stage 13: 135  Stage 14: 136
Stage 15: 137  Stage 16: 159  Stage 17: 155  Stage 18: 169  Stage 19: 196
Stage 20: 197  Stage 21: 181  Stage 22: 199  Stage 23: 211  Stage 24: 200
Total: 2913
```

**stage_sum 最大累積量**：worst case = 211 × 30 = 6330（絕對上界）；  
實際觀測 max |stage_sum| = 22.1（見動態範圍），遠低於理論上界。  
→ `Q5.10` 的整數部分（±31）在實測範圍 ±22.1 內安全，但考慮到 left_val 的 -30 極端值，建議使用 `Q6.10` 或 int32_t。

---

## 2. 動態範圍分析（Runtime，全圖插樁）

全圖總計：約 **6720 萬次** weak classifier 評估、約 **198.5 萬個** 獨立窗口。

### 2-A. feat_val_raw（integral image 加權和，除 inv_area 前）

| 統計量 | 值 |
|-------|-----|
| min | **-15,343,754** |
| max | **+13,299,927** |
| mean | -1,518 |
| p1 | -37,647 |
| p99 | +29,960 |

**整數位需求**：⌈log₂(15,343,754)⌉ = 24 位，加符號 = **25 bits 最小**。  
**Q 格式建議**：`int32_t`，不需要小數位（這是積分圖矩形和的整數加權組合，結果天然是整數）。  
int32_t 可表示 ±2,147,483,647，對 ±15.3M 有充足餘裕（無溢位風險）。

> **吸收 inv_area 後（Step 1 採用）**：比較式變為  
> `feat_val_raw < threshold × stdev × area`  
> 右邊 max ≈ 0.661 × 94.1 × 588,289 ≈ 36.6M，仍在 int32_t 範圍內（±2.1G）。  
> **中間量 `threshold × area`**：max ≈ 0.661 × 588,289 ≈ 388,886，可存 int32_t；再乘 stdev（≤94.1）需 int64_t 中間量（約 36.6M，再轉回 int32_t 無問題）。

### 2-B. feat_val_norm（除 inv_area 後，與 threshold×stdev 比較的量）

| 統計量 | 值 |
|-------|-----|
| min | -68.7 |
| max | +69.1 |
| mean | -0.802 |
| p1 | -8.78 |
| p99 | +4.88 |

**整數位需求**：⌈log₂(69.1)⌉ = 7 位 + 符號 = **8 bits 整數**。  
**Q 格式建議**：`Q7.8`（int16_t）或 `Q7.16`（int32_t，更多精度餘裕）。

### 2-C. variance

| 統計量 | 值 |
|-------|-----|
| min | 0.636 |
| max | 8,857 |
| mean | 922.9 |
| p1 | 4.66 |
| p99 | 5,810 |

**整數位需求**：⌈log₂(8857)⌉ = 14 位。  
**小數位需求**：min = 0.636，需要至少 1 小數位；實際建議 4~8 小數位以防低紋理窗口（stdev 接近 0 時精度最重要）。  
**Q 格式建議**：`Q14.10`（24 bits 用於值，放 int32_t）。  
注意：低紋理窗口 variance < 1 時，若小數位不足會讓 stdev ≈ 0，可能影響 Step 4 的符號判斷。

### 2-D. stdev

| 統計量 | 值 |
|-------|-----|
| min | 0.797 |
| max | 94.1 |
| mean | 24.6 |
| p1 | 2.16 |
| p99 | 76.2 |

**整數位需求**：⌈log₂(94.1)⌉ = 7 位 + 符號 = **8 bits**。  
**Q 格式建議**：`Q7.8`（int16_t）或 `Q7.16`（int32_t）。  
Step 4 消除 sqrt 後，stdev 本身不再需要儲存，改為直接操作 variance。

### 2-E. stage_sum（每 stage 累加後的值）

| 統計量 | 值 |
|-------|-----|
| min | -22.09 |
| max | +11.30 |
| mean | -5.16 |
| p1 | -12.30 |
| p99 | +2.27 |

**整數位需求**：⌈log₂(22.1)⌉ = 5 位 + 符號 = **6 bits 整數**。  
但 left/right_val 有 -30 的極端值，保守取 **6 bits 整數**（±63）。  
**Q 格式建議**：`Q6.10`（int32_t），與 left/right_val、stage_threshold 使用相同格式，便於直接比較。

### 2-F. threshold × stdev（weak classifier 比較的右端）

| 統計量 | 值 |
|-------|-----|
| min | -31.3 |
| max | +42.8 |
| p1 | -4.08 |
| p99 | +5.91 |

**與 feat_val_norm 相同數量級**，同用 `Q7.16`（int32_t）。

---

## 3. area 範圍

| 參數 | 值 |
|-----|-----|
| 最小 area（24×24）| 576 |
| 最大 area（~767×767）| ~588,289 |

int32_t 可表示 ±2.1G，最大 area ≈ 588,289 << 2.1G，安全。  
吸收 inv_area 後所有乘法都在 int32/int64 範圍內（見 2-A）。

---

## 4. 各步驟 Q 格式建議總表

| 變數 | 實際範圍 | 建議 Q 格式 | 型別 | 備注 |
|------|---------|------------|------|------|
| `weight` | {-1, 2, 3} | 整數 | — | 移位取代，不存 |
| `feat_val_raw` | ±15.3M | int（無小數） | `int32_t` | 積分圖整數和，天然整數 |
| `feat_val_norm` | ±69 | Q7.16 | `int32_t` | 吸收 inv_area 後改為不除 |
| `threshold` (wc) | [-0.40, +0.66] | Q1.15 | `int32_t` | 比較精度敏感 |
| `left_val` / `right_val` | [-30, +10] | Q6.10 | `int32_t` | 極端值 -30 需夠整數位 |
| `stage_sum` | [-22, +11] | Q6.10 | `int32_t` | 與 left/right_val 同格式 |
| `stage_threshold` | [-5.0, -3.0] | Q6.10 | `int32_t` | 與 stage_sum 同格式 |
| `variance` | [0.64, 8857] | Q14.10 | `int32_t` | 低紋理時小數位重要 |
| `stdev` | [0.80, 94.1] | Q7.16 | `int32_t` | Step 4 後可消除 |
| `area` | [576, 588289] | 整數 | `int32_t` | 吸收 inv_area 用 |
| `threshold × area` | ≤388K | 整數 | `int32_t` | 中間量 |
| `threshold × stdev × area` | ≤36.6M | 整數 | `int64_t` 中間 → `int32_t` | 防溢位用 int64 中間量 |

---

## 5. Step 0 結論與 Step 1~4 影響

### ✅ 確認可行

1. **Weight 全為 {-1, 2, 3}**：Step 2 可 100% 移位取代，無例外。
2. **feat_val_raw 是整數**：int32_t 即可，吸收 inv_area 後右邊也在 int32_t 範圍。
3. **stage_sum 實測範圍遠小於理論上界**：Q6.10 在 int32_t 內安全。

### ⚠️ 需要注意

1. **left/right_val 有 -30 極端值**：Q6.10（±63 整數範圍）夠，但不能縮到 Q5.x。
2. **variance 最小值 0.636**：Q14.10 的 10 小數位（精度 ~0.001）足夠；若縮到 Q14.4 會讓低紋理窗口 variance 精度不足，影響 Step 4 的符號判斷。
3. **threshold × stdev × area 中間量**：最大 36.6M，需要在乘法鏈中用 `int64_t` 暫存，最終結果才轉回 int32_t 比較。
4. **threshold 的比較精度敏感**：cascade threshold 是訓練結果，精度不足直接影響 pass/reject 翻面。`Q1.15` 精度 ~3×10⁻⁵，threshold min 差異 ~0.001，足夠；不建議縮到 Q1.8 以下。

### Step 4（sqrt 消除）的前置確認

- L（feat_val_norm）範圍 ±69 → L² ≤ 4,764（fits in int32_t）
- threshold²：max = 0.661² = 0.437
- variance：max = 8,857
- threshold² × variance：max ≈ 0.437 × 8,857 ≈ 3,870（fits in int32_t）
- **符號分析**：threshold 可正可負（[-0.402, +0.661]），Step 4 的四分支判斷是必要的。

---

*本文件為 Step 0 產出，不含任何 code 修改。*  
*下一步：Step 1 — 定點化框架 + 吸收 inv_area。*
