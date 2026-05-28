# Viola-Jones 硬體友善化優化路徑（HLS 前置）

> **目標**：把 `vj_evaluate_window`（單窗口 cascade 判斷）改寫成適合丟進 HLS 的版本——消除 `sqrtf`、消除浮點除法、把特徵加權與座標縮放的乘法盡量轉成整數移位加減、整條路徑浮點轉定點。
>
> **不在本次範圍**：`vj_detect_faces` 的多尺度掃描迴圈、`vj_group_rectangles`、`vj_detect_best_face` 這些後處理留在 PS 端軟體，不硬體化。
>
> **黃金參考**：現在這份浮點版 `vj_evaluate_window` 就是 golden reference。每一步的定點/整數版都要跟它逐窗口比對。
>
> **核心原則**：一次只改一件事，改完立刻驗證，驗證過了才往下一步。任何一步一致率掉太多，先回退、定位原因，不要疊加修改。

---

## 驗證方法（所有步驟共用）

每一步完成後，都用同一套判準驗證，**不是比最後框準不準，是比逐窗口 pass/reject 一致率**。

### 主判準：逐窗口 pass/reject 一致率

對同一張圖、同一組掃描窗口（固定 scale 序列、固定 step），讓「浮點黃金版」和「當前修改版」各自對每個窗口輸出 0/1（reject/pass），統計兩者一致的比例。

```
一致率 = (兩版判斷相同的窗口數) / (總窗口數)
```

判準門檻（建議）：
- **≥ 99.5%**：通過，可往下一步。
- **99.0% ~ 99.5%**：可接受但要記錄，觀察是否在後續步驟累積惡化。
- **< 99.0%**：不通過，回退並定位是哪個量的精度/近似出問題。

### 輔助判準：分 stage 的翻面定位

當一致率不如預期時，統計「兩版在哪個 stage 做出不同決定」——也就是某個窗口在浮點版走到 stage k 被拒、但定點版在 stage k 通過（或相反）。哪個 stage 翻面最多，通常就是那個 stage 的 `threshold` / `stage_threshold` / `stage_sum` 定點精度不足。這能精準告訴你要加哪裡的位元數。

### 最終結果對照（輔助）

逐窗口比完後，再跑一次完整 pipeline（含 grouping / best_face），確認 testt / lena / test 三張圖最後框出來的臉沒有跑掉。這是 sanity check，不是主判準。

### 測試影像要求

- 至少包含一張**複雜背景**的圖（例如 test.jpg）。定點誤差最容易在「剛好壓線」的邊緣窗口翻面，而複雜背景產生最多這種邊緣窗口。乾淨正面照（lena）本來就過得輕鬆，測不出定點化的傷。

---

## Step 0：前置數據調查（不改 code）

**這是動任何 code 之前的第一件事。沒有這份數據，定點化就是瞎猜 Q 格式。**

### 思路

定點化的每個決定（word length、Q 格式、weight 能不能用移位表達）都取決於各個量的「實際數值範圍」與「取值種類」。先把這些統計出來。

### 要統計的量（從 `vj_cascade_data.c` 靜態分析）

1. **所有 weight 的取值集合**：列出 2913 個 feature、每個 rect 的 `weight` 出現過哪些相異值（預期是 ±1, ±2, ±3 這種小整數，但**要實際確認**，不要假設）。如果出現非整數或不是小整數倍數的值，那部分就無法純用移位，需要特別處理。
2. **`threshold`（weak classifier）的範圍與分布**：min / max / 直方圖。這決定 threshold 的 Q 格式小數位需求。
3. **`left_val` / `right_val` 的範圍**：這些是投票值，通常是小數，範圍大概 ±幾。
4. **`stage_threshold` 的範圍**：每個 stage 的及格線。
5. **各 stage 的 num_weak**：確認 per-stage feature 數（畫出 stage 0=9, 1=16... 的分布），這影響 stage_sum 的累加範圍。

### 要統計的量（runtime 動態範圍，跑浮點版時插樁）

在浮點版 `vj_evaluate_window` 裡插入統計（只記錄、不改邏輯），跑過三張測試圖，收集：

6. **`feat_val`（乘 inv_area 前的原始加權和）的範圍**：這是積分圖矩形和的加權組合，整數範圍可能很大，決定整數位需求、防溢位。
7. **`feat_val * inv_area`（normalize 後）的範圍**：跟 threshold 比較的那個量。
8. **`variance` 的範圍**：決定 sqrt 消除時平方比較的位元需求。
9. **`stage_sum` 的範圍**：每個 stage 累加後的值域。

### 產出

一份 `QUANT_ANALYSIS.md`，包含上述統計，並據此**初步建議**每個量的 Q 格式（例如 `threshold` 用 Q8.8、`stage_sum` 用 Q16.16 之類）。這份建議會在 Step 1 被實際驗證、微調。

### 驗證關卡

本步驟不改 code，無一致率可驗。產出檢查點：weight 取值集合是否確認為「可移位表達的小整數」？如果**不是**，要在這裡就標記出來，影響 Step 2 的策略。

---

## Step 1：定點化框架 + 吸收 inv_area（先不碰移位、不碰 sqrt 消除）

**目標：建立浮點→定點的骨架，把變數隔離成「純粹的精度損失」，先不引入任何近似技巧。**

### 思路

這一步只做兩件事：
1. 把路徑上所有 `float` 換成定點整數表示（用 Step 0 定的 Q 格式）。
2. 順手把 `inv_area` 的浮點除法吸收掉。

**乘法暫時保留**（weight 乘法、座標縮放乘法都先用定點整數乘法做，還不要轉移位）。`sqrtf` 也暫時保留（用定點版的 sqrt 或先轉成整數平方根）。這樣這一步若一致率掉，原因就單純是「定點精度」，不會跟「移位近似」「sqrt 消除」混在一起。

### 關鍵：inv_area 怎麼吸收

現在的比較式（攤平後）本質是：

```
feat_val_raw * inv_area  <  threshold * stdev
```

其中 `inv_area = 1/(win_w*win_h)`，是浮點除法。兩邊同乘 `area = win_w*win_h`（正數，不影響不等號方向）：

```
feat_val_raw  <  threshold * stdev * area
```

`inv_area` 的除法和那個 `feat_val *= inv_area` 的乘法都消失了。右邊的 `threshold * area` 對同一個 scale 是固定的，可以每個 scale 預算一次（或併進 Step 0 的查表）。

> ⚠️ 注意：吸收 inv_area 後，比較的數值範圍變大了（少除了一個 area）。要重新確認 `feat_val_raw` 和右邊那一項的整數位足夠、不溢位。這正是 Step 0 第 6 項要量的東西。

### 關鍵：Q 格式怎麼定（思路，不是死規定）

- 從 Step 0 的數值範圍反推：整數位要 ≥ ⌈log2(max絕對值)⌉ + 1（符號位），小數位給夠精度。
- `threshold`、`stage_threshold`、`left/right_val` 這些「邊緣敏感」的量，小數位寧可多給。它們被訓練設定在「剛好讓真臉過」的邊緣，精度不足最容易翻面。
- 用 `int32_t` 當運算容器通常夠，但 `stage_sum` 累加多個 left/right_val、`feat_val_raw` 是大矩形和，要個別確認不溢位，必要時用 `int64_t` 中間量。

### 留給 Claude Code 實作

- 定義定點型別與 Q 格式巨集（如 `#define Q_THRESH 8`，`typedef int32_t fixed_t`）。
- 把 `vj_cascade_data.c` 的 float 參數另外產生一份定點版表（可寫個轉換 script，或在載入時轉）。**保留原浮點表**，因為黃金參考還要用。
- 改寫 `vj_evaluate_window` 的定點版（命名如 `vj_evaluate_window_fixed`），與浮點版**並存**，方便逐窗口對照。

### 驗證關卡

逐窗口 pass/reject 一致率 ≥ 99.5%。若不到，用分 stage 翻面定位，加對應量的小數位，重測。**這一步要先穩，後面才有乾淨的對照基準。**

---

## Step 2：feature 加權的乘法 → 整數移位加減

**目標：把 `feat_val += rect_sum * weight` 的乘法消成移位加減，這是「無乘法」卖點的核心實作。**

### 前提

Step 0 已確認 weight 都是可移位表達的小整數。**若 Step 0 發現有不可移位的 weight，這一步要先處理那些例外（可能保留少量乘法，或用 constant multiplication 拆解）。**

### 思路

Haar 特徵的 weight 是高度結構化的小整數（±1, ±2, ±3...），乘以小整數可用移位加減取代：

```
x * 2  = x << 1
x * 3  = (x << 1) + x
x * 4  = x << 2
x * -1 = -x
```

更進一步，可利用 Haar 特徵「各矩形加權和恒為零」的性質簡化。例如二矩形特徵常是「大矩形 weight=-1，內含小矩形 weight=+k」，可重寫成：

```
feat_val = k * small_rect_sum - big_rect_sum
         = (small_rect_sum << shift...) - big_rect_sum
```

### 實作策略（思路）

因為 weight 種類很少，不需要對每個 feature 寫死。可以：
- 列舉 Step 0 找到的所有 weight 種類（假設是 {-1, 2, 3} 之類）。
- 對每種 weight 寫一個移位加減的計算分支，或做成一個小的 `multiply_by_weight(rect_sum, weight_code)` 函式，用 weight 的「種類編號」查對應的移位加減 pattern。
- cascade 表裡的 weight 改存「種類編號」而非浮點值。

### 留給 Claude Code 實作

- 統計並列舉 weight 種類，產生移位加減的對應實作。
- 把 weight 乘法替換成移位加減版本。
- 保留 Step 1 的定點乘法版做對照（驗證「移位近似」是否引入額外誤差——理論上整數移位是精確的，不該有誤差，若有就是 pattern 寫錯）。

### 驗證關卡

跟 Step 1 的定點版（不是浮點版）逐窗口比對，一致率應該 **≈ 100%**（因為整數移位取代整數乘法是精確的，不是近似）。若不是 100%，代表某個 weight 的移位 pattern 寫錯了，用分 stage 定位修正。

> 同時也跟浮點黃金版比一次，確認累積一致率仍 ≥ 99.5%。

---

## Step 3：feature 座標縮放的乘法 → 預算查表（或定點）

**目標：消除 `(int)(rc->x * scale_x)` 這組浮點乘法。這可能是路徑裡乘法量最大的來源（每 feature 2~3 rect × 4 座標分量），消掉它對「無乘法」主張幫助最大。**

### 思路

現在每個矩形的每個座標都要 `rc->x * scale_x` 一次浮點乘法 + 取整。但關鍵觀察：**scale 層數很少（十幾個），每個 feature 在每個 scale 下縮放後的座標是固定的**，可以離線算好。

**路 A（推薦，最硬體友善）：預算座標查表（ROM）**

離線把每個 scale 層、每個 feature 的縮放後座標 `(rx, ry, rw, rh)` 全部算好，存成查表燒進 ROM。runtime 直接查表拿座標，完全沒有座標縮放的乘法。

- 代價：ROM 容量 = scale層數 × 2913 feature × 每feature最多3 rect × 4分量。要評估 BRAM 是否夠（這是面積換乘法的取捨）。
- 這條路最乾淨，但記憶體成本要在 Step 0 的容量規劃裡確認。

**路 B（折衷）：scale_x 定點化**

若 ROM 放不下，退而把 `scale_x` 定點化，座標縮放變成「定點乘法 + 移位取整」。乘法還在，但至少從浮點變定點。

### 取整行為要對齊

注意現在的 code 對座標用的是 `(int)` 截斷（floor for positive）。預算查表時，要**用跟浮點版相同的取整規則**離線計算，否則座標差 1 px 會讓 feature 值偏移、判斷翻面。這是這一步最容易出錯的地方。

### 留給 Claude Code 實作

- 先評估路 A 的 ROM 容量；放得下走 A，放不下走 B。
- 路 A：寫離線座標預算（產生查表），runtime 改成查表。
- 確保取整規則與浮點黃金版一致。

### 驗證關卡

逐窗口 pass/reject 一致率：
- 路 A（查表，取整規則一致）：應 ≈ 100%（精確）。
- 路 B（定點 scale_x）：可能有微小近似誤差，要求 ≥ 99.5%。

累積跟浮點黃金版比，仍須 ≥ 99.5%。若路 A 沒到 100%，幾乎肯定是取整規則沒對齊，重點查這個。

---

## Step 4：消除 sqrtf（變異數正規化的平方比較法）

**放最後，因為它涉及符號判斷，最容易出錯，要單獨驗證。**

### 思路

現在：

```c
float stdev = sqrtf(variance);
if (feat_val_normalized < threshold * stdev)  // threshold*stdev 右邊
```

`stdev` 唯一用途就是這個比較。我們不需要 stdev 的值，只要比較結果。把根號消掉：

兩邊都跟 variance 有關。`(threshold * stdev)² = threshold² * variance`。所以理論上可以比較 `feat_val_normalized²` 和 `threshold² * variance`，sqrt 被 `threshold²`（可預先算好存表）取代。

**但平方會丟符號，必須分情況：**

設左邊 `L = feat_val_normalized`，右邊 `R = threshold * stdev`。注意 `stdev ≥ 0`，`threshold` 可正可負。

要判斷 `L < R`：
- 若 `threshold ≥ 0`（R ≥ 0）：
  - 若 `L < 0`：必然 `L < R`，成立。
  - 若 `L ≥ 0`：兩邊非負，可平方比較 `L² < threshold² * variance`。
- 若 `threshold < 0`（R ≤ 0）：
  - 若 `L ≥ 0`：必然 `L ≥ R`，不成立。
  - 若 `L < 0`：兩邊非正，平方比較**反向** `L² > threshold² * variance`（負數平方後大小關係反轉）。

> ⚠️ 這個符號分情況是整步的核心，寫錯一個分支就會在一部分窗口翻面。務必把四個分支都明確實作，不要圖省事合併。

### 與 inv_area 吸收的整合

記得 Step 1 已經把 `feat_val *= inv_area` 吸收成「兩邊同乘 area」。所以這裡的 `L` 和 `R` 要用 Step 1 之後的形式（已經沒有 inv_area 的版本）來推。把 area、threshold、variance 的關係重新攤平一次，確認整數位不溢位（`threshold² * variance * area²` 之類的量可能很大，可能需要 int64 中間量，或重新安排運算順序避免溢位）。

### 「無乘法」主張的誠實邊界

這一步引入了「平方」（乘法）。所以你的「無乘法」主張**只對特徵評估路徑（Haar 加權和）成立**，變異數正規化這條路徑有乘法。RESULTS.md 目前的用詞「特徵計算這條路徑完全沒有乘法」是準的（限定在特徵計算）。**繼續守住這個限定詞，不要擴大宣稱整個 detector 無乘法。**

可選的進一步優化（非必要）：variance normalization 一個窗口只算一次（不是每 feature），所以它的乘法成本被攤薄到整個窗口。若面積允許，這幾個乘法保留也無妨；是否要進一步消除看 synthesis 後的面積報告再決定。

### 留給 Claude Code 實作

- 實作四分支的符號安全平方比較，取代 `sqrtf` + `threshold * stdev`。
- `threshold²` 預先算好存表。
- 仔細處理整數位/溢位（可能要 int64 中間量）。

### 驗證關卡

逐窗口 pass/reject 一致率 ≥ 99.5%（這步是近似消除，不像 Step 2/3 路A 那樣精確，允許微小差異）。

**特別注意**：用分 stage 翻面定位時，重點看「variance 很小的窗口」（低紋理區，如純色背景）——這些窗口 stdev 接近 0，平方比較的數值行為最敏感，最容易翻面。確認這類窗口行為正確。

---

## Step 5：整體回歸驗證 + 收尾

**目標：所有優化疊加後，做一次完整的端到端確認，並整理成可進 HLS 的乾淨版本。**

### 驗證

1. **逐窗口累積一致率**：最終定點+移位+無sqrt版 vs 浮點黃金版，三張圖（含複雜背景）整體 ≥ 99.5%。
2. **分 stage 翻面報告**：確認沒有某個 stage 異常集中翻面。
3. **完整 pipeline 結果對照**：跑完整 best_face，確認 testt / lena / test 最後框到的臉跟浮點版一致（位置、大小差異在可接受範圍）。
4. **多找幾張新圖**：用沒測過的圖（含複雜背景、不同臉大小）複驗，確認不是只對這三張 overfit。

### 收尾（為 HLS 準備，但還不寫 pragma）

- 把最終定點版 `vj_evaluate_window` 整理成自包含、依賴最少的形式：依賴閉包是 `vj_rect_sum`/`vj_rect_sum_sq`（inline）、定點 cascade 表、`vj_types.h`。
- `vj_detect_faces` 的掃描迴圈、grouping、best_face 確認仍在這支函式之外（留 PS 軟體）。
- 把 `static` 拿掉、整理介面參數，準備下一階段做 HLS 介面設計（這屬於 HLS 階段，不在本 roadmap）。

### 驗證關卡

以上四項全過 → 演算法層硬體友善化完成，可進入 HLS 階段。

---

## 步驟總覽與相依關係

| Step | 內容 | 性質 | 驗證門檻 | 比對對象 |
|------|------|------|----------|----------|
| 0 | 數值範圍 / weight 種類調查 | 不改 code | 產出檢查 | — |
| 1 | 定點化框架 + 吸收 inv_area | 精度損失 | ≥ 99.5% | 浮點黃金版 |
| 2 | weight 乘法 → 移位加減 | 精確取代 | ≈ 100% | Step 1 定點版 |
| 3 | 座標縮放 → 查表/定點 | 精確(路A)/近似(路B) | ≈100%/≥99.5% | Step 1 定點版 |
| 4 | 消除 sqrt（平方比較） | 近似消除 | ≥ 99.5% | 浮點黃金版 |
| 5 | 整體回歸 + 收尾 | 驗證 | 四項全過 | 浮點黃金版 |

**鐵律**：每步驗證過才往下；一致率掉了先回退定位，不要疊加修改；浮點黃金版全程保留不動。

---

## 重要提醒（貫穿全程）

1. **黃金參考神聖不可改**：現在這份浮點 `vj_evaluate_window` 是真理來源，全程保留、不動。所有修改版都跟它比。
2. **不重新訓練、不調 threshold**：cascade 參數是訓練好的整體平衡，你的工作是「定點版逼近浮點版」，不是重新調參。守住「忠實實作」立場。
3. **「無乘法」的限定詞**：只對特徵評估路徑成立。normalization 有乘法。不要擴大宣稱。
4. **複雜背景圖必測**：邊緣窗口才是定點誤差的試金石。
5. **這些都是 HLS 前的演算法改寫**：真正的 HLS 優化（pipeline、unroll、BRAM 配置、early-rejection 不規則性處理）是下一階段，不在本 roadmap。
