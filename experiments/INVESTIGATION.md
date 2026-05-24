# Viola-Jones Detection Investigation

測試環境：Windows 11, Python OpenCV 4.x, C VJ (MSYS2)  
圖片：testt.jpg (267×400), lena.jpg (512×512), test.jpg (1200×902)  
參數：scale_factor=1.2, minSize=(24,24)

---

## 調查一：testt.jpg 在 C minNeighbors=3 漏偵的原因

### 1-A. Raw detections 數量與分布（C vs OpenCV）

| 圖片 | C raw dets | OpenCV raw dets |
|------|-----------|-----------------|
| testt.jpg | **2** | 26 |
| lena.jpg  | 3  | 35 |
| test.jpg  | 13 | 92 |

**C raw detections 明細（testt.jpg）**

```
(104, 80,  49×49)
(100, 84,  49×49)
→ 全部同一尺度 (49 px)，位置緊鄰
```

**OpenCV raw detections 明細（testt.jpg）**

```
總計 26 個，分布在 3 個 window 尺度：
  50 px : 多個位置
  60 px : 多個位置
  72 px : 多個位置
```

### 1-B. 分群後各群的 vote count

**testt.jpg — C（Python 複製 vj_group_rectangles 算法）**

```
Groups: 1 個
  #1  votes=2   avg=(102,82,49×49)   sizes=[49]
      └─ 這就是真臉那群，2 票
```

由於 `n(2) < min_neighbors(3)`，`if (n >= min_neighbors)` 判斷失敗 → **砍掉 → 漏偵**。

**testt.jpg — OpenCV（同樣分群邏輯）**

```
Groups: 3 個
  #1  votes=23  avg=(97,80,58×58)   sizes=[50,60,72]  ← 真臉，輕鬆過 mn=3
  #2  votes= 2  avg=(157,216,29×29) sizes=[29]
  #3  votes= 1  avg=(27,42,72×72)   sizes=[72]
```

OpenCV 的真臉群有 23 票，遠超 minNeighbors=3 門檻 → **偵到**。

### 1-C. 對照 lena.jpg

| | C | OpenCV |
|--|---|--------|
| raw dets | 3 | 35 |
| 最高票群（真臉） | **3票** | **34票** |
| 過 mn=3？ | 3≥3 → **剛好過** | 34≥3 → 輕鬆過 |

C 在 lena.jpg 的真臉剛好 3 票（等於門檻，因為 `n >= min_neighbors`），是驚險通過。  
testt.jpg 的 2 票比 lena 少了 1 票，就被砍掉。

### 1-D. 解釋：為什麼 C 的 raw detections 這麼少？

**根本原因：window-size 的整數截斷 vs 四捨五入**

- C 算法：`win_w = (int)(24 * scale)` — **floor 截斷**
- OpenCV：在縮小圖上用固定 24×24 窗口，等效 window size 為 `24 * scale`，但 OpenCV 的 scale 步進與 C 不同

在 scale^4 = 1.2⁴ = 2.0736 處：
- C 得到 `int(24 × 2.0736) = int(49.77) = **49 px**`
- OpenCV 得到 **50 px**

這 1 px 的差距改變了窗口內每個 Haar feature 的位置：`rx = win_x + int(rc->x * scale_x)` 中 `scale_x = win_w/24` 微幅改變，讓某些 feature 值跨過 stage threshold，導致 C 在 scale n+1, n+2 的窗口更容易被 cascade 早期 stage 拒絕。

**結果**：

| 圖片 | C 面積通過 cascade 的尺度數 | OpenCV |
|------|---------------------------|--------|
| testt.jpg | 1 個尺度（49 px） → 2 raw dets | 3 個尺度（50/60/72）→ 23 raw dets in face group |
| lena.jpg  | 2 個尺度（123/148 px）→ 3 raw dets | 6 個尺度 → 34 raw dets |

**為什麼 testt.jpg 比 lena.jpg 更弱（即使同是 C）？**

1. **圖片尺寸小**：267×400 → 可用的 scale 層數少（face ~50 px → 只有 scale 層 4 附近命中）
2. **眼鏡遮蔽**：鏡框遮擋眼眶 Haar feature（vertical edge 類），影響 mid-stage（大約 Stage 5-12 有多個眼眶相關 weak classifier）
3. **Cascade 本身的訓練偏差**：haarcascade_frontalface_default.xml 為正面訓練，略有角度或頭部傾斜都讓通過率急劇下降

**結論**：testt.jpg 漏偵是 **兩個弱訊號疊加**：(1) C 的 floor-截斷讓通過率已低於 OpenCV；(2) 眼鏡＋圖片尺寸限制使在 C 端通過的 raw windows 只剩 2，剛好低於 mn=3 門檻。

---

## 調查二：best_face 在 testt.jpg 框到，是穩健還是「無競爭對手」？

### 2-A. testt.jpg 所有群的 vote count

**C raw dets 分群（3 群）**

```
C 端只有 2 個 raw dets，全部落在同一群：
  #1  votes=2   avg=(102,82,49×49)  ← 真臉（唯一群）
  無第 2 群
→ best_face 是「無競爭對手」獲勝，非遙遙領先
```

**OpenCV 端分群（可作為參考，說明影像本質的強弱）**

```
  #1  votes=23  avg=(97,80,58×58)   ← 真臉
  #2  votes= 2  avg=(157,216,29×29) ← 身軀/低紋理區誤偵
  #3  votes= 1  avg=(27,42,72×72)   ← 背景誤偵
  margin（真臉 vs 2nd）= 21 票
```

→ **C 端的 best_face：是「預設勝利」（唯一群），robustness 取決於背景乾淨程度。**  
→ **OpenCV 端的 best_face：21 票領先，遙遙領先，穩健。**

如果 testt.jpg 的背景有更多紋理（產生 3+ 個 raw dets 的誤偵群），C best_face 可能選到背景。

### 2-B. test.jpg — 複雜背景的壓力測試

test.jpg 是海邊照（礁石、雜物），OpenCV mn=3 時有 **4 個背景誤偵** + 1 個真臉（共 5 個偵測）。

**C raw dets 分群（13 個 raw，5 個群）**

```
  #1  votes=4   avg=(602,323,129×129)  sizes=[123,148]  ← 真臉 (face)
  #2  votes=2   avg=(792,251, 24×24)   sizes=[24]        ← 背景
  #3  votes=1   avg=(610, 96, 24×24)   sizes=[24]        ← 背景
  #4  votes=1   avg=(746,254, 24×24)   sizes=[24]        ← 背景
  #5  votes=1   (其他...)                                 ← 背景
  margin（真臉 vs 2nd）= 4 - 2 = 2 票
```

→ C best_face → **(603, 326, 123×123)** = **正確選到真臉**，但僅以 **2 票險勝**。

**OpenCV raw dets 分群（92 個 raw，27 個群）**

```
  #1  votes=38  avg=(604,326,124×124)  sizes=[103,124,149]  ← 真臉
  #2  votes=10  avg=(787,749, 79×79)   sizes=[72,86]         ← 背景誤偵
  #3  votes= 5  avg=(791,250, 25×25)   ...                   ← 背景
  #4  votes= 5  avg=(490,667, 69×69)   ...                   ← 背景
  #5  votes= 4  avg=(511,765, 50×50)   ...                   ← 背景
  margin（真臉 vs 2nd）= 38 - 10 = 28 票
```

→ OpenCV 端：**真臉遙遙領先（28 票優勢）**，非常穩健。

### 2-C. 各圖片 best_face 穩健度總結

| 圖片 | best_face 端 | 真臉票數 | 2nd 票數 | margin | 評估 |
|------|-------------|---------|---------|--------|------|
| testt.jpg | C | 2 | 無競爭 | — | **無競爭勝利** (背景若複雜會危險) |
| testt.jpg | OpenCV | 23 | 2 | 21 | **遙遙領先，穩健** |
| lena.jpg  | C | 3 | 無競爭 | — | 無競爭勝利 |
| lena.jpg  | OpenCV | 34 | 1 | 33 | 遙遙領先 |
| test.jpg  | C | 4 | 2 | **2** | **險勝（fragile）** |
| test.jpg  | OpenCV | 38 | 10 | 28 | 遙遙領先，穩健 |

### 2-D. 在什麼情況下 best_face 會選錯？

C 端的危險條件（raw dets 少的天生弱點）：

1. **複雜背景 + 弱訊號臉**：test.jpg 已顯示 C 端 margin 只有 2。如果同一張圖的背景再多產生 3~4 個 raw det 落在同一區域（形成 5+ 票的背景群），best_face 就會選到背景。
2. **多人臉**：best_face 選「最高票的群」，如果鏡頭外有側臉或更近的人，該人群票數可能超過主角。
3. **小尺寸 + 眼鏡 + 頭部旋轉**：如 testt.jpg，C 端只有 1 群時 best_face 必選，但如果背景恰好有 3 個 raw det 組成一群，就會選錯。

**OpenCV 端**：raw dets 豐富（26~92），真臉的 vote count 優勢顯著（21~33 票以上），best_face 在這三張圖上都穩健。

---

## 綜合結論

| 問題 | 答案 |
|------|------|
| testt.jpg 漏偵原因 | C raw dets 只有 2（face group 2 票 < mn=3）；根本原因是 floor-截斷 window size + 眼鏡遮蔽，導致多個 scale 層的窗口在 cascade 中途被拒絕 |
| best_face 在 testt.jpg 為何能框到 | C 端唯一的群只有 2 票，無競爭對手，best_face 預設勝利；OpenCV 端 23 票領先 21 票，真正穩健 |
| best_face 在 test.jpg 是否穩健 | C 端僅 2 票 margin（4 vs 2），相對脆弱；OpenCV 端 28 票 margin（38 vs 10），非常穩健 |
| best_face 何時會選錯 | C 端：背景複雜且臉部訊號弱，一旦有背景群票數超過臉部群即選錯；OpenCV 端：本資料集內無此風險 |
