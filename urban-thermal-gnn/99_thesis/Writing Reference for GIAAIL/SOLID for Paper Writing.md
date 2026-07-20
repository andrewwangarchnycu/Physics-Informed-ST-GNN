# SOLID for Paper writing

> 把寫 code 的結構化思維,搬到論文寫作上。
> 核心命題:**論文和程式碼本質上都是在組織知識結構,因此軟體工程的原則可以直接對應到學術寫作。**

---

## 目錄

1. [核心類比](#一核心類比)
2. [論文寫作的 SOLID 原則](#二論文寫作的-solid-原則)
3. ["呼叫函式"的寫作習慣](#三呼叫函式的寫作習慣)
4. [段落的函式結構](#四段落的函式結構)
5. [補充知識的分層架構](#五補充知識的分層架構footnote-與附錄)
6. [巢狀結構:論文即遞迴](#六巢狀結構論文即遞迴)
7. [違規偵測模式](#七違規偵測模式)
8. [判斷準則速查](#八判斷準則速查)

---

## 一、核心類比

寫 code 和寫論文都在做同一件事:**把複雜的知識,組織成讀者能逐層理解的結構**。

| 程式碼概念            | 論文對應                 |
| --------------------- | ------------------------ |
| Module / Class        | 章節 (Chapter)           |
| Function              | 段落 (Paragraph)         |
| Function signature    | 開頭句 (Topic sentence)  |
| Return value          | 結尾句 / 小結            |
| Call function         | 「詳見 §x.x」交叉引用    |
| Helper function       | 附錄 (Appendix)          |
| Inline comment        | Footnote                 |
| DRY 原則              | 同一論述只出現一次       |
| Self-documenting code | 任何層級都有清楚的入口句 |

---

## 二、論文寫作的 SOLID 原則

### S — Single Responsibility Principle(單一職責)

**每個章節只負責一件事。**

一個 function 只做一件事;一個章節只說明一個概念。

- 文獻回顧 → 只回顧,不分析自己的方法
- 方法章節 → 只說明做了什麼,不討論結果好壞
- 討論章節 → 只解釋意義,不重複貼數字

> **常見違規:** 在方法章節偷偷評論結果的優劣;在結論又重新解釋一遍實驗設計。

---

### O — Open/Closed Principle(開放封閉)

**架構對擴充開放、對修改封閉。**

好的模組設計加功能不需要動到核心;好的論文架構加新實驗不需要重寫整篇。

- 章節介面清晰,後補的 ablation study 能直接插入,不需動到緒論的論述邏輯。
- 章節間用「詳見 §x.x」連結,而不是複製貼上內容。

---

### L — Liskov Substitution Principle(里氏替換)

**子章節可以被獨立理解。**

子類別可以替換父類別;小節可以抽出來單獨閱讀,不失去意義。

- 每個小節開頭一句話說明 scope。
- 讀者跳讀某節,不需讀完前面全部內容才能理解。

> **常見違規:** 一個小節過度依賴前文,抽出來就讀不懂。

---

### I — Interface Segregation Principle(介面隔離)

**不要強迫讀者吸收不相關的資訊。**

不應強迫 client 依賴它不需要的 interface;不應把讀者不需要的細節塞進主文。

- 數學推導細節 → 附錄
- 實作參數表 → 附錄或表格
- 主文只保留**論證鏈上必要的節點**

---

### D — Dependency Inversion Principle(依賴反轉)

**高層論述不依賴低層細節,兩者都依賴抽象。**

核心論點不應被實作細節綁死。

- 緒論的 research question 是高層抽象,不該提到具體的 loss function 名稱。
- 概念說明(高層)與數學推導(低層)分開寫,用「如公式 (3) 所示」連結。
- 換掉一個 baseline,緒論不需要改動。

---

## 三、"呼叫函式"的寫作習慣

| Code 寫法                   | 論文對應寫法                    |
| --------------------------- | ------------------------------- |
| 呼叫 function,不重複實作    | 「詳見 §2.3」「如前述」         |
| 介面定義清楚再使用          | 先在術語定義節定義,後文直接用   |
| 不在 main 裡寫邏輯          | 不在摘要/緒論裡解釋細節         |
| Magic number 抽成常數       | 關鍵假設集中在假設節,不分散各處 |
| DRY — Don't Repeat Yourself | 同一段論述只出現一次,引用不複製 |

---

## 四、段落的函式結構

每個段落 = 一個 function,有明確的 signature 和 return value。

```
topic sentence          ← function signature,說明這段做什麼
  ↓
supporting sentences    ← function body,執行邏輯
  ↓
closing / transition    ← return value,交代結果或銜接下一段
```

**一段 = 一個觀點**,不可分割,不可混入第二個主張。

```
[開頭句]   這段的唯一主張(topic sentence)
[展開句]   論證、說明、例子、數據
[展開句]   進一步支撐
[收尾句]   小結,或鋪墊下一段
```

---

## 五、補充知識的分層架構(footnote 與附錄)

主幹論述只走「必要路徑」,旁支知識依重要程度分流:

```
主文段落
  │  核心論證鏈,讀者不能跳過
  │
  ├─ footnote          淺層補充
  │    不影響論點,但有助理解的背景知識
  │    術語的替代名稱、來源說明、簡短澄清
  │    原則:刪掉 footnote,主文邏輯仍完整
  │
  └─ 附錄              深層補充
       完整數學推導、實作細節、額外實驗
       讀者需要「驗證或重現」才會去翻
       原則:主文只留結論,推導過程移附錄
```

### 判斷要放哪裡

寫完一個補充句,問自己:**「拿掉這句,核心論點會不會不成立?」**

| 答案                    | 處理方式                      |
| ----------------------- | ----------------------------- |
| 會,這句是論證的一步     | 留在主文,整合進段落邏輯       |
| 不會,但有助讀者理解脈絡 | 移到 footnote                 |
| 不會,是技術細節或推導   | 移到附錄,主文加「詳見附錄 A」 |
| 不會,且不影響理解       | 直接刪掉                      |

---

## 六、巢狀結構:論文即遞迴

同一個函式結構,在每個層級自相似地重複。每一層都有相同的模式:**開頭宣告主張 → 中間展開論證 → 結尾交代結果或銜接**。

```
論文(全本)
├── 章節
│   ├── 小節
│   │   ├── 段落
│   │   │   ├── 句子
```

### 全本層級 — 論文的 main()

```
Abstract        ← 整本論文的 return value,先給結論
Chapter 1 緒論  ← 定義問題,宣告 research question
Chapter 2 文獻  ← 建立前置知識,return 研究缺口
Chapter 3 方法  ← 核心 logic,call 附錄做推導
Chapter 4 結果  ← 執行並 return 數據
Chapter 5 討論  ← 解釋 return value 的意義
Chapter 6 結論  ← 總結 side effects,提 future work
```

章與章之間靠**轉折段**銜接,功能是 `return` 上一章並說明為何要 `call` 下一章。

### 章節層級 — 章的 function body

每章開頭一段 overview,宣告主張與結構;每節結尾一句交棒句:「本節確立了 X,下節將說明 Y 如何利用 X。」

範例:

```
Chapter 3 方法

[Overview 段落]
  本章提出 VAENCA 雙模組架構,將平均流場與紊流擾動分離預測。
  §3.1 說明整體架構;§3.2 定義 AGC-NCA 平均流模組;
  §3.3 說明 VAE 條件化機制;§3.4 描述複合損失函數。

[結尾段落]
  本章完整定義模型架構與訓練目標,
  第四章將在此基礎上呈現實驗結果。
```

### 小節層級 — 小節的 function body

每個小節同樣需要開頭句宣告 scope,每段負責一個分離的觀點。

```
§3.2  AGC-NCA 平均流模組

[小節開頭句]
  AGC-NCA 以 D2Q9 MRT-LBM 的九個矩通道作為輸入,
  透過可學習碰撞算符逼近穩態平均流場。

  [段落 A]  輸入表示:九通道矩空間的物理意義
  [段落 B]  感知區塊:多受體卷積設計
  [段落 C]  更新規則:NCA 迭代機制
  [段落 D]  輸出定義:預測目標與監督信號

[小節結尾句]
  上述設計使 AGC-NCA 的更新規則與 LBM 碰撞步驟在結構上同構,
  §3.3 將說明 VAE 如何在此基礎上注入條件資訊。
```

### 整體心智模型

```
全本    problem → approach → evidence → conclusion
  │
  章     context → method/findings → implication
    │
    小節   scope → elaboration → handoff
      │
      段落   claim → support → close
        │
        句子  subject → predicate(一句一動作)
```

每一層的**開頭**都回答:「這個層級要主張什麼?」
每一層的**結尾**都回答:「這個層級證明了什麼,接下來去哪?」

---

## 七、違規偵測模式

### 違規 1:開頭句太寬泛

```
❌  This chapter discusses the methodology of this study.
✓   This study adopts a dual-module architecture to decouple
    mean flow prediction from turbulent fluctuations.
```

開頭句要說**這段的具體主張**,不是說「我要說什麼」。

---

### 違規 2:補充知識插入主幹,打斷論證節奏

```
❌  The NCA update rule applies a learned kernel K over the
    local neighborhood. Neural Cellular Automata were first
    proposed by Mordvintsev et al. (2020) as a model of
    morphogenesis, inspired by biological cell signaling.
    This kernel is shared across all cells and time steps.

✓   The NCA update rule applies a learned kernel K over the
    local neighborhood, shared across all cells and time steps.¹

    ¹ NCA was originally proposed for morphogenesis modeling
      (Mordvintsev et al., 2020); its application to physical
      simulation is a recent extension.
```

---

### 違規 3:附錄內容重複出現在主文

附錄已有完整推導,主文只保留結論與它在論證中的角色:

```
✓   Following the Chapman-Enskog expansion (derivation in
    Appendix A), the second-order moment recovers the
    Navier-Stokes equation when τ satisfies...
```

---

### 違規 4:一段塞兩個觀點

如果需要用「此外、另一方面、除了上述」把兩個想法縫在一起,**這段應該拆成兩段**。

```
❌  感知區塊採用多種卷積核以捕捉多尺度空間資訊。此外,MRT
    碰撞算符相較於 BGK 提供更好的數值穩定性。

✓  [段落 A] 感知區塊採用 Sobel、Laplacian、dilated convolution
    等多種卷積核,使單一更新步驟即可捕捉多尺度空間梯度。

   [段落 B] 碰撞步驟採用 MRT 而非 BGK 算符,因為 MRT 對各矩的
    弛豫率獨立控制,在低黏度高 Re 條件下數值穩定性顯著優於
    單弛豫模型。
```

---

## 八、判斷準則速查

### 章節層級

> **「這一節,如果我刪掉,論文的哪個論點會垮掉?」**
> 跟 code review 問「這個 function 刪掉,哪個行為會消失」是同一個問題。

- 有明確答案 → 存在合理,且只做一件事
- 說不清楚 → 可能冗餘,或職責不清

### 段落層級

> **「這段是否只主張一件事?」**
> 需要連接詞縫合 = 該拆。

### 補充句層級

> **「拿掉這句,核心論點會不會不成立?」**
> 不會但有用 → footnote / 附錄;不會也沒用 → 刪。

---

> **一句話總結:** 讀者在任何層級進入論文,都能找到一個清楚的入口句,知道接下來會讀到什麼。這就是論文版的 **self-documenting code**。
