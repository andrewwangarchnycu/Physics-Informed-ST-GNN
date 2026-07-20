---
name: academic-writing-polish
description: English academic writing clarity/conciseness reference for polishing abstracts, figure/table captions, and English prose in the PI-ST-GNN thesis project. Use when writing or revising the English abstract, English captions, or any English-language academic prose in this repo. Static reference only (no API calls).
---

# 英文學術寫作潤飾指南 (Academic Writing Polish)

## 適用範圍與分工聲明

本 skill 為**靜態參考文件**，不呼叫任何外部 API，僅提供撰寫/修訂時之寫作原則參考。

與本專案既有兩份寫作參考文件之分工關係：

| 文件 | 適用範圍 |
|---|---|
| `urban-thermal-gnn/99_thesis/Writing Reference for GIAAIL/thesis_writing_guide.md` | **中文全文**（正文章節、圖表中文說明、論證結構、台灣學術用語）— 中文內容一律以此文件為準 |
| `urban-thermal-gnn/99_thesis/Writing Reference for GIAAIL/CN_TW_wording.json` | 大陸用語 → 台灣用語替換對照表 |
| **本 skill（academic-writing-polish）** | **英文摘要（Abstract）、英文圖表標題／說明、程式碼註解以外之任何英文學術散文**之清晰度與精簡度潤飾 |

衝突時優先順序：中文本體內容以 `thesis_writing_guide.md` 為準；本 skill 僅處理英文段落，兩者不重疊、互不覆蓋。

---

## 一、清晰度原則 (Clarity)

1. **主動語態優先**：能用主動語態表達因果或動作歸屬時，避免被動語態隱藏行為者（"The model was trained" → "We trained the model" / "This study trains the model"，依全文人稱慣例統一選用）。
2. **一句一命題**：長句（英文超過 30–35 字或包含 3 個以上子句）應拆分；一個句子只承載一個可被獨立驗證的主張。
3. **避免名詞化堆疊 (Nominalization)**：「the utilization of X for the purpose of Y」→「using X to Y」；動詞優先於抽象名詞。
4. **避免堆疊模糊限定詞**：不連續使用多個弱化詞（"it may perhaps possibly suggest..."）；一個命題只用一個恰當強度的認識論限定詞（詳見下方第四節強度分級）。
5. **代名詞指涉明確**："this"／"it"／"which" 若指涉不明，一律改為「the + 具體名詞」。

## 二、精簡度原則 (Conciseness)

1. **刪除清喉嚨語**：「It is important to note that」「It should be pointed out that」「As can be seen from the above」等開場贅語直接刪除，直接陳述論點。
2. **刪除冗餘同義詞疊用**："each and every"、"first and foremost"、"basic and fundamental" 等成對近義詞僅保留一詞。
3. **精簡連接語**：避免「In order to」統一用「To」；避免「due to the fact that」統一用「because」。
4. **量化優於模糊描述**：能給出具體數字/比較基準時，不用「significantly」「greatly」等未量化形容詞獨立成句。

## 三、論證段落結構 (Argumentative Paragraph Shape)

比照 `thesis_writing_guide.md` 之中文段落四步結構，英文段落同樣遵循：

```
Claim (topic sentence) → Explanation → Evidence/Example → Transition to next point
```

- 段落開頭句＝可被檢驗的主張，不是中立背景描述或「This section presents...」之類的後設寫作句。
- 避免用「and then」串接成流水帳；改用因果連接詞（because / thus / however / given that）呈現論證推進而非時間順序報告。
- 一段只講一個觀點；需要「In addition」「On the other hand」縫合兩個想法時，應拆成兩段。

## 四、學術語氣與斷言強度分級

| 證據強度 | 建議用詞 |
|---|---|
| 已直接量測驗證之數據 | state directly ("achieves", "reaches") |
| 具機制性因果推論但未窮盡驗證 | "indicates", "suggests", "is consistent with" |
| 尚未驗證之推測/未來方向 | "may", "could potentially", explicitly flagged as future work |
| 避免使用 | "proves", "demonstrates conclusively" 除非確實為數學證明或窮舉驗證 |

此分級與中文指南之「顯示／表明／推估」vs.「證明／顯著」二分法一致，確保中英文版本之認識論強度互相對應，不因翻譯而誇大或弱化原始主張。

## 五、術語一致性

- 技術名詞全文首次出現須給出完整英文全名 + 縮寫，之後全文僅用縮寫，不重複展開（與中文指南「中文全稱（English Full Name, ABBR）」規則對應的英文版：直接以 "Physics-Informed Spatio-Temporal Graph Neural Network (PI-ST-GNN)" 首次定義，其後僅用 "PI-ST-GNN"）。
- 同一概念全文固定同一英文詞，不因換句而替換同義詞（例如 "surrogate model" 定調後不要中途換成 "proxy model"／"substitute model"）。

## 六、修訂自查清單 (Self-Check for English Passages)

- [ ] 每段開頭句是否為可檢驗主張，而非背景陳述或後設寫作句？
- [ ] 是否有句子超過 35 字或包含 3 個以上子句？是否可拆短？
- [ ] 是否有清喉嚨語（It is important to note / It should be noted）？
- [ ] 斷言強度是否對應實際證據等級（未驗證的內容是否誤用 "proves"/"demonstrates"）？
- [ ] 技術縮寫是否僅在首次出現時展開全名，其後未重複展開？
- [ ] 同一技術概念是否全文用詞一致，未出現同義詞替換？
- [ ] 主詞是否一致（如全文採第三人稱 "this study" 則不中途切換為 "we"，反之亦然）？
- [ ] 中英文摘要對照：關鍵數字（R²、RMSE 等）與斷言強度是否與中文版本一致，不因語言轉換而誇大或弱化？
