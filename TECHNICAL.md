# YOLOv7 Rice Quality Classifier — 技術架構說明

## 1. 深度學習模型 — YOLOv7 物件偵測

- **模型架構**：YOLOv7（You Only Look Once v7），為即時物件偵測領域的 SOTA 模型
- **權重檔**：`models/best.pt`，使用自訂米粒資料集訓練的模型權重
- **分類類別**：4 種米粒分類
  - `white_rice`（白米）
  - `thi_rice`（稻米）
  - `brown_rice`（糙米）
  - `black_rice`（黑米）
- **推論框架**：PyTorch（雲端部署使用 CPU 版本以減少資源佔用）
- **裝置選擇**：自動偵測 CUDA GPU，雲端環境自動 fallback 至 CPU

## 2. 影像前處理流程

| 步驟 | 技術細節 |
|------|----------|
| Letterbox 縮放 | 維持原始長寬比，使用 padding 將影像調整為 640×640 輸入尺寸 |
| 色彩空間轉換 | BGR → RGB → Tensor（3×H×W） |
| 正規化 | 像素值除以 255.0，轉為浮點數 |
| NMS 過濾 | Non-Maximum Suppression，使用 IoU 閾值過濾重疊偵測框 |
| 座標還原 | `scale_coords` 將偵測框從 letterbox 空間映射回原始圖片座標 |

## 3. Web 前端 — Streamlit

- **框架**：Streamlit，Python 原生的互動式 Web 應用框架
- **UI 設計**：透過 `st.markdown` 注入大量自訂 CSS，採用 Bento Box 風格 Dashboard 設計
- **分頁架構**：使用 `st.tabs` 實現三種偵測模式
  - 📷 圖片上傳分析
  - 🎬 影片處理分析
  - 📹 即時相機串流
- **模型快取**：`@st.cache_resource` 裝飾器確保模型只在首次載入，跨請求共享實例
- **互動式參數調整**：透過 Sidebar 的 `st.slider` 即時調整 Confidence 及 IoU 閾值

## 4. 影片處理流程

- **逐幀分析**：使用 OpenCV `cv2.VideoCapture` 逐幀讀取影片，每幀送入 YOLOv7 推論
- **影片寫出**：`cv2.VideoWriter` 輸出帶有偵測框標註的影片，支援多種 codec fallback 機制：
  ```
  mp4v → XVID → avc1 → H264
  ```
- **H.264 轉換**：透過 `subprocess` 呼叫系統 `ffmpeg`，將影片轉為 H.264 格式以確保瀏覽器相容性
- **進度追蹤**：callback 機制即時回報處理進度、已用時間、預估剩餘時間
- **偶數維度處理**：自動偵測並調整影片尺寸為偶數（H.264 編碼要求）

## 5. 即時相機串流 — WebRTC

- **核心套件**：`streamlit-webrtc` + `av`（PyAV）
- **架構設計**：繼承 `VideoProcessorBase`，在 `recv()` 方法中對每個 WebRTC 影格做即時 YOLOv7 推論
- **NAT 穿透**：使用 Google 公開 STUN Server 進行 WebRTC 連線建立
- **錯誤處理**：推論失敗時自動回傳原始影格，確保串流不中斷

## 6. 部署架構 — Streamlit Cloud

- **自動部署**：連結 GitHub Repository，`git push` 後自動觸發重新部署
- **系統依賴**：`packages.txt` 透過 `apt-get` 安裝系統套件
  - `libgl1` — OpenCV 圖形庫依賴
  - `libglib2.0-0t64` — GLib 函式庫（Debian Trixie 版本）
- **Python 依賴**：`requirements.txt` 管理 Python 套件
  - 使用 `--extra-index-url` 指定 CPU 版 PyTorch wheel（約 200MB，相較完整版約 2GB）
- **暫存檔管理**：影片處理使用 `tempfile` 模組，Session 結束後自動清理

## 7. 專案結構

```
yolov7-rice-classifier/
├── streamlit_app.py          # 主應用程式（Streamlit UI + 推論邏輯）
├── rice_classifier_app.py    # Gradio 替代介面
├── models/
│   └── best.pt               # YOLOv7 訓練權重
├── yolov7/                   # YOLOv7 原始碼
│   ├── models/               #   模型定義
│   │   └── experimental.py   #   attempt_load 模型載入
│   └── utils/                #   工具函式
│       ├── general.py        #   NMS、scale_coords
│       ├── plots.py          #   plot_one_box 繪製偵測框
│       ├── datasets.py       #   letterbox 影像前處理
│       └── torch_utils.py    #   裝置選擇
├── packages.txt              # 系統依賴（apt-get）
├── requirements.txt          # Python 依賴（pip）
└── runs/detect/              # 推論結果輸出目錄
```

## 8. 技術棧總覽

| 技術 | 用途 |
|------|------|
| **YOLOv7 + PyTorch** | 物件偵測模型推論 |
| **OpenCV** | 影像/影片讀寫與前處理 |
| **Streamlit** | Web UI 框架 |
| **streamlit-webrtc** | 即時相機串流（WebRTC） |
| **PyAV** | 影格格式轉換 |
| **ffmpeg / ffprobe** | 影片格式轉換與元資料分析 |
| **Custom CSS (Bento Grid)** | Dashboard UI 樣式設計 |
| **Streamlit Cloud** | 雲端部署平台 |
| **GitHub** | 版本控制與 CI/CD 觸發 |
