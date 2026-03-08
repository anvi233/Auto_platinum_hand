from ultralytics import YOLO
import os
import torch

if __name__ == '__main__':
    # 1. 徹底遷移到 E 槽
    base_dir = r'E:\Myst_Project'
    if not os.path.exists(base_dir): os.makedirs(base_dir)
    os.chdir(base_dir)

    # 2. 顯存碎片清理 (預防措施)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 3. 載入模型
    model = YOLO('yolov8n.pt') 

    # 4. 極限保守訓練配置
    model.train(
        data=r'F:\PySpace\Auto_platinum_hand\data.yaml', 
        epochs=100, 
        imgsz=640, 
        device=0,      # 確保使用 GPU
        batch=4,       # 再次確認：小 batch 雖然慢一點點，但絕對不會崩潰
        workers=0,     # 🌟 解決 WinError 1455 的唯一靈丹妙藥
        project=base_dir, 
        name='v1_safe_run',
        plots=False,   # 關閉繪圖，防止 OpenCV 再次觸發內存溢出
        cache=False,   # 不佔用額外硬碟/內存快取
        deterministic=True # 增加穩定性
    )