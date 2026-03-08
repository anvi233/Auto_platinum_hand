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
    model_v1_path = r'E:\Myst_Project\v1_safe_run2\weights\best.pt'
    model = YOLO(model_v1_path) 

    # 4. 極限保守訓練配置
    model.train(
        data=r'F:\PySpace\Auto_platinum_hand\data.yaml', 
        epochs=100,        # 再跑 100 輪
        imgsz=640, 
        device=0,
        batch=8,           # 現在穩定後可以嘗試 batch 8 提高速度
        workers=0,         # 依然保持 0 防止內存報錯
        project=r'E:\Myst_Project', 
        name='v2_refined_260_samples',
        plots=False,
        save_period=50
    )