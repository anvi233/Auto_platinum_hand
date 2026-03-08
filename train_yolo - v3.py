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
        epochs=150,        # 再跑 100 輪
        imgsz=640, 
        device=0,
        batch=8,           # 現在穩定後可以嘗試 batch 8 提高速度
        workers=0,         # 依然保持 0 防止內存報錯
        project=base_dir, 
        name='v2_ultimate_1100',
        plots=True,
        cache=True,
    )

    # 2. 🌟 訓練完成後，立即生成這 1100 張圖的最新標籤
    # 這樣你以後 check 時，直接去這個文件夾改 .txt 就好
    print("\n🚀 正在為 1100 張圖生成最新標籤庫...")
    best_model = YOLO(os.path.join(base_dir, 'v2_ultimate_1100', 'weights', 'best.pt'))
    best_model.predict(
        source=r'F:\PySpace\Auto_platinum_hand\dataset\images',
        save_txt=True,
        save=True,  # 同時生成帶框的圖方便你肉眼 check
        project=base_dir,
        name='v2_final_check_labels',
        conf=0.5
    )

    print(f"\n🏆 全部完成！")
    print(f"最新標籤在: {base_dir}\\v2_final_check_labels\\labels")