from ultralytics import YOLO
import os

if __name__ == '__main__':
    base_dir = r'E:\Myst_Project'
    os.chdir(base_dir)

    # 🌟 站在巨人的肩膀上：載入 V2 的完美權重
    model = YOLO(r'E:\Myst_Project\v2_ultimate_11002\weights\best.pt') 

    # 🌟 1900 張圖的終極訓練
    model.train(
        data=r'F:\PySpace\Auto_platinum_hand\data.yaml', 
        epochs=100,        # 基礎已經很好了，100 輪足夠收斂
        imgsz=640, 
        device=0,
        batch=8,           # 保持安全配置
        workers=0,
        project=base_dir, 
        name='v3_final_1900',
        cache=True,
        plots=True
    )
    print("\n🏆 1900 樣本終極視覺引擎誕生！")