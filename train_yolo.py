from ultralytics import YOLO
import os

if __name__ == '__main__':
    base_dir = r'E:\Myst_Project'
    os.chdir(base_dir)

    # 載入你原本已經訓練好的 best.pt
    model = YOLO(r'E:\Myst_Project\v3_final_1900\weights\best.pt')

    # 使用較低的學習率 (lr0=0.001) 進行微調，保護原有的 1900 張樣本特徵
    # 加入數據增強參數，強制模型適應 Chiaki 實機的色差、模糊與比例差異
    results = model.train(
        data=r'F:\PySpace\Auto_platinum_hand\data.yaml', 
        epochs=50,             # 50 輪微調通常已足夠
        lr0=0.001,             # 降低初始學習率
        hsv_h=0.05,            # 輕微色調變換
        hsv_s=0.6,             # 較強的飽和度變換 (對抗串流泛白)
        hsv_v=0.6,             # 較強的亮度變換
        scale=0.3,             # 30% 縮放干擾 (對抗實機解析度差異)
        translate=0.1,         # 10% 輕微平移
        fliplr=0.0,            # 絕對禁止左右翻轉 (指針方向特徵不能亂)
        erasing=0.1,           # 10% 隨機擦除 (模擬串流馬賽克破圖)
        batch=16,
        workers=4
    )