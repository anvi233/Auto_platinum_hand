from ultralytics import YOLO
import os

# 1. 載入剛出爐的模型
model_path = r'E:\Myst_Project\v1_safe_run2\weights\best.pt'
model = YOLO(model_path)

# 2. 批量推理 1000+ 張圖 (dataset/images 資料夾)
# save=True 會把結果存成圖片，你可以直接看
# conf=0.5 設定信心門檻，低於 50% 的不顯示
results = model.predict(
    source=r'F:\PySpace\Auto_platinum_hand\dataset\youtube', 
    save=True, 
    conf=0.5, 
    project=r'E:\Myst_Project\Inference_Test', 
    name='full_check'
)

print(f"✅ 巡檢完成！請去 E:\Myst_Project\Inference_Test\full_check 裡看畫好框的圖。")