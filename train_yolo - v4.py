from ultralytics import YOLO

if __name__ == '__main__':
    # 🌟 載入真正成功的那個模型 (注意資料夾結尾的 2)
    model_path = r'E:\Myst_Project\v2_ultimate_11002\weights\best.pt'
    best_model = YOLO(model_path)

    # 🌟 執行剛才沒跑完的自動標註
    print("\n🚀 正在為 1100 張圖生成最新標籤庫...")
    best_model.predict(
        source=r'F:\PySpace\Auto_platinum_hand\dataset\images',
        save_txt=True,
        save=True,  # 依然保存圖片方便你肉眼 Check
        project=r'E:\Myst_Project',
        name='v2_final_check_labels',
        conf=0.5
    )
    
    print("\n🏆 標籤與預覽圖生成完畢！")