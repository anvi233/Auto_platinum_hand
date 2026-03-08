from ultralytics import YOLO

if __name__ == '__main__':
    # 🌟 載入剛剛練成的最強大腦 (注意路徑是 11002)
    model_path = r'E:\Myst_Project\v2_ultimate_11002\weights\best.pt'
    model = YOLO(model_path)

    print("\n🚀 正在對最後的 800 張未知圖進行盲測...")
    model.predict(
        # 🌟 指向你存放剩下 800 張圖的專屬資料夾
        source=r'F:\PySpace\Auto_platinum_hand\dataset\images_test800',
        save=True,      # 存成圖片，方便你像翻幻燈片一樣快速肉眼驗收
        save_txt=True,  # 順便把 txt 也吐出來，如果有極個別的錯可以直接改
        conf=0.5,       # 保持實戰門檻
        project=r'E:\Myst_Project\Final_Test',
        name='blind_test_800'
    )
    print("\n🏆 盲測預測完畢！請前往 E:\Myst_Project\Final_Test\blind_test_800 驗收。")