import cv2
import numpy as np
import bettercam
import time
from playwright.sync_api import sync_playwright

def run_gpu_crop_test():
    with sync_playwright() as p:
        # 1. 獲取 Playwright 的「真理圖」作為基準
        print("🔗 正在連接 Chrome 並獲取 Playwright 標準截圖...")
        browser = p.chromium.connect_over_cdp("http://localhost:9222")
        page = next((pg for pg in browser.contexts[0].pages if "youtube" in pg.url), browser.contexts[0].pages[0])
        video = page.wait_for_selector("video")
        
        # 拿到無視網址列的純淨影片圖
        p_bytes = video.screenshot(type="jpeg", quality=100)
        p_frame = cv2.imdecode(np.frombuffer(p_bytes, np.uint8), cv2.IMREAD_COLOR)
        
        # 2. 初始化單個 Bettercam 實例 (全屏模式)
        print("🔍 啟動 GPU 全屏抓取並進行像素校準...")
        camera = bettercam.create() 
        full_screen = camera.grab()
        if full_screen is None:
            print("❌ 錯誤：GPU 無法抓取螢幕。")
            return
            
        full_screen_bgr = cv2.cvtColor(full_screen, cv2.COLOR_RGB2BGR)
        
        # 3. 自動計算物理偏移量
        res = cv2.matchTemplate(full_screen_bgr, p_frame, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        
        if max_val < 0.8:
            print(f"❌ 校準失敗 (置信度 {max_val:.2f})，請檢查影片是否被遮擋。")
            return

        # 4. 識別影片內部的彩色遊戲框 (rx, ry, rw, rh)
        gray_p = cv2.cvtColor(p_frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.medianBlur(gray_p, 35)
        edges = cv2.Canny(blurred, 30, 100)
        cnts, _ = cv2.findContours(cv2.dilate(edges, None, iterations=2), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rx, ry, rw, rh = cv2.boundingRect(max(cnts, key=cv2.contourArea))

        # 計算最終物理座標 (相對於螢幕)
        real_x, real_y = max_loc[0] + rx, max_loc[1] + ry
        print(f"🎯 校準成功！物理座標: ({real_x}, {real_y}), 尺寸: {rw}x{rh}")

        # 5. 進入監控循環：全屏抓取 + 內存裁切
        cv2.namedWindow("GPU_Crop_Test", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("GPU_Crop_Test", rw, rh)

        print("✅ 測試啟動！監視窗應顯示 100% 純淨畫面。按 'q' 退出。")
        while True:
            full_rgb = camera.grab()
            if full_rgb is not None:
                # 💥 核心：直接在內存中裁切出遊戲區域，避開單例 Bug
                # 注意：Bettercam 抓到的是 (Height, Width, Channel)
                crop_rgb = full_rgb[real_y : real_y + rh, real_x : real_x + rw]
                
                # 轉換顯示
                display = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
                
                # 畫十字線驗證中心點
                cv2.line(display, (rw//2, 0), (rw//2, rh), (0, 255, 255), 1)
                cv2.line(display, (0, rh//2), (rw, rh//2), (0, 255, 255), 1)
                
                cv2.imshow("GPU_Crop_Test", display)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_gpu_crop_test()