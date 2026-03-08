import cv2
import numpy as np
import os
import time
import dxcam 
import win32gui
import win32api
import win32con

# ==========================================
# 🎯 設定你要測試/採集的目標
# ==========================================
MODE = "YOUTUBE"  # 測試完 YouTube 後，改成 "CHIAKI" 繼續測

class VisualCollector:
    def __init__(self, mode):
        self.mode = mode
        self.camera = dxcam.create(output_idx=0, output_color="BGR")
        
        self.save_dir = f"dataset/{self.mode.lower()}"
        os.makedirs(self.save_dir, exist_ok=True)
        self.img_count = 0
        
        # 🔒 新增：用於鎖死裁切座標
        self.locked_roi = None 

    def get_pure_game_scene(self, bgr_image):
        """核心算法：自動切除瀏覽器UI與影片黑邊"""
        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        
        _, dark_mask = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY_INV)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        dark_mask_closed = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel)
        
        cnts_dark, _ = cv2.findContours(dark_mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts_dark: return 0, 0, bgr_image.shape[1], bgr_image.shape[0]
            
        largest_dark_cnt = max(cnts_dark, key=cv2.contourArea)
        cx, cy, cw, ch = cv2.boundingRect(largest_dark_cnt)
        
        container_roi_gray = gray[cy:cy+ch, cx:cx+cw]
        _, game_mask = cv2.threshold(container_roi_gray, 20, 255, cv2.THRESH_BINARY)
        game_mask = cv2.morphologyEx(game_mask, cv2.MORPH_CLOSE, kernel)
        game_mask = cv2.morphologyEx(game_mask, cv2.MORPH_OPEN, kernel)
        
        cnts_game, _ = cv2.findContours(game_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts_game: return cx, cy, cw, ch
            
        largest_game_cnt = max(cnts_game, key=cv2.contourArea)
        gx, gy, gw, gh = cv2.boundingRect(largest_game_cnt)
        
        return cx + gx, cy + gy, gw, gh

    def get_window_client_area(self, keyword):
        """獲取軟體內部渲染區域"""
        hwnds = []
        def enum_cb(hwnd, param):
            if win32gui.IsWindowVisible(hwnd):
                title = win32gui.GetWindowText(hwnd).lower()
                if keyword in title and "powershell" not in title and "code" not in title:
                    param.append(hwnd)
        win32gui.EnumWindows(enum_cb, hwnds)
        
        if not hwnds: return None
            
        target_hwnd = hwnds[0]
        rect = win32gui.GetClientRect(target_hwnd)
        left, top = win32gui.ClientToScreen(target_hwnd, (rect[0], rect[1]))
        right, bottom = win32gui.ClientToScreen(target_hwnd, (rect[2], rect[3]))
        
        w = right - left
        h = bottom - top
        if w <= 0 or h <= 0: return None
        return left, top, w, h

    def run(self):
        print(f"\n🔍 正在尋找 {self.mode} 視窗...")
        keyword = "youtube" if self.mode == "YOUTUBE" else "chiaki"
        rect = self.get_window_client_area(keyword)
        
        while rect is None:
            print(f"❌ 找不到 {self.mode}，請確認視窗已開啟...")
            time.sleep(1)
            rect = self.get_window_client_area(keyword)
            
        tx, ty, tw, th = rect
        self.camera.start(target_fps=30, video_mode=True)
        
        print(f"✅ 視窗鎖定成功！")
        print("👁️ 請查看彈出的 [YOLO Vision Preview] 預覽視窗。")
        print("👉 按下 [空白鍵] 開始/暫停每秒截圖。")
        print("👉 按下 [R] 重新計算並鎖定遊戲畫面範圍。")
        print("👉 按下 [Q] 退出程式。")

        recording = False
        last_save_time = time.time()
        
        cv2.namedWindow("YOLO Vision Preview", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("YOLO Vision Preview", 800, 450)

        while True:
            full_frame = self.camera.get_latest_frame()
            if full_frame is None: continue
            
            # 1. 抓取軟體視窗
            window_img = full_frame[ty:ty+th, tx:tx+tw]
            
            # 2. 🛡️ 畫面鎖定邏輯
            if self.locked_roi is None:
                ix, iy, iw, ih = self.get_pure_game_scene(window_img)
                # 確保抓到的不是一個太小的雜訊點
                if iw > 200 and ih > 200:
                    self.locked_roi = (ix, iy, iw, ih)
                    print(f"\n🔒 畫面裁切範圍已永久鎖定！尺寸: {iw}x{ih}")
            else:
                ix, iy, iw, ih = self.locked_roi

            # 3. 根據鎖定的座標進行裁切
            pure_game_img = window_img[iy:iy+ih, ix:ix+iw]
            
            # 4. 實時顯示給你看
            display_img = pure_game_img.copy()
            if recording:
                # 畫一個紅框提示正在錄製
                cv2.rectangle(display_img, (0, 0), (iw-1, ih-1), (0, 0, 255), 4)
                cv2.putText(display_img, f"RECORDING... {self.img_count}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            else:
                cv2.putText(display_img, "PAUSED (Press Space)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            
            cv2.imshow("YOLO Vision Preview", display_img)
            
            # 5. 存圖邏輯
            curr_time = time.time()
            if recording and (curr_time - last_save_time > 1.0):
                img_name = f"{self.save_dir}/{self.mode.lower()}_{self.img_count:04d}.jpg"
                cv2.imwrite(img_name, pure_game_img) # 保存乾淨的截圖
                self.img_count += 1
                last_save_time = curr_time
                print(f"✅ 已儲存截圖: {img_name}")

            # 6. 熱鍵控制
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): 
                break
            if key == ord('r'):
                self.locked_roi = None # 解除鎖定，讓下一幀重新計算
                recording = False
                print("🔄 重新尋找並鎖定畫面範圍...")
            if key == ord(' '):
                if self.locked_roi is not None:
                    recording = not recording
                    print(f"狀態: {'📸 開始錄製' if recording else '⏸️ 暫停錄製'}")
                else:
                    print("⚠️ 還沒抓到遊戲畫面，無法開始錄製！")

        self.camera.stop()
        cv2.destroyAllWindows()
        print(f"\n🎉 測試與採集結束！共收集 {self.img_count} 張圖片。")

if __name__ == "__main__":
    collector = VisualCollector(MODE)
    collector.run()