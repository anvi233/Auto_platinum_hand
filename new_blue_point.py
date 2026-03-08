import cv2
import numpy as np
import math
import time
import dxcam 
import win32gui
import win32api
import win32con
from ultralytics import YOLO

class AutoPlatinumHand:
    def __init__(self, model_path=r'E:\Myst_Project\v3_final_1900\weights\best.pt'):
        # --- 視覺引擎初始化 ---
        print("🧠 載入 YOLO V3 視覺引擎...")
        self.model = YOLO(model_path)
        self.camera = dxcam.create(output_idx=0, output_color="BGR")
        
        # --- 狀態與窗口 ---
        self.state = "INIT"
        self.chiaki_hwnd = None
        self.yt_locked_roi = None
        self.ck_locked_roi = None
        
        # 鍵盤狀態追蹤，防止重複發送
        self.key_states = {'up': False, 'down': False, 'left': False, 'right': False}

    # ==========================================
    # 區塊 A：畫面提取與 YOLO 推理
    # ==========================================

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
        if "chiaki" in keyword: self.chiaki_hwnd = target_hwnd
            
        rect = win32gui.GetClientRect(target_hwnd)
        left, top = win32gui.ClientToScreen(target_hwnd, (rect[0], rect[1]))
        right, bottom = win32gui.ClientToScreen(target_hwnd, (rect[2], rect[3]))
        
        w = right - left
        h = bottom - top
        if w <= 0 or h <= 0: return None
        return left, top, w, h

    def get_pure_game_scene(self, bgr_image):
        """自動切除瀏覽器UI與影片黑邊"""
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

    def process_target_yolo(self, full_frame, keyword, locked_roi):
        """處理單一目標，返回歸一化座標 (0~1) 和新的鎖定 ROI"""
        rect = self.get_window_client_area(keyword)
        if rect is None:
            return None, None, None, locked_roi
            
        tx, ty, tw, th = rect
        window_img = full_frame[ty:ty+th, tx:tx+tw]
        
        if locked_roi is None:
            ix, iy, iw, ih = self.get_pure_game_scene(window_img)
            if iw > 200 and ih > 200:
                locked_roi = (ix, iy, iw, ih)
                print(f"🔒 [{keyword}] 裁切鎖定: {iw}x{ih}")
            else:
                return None, None, None, locked_roi
        else:
            ix, iy, iw, ih = locked_roi

        pure_game_img = window_img[iy:iy+ih, ix:ix+iw]
        
        # YOLO 推理
        results = self.model.predict(pure_game_img, conf=0.6, verbose=False)
        
        if len(results[0].boxes) > 0:
            box = sorted(results[0].boxes, key=lambda x: x.conf, reverse=True)[0]
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            
            # 轉換為 0~1 的相對座標
            rel_x = cx / iw
            rel_y = cy / ih
            cls_name = self.model.names[int(box.cls)]
            
            return rel_x, rel_y, cls_name, locked_roi
            
        return None, None, None, locked_roi

    # ==========================================
    # 區塊 B：按鍵控制與移動邏輯
    # ==========================================

    def update_key_bg(self, key_str, press):
        """底層：發送按鍵到 Chiaki"""
        if not self.chiaki_hwnd: return
        vk_map = {'up': win32con.VK_UP, 'down': win32con.VK_DOWN, 'left': win32con.VK_LEFT, 'right': win32con.VK_RIGHT}
        vk = vk_map.get(key_str)
        if self.key_states.get(key_str) != press:
            if press: win32api.PostMessage(self.chiaki_hwnd, win32con.WM_KEYDOWN, vk, 0)
            else: win32api.PostMessage(self.chiaki_hwnd, win32con.WM_KEYUP, vk, 0)
            self.key_states[key_str] = press

    def release_all_keys(self):
        for k in ['up', 'down', 'left', 'right']: self.update_key_bg(k, False)

    def move_toward_target(self, current_x, current_y, target_x, target_y):
        """基於歸一化誤差 (0~1) 決定按鍵策略"""
        dx = target_x - current_x
        dy = target_y - current_y
        dist = math.hypot(dx, dy)

        # 🛑 誤差小於 1% (0.01) 視為到達目標
        if dist < 0.01:
            self.release_all_keys()
            return True # 到達標記

        # 🏃‍♂️ 大於 5% 誤差，長按全速移動
        if abs(dx) > 0.05:
            self.update_key_bg('right', dx > 0)
            self.update_key_bg('left', dx < 0)
        else:
            # 微調階段，短按或釋放
            self.update_key_bg('right', dx > 0.01)
            self.update_key_bg('left', dx < -0.01)

        if abs(dy) > 0.05:
            self.update_key_bg('down', dy > 0)
            self.update_key_bg('up', dy < 0)
        else:
            self.update_key_bg('down', dy > 0.01)
            self.update_key_bg('up', dy < -0.01)
            
        return False

    # ==========================================
    # 執行循環
    # ==========================================

    def run_follower_test(self):
        print("🚀 啟動視覺跟隨測試...")
        print("請手動播放 YouTube 影片，腳本將嘗試讓 Chiaki 指針跟隨。")
        print("👉 按 [Q] 退出。")
        print("-" * 50)
        
        self.camera.start(target_fps=30, video_mode=True)
        last_print_time = time.time()

        try:
            while True:
                full_frame = self.camera.get_latest_frame()
                if full_frame is None: continue

                # 1. 獲取 YouTube 影片目標位置 (0~1)
                yt_x, yt_y, yt_cls, self.yt_locked_roi = self.process_target_yolo(
                    full_frame, "youtube", self.yt_locked_roi
                )
                
                # 2. 獲取 Chiaki 實機當前位置 (0~1)
                ck_x, ck_y, ck_cls, self.ck_locked_roi = self.process_target_yolo(
                    full_frame, "chiaki", self.ck_locked_roi
                )

                # 3. 執行移動邏輯
                if yt_x is not None and ck_x is not None:
                    is_arrived = self.move_toward_target(ck_x, ck_y, yt_x, yt_y)
                    status_str = "🛑 已鎖定" if is_arrived else "🏃‍♂️ 移動中"
                else:
                    self.release_all_keys()
                    status_str = "👀 尋找目標中"

                # 每秒打印狀態
                curr_time = time.time()
                if curr_time - last_print_time >= 1.0:
                    y_str = f"YT:({yt_x:.3f}, {yt_y:.3f})" if yt_x else "YT: --"
                    c_str = f"CK:({ck_x:.3f}, {ck_y:.3f})" if ck_x else "CK: --"
                    print(f"⏱️ {status_str} | {y_str}  ->  {c_str}")
                    last_print_time = curr_time

                # 退出機制
                if win32api.GetAsyncKeyState(ord('Q')) & 0x8000:
                    break
                    
        finally:
            self.release_all_keys()
            self.camera.stop()
            print("🛑 測試結束。")

if __name__ == "__main__":
    agent = AutoPlatinumHand()
    agent.run_follower_test()