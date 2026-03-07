import cv2
import numpy as np
import math
import time
import dxcam 
import torch 
import win32gui
import win32api
import win32con
from playwright.sync_api import sync_playwright
# [修改解釋]: 引入 YOLO 官方庫以支持 3060 硬件加速識別
from ultralytics import YOLO

class AutoPlatinumHand:
    def __init__(self, youtube_url, cursor_path='cursor.png', waiting_path='waiting.png', chiaki_cursor_path='cursor2.png'):
        # --- 設備與資源初始化 ---
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # [修改解釋]: 初始化 YOLO 模型。初始使用 yolov8n.pt，訓練後可替換為自定義模型路徑
        self.model = YOLO('yolov8n.pt') 
        
        self.cursor_tpl = cv2.imread(cursor_path, 0)
        self.chiaki_cursor_tpl = cv2.imread(chiaki_cursor_path, 0)
        self.waiting_tpl = cv2.imread(waiting_path, 0)
        
        # --- 狀態與隊列 ---
        self.youtube_url = youtube_url
        self.state = "INIT"  # HOVER, MOVE, WAIT
        self.queue = []      # 任務隊列：存放 hover, click, scene_change_check
        self.click_count = 0
        self.scene_change_count = 0
        
        # --- 座標追蹤變量 (1:1 映射) ---
        self.last_known_cursor_pos = None  # 影片指針位置
        self.chaiki_cursor_pos = None      # Chaiki 實機指針位置
        
        # --- 窗口同步與偏移 ---
        self.camera = None
        self.chiaki_hwnd = None
        
        # 絕對座標儲存
        self.yt_x, self.yt_y, self.yt_w, self.yt_h = 0, 0, 0, 0
        self.chiaki_x, self.chiaki_y, self.chiaki_w, self.chiaki_h = 0, 0, 0, 0
        self.scale_x = 1.0
        self.scale_y = 1.0
        
        # --- 暫停與免責期邏輯 ---
        self.last_waiting_ms = 0.0
        self.last_time_ms = 0.0
        self.last_full_gray_np = None

        # --- 補齊：移動控制參數 ---
        self.deadzone = 8
        self.key_states = {'up': False, 'down': False, 'left': False, 'right': False}
        self.prev_roi_patch = None
        self.roi_box = None

    # ==========================================
    # 區塊 A：初始化與窗口對齊 (不動)
    # ==========================================

    def get_pure_game_scene(self, bgr_image):
        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        _, dark_mask = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY_INV)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        dark_mask_closed = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel)
        cnts_dark, _ = cv2.findContours(dark_mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts_dark: return 0, 0, bgr_image.shape[1], bgr_image.shape[0]
        largest_dark_cnt = max(cnts_dark, key=cv2.contourArea)
        cx, cy, cw, ch = cv2.boundingRect(largest_dark_cnt)
        container_roi_gray = gray[cy:cy+ch, cx:cx+cw]
        _, game_mask = cv2.threshold(container_roi_gray, 15, 255, cv2.THRESH_BINARY)
        game_mask = cv2.morphologyEx(game_mask, cv2.MORPH_CLOSE, kernel)
        game_mask = cv2.morphologyEx(game_mask, cv2.MORPH_OPEN, kernel)
        cnts_game, _ = cv2.findContours(game_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts_game: return cx, cy, cw, ch
        largest_game_cnt = max(cnts_game, key=cv2.contourArea)
        gx, gy, gw, gh = cv2.boundingRect(largest_game_cnt)
        return cx + gx, cy + gy, gw, gh

    def get_absolute_game_rect(self, keyword, full_screen_frame):
        hwnds = []
        def enum_cb(hwnd, param):
            if win32gui.IsWindowVisible(hwnd):
                title = win32gui.GetWindowText(hwnd).lower()
                if keyword in title and "powershell" not in title:
                    param.append(hwnd)
        win32gui.EnumWindows(enum_cb, hwnds)
        if not hwnds: return None
        hwnd = hwnds[0]
        if "chiaki" in keyword: self.chiaki_hwnd = hwnd
        rect = win32gui.GetWindowRect(hwnd)
        sh, sw = full_screen_frame.shape[:2]
        x1, y1, x2, y2 = max(0, rect[0]), max(0, rect[1]), min(sw, rect[2]), min(sh, rect[3])
        if x1 >= x2 or y1 >= y2: return None
        window_img = full_screen_frame[y1:y2, x1:x2]
        ix, iy, iw, ih = self.get_pure_game_scene(window_img)
        return x1 + ix, y1 + iy, iw, ih

    def auto_align_chiaki(self, full_grab):
        chiaki_rect = self.get_absolute_game_rect("chiaki", full_grab)
        if chiaki_rect:
            self.chiaki_x, self.chiaki_y, self.chiaki_w, self.chiaki_h = chiaki_rect
            print(f"✅ 實機精確鎖定: ({self.chiaki_x}, {self.chiaki_y})")
            return True
        return False

    def chaiki_ready(self, full_grab):
        yt_rect = self.get_absolute_game_rect("youtube", full_grab)
        if yt_rect:
            self.yt_x, self.yt_y, self.yt_w, self.yt_h = yt_rect
        self.auto_align_chiaki(full_grab)
        if self.yt_w > 0 and self.chiaki_w > 0:
            self.scale_x = self.yt_w / self.chiaki_w
            self.scale_y = self.yt_h / self.chiaki_h

    # ==========================================
    # 區塊 B：指針追蹤與移動邏輯 (YOLO 改裝區)
    # ==========================================

    def Realtime_cursor_position(self, bgr_frame, last_pos, target_class='pointer'):
        """🚀 YOLO 數據源：直接獲取識別物體的相對坐標，取代繁瑣的影像過濾"""
        # [修改解釋]: 使用 YOLO 進行推理。stream=True 可進一步提升 3060 處理流媒體的性能
        results = self.model.predict(bgr_frame, conf=0.5, verbose=False)
        
        best_pos = None
        for result in results:
            for box in result.boxes:
                # [修改解釋]: 根據標籤篩選目標。如果是訓練好的模型，這裡可區分 'move' 或 'click'
                # 暫時為了通用，獲取第一個偵測到的物體中心
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                best_pos = (int(cx), int(cy))
                break # 優先取置信度最高的
        
        # [修改解釋]: 抹除微動：如果 YOLO 偵測到的位移小於 3 像素，視為原地不動，防止指針發抖
        if best_pos and last_pos:
            dist = math.hypot(best_pos[0] - last_pos[0], best_pos[1] - last_pos[1])
            if dist < 3: return last_pos

        return best_pos if best_pos else last_pos

    def track_video_cursor(self, target_pos, current_pos):
        if not target_pos or not current_pos: return False
        tx, ty = int(target_pos[0] / self.scale_x), int(target_pos[1] / self.scale_y)
        dx, dy = tx - current_pos[0], ty - current_pos[1]
        dist = math.hypot(dx, dy)
        if dist <= self.deadzone:
            self.release_all_keys()
            return True
        self.update_key_bg('right', dx > (self.deadzone if dist > 35 else 2))
        self.update_key_bg('left', dx < -(self.deadzone if dist > 35 else 2))
        self.update_key_bg('down', dy > (self.deadzone if dist > 35 else 2))
        self.update_key_bg('up', dy < -(self.deadzone if dist > 35 else 2))
        return False

    def update_key_bg(self, key_str, press):
        if not self.chiaki_hwnd: return
        vk_map = {'up': win32con.VK_UP, 'down': win32con.VK_DOWN, 'left': win32con.VK_LEFT, 'right': win32con.VK_RIGHT, 'enter': win32con.VK_RETURN}
        vk = vk_map.get(key_str)
        if self.key_states.get(key_str) != press:
            if press: win32api.PostMessage(self.chiaki_hwnd, win32con.WM_KEYDOWN, vk, 0)
            else: win32api.PostMessage(self.chiaki_hwnd, win32con.WM_KEYUP, vk, 0)
            self.key_states[key_str] = press

    def release_all_keys(self):
        for k in ['up', 'down', 'left', 'right']: self.update_key_bg(k, False)

    # ==========================================
    # 區塊 D：視覺化 (保留以便調試)
    # ==========================================

    def draw_8_lines(self, overlay, cx, cy, w, h, color):
        cx, cy = int(cx), int(cy)
        cv2.line(overlay, (cx, cy), (cx, 0), color, 1) 
        cv2.line(overlay, (cx, cy), (cx, h), color, 1) 
        cv2.line(overlay, (cx, cy), (0, cy), color, 1) 
        cv2.line(overlay, (cx, cy), (w, cy), color, 1) 

    def draw_sync_hud(self, frame, video_pos, chiaki_pos, fps):
        overlay = frame.copy()
        if video_pos: self.draw_8_lines(overlay, video_pos[0], video_pos[1], self.yt_w, self.yt_h, (0, 165, 255))
        if chiaki_pos: self.draw_8_lines(overlay, chiaki_pos[0]*self.scale_x, chiaki_pos[1]*self.scale_y, self.yt_w, self.yt_h, (255, 255, 0))
        display = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        cv2.putText(display, f"FPS: {int(fps)} | YOLO Active", (10, 30), 0, 0.7, (0, 255, 0), 2)
        return display

    # ==========================================
    # 執行循環
    # ==========================================

    def run_live_sync(self, start_time_sec=153):
        with sync_playwright() as p:
            browser = p.chromium.connect_over_cdp("http://localhost:9222")
            page = next((pg for pg in browser.contexts[0].pages if "youtube" in pg.url), browser.contexts[0].pages[0])
            page.evaluate(f"document.querySelector('video').pause(); document.querySelector('video').currentTime = {start_time_sec};")
            
            self.camera = dxcam.create(output_idx=0, output_color="BGR") 
            full_grab = self.camera.grab()
            while full_grab is None: full_grab = self.camera.grab()
            self.chaiki_ready(full_grab)
            self.camera.start(target_fps=30, video_mode=True)
            
            recording, prev_time = False, time.time()
            while True:
                full_frame = self.camera.get_latest_frame()
                if full_frame is None: continue
                yt_frame = full_frame[self.yt_y:self.yt_y+self.yt_h, self.yt_x:self.yt_x+self.yt_w]
                chiaki_frame = full_frame[self.chiaki_y:self.chiaki_y+self.chiaki_h, self.chiaki_x:self.chiaki_x+self.chiaki_w]
                
                curr_time = time.time()
                fps = 1 / (curr_time - prev_time)
                prev_time = curr_time

                # [修改解釋]: 統一調用 YOLO 數據源
                self.last_known_cursor_pos = self.Realtime_cursor_position(yt_frame, self.last_known_cursor_pos)
                self.chaiki_cursor_pos = self.Realtime_cursor_position(chiaki_frame, self.chaiki_cursor_pos)
                
                key = cv2.waitKey(1) & 0xFF
                if recording: self.track_video_cursor(self.last_known_cursor_pos, self.chaiki_cursor_pos)
                if key == ord(' '):
                    recording = not recording
                    page.evaluate(f"document.querySelector('video').{'play' if recording else 'pause'}()")
                    if not recording: self.release_all_keys()
                
                cv2.imshow("Platinum Vision", self.draw_sync_hud(yt_frame, self.last_known_cursor_pos, self.chaiki_cursor_pos, fps))
                if key == ord('q'): break
            self.camera.stop(); browser.close(); cv2.destroyAllWindows()

if __name__ == "__main__":
    agent = AutoPlatinumHand("https://www.youtube.com/watch?v=7K_NimshHUI")
    agent.run_live_sync()