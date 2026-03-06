import cv2
import numpy as np
import math
import time
import dxcam 
import torch 
import win32gui
from playwright.sync_api import sync_playwright

class AutoPlatinumHand:
    def __init__(self, youtube_url, cursor_path='cursor.png', waiting_path='waiting.png', chiaki_cursor_path='cursor2.png'):
        # --- 設備與資源初始化 ---
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    # ==========================================
    # 區塊 A：初始化與窗口對齊 (Ready 階段)
    # ==========================================

    def get_pure_game_scene(self, bgr_image):
        """核心過濾：找尋畫面中最大面積的彩色長方形"""
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
        """抓取目標視窗並裁切黑邊，返回絕對座標"""
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
        """鎖定 Chiaki 窗口並提取遊戲 ROI"""
        chiaki_rect = self.get_absolute_game_rect("chiaki", full_grab)
        if chiaki_rect:
            self.chiaki_x, self.chiaki_y, self.chiaki_w, self.chiaki_h = chiaki_rect
            print(f"✅ 實機精確鎖定: ({self.chiaki_x}, {self.chiaki_y}), 尺寸 {self.chiaki_w}x{self.chiaki_h}")
            return True
        print("❌ 找不到 Chiaki 視窗！")
        return False

    def chaiki_ready(self, full_grab):
        """完成 1:1 強制對齊計算"""
        yt_rect = self.get_absolute_game_rect("youtube", full_grab)
        if yt_rect:
            self.yt_x, self.yt_y, self.yt_w, self.yt_h = yt_rect
            print(f"🎯 影片精確鎖定: ({self.yt_x}, {self.yt_y}), 尺寸 {self.yt_w}x{self.yt_h}")
        else:
            print("❌ 找不到 YouTube 視窗！")

        self.auto_align_chiaki(full_grab)

        if self.yt_w > 0 and self.chiaki_w > 0:
            self.scale_x = self.yt_w / self.chiaki_w
            self.scale_y = self.yt_h / self.chiaki_h
            print(f"⚖️ 邏輯縮放比 - X軸: {self.scale_x:.4f}, Y軸: {self.scale_y:.4f}")

    # ==========================================
    # 區塊 B：指針追蹤與移動邏輯 (Movement)
    # ==========================================

    def detect_template(self, gray_frame, template, threshold=0.70):
        if template is None or gray_frame is None or gray_frame.size == 0: return None
        res = cv2.matchTemplate(gray_frame, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        if max_val > threshold:
            return (max_loc[0] + template.shape[1]//2, max_loc[1] + template.shape[0]//2)
        return None

    def track_video_cursor(self, current_gray, prev_gray):
        pass

    def Realtime_cursor_position(self):
        pass

    def move_to_target(self, target_pos):
        pass

    def refine_move_to_target(self, target_pos):
        pass

    # ==========================================
    # 區塊 C：點擊決策與同步暫停 (Logic & Sync)
    # ==========================================

    def is_scene_change(self, frame_hash, prev_hash):
        pass

    def is_roi_change(self, current_roi, prev_roi):
        pass

    def is_cursor_change(self, frame):
        pass

    def click_check(self, is_scene, is_roi, is_wait, current_ms):
        pass

    def need_stop_for_sync(self):
        pass

    def execute_click(self, queue_item):
        pass

    def execute_queue_logic(self):
        pass

    # ==========================================
    # 區塊 D：視覺化與日誌 (UI & Log)
    # ==========================================

    def draw_8_lines(self, overlay, cx, cy, w, h, color):
        """繪製八向百分比輔助線"""
        diag = math.hypot(w, h)
        left_pct = int((cx / w) * 100) if w > 0 else 0
        right_pct = int(((w - cx) / w) * 100) if w > 0 else 0
        top_pct = int((cy / h) * 100) if h > 0 else 0
        bottom_pct = int(((h - cy) / h) * 100) if h > 0 else 0
        tl_pct = int((math.hypot(cx, cy) / diag) * 100) if diag > 0 else 0
        tr_pct = int((math.hypot(w - cx, cy) / diag) * 100) if diag > 0 else 0
        bl_pct = int((math.hypot(cx, h - cy) / diag) * 100) if diag > 0 else 0
        br_pct = int((math.hypot(w - cx, h - cy) / diag) * 100) if diag > 0 else 0

        cv2.line(overlay, (cx, cy), (cx, 0), color, 1) 
        cv2.putText(overlay, f"{top_pct}%", (cx + 5, cy // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        cv2.line(overlay, (cx, cy), (cx, h), color, 1) 
        cv2.putText(overlay, f"{bottom_pct}%", (cx + 5, cy + (h - cy) // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        cv2.line(overlay, (cx, cy), (0, cy), color, 1) 
        cv2.putText(overlay, f"{left_pct}%", (cx // 2, cy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        cv2.line(overlay, (cx, cy), (w, cy), color, 1) 
        cv2.putText(overlay, f"{right_pct}%", (cx + (w - cx) // 2, cy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        cv2.line(overlay, (cx, cy), (0, 0), color, 1) 
        cv2.putText(overlay, f"{tl_pct}%", (cx // 2, cy // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        cv2.line(overlay, (cx, cy), (w, 0), color, 1) 
        cv2.putText(overlay, f"{tr_pct}%", (cx + (w - cx) // 2, cy // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        cv2.line(overlay, (cx, cy), (0, h), color, 1) 
        cv2.putText(overlay, f"{bl_pct}%", (cx // 2, cy + (h - cy) // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        cv2.line(overlay, (cx, cy), (w, h), color, 1) 
        cv2.putText(overlay, f"{br_pct}%", (cx + (w - cx) // 2, cy + (h - cy) // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    def draw_sync_hud(self, frame, video_pos, chiaki_pos, fps):
        """獨立繪製 GUI：整合輔助線繪製與縮放邏輯"""
        overlay = frame.copy()
        
        # 繪製影片橘色 8 線
        if video_pos:
            self.draw_8_lines(overlay, video_pos[0], video_pos[1], self.yt_w, self.yt_h, (0, 165, 255))
        
        # 繪製實機藍色 8 線 (需套用邏輯縮放比)
        mapped_cx, mapped_cy = 0, 0
        if chiaki_pos:
            mapped_cx = int(chiaki_pos[0] * self.scale_x)
            mapped_cy = int(chiaki_pos[1] * self.scale_y)
            self.draw_8_lines(overlay, mapped_cx, mapped_cy, self.yt_w, self.yt_h, (255, 255, 0))
            
        display = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # 狀態與誤差顯示
        if video_pos and chiaki_pos:
            dist = math.hypot(mapped_cx - video_pos[0], mapped_cy - video_pos[1])
            status_color = (0, 255, 0) if dist < 5 else (0, 0, 255)
            cv2.putText(display, f"Distance: {int(dist)}px | FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

        return display

    def log_sync_event(self, event_type, details):
        pass

    def get_sparse_hash(self, gray_frame):
        h, w = gray_frame.shape
        dh, dw = int(h * 0.15), int(w * 0.15)
        step = 4 
        top = gray_frame[0:dh, ::step].flatten()
        bottom = gray_frame[h-dh:h, ::step].flatten()
        left = gray_frame[dh:h-dh, 0:dw:step].flatten()
        right = gray_frame[dh:h-dh, w-dw:w:step].flatten()
        border_pixels = np.concatenate((top, bottom, left, right))
        return border_pixels.astype(np.int16)

    # ==========================================
    # 執行循環
    # ==========================================

    def run_live_sync(self, start_time_sec=35):
        with sync_playwright() as p:
            # --- Playwright 與影片同步啟動 ---
            browser = p.chromium.connect_over_cdp("http://localhost:9222")
            page = next((pg for pg in browser.contexts[0].pages if "youtube" in pg.url), browser.contexts[0].pages[0])
            video = page.wait_for_selector("video")
            page.evaluate(f"document.querySelector('video').pause(); document.querySelector('video').currentTime = {start_time_sec};")
            time.sleep(0.5)
            
            # --- 初始截圖與播放器定位 ---
            self.camera = dxcam.create(output_idx=0, output_color="BGR") 
            
            full_grab = self.camera.grab()
            while full_grab is None:
                full_grab = self.camera.grab()
                time.sleep(0.01)
                
            self.chaiki_ready(full_grab)
            
            # 🎯 啟動背景線程模式 (事件驅動)
            self.camera.start(target_fps=30, video_mode=True)

            recording = False
            start_tick = 0
            prev_time = time.time()

            cv2.namedWindow("Platinum Vision")
            cv2.moveWindow("Platinum Vision", 1920 - self.yt_w - 50, 1080 - self.yt_h - 100)

            with torch.no_grad():
                while True:
                    # 🎯 阻塞等待新幀
                    full_frame = self.camera.get_latest_frame()
                    if full_frame is None: continue
                    
                    yt_frame = full_frame[self.yt_y:self.yt_y+self.yt_h, self.yt_x:self.yt_x+self.yt_w]
                    chiaki_frame = full_frame[self.chiaki_y:self.chiaki_y+self.chiaki_h, self.chiaki_x:self.chiaki_x+self.chiaki_w]
                    
                    yt_gray = cv2.cvtColor(yt_frame, cv2.COLOR_BGR2GRAY)
                    chiaki_gray = cv2.cvtColor(chiaki_frame, cv2.COLOR_BGR2GRAY)
                    
                    curr_time = time.time()
                    ms = ((cv2.getTickCount() - start_tick) / cv2.getTickFrequency()) * 1000 if recording else 0
                    fps = 1 / (curr_time - prev_time) if curr_time > prev_time else 60
                    self.last_time_ms = ms
                    prev_time = curr_time

                    # 雙指針辨識
                    v_pos = self.detect_template(yt_gray, self.cursor_tpl, 0.70)
                    if v_pos: self.last_known_cursor_pos = v_pos
                    
                    c_pos = self.detect_template(chiaki_gray, self.chiaki_cursor_tpl, 0.70)
                    if c_pos: self.chaiki_cursor_pos = c_pos

                    key = cv2.waitKey(1) & 0xFF

                    if recording:
                        # --- 後續點擊與隊列處理 (佔位) ---
                        pass

                    if key == ord(' '):
                        if not recording:
                            page.evaluate("document.querySelector('video').play();")
                            start_tick, recording = cv2.getTickCount(), True
                            print("▶️ 即時對齊模式啟動")
                        else:
                            page.evaluate("document.querySelector('video').pause();")
                            recording = False
                            print("⏸️ 即時對齊模式暫停")
                    
                    self.last_full_gray_np = yt_gray.copy()
                    
                    # 繪製 HUD (底圖直接使用影片區域)
                    display = self.draw_sync_hud(yt_frame, self.last_known_cursor_pos, self.chaiki_cursor_pos, fps)
                    cv2.imshow("Platinum Vision", display)
                    
                    if key == ord('q'): break

            self.camera.stop()
            browser.close()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    agent = AutoPlatinumHand("https://www.youtube.com/watch?v=7K_NimshHUI")
    agent.run_live_sync()