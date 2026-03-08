import cv2
import numpy as np
import math
import time
import dxcam
import torch
import win32gui
import win32api                     # added for cursor control
import win32con
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
        self.last_click_ms = 0
        self.sync_failed = 0
        self.sync_paused = False
        self.frame_counter = 0
        self.sync_log = []

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
    def Realtime_cursor_position(self):
        """回傳目前影片與實機的指針位置 (tuple 或 None)"""
        return self.last_known_cursor_pos, self.chaiki_cursor_pos

    def move_action(self, target_pos):
        """立即把 Chaiki 游標移到目標（relative to chiaki window）"""
        if target_pos is None: return
        abs_x = self.chiaki_x + int(target_pos[0])
        abs_y = self.chiaki_y + int(target_pos[1])
        win32api.SetCursorPos((abs_x, abs_y))
        self.chaiki_cursor_pos = (target_pos[0], target_pos[1])
        print(f"⇨ moved chiaki cursor to {abs_x},{abs_y}")

    def refine_move(self, target_pos):
        """小幅調整；這裡簡單地向目標靠近一半距離"""
        if target_pos is None: return
        cur = self.chaiki_cursor_pos or (0, 0)
        dx = target_pos[0] - cur[0]
        dy = target_pos[1] - cur[1]
        step = (cur[0] + dx * 0.5, cur[1] + dy * 0.5)
        abs_x = self.chiaki_x + int(step[0])
        abs_y = self.chiaki_y + int(step[1])
        win32api.SetCursorPos((abs_x, abs_y))
        self.chaiki_cursor_pos = step
        print(f"⇨ refined move to {abs_x},{abs_y}")

    def click_action(self, target_pos):
        """在 Chaiki 上點擊（假定目標座標已轉為相對）"""
        if target_pos is None: return
        abs_x = self.chiaki_x + int(target_pos[0])
        abs_y = self.chiaki_y + int(target_pos[1])
        win32api.SetCursorPos((abs_x, abs_y))
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, abs_x, abs_y, 0, 0)
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, abs_x, abs_y, 0, 0)
        self.click_count += 1
        print(f"✳ clicked chiaki at {abs_x},{abs_y}")

    # ==========================================
    # queue helpers
    # ==========================================
    def enqueue_click(self, video_pos):
        now = time.time() * 1000
        if video_pos is None:
            return
        if now - self.last_click_ms < 200:   # 200 ms 最小間隔
            return
        self.last_click_ms = now
        self.queue.append({'video_pos': video_pos, 'ts': now})
        print(f"[queue] enqueued {video_pos}")

    def process_queue(self):
        while self.queue:
            item = self.queue.pop(0)
            vpos = item['video_pos']
            tx = vpos[0] / self.scale_x
            ty = vpos[1] / self.scale_y
            self.move_action((tx, ty))
            self.click_action((tx, ty))

    # ==========================================
    # 區塊 C：點擊決策與同步暫停 (Logic & Sync)
    # ==========================================
    def sync_check(self, yt_gray, chiaki_gray, page):
        """每隔一段時間檢查邊緣 hash，同步失敗時暫停/恢復影片"""
        if yt_gray is None or chiaki_gray is None:
            return
        h1 = self.get_sparse_hash(yt_gray)
        h2 = self.get_sparse_hash(chiaki_gray)
        diff = np.mean(np.abs(h1 - h2))
        threshold = 20.0
        self.frame_counter += 1
        if self.frame_counter % 60 != 0:
            return
        if diff > threshold:
            self.sync_failed += 1
            print(f"[sync] diff {diff:.1f}, failed {self.sync_failed}")
        else:
            self.sync_failed = 0
            if self.sync_paused:
                page.evaluate("document.querySelector('video').play();")
                self.sync_paused = False
                self.log_sync_event("resync", f"diff={diff:.1f}")
        if self.sync_failed >= 3 and not self.sync_paused:
            page.evaluate("document.querySelector('video').pause();")
            self.sync_paused = True
            self.log_sync_event("pause_for_sync", f"diff={diff:.1f}")

    def log_sync_event(self, event_type, details):
        ts = time.time()
        print(f"[{ts:.2f}] {event_type}: {details}")
        self.sync_log.append((ts, event_type, details))

    # ==========================================
    # 執行循環 (需要對應的修改)
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
                    if v_pos:
                        self.last_known_cursor_pos = v_pos
                    c_pos = self.detect_template(chiaki_gray, self.chiaki_cursor_tpl, 0.70)
                    if c_pos:
                        self.chaiki_cursor_pos = c_pos

                    key = cv2.waitKey(1) & 0xFF

                    if recording:
                        # --- 檢查變化並入隊 ---
                        if self.last_full_gray_np is not None:
                            last_hash = self.get_sparse_hash(self.last_full_gray_np)
                            curr_hash = self.get_sparse_hash(yt_gray)
                            if (np.mean(np.abs(curr_hash - last_hash)) > 30
                                    or (v_pos and self.last_known_cursor_pos and v_pos != self.last_known_cursor_pos)):
                                self.enqueue_click(v_pos)

                        # 同步檢查
                        self.sync_check(yt_gray, chiaki_gray, page)

                        # 處理隊列中的任務
                        self.process_queue()

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

                    # 繪製 HUD
                    display = self.draw_sync_hud(yt_frame, self.last_known_cursor_pos, self.chaiki_cursor_pos, fps)
                    cv2.imshow("Platinum Vision", display)

                    if key == ord('q'): break

            self.camera.stop()
            browser.close()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    agent = AutoPlatinumHand("https://www.youtube.com/watch?v=7K_NimshHUI")
    agent.run_live_sync()