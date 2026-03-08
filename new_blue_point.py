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
from ultralytics import YOLO

class AutoPlatinumHand:
    def __init__(self, youtube_url, model_path=r'E:\Myst_Project\v3_final_1900\weights\best.pt'):
        # --- 設備與資源初始化 ---
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("🧠 載入 YOLO V3 視覺引擎...")
        self.model = YOLO(model_path)
        
        # --- 狀態 ---
        self.youtube_url = youtube_url
        self.state = "INIT"
        self.pending_click = False   # 🌟 核心：待執行點擊信號燈
        self.last_click_time = 0.0   # 確保最小點擊判斷間隔 0.5 秒
        
        # --- 座標追蹤變量 (1:1 映射) ---
        self.last_known_cursor_pos = None  # 影片指針位置 (歸一化)
        self.chaiki_cursor_pos = None      # Chaiki 實機指針位置 (歸一化)
        self.last_yt_cls = None            # 追蹤影片指針型態變化
        
        # --- 窗口同步與偏移 ---
        self.camera = None
        self.chiaki_hwnd = None
        
        # 絕對座標儲存
        self.yt_x, self.yt_y, self.yt_w, self.yt_h = 0, 0, 0, 0
        self.chiaki_x, self.chiaki_y, self.chiaki_w, self.chiaki_h = 0, 0, 0, 0
        self.scale_x = 1.0
        self.scale_y = 1.0
        
        # --- 暫停、按鍵與免責期邏輯 ---
        self.last_waiting_ms = 0.0
        self.last_time_ms = 0.0
        self.last_full_gray_np = None
        
        self.key_states = {'up': False, 'down': False, 'left': False, 'right': False}
        self.frame_counter = 0
        self.sync_fail_count = 0
        self.is_paused_by_sync = False

        # 佇列、最後一個影片游標座標
        self.queue = []                  # [(abs_x, abs_y), …]
        self.last_video_rel = None       # (0‑1, 0‑1) or None

    # ==========================================
    # 區塊 A：初始化與窗口對齊 ( Ready 階段 )
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
            print(f"✅ 實機精確鎖定: ({self.chiaki_x}, {self.chiaki_y}), 尺寸 {self.chiaki_w}x{self.chiaki_h}")
            return True
        print("❌ 找不到 Chiaki 視窗！")
        return False

    def chaiki_ready(self, full_grab):
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

    # ==========================================
    # 區塊 B：指針追蹤與移動邏輯 (Movement)
    # ==========================================

    def _get_lparam(self, vk, down=True):
        """
        根據 Chiaki 映射表構造精確的 lParam。
        方向鍵 (VK_UP/DOWN/LEFT/RIGHT) 和 Return 鍵在 Win32 中均屬於擴展鍵。
        """
        scan_code = win32api.MapVirtualKey(vk, 0)
        # 🌟 核心修正：根據截圖，這些按鍵在實體鍵盤上均帶有擴展位元 (Extended bit)
        extended = 1 if vk in [win32con.VK_UP, win32con.VK_DOWN, win32con.VK_LEFT, win32con.VK_RIGHT, win32con.VK_RETURN] else 0
        
        # bit 0-15: Repeat count (1)
        # bit 16-23: Scan code
        # bit 24: Extended key flag
        # bit 29: Context code (0 for WM_KEYDOWN)
        # bit 30: Previous key state
        # bit 31: Transition state
        lparam = 1 | (scan_code << 16) | (extended << 24)
        if not down:
            lparam |= (1 << 30) | (1 << 31)
        return lparam

    def update_key_bg(self, key_str, press):
        """底層按鍵發送，增加終端打印日誌"""
        if not self.chiaki_hwnd: return
        vk_map = {'up': win32con.VK_UP, 'down': win32con.VK_DOWN, 'left': win32con.VK_LEFT, 'right': win32con.VK_RIGHT}
        vk = vk_map.get(key_str)
        
        if self.key_states.get(key_str) != press:
            msg_name = "KEYDOWN" if press else "KEYUP"
            # 打印按鍵動作
            print(f"⌨️  [Chiaki Control] {key_str.upper()}: {msg_name}")
            
            msg = win32con.WM_KEYDOWN if press else win32con.WM_KEYUP
            win32api.PostMessage(self.chiaki_hwnd, msg, vk, self._get_lparam(vk, press))
            self.key_states[key_str] = press

    def release_all_keys(self):
        for k in ['up', 'down', 'left', 'right']: self.update_key_bg(k, False)

    def Realtime_cursor_position(self, bgr_frame):
        """YOLO 即時推論，返回: 歸一化座標(0-1), 類別名稱, 絕對像素座標"""
        results = self.model.predict(bgr_frame, conf=0.6, verbose=False)
        if len(results[0].boxes) > 0:
            box = sorted(results[0].boxes, key=lambda x: x.conf, reverse=True)[0]
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            h, w = bgr_frame.shape[:2]
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            
            rel_x = round(float(cx / w), 4)
            rel_y = round(float(cy / h), 4)
            cls_name = self.model.names[int(box.cls)]
            return (rel_x, rel_y), cls_name, (int(cx), int(cy))
        return None, None, None

    def move_toward_target(self, current_pos, target_pos):
        """歸一化跟隨：呼叫合併後的移動邏輯處理 X/Y 軸"""
        dist = math.hypot(target_pos[0] - current_pos[0], target_pos[1] - current_pos[1])

        if dist < 0.01: # 🎯 誤差小於 1% 視為對齊
            self.release_all_keys()
            return True

        # 執行合併後的移動決策
        self.move_action(current_pos, target_pos)
        return False

    def move_action(self, current_pos, target_pos):
        """
        合併後的移動決策：
        - 打印當前與目標座標
        - 處理長按與 100ms 微調
        """
        dx = target_pos[0] - current_pos[0]
        dy = target_pos[1] - current_pos[1]
        
        # 終端打印當前座標狀態
        print(f"📍 Pos: Curr({current_pos[0]:.3f}, {current_pos[1]:.3f}) -> Target({target_pos[0]:.3f}, {target_pos[1]:.3f}) | Δ:({dx:.3f}, {dy:.3f})")

        tap_keys = []

        # --- X 軸處理 ---
        if abs(dx) > 0.05:
            self.update_key_bg('right', dx > 0)
            self.update_key_bg('left', dx < 0)
        else:
            self.update_key_bg('right', False)
            self.update_key_bg('left', False)
            if abs(dx) >= 0.01:
                tap_keys.append('right' if dx > 0 else 'left')

        # --- Y 軸處理 ---
        if abs(dy) > 0.05:
            self.update_key_bg('down', dy > 0)
            self.update_key_bg('up', dy < 0)
        else:
            self.update_key_bg('down', False)
            self.update_key_bg('up', False)
            if abs(dy) >= 0.01:
                tap_keys.append('down' if dy > 0 else 'up')

        # --- 執行微調 ---
        if tap_keys:
            print(f"🤏 [Micro-Adjust] Tapping {tap_keys} for 100ms...")
            vk_map = {'up': win32con.VK_UP, 'down': win32con.VK_DOWN, 'left': win32con.VK_LEFT, 'right': win32con.VK_RIGHT}
            for key in tap_keys:
                vk = vk_map[key]
                win32api.PostMessage(self.chiaki_hwnd, win32con.WM_KEYDOWN, vk, self._get_lparam(vk, True))
            
            time.sleep(0.1) 
            
            for key in tap_keys:
                vk = vk_map[key]
                win32api.PostMessage(self.chiaki_hwnd, win32con.WM_KEYUP, vk, self._get_lparam(vk, False))

    def click_action(self):
        """執行 Cross (Return 鍵) 點擊"""
        if not self.chiaki_hwnd: return
        vk = win32con.VK_RETURN
        
        # 💡 模擬真實物理按壓時長，Chiaki 對於太快的訊號有時會丟失
        win32api.PostMessage(self.chiaki_hwnd, win32con.WM_KEYDOWN, vk, self._get_lparam(vk, True))
        time.sleep(0.08) # 稍微拉長至 80ms 確保模擬器捕捉到
        win32api.PostMessage(self.chiaki_hwnd, win32con.WM_KEYUP, vk, self._get_lparam(vk, False))
        print("✅ [實機] 執行 Cross (Return) 點擊！")

    # ==========================================
    # 區塊 C：點擊決策與同步暫停 (Logic & Sync)
    # ==========================================
    
    def is_scene_change(self, current_gray):
        if self.last_full_gray_np is None: return False
        curr_hash = self.get_sparse_hash(current_gray)
        prev_hash = self.get_sparse_hash(self.last_full_gray_np)
        return np.mean(cv2.absdiff(curr_hash, prev_hash)) > 35

    def is_roi_change(self, current_gray, cx, cy):
        if self.last_full_gray_np is None: return False
        h, w = current_gray.shape
        x1, y1 = max(0, cx-40), max(0, cy-40)
        x2, y2 = min(w, cx+40), min(h, cy+40)
        curr_roi = current_gray[y1:y2, x1:x2].copy()
        prev_roi = self.last_full_gray_np[y1:y2, x1:x2].copy()
        rh, rw = curr_roi.shape
        if rh > 20 and rw > 20:
            curr_roi[rh//2-10:rh//2+10, rw//2-10:rw//2+10] = 0
            prev_roi[rh//2-10:rh//2+10, rw//2-10:rw//2+10] = 0
        return np.mean(cv2.absdiff(curr_roi, prev_roi)) > 25  # 🌟 提高 ROI 門檻防干擾

    def sync_check(self, yt_gray, chiaki_gray):
        """模糊匹配遊戲邊沿 15%，判斷是否同步"""
        yt_hash = self.get_sparse_hash(yt_gray)
        ck_hash = self.get_sparse_hash(cv2.resize(chiaki_gray, (yt_gray.shape[1], yt_gray.shape[0])))
        return np.mean(cv2.absdiff(yt_hash, ck_hash)) < 25  # 🌟 縮緊同步閾值，不同步立刻抓
    
    def enqueue_video_pos(self, pos):
        """將目標座標加入隊列，避免連續幀塞入完全相同的座標"""
        if not self.queue or self.queue[-1] != pos:
            self.queue.append(pos)

    def process_queue(self):
        """處理佇列，並打印隊列狀態"""
        if self.queue:
            print(f"📜 [Queue] Pending Tasks: {len(self.queue)} | Current Target: {self.queue[0]}")
            target_pos = self.queue[0]
            if self.chaiki_cursor_pos:
                # 呼叫合併後的 move_action
                is_arrived = (math.hypot(target_pos[0]-self.chaiki_cursor_pos[0], 
                                         target_pos[1]-self.chaiki_cursor_pos[1]) < 0.01)
                
                if is_arrived:
                    print("🎯 [Queue] Arrived! Executing Click...")
                    self.click_action()
                    self.queue.pop(0)
                else:
                    self.move_action(self.chaiki_cursor_pos, target_pos)
            else:
                print("⚠️ [Queue] Waiting for Chiaki cursor detection...")
                self.release_all_keys()
        else:
            # 如果隊列空了，但偵測到影片有新座標，則實時跟隨
            if self.chaiki_cursor_pos and self.last_known_cursor_pos:
                self.move_action(self.chaiki_cursor_pos, self.last_known_cursor_pos)
            else:
                self.release_all_keys()

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
            
            # 🎯 啟動背景線程模式
            self.camera.start(target_fps=30, video_mode=True)

            # 🌟 直接進入全自動模式
            recording = True
            
            page.evaluate("document.querySelector('video').play();")
            print("▶️ 程式啟動，影片自動播放，進入全自動追蹤與同步模式！")

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

                    # 1. YOLO 推論
                    yt_rel_pos, yt_cls, yt_abs_pos = self.Realtime_cursor_position(yt_frame)
                    ck_rel_pos, ck_cls, ck_abs_pos = self.Realtime_cursor_position(chiaki_frame)

                    # **注意：使用 is not None 判斷，避免 (0.0,0.0) 被丟掉**
                    if yt_rel_pos is not None:
                        # 偵測到影片游標
                        # 如果停在同一個位置就把座標送到佇列
                        if self.last_video_rel is not None and yt_rel_pos == self.last_video_rel:
                            self.enqueue_video_pos(yt_rel_pos)
                        self.last_video_rel = yt_rel_pos
                        self.last_known_cursor_pos = yt_rel_pos

                    if ck_rel_pos is not None:
                        self.chaiki_cursor_pos = ck_rel_pos

                    if recording:
                        # 決定是否發出 pending click
                        if yt_rel_pos is not None:
                            roi_changed = self.is_roi_change(yt_gray,
                                                             yt_abs_pos[0] if yt_abs_pos else 0,
                                                             yt_abs_pos[1] if yt_abs_pos else 0)
                            cursor_changed = (yt_cls != self.last_yt_cls
                                          and yt_cls in ('Hold', 'Keep'))
                            self.last_yt_cls = yt_cls
                            if (roi_changed or cursor_changed):
                                self.pending_click = True
                                print('📥 point click pending')

                        # 如果待辦點擊燈號亮，先把當前游標位置入佇列
                        if self.pending_click and self.last_known_cursor_pos is not None:
                            self.enqueue_video_pos(self.last_known_cursor_pos)
                            self.pending_click = False

                        # 佇列處理（移動 + 點擊）
                        self.process_queue()

                    # 4. 自動同步檢查：每 30 幀檢查一次
                    self.frame_counter += 1
                    if self.frame_counter >= 30:
                        self.frame_counter = 0
                        is_synced = self.sync_check(yt_gray, chiaki_gray)
                        
                        if not is_synced:
                            self.sync_fail_count += 1
                            if self.sync_fail_count >= 3 and not self.is_paused_by_sync:
                                page.evaluate("document.querySelector('video').pause();")
                                self.is_paused_by_sync = True
                                self.release_all_keys()
                                print("⏸️ 畫面不同步持續約 3 秒，自動暫停影片等待實機...")
                        else:
                            self.sync_fail_count = 0
                            if self.is_paused_by_sync:
                                page.evaluate("document.querySelector('video').play();")
                                self.is_paused_by_sync = False
                                print("▶️ 畫面已重新對齊，自動恢復播放！")

                    self.last_full_gray_np = yt_gray.copy()
                    
                    # 全局熱鍵退出
                    if win32api.GetAsyncKeyState(ord('Q')) & 0x8000:
                        print("🛑 收到 Q 鍵，系統安全退出...")
                        break

            self.camera.stop()
            browser.close()

if __name__ == "__main__":
    agent = AutoPlatinumHand("https://www.youtube.com/watch?v=7K_NimshHUI")
    agent.run_live_sync()