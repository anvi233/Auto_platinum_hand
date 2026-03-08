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
    def __init__(self, youtube_url, model_path=r'F:\PySpace\Auto_platinum_hand\runs\detect\train2\weights\best.pt'):
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
        
        # 💡 增加 'cross' 鍵位狀態初始化
        self.key_states = {'up': False, 'down': False, 'left': False, 'right': False, 'cross': False}
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

    def _get_lparam(self, vk, down=True):
        """構造 lParam，移除 VK_RETURN 的擴展位元防止 Phantom Click"""
        scan_code = win32api.MapVirtualKey(vk, 0)
        # 💡 合理解釋：移除了 VK_RETURN。一般鍵盤的 Enter 屬於標準鍵，不該帶有 Extended Bit，否則模擬器不認。
        extended = 1 if vk in [win32con.VK_UP, win32con.VK_DOWN, win32con.VK_LEFT, win32con.VK_RIGHT] else 0
        
        lparam = 1 | (scan_code << 16) | (extended << 24)
        if not down:
            lparam |= (1 << 30) | (1 << 31)
        return lparam

    def update_key_bg(self, key_str, press):
        """底層按鍵發送，增加終端打印日誌與跨鍵支持"""
        if not self.chiaki_hwnd: return
        vk_map = {
            'up': win32con.VK_UP, 'down': win32con.VK_DOWN, 
            'left': win32con.VK_LEFT, 'right': win32con.VK_RIGHT,
            'cross': win32con.VK_RETURN
        }
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
        results = self.model.predict(bgr_frame, conf=0.2, verbose=False)
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
        引入實機測速常數 V=0.85 與起步補償，解決 20ms 只走 1px 的問題。
        """
        dx = target_pos[0] - current_pos[0]
        dy = target_pos[1] - current_pos[1]
        
        print(f"📍 Pos: Curr({current_pos[0]:.3f}, {current_pos[1]:.3f}) -> Target({target_pos[0]:.3f}, {target_pos[1]:.3f}) | Δ:({dx:.3f}, {dy:.3f})")

        def control_axis(delta, pos_key, neg_key):
            abs_d = abs(delta)
            if abs_d > 0.25: 
                # 大於 25% 距離，長按全速巡航
                self.update_key_bg(pos_key, delta > 0)
                self.update_key_bg(neg_key, delta < 0)
                return 0
            else:
                self.update_key_bg(pos_key, False)
                self.update_key_bg(neg_key, False)
                # 微調區間：下探至 0.015 (1.5%)
                if abs_d >= 0.015:
                    # 💡 核心：使用測速常數 V=0.85 計算理論時間
                    raw_time = abs_d / 0.85
                    
                    # 💡 核心：加上 0.035 秒的起步補償，抵消輪詢延遲與遊戲死區
                    # 並且將下限設為 0.04 秒 (40ms)，確保 Chiaki 能確實讀取為連續輸入
                    pulse_time = max(0.04, min(0.18, raw_time + 0.035))
                    
                    key = pos_key if delta > 0 else neg_key
                    return (key, pulse_time)
                return 0

        tap_x = control_axis(dx, 'right', 'left')
        tap_y = control_axis(dy, 'down', 'up')
        taps = [t for t in (tap_x, tap_y) if t != 0]
        
        if taps:
            max_pulse = max([t[1] for t in taps])
            tap_keys = [t[0] for t in taps]
            
            print(f"🤏 [Math Adjust] Tapping {tap_keys} for {max_pulse*1000:.0f}ms (Compensated)...")
            vk_map = {'up': win32con.VK_UP, 'down': win32con.VK_DOWN, 'left': win32con.VK_LEFT, 'right': win32con.VK_RIGHT}
            
            for key in tap_keys:
                vk = vk_map[key]
                win32api.PostMessage(self.chiaki_hwnd, win32con.WM_KEYDOWN, vk, self._get_lparam(vk, True))
            
            time.sleep(max_pulse)
            
            for key in tap_keys:
                vk = vk_map[key]
                win32api.PostMessage(self.chiaki_hwnd, win32con.WM_KEYUP, vk, self._get_lparam(vk, False))
                
            # 冷卻期：等待畫面回傳，避免殘影導致重複計算
            time.sleep(0.1)

    def click_action(self):
        """執行 Cross (Return 鍵) 點擊"""
        if not self.chiaki_hwnd: return
        vk = win32con.VK_RETURN
        
        # 💡 模擬真實物理按壓，將 80ms 延長到 150ms 確保模擬器捕捉到
        win32api.PostMessage(self.chiaki_hwnd, win32con.WM_KEYDOWN, vk, self._get_lparam(vk, True))
        time.sleep(0.15) 
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

    def is_scene_changed(self, current_gray, last_gray, cursor_x, cursor_y, mask_size=100):
        """
        💡 採用全局背景對比：將游標周圍 mask_size 大小的區域塗黑後，對比兩幀差異。
        徹底防範游標閃爍造成的誤判。
        """
        if current_gray is None or last_gray is None:
            return False
            
        curr_masked = current_gray.copy()
        last_masked = last_gray.copy()
        h, w = curr_masked.shape
        
        # 計算遮罩的邊界，防止超出畫面
        x1 = max(0, int(cursor_x - mask_size/2))
        x2 = min(w, int(cursor_x + mask_size/2))
        y1 = max(0, int(cursor_y - mask_size/2))
        y2 = min(h, int(cursor_y + mask_size/2))
        
        # 將游標區域塗黑
        curr_masked[y1:y2, x1:x2] = 0
        last_masked[y1:y2, x1:x2] = 0
        
        # 計算差異
        diff = cv2.absdiff(curr_masked, last_masked)
        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        changed_pixels = cv2.countNonZero(thresh)
        total_pixels = w * h - ((x2 - x1) * (y2 - y1))
        
        if total_pixels <= 0: return False
        change_ratio = changed_pixels / total_pixels
        
        return change_ratio > 0.05

    def sync_check(self, yt_gray, chiaki_gray):
        """模糊匹配遊戲邊沿 15%，判斷是否同步"""
        yt_hash = self.get_sparse_hash(yt_gray)
        ck_hash = self.get_sparse_hash(cv2.resize(chiaki_gray, (yt_gray.shape[1], yt_gray.shape[0])))
        return np.mean(cv2.absdiff(yt_hash, ck_hash)) < 25  # 🌟 縮緊同步閾值，不同步立刻抓
    
    def enqueue_video_pos(self, pos):
        """將目標座標加入隊列，避免連續幀塞入完全相同的座標"""
        if not self.queue or self.queue[-1] != pos:
            self.queue.append(pos)

    def process_queue(self, chiaki_gray=None):
        """處理佇列：使用實機自身畫面比對來驗證點擊是否有效"""
        if self.queue:
            target_pos = self.queue[0]
            current_t = time.time()
            
            if not hasattr(self, 'click_validating'):
                self.click_validating = False
                self.val_start_time = 0.0
                self.click_retry_count = 0

            if self.click_validating:
                if current_t - self.val_start_time > 1.5:
                    # 💡 拋棄 sync_fail_count，改用 Chiaki 自己點擊前後的畫面做比對
                    if chiaki_gray is not None and hasattr(self, 'chiaki_gray_before_click'):
                        ck_abs_x = int(self.chaiki_cursor_pos[0] * self.chiaki_w) if self.chaiki_cursor_pos else 0
                        ck_abs_y = int(self.chaiki_cursor_pos[1] * self.chiaki_h) if self.chaiki_cursor_pos else 0
                        
                        chiaki_reacted = self.is_scene_changed(chiaki_gray, self.chiaki_gray_before_click, ck_abs_x, ck_abs_y)
                        
                        if not chiaki_reacted:
                            self.click_retry_count += 1
                            if self.click_retry_count <= 3:
                                import random
                                offset_x = random.uniform(-0.03, 0.03)
                                offset_y = random.uniform(-0.03, 0.03)
                                self.queue[0] = (target_pos[0] + offset_x, target_pos[1] + offset_y)
                                print(f"🔄 [Validation Failed] 實機無反應！微調目標 -> 新座標: {self.queue[0]}")
                            else:
                                print("⚠️ [Validation Failed] 點擊 3 次實機均無反應，強制放棄任務。")
                                self.queue.pop(0)
                                self.click_retry_count = 0
                            
                            self.click_validating = False
                        else:
                            print("✨ [Validation Success] 實機畫面成功切換！任務完成。")
                            self.queue.pop(0)
                            self.click_validating = False
                            self.click_retry_count = 0
                            
                            # ✨ 防抖冷卻：成功切換後，強制忽略接下來 1.5 秒內的影片變動
                            self.ignore_tasks_until = current_t + 1.5
                    return
                else:
                    self.release_all_keys()
                return

            # --- 正常的移動與點擊邏輯 ---
            if self.chaiki_cursor_pos:
                dist = math.hypot(target_pos[0]-self.chaiki_cursor_pos[0], 
                                  target_pos[1]-self.chaiki_cursor_pos[1])
                
                is_arrived = dist < 0.015
                
                if is_arrived:
                    print(f"🎯 [Queue] 到達目標！(誤差: {dist:.3f}) 執行點擊...")
                    self.click_action()
                    self.click_validating = True
                    self.val_start_time = current_t
                    # 記錄點擊瞬間的實機畫面
                    if chiaki_gray is not None:
                        self.chiaki_gray_before_click = chiaki_gray.copy()
                else:
                    self.move_action(self.chaiki_cursor_pos, target_pos)
            else:
                self.release_all_keys()
        else:
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

    def run_live_sync(self, start_time_sec=34):
        with sync_playwright() as p:
            # --- Playwright 與影片同步啟動 ---
            browser = p.chromium.connect_over_cdp("http://localhost:9222")
            page = next((pg for pg in browser.contexts[0].pages if "youtube" in pg.url), browser.contexts[0].pages[0])
            video = page.wait_for_selector("video")
            
            # 確保影片暫停在起始時間點
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

            # ==========================================
            # 🌟 新增：手動對齊準備階段
            # ==========================================
            print("\n" + "="*50)
            print("⏳ 進入手動對齊模式...")
            print("請手動點擊 Chiaki 視窗確保其獲得焦點，並將實機指針與影片大致對齊。")
            print("👉 對齊完成後，請按下 [空白鍵 (SPACE)] 正式啟動自動控制！")
            print("👉 隨時可按下 [Q] 鍵退出程式。")
            print("="*50 + "\n")

            aligned = False
            align_frame_count = 0
            
            with torch.no_grad():
                while not aligned:
                    full_frame = self.camera.get_latest_frame()
                    if full_frame is None: continue

                    # 抓取畫面
                    yt_frame = full_frame[self.yt_y:self.yt_y+self.yt_h, self.yt_x:self.yt_x+self.yt_w]
                    chiaki_frame = full_frame[self.chiaki_y:self.chiaki_y+self.chiaki_h, self.chiaki_x:self.chiaki_x+self.chiaki_w]
                    
                    # 每 15 幀 (約 0.5 秒) 打印一次當前檢測狀態，輔助人工對齊
                    align_frame_count += 1
                    if align_frame_count >= 15:
                        yt_rel_pos, yt_cls, _ = self.Realtime_cursor_position(yt_frame)
                        ck_rel_pos, ck_cls, _ = self.Realtime_cursor_position(chiaki_frame)
                        
                        yt_str = f"{yt_rel_pos[0]:.3f}, {yt_rel_pos[1]:.3f}" if yt_rel_pos else "未檢測到"
                        ck_str = f"{ck_rel_pos[0]:.3f}, {ck_rel_pos[1]:.3f}" if ck_rel_pos else "未檢測到"
                        print(f"🔧 [對齊中] 影片指針: ({yt_str}) | 實機指針: ({ck_str})")
                        align_frame_count = 0

                    # 檢測空白鍵 (SPACE) 開始
                    if win32api.GetAsyncKeyState(win32con.VK_SPACE) & 0x8000:
                        print("\n🚀 收到 [空白鍵]！結束對齊模式，正式啟動自動控制系統！\n")
                        aligned = True
                        time.sleep(0.5) # 防止按鍵連擊干擾後續邏輯

                    # 檢測 Q 鍵安全退出
                    if win32api.GetAsyncKeyState(ord('Q')) & 0x8000:
                        print("🛑 收到 Q 鍵，系統安全退出...")
                        self.camera.stop()
                        browser.close()
                        return
                
                # ==========================================
                # 🌟 正式進入全自動模式
                # ==========================================
                self.frame_counter = 0 
                self.queue.clear() 
                self.last_yt_cls = None        
                self.last_full_gray_np = None  
                recording = True
                
                # 💡 注入 CSS 徹底隱藏 YouTube 播放列，防止淡出動畫觸發誤判
                js_code = """
                document.querySelector('video').play();
                var style = document.createElement('style');
                style.innerHTML = '.ytp-chrome-bottom, .ytp-chrome-top, .ytp-gradient-bottom, .ytp-gradient-top, .ytp-watermark { display: none !important; }';
                document.head.appendChild(style);
                """
                page.evaluate(js_code)
                print("▶️ 影片自動播放，並已屏蔽 YouTube UI 干擾，進入全自動模式！")

                while True:
                    # 🎯 阻塞等待新幀
                    full_frame = self.camera.get_latest_frame()
                    if full_frame is None: continue
                    
                    yt_frame = full_frame[self.yt_y:self.yt_y+self.yt_h, self.yt_x:self.yt_x+self.yt_w]
                    chiaki_frame = full_frame[self.chiaki_y:self.chiaki_y+self.chiaki_h, self.chiaki_x:self.chiaki_x+self.chiaki_w]
                    
                    yt_gray = cv2.cvtColor(yt_frame, cv2.COLOR_BGR2GRAY)
                    chiaki_gray = cv2.cvtColor(chiaki_frame, cv2.COLOR_BGR2GRAY)
                    
                    # 1. YOLO 推論
                    yt_rel_pos, yt_cls, yt_abs_pos = self.Realtime_cursor_position(yt_frame)
                    ck_rel_pos, ck_cls, ck_abs_pos = self.Realtime_cursor_position(chiaki_frame)

                    # **注意：使用 is not None 判斷，避免 (0.0,0.0) 被丟掉**
                    if yt_rel_pos is not None:
                        self.last_video_rel = yt_rel_pos
                        self.last_known_cursor_pos = yt_rel_pos

                        # 💡 捕捉最後 Hover 位置
                        if not hasattr(self, 'hover_frame_count'):
                            self.hover_frame_count = 0
                            self.last_hover_pos = yt_rel_pos
                            self.last_yt_rel_pos_for_hover = yt_rel_pos

                        if self.last_yt_rel_pos_for_hover is not None:
                            dist = math.hypot(yt_rel_pos[0] - self.last_yt_rel_pos_for_hover[0],
                                              yt_rel_pos[1] - self.last_yt_rel_pos_for_hover[1])
                            if dist < 0.005:
                                self.hover_frame_count += 1
                                if self.hover_frame_count >= 5:
                                    self.last_hover_pos = yt_rel_pos
                            else:
                                self.hover_frame_count = 0
                        self.last_yt_rel_pos_for_hover = yt_rel_pos

                    if ck_rel_pos is not None:
                        self.chaiki_cursor_pos = ck_rel_pos

                    if recording:
                        current_t = time.time()
                        # 💡 隊列鎖：驗證期間或冷卻期內，徹底鎖死，無視影片任何變動
                        is_validating = getattr(self, 'click_validating', False)
                        is_cooling_down = current_t < getattr(self, 'ignore_tasks_until', 0)
                        is_locked = is_validating or is_cooling_down

                        if yt_rel_pos is not None and not is_locked:
                            # 呼叫全局遮罩對比
                            scene_changed = self.is_scene_changed(yt_gray, self.last_full_gray_np, 
                                                                  yt_abs_pos[0] if yt_abs_pos else 0,
                                                                  yt_abs_pos[1] if yt_abs_pos else 0)
                            
                            if self.last_yt_cls is None:
                                cursor_changed = False
                                scene_changed = False 
                            else:
                                cursor_changed = (yt_cls != self.last_yt_cls and yt_cls in ('Hold', 'Keep'))
                                
                            self.last_yt_cls = yt_cls

                            # ==========================================
                            # 💡 連續幀場景變化狀態機 & 2 秒超時防護
                            # ==========================================
                            if not hasattr(self, 'video_scene_unstable'):
                                self.video_scene_unstable = False
                                self.unstable_start_time = 0.0

                            trigger_click = False

                            if scene_changed:
                                if not self.video_scene_unstable:
                                    self.video_scene_unstable = True
                                    self.unstable_start_time = current_t
                                    trigger_click = True
                                else:
                                    if current_t - self.unstable_start_time > 2.0:
                                        self.video_scene_unstable = False
                            else:
                                if self.video_scene_unstable:
                                    self.video_scene_unstable = False

                            if cursor_changed:
                                trigger_click = True
                            # ==========================================

                            if trigger_click:
                                self.pending_click = True
                                print(f'📥 point click pending (Scene_transition_start: {scene_changed}, Cursor_changed: {cursor_changed})')

                        if self.pending_click and not is_locked:
                            # 使用 Hover 位置作為目標，免疫過場動畫期間的手把滑動
                            click_target = getattr(self, 'last_hover_pos', self.last_known_cursor_pos)
                            print(f"🎯 [Hover Target] 鎖定最後停留位置: {click_target} (排除過場偏移)")
                            self.enqueue_video_pos(click_target)
                            
                            self.pending_click = False

                        # 💡 傳入 chiaki_gray 進行實機驗證
                        self.process_queue(chiaki_gray)

                    # 4. 自動同步檢查
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
                    
                    if win32api.GetAsyncKeyState(ord('Q')) & 0x8000:
                        print("🛑 收到 Q 鍵，系統安全退出...")
                        break

            self.camera.stop()
            browser.close()



if __name__ == "__main__":
    agent = AutoPlatinumHand("https://www.youtube.com/watch?v=7K_NimshHUI")
    agent.run_live_sync()