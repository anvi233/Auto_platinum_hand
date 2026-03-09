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
                # 微調區間：下探至 0.005 (0.5%)
                if abs_d >= 0.005:
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

    def is_scene_changed(self, current_gray, last_gray, cursor_x, cursor_y, is_sync_mode=False):
        if current_gray is None or last_gray is None: return False
        
        # 💡 模擬人眼：將畫面極度簡化為 32x32 網格，只看大體色塊跳變
        curr_s = cv2.resize(current_gray, (32, 32))
        last_s = cv2.resize(last_gray, (32, 32))
        
        # 計算平均絕對誤差 (MAE)
        diff_score = np.mean(cv2.absdiff(curr_s, last_s)) / 255.0
        
        if is_sync_mode:
            # 同步模式：門檻調高，只有大範圍不一致才判定
            return diff_score > 0.08
        else:
            # 觸發模式：3% 的變化足以捕捉 PS1 點擊閃爍
            return diff_score > 0.03

    def sync_check(self, yt_gray, chiaki_gray):
        """💡 職能解耦：僅用於判斷是否暫停影片。"""
        if yt_gray is None or chiaki_gray is None: return True
        ck_res = cv2.resize(chiaki_gray, (self.yt_w, self.yt_h))
        # 僅使用全局判定
        is_different = self.is_scene_changed(yt_gray, ck_res, 0, 0, is_sync_mode=True)
        return not is_different
    
    def enqueue_video_pos(self, pos):
        """將目標座標加入隊列，避免連續幀塞入完全相同的座標"""
        if not self.queue or self.queue[-1] != pos:
            self.queue.append(pos)

    def process_queue(self, chiaki_gray=None, yt_gray=None):
        """
        💡 職能解耦診斷版：
        1. 沒任務時執行 Roaming 跟隨。
        2. 有任務時，物理距離 (dist) 是執行點擊的唯一門檻。
        3. 任務彈出 (pop) 僅依賴點擊動作的完成，不再受 sync_check 干擾。
        """
        if not self.queue:
            if self.chaiki_cursor_pos and self.last_known_cursor_pos:
                # 只有在沒任務時才打印漫遊狀態，減少日誌噪音
                dist_roaming = math.hypot(self.chaiki_cursor_pos[0]-self.last_known_cursor_pos[0], 
                                          self.chaiki_cursor_pos[1]-self.last_known_cursor_pos[1])
                if dist_roaming > 0.05:
                    print(f"👣 [Roaming] 跟隨影片指針中... 距離: {dist_roaming:.4f}")
                self.move_action(self.chaiki_cursor_pos, self.last_known_cursor_pos)
            else:
                self.release_all_keys()
            return

        # 鎖定隊列首位任務
        target_pos = self.queue[0]
        current_t = time.time()
        
        # 獲取實機與當前目標的物理距離
        dist = math.hypot(target_pos[0]-self.chaiki_cursor_pos[0], target_pos[1]-self.chaiki_cursor_pos[1])

        # --- 狀態 A：執行點擊後的固定冷卻與銷毀 ---
        if getattr(self, 'click_validating', False):
            wait_time = current_t - self.val_start_time
            # 💡 模擬人眼邏輯：點擊完畢給予 0.6s 緩衝讓畫面反應，隨即彈出
            if wait_time > 0.6:
                print(f"✅ [Task Finished] 物理到位並執行點擊。彈出任務: {target_pos} | 剩餘: {len(self.queue)-1}")
                self.queue.pop(0) 
                self.click_validating = False
                self.ignore_tasks_until = current_t + 0.1 # 防止連續誤點擊的微小保護
            return

        # --- 狀態 B：執行移動與點擊決策 ---
        # 1. 物理到位判定 (誤差小於 1.8%)
        if dist < 0.018:
            # 2. 影片穩定判定 (模擬人眼：畫面不閃爍時才點)
            if getattr(self, 'video_is_stable', True):
                print(f"🔥 [EXECUTE CLICK] 🔥")
                print(f"   - 隊列首位 (Target): {target_pos}")
                print(f"   - 實機位置 (Actual): {self.chaiki_cursor_pos}")
                print(f"   - 物理誤差 (Dist): {dist:.4f}")
                print(f"   - 隊列總數 (Len): {len(self.queue)}")
                
                self.click_action() # 執行物理點擊
                self.click_validating = True
                self.val_start_time = current_t
            else:
                # 影片還在劇烈變化中，為了精準度選擇原地等待一幀
                self.release_all_keys()
        else:
            # 距離還遠，僅執行移動，排除一切干擾專心走位
            if not hasattr(self, '_last_move_log') or current_t - self._last_move_log > 1.0:
                print(f"🚚 [Moving] 鎖定任務點: {target_pos} | 當前距離: {dist:.4f}")
                self._last_move_log = current_t
            self.move_action(self.chaiki_cursor_pos, target_pos)

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
                
                # 💡 利用 Playwright 原生 API 安全注入 CSS 隱藏 YouTube UI，並啟動播放
                page.evaluate("document.querySelector('video').play();")
                page.add_style_tag(content='.ytp-chrome-bottom, .ytp-chrome-top, .ytp-gradient-bottom, .ytp-gradient-top, .ytp-watermark { display: none !important; }')
                print("▶️ 影片自動播放，並已屏蔽 YouTube UI 干擾，進入全自動模式！")

                # 💡 在啟動播放後加入開場保護計數
                self.observation_frames = 0
                self.prev_intent_pos = None  # 🌟 新增：用於記錄點擊發生前的意圖座標

                while True:
                    full_frame = self.camera.get_latest_frame()
                    if full_frame is None: continue
                    
                    yt_frame = full_frame[self.yt_y:self.yt_y+self.yt_h, self.yt_x:self.yt_x+self.yt_w]
                    chiaki_frame = full_frame[self.chiaki_y:self.chiaki_y+self.chiaki_h, self.chiaki_x:self.chiaki_x+self.chiaki_w]
                    
                    yt_gray = cv2.cvtColor(yt_frame, cv2.COLOR_BGR2GRAY)
                    chiaki_gray = cv2.cvtColor(chiaki_frame, cv2.COLOR_BGR2GRAY)

                    if not hasattr(self, 'reference_gray') or self.reference_gray is None:
                        self.reference_gray = yt_gray.copy()
                        self.ref_frame_timer = 0
                    
                    # 🌟 核心修正：在更新座標前，備份上一幀正在追趕的目標
                    # 這是為了落實「點擊回饋」約定：場景變了，兇手一定是「剛才追的地方」
                    if self.last_known_cursor_pos:
                        self.prev_intent_pos = (float(self.last_known_cursor_pos[0]), float(self.last_known_cursor_pos[1]))

                    # 執行視覺偵測
                    yt_rel_pos, yt_cls, yt_abs_pos = self.Realtime_cursor_position(yt_frame)
                    ck_rel_pos, ck_cls, ck_abs_pos = self.Realtime_cursor_position(chiaki_frame)

                    # 開場跳幀保護
                    if self.observation_frames < 10:
                        self.observation_frames += 1
                        self.release_all_keys()
                        continue

                    # --- 決策邏輯 ---
                    current_t = time.time()
                    is_locked = getattr(self, 'click_validating', False) or (current_t < getattr(self, 'ignore_tasks_until', 0))

                    scene_changed = False
                    if not is_locked:
                        # 使用保底座標防止 YOLO 丟失時報錯
                        sx = yt_abs_pos[0] if yt_abs_pos else int((self.last_known_cursor_pos[0] if self.last_known_cursor_pos else 0.5) * self.yt_w)
                        sy = yt_abs_pos[1] if yt_abs_pos else int((self.last_known_cursor_pos[1] if self.last_known_cursor_pos else 0.5) * self.yt_h)
                        
                        try:
                            scene_changed = self.is_scene_changed(yt_gray, self.reference_gray, sx, sy)
                        except:
                            scene_changed = False
                        
                        # 影片穩定性監控 (3 幀)
                        if not hasattr(self, 'video_stable_frames'): self.video_stable_frames = 0
                        if scene_changed:
                            self.video_stable_frames = 0
                            self.video_is_stable = False
                            self.reference_gray = yt_gray.copy()
                            self.ref_frame_timer = 0
                        else:
                            self.video_stable_frames += 1
                            if self.video_stable_frames >= 3: self.video_is_stable = True
                            self.ref_frame_timer += 1
                            if self.ref_frame_timer >= 30:
                                self.reference_gray = yt_gray.copy()
                                self.ref_frame_timer = 0

                    # 更新影片與實機位置快照
                    if yt_rel_pos is not None:
                        self.last_video_rel, self.last_known_cursor_pos = yt_rel_pos, yt_rel_pos
                    else:
                        if not self.queue: self.release_all_keys()

                    if ck_rel_pos is not None: 
                        self.chaiki_cursor_pos = ck_rel_pos

                    # --- 任務排隊邏輯 (關鍵修正) ---
                    # --- 任務入隊邏輯 ---
                    if recording and not is_locked:
                        trigger = scene_changed or (yt_rel_pos and self.last_yt_cls and yt_cls != self.last_yt_cls and yt_cls in ('Hold', 'Keep'))
                        
                        if trigger:
                            # 🌟 落實回饋約定：使用 prev_intent_pos
                            priority_target = self.prev_intent_pos if self.prev_intent_pos else self.last_known_cursor_pos
                            
                            if priority_target:
                                new_task = (float(priority_target[0]), float(priority_target[1]))
                                # 嚴格排隊，不准覆蓋。同一個點 4% 距離內不重複入隊
                                if not self.queue or math.hypot(new_task[0]-self.queue[-1][0], new_task[1]-self.queue[-1][1]) > 0.04:
                                    self.queue.append(new_task)
                                    print(f"📥 [New Task] {new_task} 加入隊列 | 總數: {len(self.queue)}")
                        
                        self.last_yt_cls = yt_cls# --- 任務入隊邏輯 ---
                    if recording and not is_locked:
                        # 💡 只有在非移動（Delta小）時才判斷場景變化，防止移動模糊誤觸
                        is_moving = self.chaiki_cursor_pos and self.last_known_cursor_pos and \
                                    math.hypot(self.chaiki_cursor_pos[0]-self.last_known_cursor_pos[0], 
                                               self.chaiki_cursor_pos[1]-self.last_known_cursor_pos[1]) > 0.02

                        if not is_moving and scene_changed:
                            # 捕捉發生變化前的座標快照 (點擊回饋)
                            priority_target = self.prev_intent_pos if self.prev_intent_pos else self.last_known_cursor_pos
                            if priority_target:
                                new_task = (float(priority_target[0]), float(priority_target[1]))
                                if not self.queue or math.hypot(new_task[0]-self.queue[-1][0], new_task[1]-self.queue[-1][1]) > 0.04:
                                    self.queue.append(new_task)
                                    print(f"📥 [Vision Captured] 全景跳變入隊: {new_task}")

                    # --- 5秒一次的心跳同步 (假設 30fps) ---
                    self.sync_timer = getattr(self, 'sync_timer', 0) + 1
                    if self.sync_timer >= 150:
                        self.sync_timer = 0
                        if not self.sync_check(yt_gray, chiaki_gray):
                            print("🛑 [Sync Guard] 兩邊差距過大，暫停 5s 等待實機...")
                            page.evaluate("document.querySelector('video').pause();")
                            time.sleep(5.0)
                            page.evaluate("document.querySelector('video').play();")

                    # 執行任務處理 (移動/點擊/驗證)
                    self.process_queue(chiaki_gray, yt_gray)

                    # --- 定期同步檢查 ---
                    self.frame_counter += 1
                    if self.frame_counter >= 45:
                        self.frame_counter = 0
                        if not is_locked and not self.sync_check(yt_gray, chiaki_gray):
                            self.sync_fail_count += 1
                            if self.sync_fail_count >= 3 and not self.is_paused_by_sync:
                                page.evaluate("document.querySelector('video').pause();")
                                self.is_paused_by_sync = True
                                print("⏸️ 畫面不同步，暫停影片...")
                        else:
                            self.sync_fail_count = 0
                            if self.is_paused_by_sync:
                                page.evaluate("document.querySelector('video').play();")
                                self.is_paused_by_sync = False
                                print("▶️ 重新對齊，恢復播放！")

                    if win32api.GetAsyncKeyState(ord('Q')) & 0x8000: 
                        break

            self.camera.stop()
            browser.close()



if __name__ == "__main__":
    agent = AutoPlatinumHand("https://www.youtube.com/watch?v=7K_NimshHUI")
    agent.run_live_sync()