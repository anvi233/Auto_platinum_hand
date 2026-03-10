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
        
        # --- 座標追蹤變量 (1:1 映射) ---
        self.last_known_cursor_pos = None  # 影片指針位置 (歸一化)
        self.chaiki_cursor_pos = None      # Chaiki 實機指針位置 (歸一化)
        self.last_yt_cls = None            # 追蹤影片指針型態變化
        self.prev_intent_pos = None        # 意圖快照：點擊發生前的穩定座標
        
        # --- 窗口同步與偏移 ---
        self.camera = None
        self.chiaki_hwnd = None
        
        self.yt_x, self.yt_y, self.yt_w, self.yt_h = 0, 0, 0, 0
        self.chiaki_x, self.chiaki_y, self.chiaki_w, self.chiaki_h = 0, 0, 0, 0
        self.scale_x = 1.0
        self.scale_y = 1.0
        
        # --- 暫停、按鍵與免責期邏輯 ---
        self.last_full_gray_np = None
        self.key_states = {'up': False, 'down': False, 'left': False, 'right': False, 'cross': False}
        self.frame_counter = 0
        self.sync_fail_count = 0
        self.is_paused_by_sync = False

        # --- 佇列與狀態鎖 (核心修改區) ---
        self.queue = []                  # 任務隊列：[(x, y), ...]
        
        # 🎬 動畫結界鎖 (影片端專用)
        self.is_scene_changing = False   # 標記是否正在播放過場動畫
        self.scene_stable_frames = 0     # 畫面穩定計數器
        
        # 🎮 實機確認鎖 (實機端專用)
        self.ck_stable_frames = 0        # 實機雙幀到位確認計數器
        self.click_validating = False    # 標記是否正在等待點擊動作完成
        self.val_start_time = 0.0        # 點擊動作開始的時間

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
        scan_code = win32api.MapVirtualKey(vk, 0)
        extended = 1 if vk in [win32con.VK_UP, win32con.VK_DOWN, win32con.VK_LEFT, win32con.VK_RIGHT] else 0
        lparam = 1 | (scan_code << 16) | (extended << 24)
        if not down:
            lparam |= (1 << 30) | (1 << 31)
        return lparam

    def update_key_bg(self, key_str, press):
        if not self.chiaki_hwnd: return
        vk_map = {
            'up': win32con.VK_UP, 'down': win32con.VK_DOWN, 
            'left': win32con.VK_LEFT, 'right': win32con.VK_RIGHT,
            'cross': win32con.VK_RETURN
        }
        vk = vk_map.get(key_str)
        
        if press:
            # 💡 核心修復：拔掉狀態鎖！只要需要按，每幀都狂發 KEYDOWN。
            # 這能完美模擬真實鍵盤的長按連發，徹底粉碎模擬器吞鍵或軸衝突的問題。
            win32api.PostMessage(self.chiaki_hwnd, win32con.WM_KEYDOWN, vk, self._get_lparam(vk, True))
            self.key_states[key_str] = True
        else:
            # 鬆開時才檢查狀態，避免發送多餘的 KEYUP
            if self.key_states.get(key_str) == True:
                win32api.PostMessage(self.chiaki_hwnd, win32con.WM_KEYUP, vk, self._get_lparam(vk, False))
                self.key_states[key_str] = False

    def release_all_keys(self):
        for k in ['up', 'down', 'left', 'right']: self.update_key_bg(k, False)

    def Realtime_cursor_position(self, bgr_frame):
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

    def move_action(self, current_pos, target_pos):
        dx = target_pos[0] - current_pos[0]
        dy = target_pos[1] - current_pos[1]
        
        abs_dx = abs(dx)
        abs_dy = abs(dy)
        
        vk_map = {'up': win32con.VK_UP, 'down': win32con.VK_DOWN, 'left': win32con.VK_LEFT, 'right': win32con.VK_RIGHT}
        executed_micro = False

        # ==========================================
        # 1. 絕對獨立處理 X 軸 (左右)
        # ==========================================
        if abs_dx > 0.25:
            # 距離遠：開啟長按巡航，交給後台跑
            self.update_key_bg('right', dx > 0)
            self.update_key_bg('left', dx < 0)
        else:
            # 距離近：立刻關閉長按，準備精準微調
            self.update_key_bg('right', False)
            self.update_key_bg('left', False)
            
            # 進入微調脈衝
            if abs_dx >= 0.005:
                x_key = 'right' if dx > 0 else 'left'
                x_time = max(0.09, min(0.18, (abs_dx / 0.85) + 0.035))
                
                win32api.PostMessage(self.chiaki_hwnd, win32con.WM_KEYDOWN, vk_map[x_key], self._get_lparam(vk_map[x_key], True))
                time.sleep(x_time)
                win32api.PostMessage(self.chiaki_hwnd, win32con.WM_KEYUP, vk_map[x_key], self._get_lparam(vk_map[x_key], False))
                executed_micro = True

        # ==========================================
        # 2. 絕對獨立處理 Y 軸 (上下)
        # ==========================================
        if abs_dy > 0.25:
            # 距離遠：開啟長按巡航，交給後台跑
            self.update_key_bg('down', dy > 0)
            self.update_key_bg('up', dy < 0)
        else:
            # 距離近：立刻關閉長按，準備精準微調
            self.update_key_bg('down', False)
            self.update_key_bg('up', False)
            
            # 進入微調脈衝
            if abs_dy >= 0.005:
                y_key = 'down' if dy > 0 else 'up'
                y_time = max(0.09, min(0.18, (abs_dy / 0.85) + 0.035))
                
                win32api.PostMessage(self.chiaki_hwnd, win32con.WM_KEYDOWN, vk_map[y_key], self._get_lparam(vk_map[y_key], True))
                time.sleep(y_time)
                win32api.PostMessage(self.chiaki_hwnd, win32con.WM_KEYUP, vk_map[y_key], self._get_lparam(vk_map[y_key], False))
                executed_micro = True

        # ==========================================
        # 3. 模擬器狀態防丟失保護 (終極保險)
        # ==========================================
        if executed_micro:
            # 為什麼要這段？因為 Chiaki 的十字鍵/搖桿底層邏輯有互斥性。
            # 如果 X 軸剛剛做完了微調並發送了 KEYUP，它可能會把正在長按趕路的 Y 軸也給打斷。
            # 所以微調結束後，我們強行喚醒一次所有應該在「長按」狀態的按鍵！
            for k, is_pressed in self.key_states.items():
                if is_pressed:
                    win32api.PostMessage(self.chiaki_hwnd, win32con.WM_KEYDOWN, vk_map[k], self._get_lparam(vk_map[k], True))
            
            time.sleep(0.08) # 給予畫面更新的緩衝時間

    def click_action(self):
        if not self.chiaki_hwnd: return
        vk = win32con.VK_RETURN
        win32api.PostMessage(self.chiaki_hwnd, win32con.WM_KEYDOWN, vk, self._get_lparam(vk, True))
        time.sleep(0.15) 
        win32api.PostMessage(self.chiaki_hwnd, win32con.WM_KEYUP, vk, self._get_lparam(vk, False))
        print("✅ [實機] 執行 Cross (Return) 點擊！")


    # ==========================================
    # 區塊 C：點擊決策與同步暫停 (職能解耦)
    # ==========================================

    def is_scene_changed(self, current_gray, last_gray, curr_cx, curr_cy, last_cx, last_cy, is_sync_mode=False):
        if current_gray is None or last_gray is None: return False
        
        curr_clean = current_gray.copy()
        last_clean = last_gray.copy()
        h, w = curr_clean.shape
        
        # 確保塗黑區域嚴格限制為 35x35 (半徑 17)
        mask_r = 18 
        
        def mask_out(cx, cy):
            mx1, my1 = max(0, int(cx - mask_r)), max(0, int(cy - mask_r))
            mx2, my2 = min(w, int(cx + mask_r)), min(h, int(cy + mask_r))
            curr_clean[my1:my2, mx1:mx2] = 0
            last_clean[my1:my2, mx1:mx2] = 0

        # 將 A 點 (舊指針) 與 B 點 (新指針) 在兩張圖中同時塗黑 35x35
        mask_out(curr_cx, curr_cy)
        mask_out(last_cx, last_cy)

        # 對位像素相減：計算所有剩餘純背景像素的絕對差值
        diff_full = cv2.absdiff(curr_clean, last_clean)
        
        # 灰階差大於 3 (約 1%) 的像素標記為有效變化
        _, thresh_full = cv2.threshold(diff_full, 3, 255, cv2.THRESH_BINARY)

        # --- 全背景計算 ---
        total_pixels = w * h
        global_changed_pixels = np.count_nonzero(thresh_full)
        
        if is_sync_mode:
            return global_changed_pixels > (total_pixels * 0.08)

        # --- 1/3 區域計算 ---
        bw, bh = w // 3, h // 3
        x1, y1 = max(0, int(curr_cx - bw // 2)), max(0, int(curr_cy - bh // 2))
        x2, y2 = min(w, int(curr_cx + bw // 2)), min(h, int(curr_cy + bh // 2))
        
        local_changed = False
        if x2 > x1 and y2 > y1:
            # 直接從對位相減的結果矩陣中切出 100% 重合的局部區域
            thresh_local = thresh_full[y1:y2, x1:x2]
            local_changed_pixels = np.count_nonzero(thresh_local)
            
            # 絕對數量判定：焦點區域內變動超過 100 個像素即算場景改變
            if local_changed_pixels > 100:
                local_changed = True

        # 全背景變化 > 1%，或 1/3 區域變化像素 > 100
        return (global_changed_pixels > total_pixels * 0.01) or local_changed

    def sync_check(self, yt_gray, chiaki_gray):
        if yt_gray is None or chiaki_gray is None: return True
        ck_res = cv2.resize(chiaki_gray, (self.yt_w, self.yt_h))
        # 💡 補上缺少的指針參數 (0, 0, 0, 0 作為佔位)，避免 TypeError
        is_different = self.is_scene_changed(yt_gray, ck_res, 0, 0, 0, 0, is_sync_mode=True)
        return not is_different

    # ------------------------------------------
    # 模塊 1: 影片端專屬 (捕捉因)
    # ------------------------------------------
    def save_debug_screenshot(self, frame, rel_pos, timestamp):
        """
        保存帶有預判點擊位置的截圖，用於後續狀態機比對與除錯。
        """
        import os
        os.makedirs('shotscreen', exist_ok=True)
        
        debug_frame = frame.copy()
        h, w = debug_frame.shape[:2]
        
        # 轉換回絕對物理座標
        abs_x = int(rel_pos[0] * w)
        abs_y = int(rel_pos[1] * h)
        
        # 畫一個紅實心圓標示 YOLO 預計點擊的位置
        cv2.circle(debug_frame, (abs_x, abs_y), 6, (0, 0, 255), -1)
        
        # 以系統時間戳為檔名保存
        filename = f"shotscreen/{int(timestamp)}.jpg"
        cv2.imwrite(filename, debug_frame)
        print(f"📸 [Screenshot] 已保存點擊瞬間截圖至: {filename}")
        
    def detect_and_enqueue_click(self, yt_rel_pos, yt_cls, scene_changed, yt_frame):
        # 如果正在暫停，我們需要不斷更新『基準時間』
        if getattr(self, 'is_paused_by_sync', False):
            self.last_yt_trigger_time = time.time()
            return

        # --- 動畫鎖邏輯 ---
        scene_jump_triggered = False
        if scene_changed:
            self.scene_stable_frames = 0
            if not self.is_scene_changing:
                self.is_scene_changing = True
                scene_jump_triggered = True 
        else:
            self.scene_stable_frames += 1
            if self.scene_stable_frames >= 10:
                self.is_scene_changing = False

        cursor_changed = (yt_rel_pos and self.last_yt_cls and yt_cls != self.last_yt_cls and yt_cls in ('Hold', 'Keep', 'Waiting'))
        trigger = scene_jump_triggered or (cursor_changed and not self.is_scene_changing)

        if trigger:
            priority_target = self.prev_intent_pos if self.prev_intent_pos else self.last_known_cursor_pos
            if priority_target:
                new_pos = (float(priority_target[0]), float(priority_target[1]))
                
                current_real_t = time.time()
                if not hasattr(self, 'last_yt_trigger_time'):
                    self.last_yt_trigger_time = current_real_t
                
                dt_gap = current_real_t - self.last_yt_trigger_time
                
                # 0.5s 物理防抖
                if not hasattr(self, 'last_enqueue_real_t') or (current_real_t - self.last_enqueue_real_t > 0.5):
                    self.last_yt_trigger_time = current_real_t 
                    
                    self.queue.append({'pos': new_pos, 'dt': dt_gap})
                    self.last_enqueue_real_t = current_real_t
                    print(f"📥 [New Task] {new_pos} 入隊 | 影片真實間隔: {dt_gap:.2f}s | 總數: {len(self.queue)}")
                    
                    # 💡 呼叫獨立出來的截圖函數
                    self.save_debug_screenshot(yt_frame, new_pos, current_real_t)

                    self.prev_intent_pos = None 

        self.last_yt_cls = yt_cls

    # ------------------------------------------
    # 模塊 2: 實機端專屬 (執行果)
    # ------------------------------------------
    def execute_and_dequeue_click(self, current_t):
        if not self.queue: return
        if not self.chaiki_cursor_pos: return

        task = self.queue[0]
        target_pos = task['pos']
        required_dt = task['dt']
        
        # 【解釋】計算當前指針到目標的距離
        dist = math.hypot(target_pos[0]-self.chaiki_cursor_pos[0], target_pos[1]-self.chaiki_cursor_pos[1])

        # 【解釋】如果距離小於 0.012，準備開火
        if dist < 0.012:
            # 【解釋】剎車，重置按鍵狀態，並累加雙幀確認計數
            self.release_all_keys()
            self.ck_stable_frames += 1
            
            # 【解釋】連續兩幀到位，進入時間判定
            if self.ck_stable_frames >= 2:
                # 【解釋】計算距離上次點擊過去了多久
                time_since_last_click = current_t - getattr(self, 'last_ck_click_finish_time', 0.0)
                
                if time_since_last_click >= required_dt:
                    print(f"🔥 [EXECUTE CLICK] 滿足影片間隔 ({required_dt:.2f}s)，執行點擊！")
                    self.click_action()
                    
                    # 【解釋】記錄本次點擊完成時間
                    self.last_ck_click_finish_time = time.time()
                    print(f"✅ [Task Finished] 彈出: {target_pos}")
                    self.queue.pop(0) 
                    
                    # 💡 這裡必須加回來！點擊完成並彈出任務後，必須重置穩定幀，否則下一個任務一進來就會被瞬間誤判為「已穩定」
                    self.ck_stable_frames = 0
                else:
                    pass
        else:
            # 【解釋】距離不夠，重置雙幀計數並繼續移動
            self.ck_stable_frames = 0
            self.move_action(self.chaiki_cursor_pos, target_pos)


    # ==========================================
    # 執行循環 (主幹瘦身)
    # ==========================================

    def run_live_sync(self, start_time_sec=34):
        with sync_playwright() as p:
            browser = p.chromium.connect_over_cdp("http://localhost:9222")
            page = next((pg for pg in browser.contexts[0].pages if "youtube" in pg.url), browser.contexts[0].pages[0])
            video = page.wait_for_selector("video")
            
            page.evaluate(f"document.querySelector('video').pause(); document.querySelector('video').currentTime = {start_time_sec};")
            time.sleep(0.5)
            
            self.camera = dxcam.create(output_idx=0, output_color="BGR") 
            full_grab = self.camera.grab()
            while full_grab is None:
                full_grab = self.camera.grab()
                time.sleep(0.01)
                
            self.chaiki_ready(full_grab)
            self.camera.start(target_fps=30, video_mode=True)

            print("\n" + "="*50)
            print("⏳ 進入手動對齊模式... 按下 [空白鍵 (SPACE)] 啟動。")
            print("="*50 + "\n")

            aligned = False
            with torch.no_grad():
                while not aligned:
                    if win32api.GetAsyncKeyState(win32con.VK_SPACE) & 0x8000:
                        print("\n🚀 啟動自動控制系統！\n")
                        aligned = True
                        time.sleep(0.5)
                    if win32api.GetAsyncKeyState(ord('Q')) & 0x8000:
                        self.camera.stop(); browser.close(); return

                self.queue.clear() 
                self.last_yt_cls = None        
                recording = True
                
                page.evaluate("document.querySelector('video').play();")
                page.add_style_tag(content='.ytp-chrome-bottom, .ytp-chrome-top, .ytp-gradient-bottom, .ytp-gradient-top, .ytp-watermark { display: none !important; }')

                self.observation_frames = 0

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

                    yt_rel_pos, yt_cls, yt_abs_pos = self.Realtime_cursor_position(yt_frame)
                    ck_rel_pos, ck_cls, ck_abs_pos = self.Realtime_cursor_position(chiaki_frame)

                    # 開場跳幀保護
                    if self.observation_frames < 10:
                        self.observation_frames += 1
                        self.release_all_keys()
                        continue

                    # 更新已知意圖座標 (確保在場景變換前一刻留存)
                    if yt_rel_pos is not None:
                        self.last_known_cursor_pos = yt_rel_pos
                        self.prev_intent_pos = (float(yt_rel_pos[0]), float(yt_rel_pos[1]))
                    else:
                        if not self.queue: self.release_all_keys()

                    if ck_rel_pos is not None: 
                        self.chaiki_cursor_pos = ck_rel_pos

                    current_t = time.time()
                    
                    # 💡 確保算出當前幀的絕對座標 (sx, sy)
                    try:
                        sx = yt_abs_pos[0] if yt_abs_pos else int((self.last_known_cursor_pos[0] if self.last_known_cursor_pos else 0.5) * self.yt_w)
                        sy = yt_abs_pos[1] if yt_abs_pos else int((self.last_known_cursor_pos[1] if self.last_known_cursor_pos else 0.5) * self.yt_h)
                    except:
                        sx, sy = int(0.5 * self.yt_w), int(0.5 * self.yt_h)

                    # 💡 初始化底圖的指針座標 (給第一幀使用)
                    if not hasattr(self, 'ref_cursor_x'):
                        self.ref_cursor_x, self.ref_cursor_y = sx, sy

                    # 隊列滿 3 暫停影片邏輯
                    if len(self.queue) >= 3 and not self.is_paused_by_sync:
                        page.evaluate("document.querySelector('video').pause();")
                        self.is_paused_by_sync = True
                        print("⏸️ 隊列任務積壓達到 3 個，暫停影片並關閉檢測...")
                        self.reference_gray = None # 清空底圖避免暫停 UI 污染
                    elif len(self.queue) < 3 and getattr(self, 'sync_fail_count', 0) < 3 and self.is_paused_by_sync:
                        page.evaluate("document.querySelector('video').play();")
                        self.is_paused_by_sync = False
                        print("▶️ 隊列與畫面均已對齊，恢復播放！")
                        self.reference_gray = None # 恢復播放後重新獲取乾淨底圖

                    # 物理隔離！暫停期間絕對不執行場景變化判定與入隊
                    if not self.is_paused_by_sync:
                        scene_changed = False
                        is_locked = getattr(self, 'click_validating', False)
                        
                        if not is_locked:
                            # 🛡️ 安全機制：如果底圖被清空(例如剛從暫停恢復)，立即重新獲取底圖和底圖指針
                            if self.reference_gray is None:
                                self.reference_gray = yt_gray.copy()
                                self.ref_cursor_x, self.ref_cursor_y = sx, sy
                                self.ref_frame_timer = 0

                            try:
                                # 💡 傳入 6 個參數：當前圖, 底圖, 當前指針X/Y, 底圖指針X/Y
                                scene_changed = self.is_scene_changed(
                                    yt_gray, self.reference_gray, 
                                    sx, sy, 
                                    self.ref_cursor_x, self.ref_cursor_y
                                )
                            except Exception as e:
                                scene_changed = False
                            
                            # 維護底圖與其對應的指針座標
                            if scene_changed:
                                self.reference_gray = yt_gray.copy()
                                self.ref_cursor_x, self.ref_cursor_y = sx, sy
                                self.ref_frame_timer = 0
                            else:
                                self.ref_frame_timer += 1
                                if self.ref_frame_timer >= 30: # 每秒定期刷新底圖
                                    self.reference_gray = yt_gray.copy()
                                    self.ref_cursor_x, self.ref_cursor_y = sx, sy
                                    self.ref_frame_timer = 0

                            # 影片端專屬：入隊邏輯 (判斷因)
                            if recording:
                                self.detect_and_enqueue_click(yt_rel_pos, yt_cls, scene_changed, yt_frame)

                    # 3. 實機端專屬：出隊邏輯 (執行果) - 實機必須持續處理隊列，不受暫停阻斷
                    self.execute_and_dequeue_click(current_t)

                    # --- 定期同步檢查 (兼容舊版邏輯並加上防禦) ---
                    self.frame_counter += 1
                    if self.frame_counter >= 45:
                        self.frame_counter = 0
                        if not getattr(self, 'click_validating', False) and not self.sync_check(yt_gray, chiaki_gray):
                            self.sync_fail_count += 1
                            if self.sync_fail_count >= 3 and not self.is_paused_by_sync:
                                page.evaluate("document.querySelector('video').pause();")
                                self.is_paused_by_sync = True
                                print("⏸️ 畫面不同步，暫停影片...")
                                self.reference_gray = None # 確保同步暫停時也清空底圖
                        else:
                            self.sync_fail_count = 0
                            if self.is_paused_by_sync and len(self.queue) < 3:
                                page.evaluate("document.querySelector('video').play();")
                                self.is_paused_by_sync = False
                                print("▶️ 重新對齊，恢復播放！")
                                self.reference_gray = None

                    if win32api.GetAsyncKeyState(ord('Q')) & 0x8000: 
                        break

            self.camera.stop()
            browser.close()

if __name__ == "__main__":
    agent = AutoPlatinumHand("https://www.youtube.com/watch?v=7K_NimshHUI")
    agent.run_live_sync()