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

        # --- 新增的異步控制變數 ---
        self.last_click_time = 0.0
        self.frame_counter = 0
        self.sync_fail_count = 0
        self.is_paused_by_sync = False
        self.key_states = {'up': False, 'down': False, 'left': False, 'right': False}

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
        """支持 4 個方向動態旋轉匹配的指針檢測 (臨時替代 YOLO 確保能看見指針)"""
        if template is None or gray_frame is None or gray_frame.size == 0: return None
        
        best_val = -1
        best_loc = None
        best_shape = None
        
        templates_4_dirs = [
            template,
            cv2.rotate(template, cv2.ROTATE_90_CLOCKWISE),
            cv2.rotate(template, cv2.ROTATE_180),
            cv2.rotate(template, cv2.ROTATE_90_COUNTERCLOCKWISE)
        ]
        
        for tpl in templates_4_dirs:
            res = cv2.matchTemplate(gray_frame, tpl, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            
            if max_val > best_val:
                best_val = max_val
                best_loc = max_loc
                best_shape = tpl.shape
                
        if best_val > threshold:
            return (best_loc[0] + best_shape[1]//2, best_loc[1] + best_shape[0]//2)
            
        return None

    def Realtime_cursor_position(self):
        # 佔位，目前暫時用 detect_template 解決 YOLO 尚未訓練的問題
        pass

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

    def move_action(self, dx, dy):
        """長按大範圍移動"""
        self.update_key_bg('right', dx > 15)
        self.update_key_bg('left', dx < -15)
        self.update_key_bg('down', dy > 15)
        self.update_key_bg('up', dy < -15)

    def refine_move(self, dx, dy):
        """短按微調"""
        self.update_key_bg('right', dx > 2)
        self.update_key_bg('left', dx < -2)
        self.update_key_bg('down', dy > 2)
        self.update_key_bg('up', dy < -2)

    def click_action(self):
        """執行物理點擊"""
        win32api.PostMessage(self.chiaki_hwnd, win32con.WM_KEYDOWN, win32con.VK_RETURN, 0)
        time.sleep(0.05)
        win32api.PostMessage(self.chiaki_hwnd, win32con.WM_KEYUP, win32con.VK_RETURN, 0)
        print("✅ 實機點擊執行完畢")

    # ==========================================
    # 區塊 C：點擊決策與同步暫停 (Logic & Sync)
    # ==========================================

    def sync_check(self, yt_gray, chiaki_gray):
        """每 60 幀模糊匹配邊沿 15% 確保畫面同步"""
        yt_hash = self.get_sparse_hash(yt_gray)
        ck_hash = self.get_sparse_hash(cv2.resize(chiaki_gray, (yt_gray.shape[1], yt_gray.shape[0])))
        diff = np.mean(cv2.absdiff(yt_hash, ck_hash))
        return diff < 60  # 放寬至 60 避免誤判

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
            last_print_time = time.time() # 新增：用於控制每秒打印頻率

            print("🚀 系統啟動：[Space] 開始/暫停 | [Q] 退出")

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

                    # 雙指針辨識 (使用 4 向檢測函數)
                    v_pos = self.detect_template(yt_gray, self.cursor_tpl, 0.70)
                    c_pos = self.detect_template(chiaki_gray, self.chiaki_cursor_tpl, 0.70)
                    
                    if v_pos: self.last_known_cursor_pos = v_pos
                    if c_pos: self.chaiki_cursor_pos = c_pos

                    # ⏱️ 每秒打印一次座標狀態
                    if curr_time - last_print_time >= 1.0:
                        print(f"⏱️ [座標即時監控] 影片目標: {self.last_known_cursor_pos} | 實機位置: {self.chaiki_cursor_pos}")
                        last_print_time = curr_time

                    if recording:
                        # --- A. 事件檢測 (場景變化 & ROI 變化) ---
                        scene_changed = False
                        roi_changed = False
                        
                        if self.last_full_gray_np is not None:
                            # 檢測 Scene Change (放寬閾值到 35 防誤判)
                            yt_hash = self.get_sparse_hash(yt_gray)
                            prev_hash = self.get_sparse_hash(self.last_full_gray_np)
                            if np.mean(cv2.absdiff(yt_hash, prev_hash)) > 35: 
                                scene_changed = True

                            # 檢測 ROI Change
                            if v_pos:
                                cx, cy = int(v_pos[0]), int(v_pos[1])
                                h, w = yt_gray.shape
                                x1, y1 = max(0, cx-40), max(0, cy-40)
                                x2, y2 = min(w, cx+40), min(h, cy+40)
                                curr_roi = yt_gray[y1:y2, x1:x2].copy()
                                prev_roi = self.last_full_gray_np[y1:y2, x1:x2].copy()
                                
                                rh, rw = curr_roi.shape
                                if rh > 20 and rw > 20:
                                    curr_roi[rh//2-10:rh//2+10, rw//2-10:rw//2+10] = 0
                                    prev_roi[rh//2-10:rh//2+10, rw//2-10:rw//2+10] = 0
                                if np.mean(cv2.absdiff(curr_roi, prev_roi)) > 15:
                                    roi_changed = True

                        # --- B. 隊列注入 (事件驅動入隊) ---
                        # 觸發條件：點擊間隔 > 200ms 且 (ROI背景變化 或 場景切換)
                        if (curr_time - self.last_click_time > 0.2) and v_pos:
                            if roi_changed or scene_changed:
                                if not any(q['pos'] == v_pos for q in self.queue):
                                    self.queue.append({'pos': v_pos})
                                    self.last_click_time = curr_time
                                    event_name = "場景切換" if scene_changed else "ROI變化"
                                    print(f"📥 點擊事件入隊 ({event_name})，目標: {v_pos}")

                        # --- C. 隊列延遲執行 (Chiaki 移動與點擊) ---
                        if self.queue:
                            target_v_pos = self.queue[0]['pos']
                            if self.chaiki_cursor_pos:
                                tx = target_v_pos[0] / self.scale_x
                                ty = target_v_pos[1] / self.scale_y
                                dx = tx - self.chaiki_cursor_pos[0]
                                dy = ty - self.chaiki_cursor_pos[1]
                                dist = math.hypot(dx, dy)

                                if dist <= 8:
                                    self.release_all_keys()
                                    self.click_action()
                                    self.queue.pop(0)
                                elif dist > 50:
                                    self.move_action(dx, dy)
                                else:
                                    self.refine_move(dx, dy)
                            else:
                                # Chiaki 指針短暫丟失 -> 什麼都不做，等待下一幀重新識別
                                self.release_all_keys()
                        else:
                            self.release_all_keys()

                        # --- D. 每 60 幀邊沿 Sync Check ---
                        self.frame_counter += 1
                        if self.frame_counter >= 60:
                            self.frame_counter = 0
                            is_synced = self.sync_check(yt_gray, chiaki_gray)
                            
                            if not is_synced:
                                self.sync_fail_count += 1
                                if self.sync_fail_count >= 3 and not self.is_paused_by_sync:
                                    page.evaluate("document.querySelector('video').pause();")
                                    self.is_paused_by_sync = True
                                    print("⏸️ 畫面不同步，自動暫停影片")
                            else:
                                self.sync_fail_count = 0
                                if self.is_paused_by_sync:
                                    page.evaluate("document.querySelector('video').play();")
                                    self.is_paused_by_sync = False
                                    print("▶️ 畫面已同步，自動恢復播放")

                    self.last_full_gray_np = yt_gray.copy()

                    key = cv2.waitKey(1) & 0xFF
                    
                    if win32api.GetAsyncKeyState(ord('Q')) & 0x8000: break
                    if win32api.GetAsyncKeyState(win32con.VK_SPACE) & 0x01:
                        recording = not recording
                        page.evaluate(f"document.querySelector('video').{'play' if recording else 'pause'}()")
                        if not recording: self.release_all_keys()
                        print(f"狀態切換: {'執行中' if recording else '暫停'}")

            self.camera.stop()
            browser.close()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    agent = AutoPlatinumHand("https://www.youtube.com/watch?v=7K_NimshHUI")
    agent.run_live_sync()