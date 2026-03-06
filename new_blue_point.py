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

        # --- 補齊：移動控制參數 ---
        self.deadzone = 8
        self.key_states = {'up': False, 'down': False, 'left': False, 'right': False}
        self.prev_roi_patch = None
        self.roi_box = None

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
        return (max_loc[0] + template.shape[1]//2, max_loc[1] + template.shape[0]//2) if max_val > threshold else None

    def track_video_cursor(self, target_pos, current_pos):
        """🚀 追趕邏輯：傳入影片與實機的絕對座標，進行移動"""
        if not target_pos or not current_pos: return False

        # 套用縮放比映射到實機座標
        chiaki_target_x = int(target_pos[0] / self.scale_x)
        chiaki_target_y = int(target_pos[1] / self.scale_y)
        chiaki_target = (chiaki_target_x, chiaki_target_y)
        
        dx = chiaki_target_x - current_pos[0]
        dy = chiaki_target_y - current_pos[1]
        dist = math.hypot(dx, dy)

        if dist <= self.deadzone:
            self.release_all_keys()
            return True

        if dist > 35:
            self.move_to_target(chiaki_target)
        else:
            self.refine_move_to_target(chiaki_target)
        return False

    def Realtime_cursor_position(self, bgr_frame, template_gray, last_pos, tpl_path='cursor.png'):
        """🚀 終極動態追蹤：雙重遮罩 (幀差法 Motion + HSV色值驗證 Feature)"""
        # 0. 初始化狀態與顏色特徵 (懶加載，不改 __init__)
        attr_prev = f"prev_bgr_{tpl_path.split('.')[0]}"
        attr_bounds = f"hsv_bounds_{tpl_path.split('.')[0]}"

        if not hasattr(self, attr_bounds):
            tpl_bgr = cv2.imread(tpl_path, cv2.IMREAD_COLOR)
            if tpl_bgr is not None:
                th, tw = tpl_bgr.shape[:2]
                # 提取中心區域(手型核心)顏色作為「純血色值」
                center_patch = cv2.cvtColor(tpl_bgr[th//2-3:th//2+3, tw//2-3:tw//2+3], cv2.COLOR_BGR2HSV)
                h, s, v = cv2.mean(center_patch)[:3]
                # 建立色彩寬容範圍 (容忍壓縮與光影變化)
                setattr(self, attr_bounds, (
                    np.array([max(0, h-30), max(0, s-60), max(0, v-60)]),
                    np.array([min(179, h+30), min(255, s+60), min(255, v+60)])
                ))
            else:
                setattr(self, attr_bounds, None)

        prev_bgr = getattr(self, attr_prev, None)
        setattr(self, attr_prev, bgr_frame.copy()) # 更新歷史幀供下一幀對沖

        gray_full = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
        if last_pos is None or prev_bgr is None:
            return self.detect_template(gray_full, template_gray, 0.70)

        cx, cy = last_pos
        h, w = bgr_frame.shape[:2]
        roi_size = 120  # 擴大範圍到 240x240，防止滑鼠甩太快飛出

        x1, y1 = max(0, cx - roi_size), max(0, cy - roi_size)
        x2, y2 = min(w, cx + roi_size), min(h, cy + roi_size)

        th, tw = template_gray.shape
        if x2 - x1 <= tw or y2 - y1 <= th:
            return self.detect_template(gray_full, template_gray, 0.70)

        roi_curr = bgr_frame[y1:y2, x1:x2]
        roi_prev = prev_bgr[y1:y2, x1:x2]

        # 1. 第一層過濾：動態遮罩 (Frame Differencing)
        diff = cv2.absdiff(roi_curr, roi_prev)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, motion_mask = cv2.threshold(diff_gray, 20, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)

        # 💤 休眠省電機制：若畫面靜止(動態像素極少)，判定為沒動，直接回傳上次位置
        if cv2.countNonZero(motion_mask) < 15:
            return last_pos

        # 2. 第二層過濾：色彩遮罩 (HSV Validation)
        bounds = getattr(self, attr_bounds, None)
        if bounds:
            roi_hsv = cv2.cvtColor(roi_curr, cv2.COLOR_BGR2HSV)
            color_mask = cv2.inRange(roi_hsv, bounds[0], bounds[1])
            # 🎯 雙重遮罩交集：必須「在動」且「顏色對」
            final_mask = cv2.bitwise_and(motion_mask, color_mask)
        else:
            final_mask = motion_mask

        # 3. 形狀與雜訊過濾
        cnts, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            largest_cnt = max(cnts, key=cv2.contourArea)
            area = cv2.contourArea(largest_cnt)
            # 過濾太小的雜訊(面積<10)，以及過渡動畫的巨大閃爍(面積>2500)
            if 10 < area < 2500:
                M = cv2.moments(largest_cnt)
                if M["m00"] != 0:
                    new_cx = x1 + int(M["m10"] / M["m00"])
                    new_cy = y1 + int(M["m01"] / M["m00"])
                    return (new_cx, new_cy)

        # 🚨 全域重捕：如果在 ROI 內動的東西不符合特徵，觸發全螢幕尋找
        return self.detect_template(gray_full, template_gray, 0.70)

    def move_to_target(self, target_pos):
        """長按大範圍移動。"""
        dx = target_pos[0] - self.chaiki_cursor_pos[0]
        dy = target_pos[1] - self.chaiki_cursor_pos[1]
        self.update_key_bg('right', dx > self.deadzone)
        self.update_key_bg('left', dx < -self.deadzone)
        self.update_key_bg('down', dy > self.deadzone)
        self.update_key_bg('up', dy < -self.deadzone)

    def refine_move_to_target(self, target_pos):
        """🎯 按幀計算的像素級微調。"""
        dx = target_pos[0] - self.chaiki_cursor_pos[0]
        dy = target_pos[1] - self.chaiki_cursor_pos[1]
        fine_dz = 2 
        self.update_key_bg('right', dx > fine_dz)
        self.update_key_bg('left', dx < -fine_dz)
        self.update_key_bg('down', dy > fine_dz)
        self.update_key_bg('up', dy < -fine_dz)

    def update_key_bg(self, key_str, press):
        """後台注入按鍵消息"""
        if not self.chiaki_hwnd: return
        vk_map = {'up': win32con.VK_UP, 'down': win32con.VK_DOWN, 'left': win32con.VK_LEFT, 'right': win32con.VK_RIGHT, 'enter': win32con.VK_RETURN}
        vk = vk_map.get(key_str)
        if self.key_states.get(key_str) != press:
            if press: win32api.PostMessage(self.chiaki_hwnd, win32con.WM_KEYDOWN, vk, 0)
            else: win32api.PostMessage(self.chiaki_hwnd, win32con.WM_KEYUP, vk, 0)
            self.key_states[key_str] = press

    def release_all_keys(self):
        """釋放所有方向鍵"""
        for k in ['up', 'down', 'left', 'right']: self.update_key_bg(k, False)

    # ==========================================
    # 區塊 C：點擊決策與同步暫停 (Logic & Sync)
    # ==========================================

    def execute_queue_logic(self):
        """🚀 任務分派器：監聽 Queue，Hover 完成後執行 Click 並掛起 [cite: 2026-03-05]。"""
        if not self.queue: return 
        task = self.queue[0]
        if task.get("type") in ["HOVER", "MOVE"]:
            if self.Realtime_cursor_position(): self.queue.pop(0)
            return
        if task.get("type") == "CLICK":
            if self.Realtime_cursor_position():
                win32api.PostMessage(self.chiaki_hwnd, win32con.WM_KEYDOWN, win32con.VK_RETURN, 0)
                time.sleep(0.05)
                win32api.PostMessage(self.chiaki_hwnd, win32con.WM_KEYUP, win32con.VK_RETURN, 0)
                self.queue.pop(0)
                self.state = "IDLE"

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

    def process_roi_focus(self, gray_frame, anchor_pos, roi_size=60, cursor_size=15):
        """🚀 焦點核心 (單幀處理)"""
        if gray_frame is None or anchor_pos is None: return None, None
        cx, cy = anchor_pos
        h, w = gray_frame.shape
        x1, y1 = max(0, cx - roi_size), max(0, cy - roi_size)
        x2, y2 = min(w, cx + roi_size), min(h, cy + roi_size)
        roi_box = (x1, y1, x2, y2)
        if x2 - x1 <= cursor_size * 2 or y2 - y1 <= cursor_size * 2: return None, roi_box
        patch = cv2.GaussianBlur(gray_frame[y1:y2, x1:x2].copy(), (5, 5), 0)
        ph, pw = patch.shape
        ix1, iy1 = max(0, pw//2 - cursor_size), max(0, ph//2 - cursor_size)
        ix2, iy2 = min(pw, pw//2 + cursor_size), min(ph, ph//2 + cursor_size)
        patch[iy1:iy2, ix1:ix2] = 0
        return patch, roi_box
    
    def draw_sync_hud(self, frame, video_pos, chiaki_pos, fps):
        """獨立繪製 GUI：整合輔助線繪製與縮放邏輯"""
        self.current_roi_patch, self.roi_box = self.process_roi_focus(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), video_pos)
        overlay = frame.copy()
        if self.roi_box:
            cv2.rectangle(overlay, (self.roi_box[0], self.roi_box[1]), (self.roi_box[2], self.roi_box[3]), (0, 255, 0), 2)
        
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

    def run_live_sync(self, start_time_sec=153):
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
            self.camera.start(target_fps=30, video_mode=True)
            recording, start_tick, prev_time = False, 0, time.time()
            cv2.namedWindow("Platinum Vision")
            with torch.no_grad():
                while True:
                    full_frame = self.camera.get_latest_frame()
                    if full_frame is None: continue
                    yt_frame = full_frame[self.yt_y:self.yt_y+self.yt_h, self.yt_x:self.yt_x+self.yt_w]
                    chiaki_frame = full_frame[self.chiaki_y:self.chiaki_y+self.chiaki_h, self.chiaki_x:self.chiaki_x+self.chiaki_w]
                    
                    yt_gray = cv2.cvtColor(yt_frame, cv2.COLOR_BGR2GRAY)
                    chiaki_gray = cv2.cvtColor(chiaki_frame, cv2.COLOR_BGR2GRAY)
                    
                    curr_time = time.time()
                    fps = 1 / (curr_time - prev_time) if curr_time > prev_time else 60
                    prev_time = curr_time
                    # 1. 影片指針追蹤 (傳入 BGR 彩色幀與模板圖檔名，直接覆寫以清除殘影)
                    self.last_known_cursor_pos = self.Realtime_cursor_position(
                        yt_frame, self.cursor_tpl, self.last_known_cursor_pos, 'cursor.png'
                    )
                    
                    # 2. 實機指針追蹤
                    self.chaiki_cursor_pos = self.Realtime_cursor_position(
                        chiaki_frame, self.chiaki_cursor_tpl, self.chaiki_cursor_pos, 'cursor2.png'
                    )
                    
                    key = cv2.waitKey(1) & 0xFF
                    
                    # 3. 處理追趕邏輯
                    if recording: 
                      self.track_video_cursor(self.last_known_cursor_pos, self.chaiki_cursor_pos)
                    if key == ord(' '):
                        if not recording:
                            page.evaluate("document.querySelector('video').play()")
                            recording = True
                        else:
                            page.evaluate("document.querySelector('video').pause()")
                            recording = False
                            self.release_all_keys()
                    display = self.draw_sync_hud(yt_frame, self.last_known_cursor_pos, self.chaiki_cursor_pos, fps)
                    cv2.imshow("Platinum Vision", display)
                    if key == ord('q'): break
            self.camera.stop(); browser.close(); cv2.destroyAllWindows()

if __name__ == "__main__":
    agent = AutoPlatinumHand("https://www.youtube.com/watch?v=7K_NimshHUI")
    agent.run_live_sync()