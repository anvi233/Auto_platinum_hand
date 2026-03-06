import cv2
import numpy as np
import math
import time
import dxcam
import torch
import win32gui
import win32api
import win32con
import ctypes
import keyboard
from playwright.sync_api import sync_playwright

# 強制開啟 DPI 感知，確保抓圖與視窗絕對座標 1:1 對應，不受 Windows 縮放影響
ctypes.windll.user32.SetProcessDPIAware()

class AutoPlatinumHand:
    def __init__(self, youtube_url, hex_color="#f2b877", cursor_path='cursor.png', waiting_path='waiting.png'):
        self.youtube_url = youtube_url
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🚀 啟動 PyTorch 運算設備: {self.device}")
        
        self.cursor_tpl = cv2.imread(cursor_path, 0)
        self.waiting_tpl = cv2.imread(waiting_path, 0)
        self.lower_hsv, self.upper_hsv = self.hex_to_hsv_fuzzy(hex_color, h_tol=10, s_tol=40, v_tol=40)
        
        # 狀態控制
        self.anchor_pos = None      
        self.last_time_ms = 0.0
        self.click_cooldown = 0.0
        self.scene_change_count = 0
        self.click_count = 0
        
        # 場景防抖與同步控制變量
        self.last_global_hash = None
        self.is_transitioning = False
        self.is_waiting = False
        self.ignore_next_scene_click = False
        self.transition_start_ms = 0.0
        self.stable_frames_count = 0
        self.last_full_gray_np = None

        # 🎯 導航控制器與背景按鍵映射
        self.key_states = {"up": False, "down": False, "left": False, "right": False, "enter": False}
        self.pending_clicks = []
        self.deadzone = 8
        
        # VK Codes (虛擬按鍵碼) 對應
        self.VK_MAP = {
            'up': win32con.VK_UP,
            'down': win32con.VK_DOWN,
            'left': win32con.VK_LEFT,
            'right': win32con.VK_RIGHT,
            'enter': win32con.VK_RETURN
        }
        
        # 🎯 運行狀態機 (0: 等待對齊, 1: 等待開始, 2: 運行中)
        self.run_state = 0 
        self.chiaki_x = 0
        self.chiaki_y = 0
        self.chiaki_hwnd = None
        
        # 系統全局防抖與自動暫停計時器
        self.last_space_time = 0.0
        self.is_auto_paused = False
        self.desync_start_time = 0.0 # 🎯 場景不同步的 2 秒計時器

    # ---------------------------------------------------------
    # 🕹️ 背景按鍵注入系統 (等同 AHK ControlSend)
    # ---------------------------------------------------------
    def update_key_bg(self, key_str, press):
        """使用 PostMessage 在背景向 Chiaki 注入按鍵，不搶佔滑鼠焦點"""
        if not self.chiaki_hwnd: return
        vk = self.VK_MAP.get(key_str)
        if not vk: return

        if self.key_states[key_str] != press:
            if press:
                win32api.PostMessage(self.chiaki_hwnd, win32con.WM_KEYDOWN, vk, 0)
            else:
                win32api.PostMessage(self.chiaki_hwnd, win32con.WM_KEYUP, vk, 0)
            self.key_states[key_str] = press

    def release_all_keys(self):
        """緊急煞車：釋放所有方向鍵與確認鍵"""
        for k in self.key_states.keys():
            self.update_key_bg(k, False)

    # ---------------------------------------------------------
    # 🖼️ 視覺與影像處理工具
    # ---------------------------------------------------------
    def hex_to_hsv_fuzzy(self, hex_code, h_tol, s_tol, v_tol):
        hex_code = hex_code.lstrip('#')
        rgb = tuple(int(hex_code[i:i+2], 16) for i in (0, 2, 4))
        bgr_pixel = np.uint8([[[rgb[2], rgb[1], rgb[0]]]])
        hsv_pixel = cv2.cvtColor(bgr_pixel, cv2.COLOR_BGR2HSV)[0][0]
        h, s, v = int(hsv_pixel[0]), int(hsv_pixel[1]), int(hsv_pixel[2])
        return np.array([max(0, h - h_tol), max(0, s - s_tol), max(0, v - v_tol)]), \
               np.array([min(179, h + h_tol), min(255, s + s_tol), min(255, v + v_tol)])

    def get_sparse_hash(self, gray_frame):
        """邊框高密度採樣：跳過中心動畫，精準提取四周 15% 區域"""
        h, w = gray_frame.shape
        dh, dw = int(h * 0.15), int(w * 0.15)
        step = 4 
        top = gray_frame[0:dh, ::step].flatten()
        bottom = gray_frame[h-dh:h, ::step].flatten()
        left = gray_frame[dh:h-dh, 0:dw:step].flatten()
        right = gray_frame[dh:h-dh, w-dw:w:step].flatten()
        return np.concatenate((top, bottom, left, right)).astype(np.int16)

    def detect_template(self, gray_frame, template, threshold=0.75):
        if template is None or gray_frame is None or gray_frame.size == 0: return None
        res = cv2.matchTemplate(gray_frame, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        return (max_loc[0] + template.shape[1]//2, max_loc[1] + template.shape[0]//2) if max_val > threshold else None

    def get_hsv_cursor_pos(self, frame, frame_w, frame_h, ref_pos=None):
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        if ref_pos:
            r = int(frame_h / 6)
            cx, cy = ref_pos
            x1, y1 = max(0, cx - r), max(0, cy - r)
            x2, y2 = min(frame_w, cx + r), min(frame_h, cy + r)
            mask = cv2.inRange(hsv_frame[y1:y2, x1:x2], self.lower_hsv, self.upper_hsv)
            conts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if conts:
                M = cv2.moments(max(conts, key=cv2.contourArea))
                if M["m00"] > 8: return (int(M["m10"]/M["m00"])+x1, int(M["m01"]/M["m00"])+y1), mask, (x1, y1, x2, y2)
        
        mask = cv2.inRange(hsv_frame, self.lower_hsv, self.upper_hsv)
        conts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if conts:
            M = cv2.moments(max(conts, key=cv2.contourArea))
            if M["m00"] > 8: return (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])), mask, None
        return None, mask, None

    def draw_hud_overlay(self, frame, hsv_mask, roi_box, anchor_pos, chiaki_pos, fps_val):
        overlay = frame.copy()
        h, w = frame.shape[:2]
        
        if hsv_mask is not None and roi_box:
            x1, y1, x2, y2 = roi_box
            overlay[y1:y2, x1:x2][hsv_mask > 0] = [0, 255, 0]
            
        if anchor_pos:
            cx, cy = anchor_pos
            cv2.line(overlay, (cx, cy), (cx, 0), (0, 255, 255), 1)
            cv2.line(overlay, (cx, cy), (cx, h), (0, 255, 255), 1)
            cv2.line(overlay, (cx, cy), (0, cy), (0, 255, 255), 1)
            cv2.line(overlay, (cx, cy), (w, cy), (0, 255, 255), 1)
            cv2.putText(overlay, f"YT: ({cx}, {cy})", (cx + 15, cy - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        if roi_box: cv2.rectangle(overlay, (roi_box[0], roi_box[1]), (roi_box[2], roi_box[3]), (0, 165, 255), 2)
        
        if chiaki_pos:
            cv2.drawMarker(overlay, chiaki_pos, (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
            cv2.putText(overlay, f"CHK: ({chiaki_pos[0]}, {chiaki_pos[1]})", (chiaki_pos[0]+15, chiaki_pos[1]+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        display = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        info = f"FPS: {int(fps_val)}  SCENES: {self.scene_change_count}  CLICKS: {self.click_count}  QUEUE: {len(self.pending_clicks)}"
        cv2.putText(display, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        status_msg = ""
        color = (0, 255, 0)
        if self.run_state == 0:
            status_msg = "PRESS [SPACE] TO ALIGN CHIAKI WINDOW"
            color = (0, 165, 255)
        elif self.run_state == 1:
            status_msg = "ALIGNED! PRESS [SPACE] TO START AUTO-DRIVE"
            color = (0, 255, 255)
            
        if status_msg:
            cv2.putText(display, status_msg, (w//2-250, h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
        return display

    # ---------------------------------------------------------
    # 🎯 自動對齊與驗證邏輯
    # ---------------------------------------------------------
    def get_yt_video_size(self, page_obj):
        """
        透過 JavaScript 直接從瀏覽器獲取 video 標籤的物理渲染寬高（紅框）
        """
        try:
            size = page_obj.evaluate("""
                () => {
                    const video = document.querySelector('video');
                    if (!video) return null;
                    const rect = video.getBoundingClientRect();
                    return { w: Math.round(rect.width), h: Math.round(rect.height) };
                }
            """)
            return size['w'], size['h']
        except:
            return None, None

    # ---------------------------------------------------------
    # 🎯 修正後的對齊邏輯：手動對齊，但精確識別遊戲區 (去黑框)
    # ---------------------------------------------------------
    def auto_align_chiaki(self, camera, yt_frame, rw, rh):
        """鎖定 Chiaki 視窗位置，並精確識別內部的遊戲畫面區域（去黑框）"""
        self.yt_w, self.yt_h = rw, rh
        hwnds = []
        # 尋找 Chiaki 視窗
        win32gui.EnumWindows(lambda hwnd, param: param.append(hwnd) if win32gui.IsWindowVisible(hwnd) and "chiaki" in win32gui.GetWindowText(hwnd).lower() else None, hwnds)
        
        if not hwnds:
            print("❌ 找不到 Chiaki 視窗！請確認已開啟。")
            return False
            
        hwnd = hwnds[0]
        self.chiaki_hwnd = hwnd
        # win32gui.SetForegroundWindow(hwnd) # 註釋掉，避免搶奪焦點
        
        # 1. 獲取當前視窗的外部矩形 (含邊框)
        rect = win32gui.GetWindowRect(hwnd)
        wx, wy, ww, wh = rect[0], rect[1], rect[2]-rect[0], rect[3]-rect[1]
        
        # 2. 截圖並識別當前 Chiaki 內部的遊戲畫面區域
        full_grab = camera.grab()
        if full_grab is None: return False
        chiaki_window_img = full_grab[wy:wy+wh, wx:wx+ww]
        
        gray_c = cv2.cvtColor(chiaki_window_img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(cv2.medianBlur(gray_c, 35), 30, 100)
        cnts, _ = cv2.findContours(cv2.dilate(edges, None, iterations=2), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_rects = [cv2.boundingRect(c) for c in cnts if cv2.boundingRect(c)[2] > 300]
        
        if not valid_rects:
            print("❌ 無法識別 Chiaki 內的遊戲畫面邊界！請確認畫面不是全黑。")
            return False
            
        # 取得當前 Chiaki 內部遊戲畫面的實際尺寸 (cw, ch) 與相對於視窗的偏移 (cx, cy)
        cx, cy, cw, ch = max(valid_rects, key=lambda r: r[2]*r[3])
        
        # 🎯 這裡去掉了原有的 win32gui.MoveWindow 邏輯，不強制縮放視窗
        
        # 3. 更新對齊後的座標，確保後續截圖與點擊是「純遊戲區」
        # 我們將起點設在遊戲畫面的第一個像素，徹底去掉黑框偏移
        self.chiaki_x = wx + cx
        self.chiaki_y = wy + cy
        self.chiaki_w = cw
        self.chiaki_h = ch
        
        print(f"✅ Chiaki 遊戲區鎖定完成！")
        print(f"📏 遊戲區尺寸: {cw}x{ch} (相對於視窗偏移: x={cx}, y={cy})")
        print(f"💡 YouTube 遊戲區尺寸: {rw}x{rh}")
        
        # 4. 背景驗證 (維持原有的像素驗證邏輯)
        full_grab = camera.grab()
        chiaki_game_frame = full_grab[self.chiaki_y:self.chiaki_y+ch, self.chiaki_x:self.chiaki_x+cw]
        
        # 如果尺寸不同，先 resize 再做驗證
        yt_resized = cv2.resize(cv2.cvtColor(yt_frame, cv2.COLOR_BGR2GRAY), (cw, ch))
        diff = cv2.absdiff(yt_resized, cv2.cvtColor(chiaki_game_frame, cv2.COLOR_BGR2GRAY))
        mean_diff = np.mean(diff)
        
        if mean_diff < 55:
            print(f"✅ [鎖定成功] 影像匹配，誤差值: {mean_diff:.2f}")
        else:
            print(f"⚠️ [鎖定完成] 尺寸已記錄，但場景色差較大 ({mean_diff:.2f})。")
            
        return True

    # ---------------------------------------------------------
    # 🚀 主迴圈：雙重視角與實時導航
    # ---------------------------------------------------------
    def process_with_playwright(self, start_time_sec=35):
        with sync_playwright() as p:
            print("🔗 正在連接 Chrome (port 9222)...")
            browser = p.chromium.connect_over_cdp("http://localhost:9222")
            page = next((pg for pg in browser.contexts[0].pages if "youtube" in pg.url), browser.contexts[0].pages[0])
            video = page.wait_for_selector("video")
            
            page.evaluate(f"document.querySelector('video').pause(); document.querySelector('video').currentTime = {start_time_sec};")
            
            p_bytes = video.screenshot(type="jpeg", quality=100)
            p_frame = cv2.imdecode(np.frombuffer(p_bytes, np.uint8), cv2.IMREAD_COLOR)
            
            camera = dxcam.create(output_idx=0, output_color="BGR") 
            full_init = camera.grab()
            while full_init is None:
                full_init = camera.grab()
                time.sleep(0.01)
            
            gray_p = cv2.cvtColor(p_frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(cv2.medianBlur(gray_p, 35), 30, 100)
            cnts, _ = cv2.findContours(cv2.dilate(edges, None, iterations=2), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            valid_rects = [cv2.boundingRect(c) for c in cnts if cv2.boundingRect(c)[2] > 300]
            rx, ry, rw, rh = max(valid_rects, key=lambda r: r[2]*r[3]) if valid_rects else (0,0,p_frame.shape[1],p_frame.shape[0])

            res = cv2.matchTemplate(full_init, p_frame, cv2.TM_CCOEFF_NORMED)
            _, _, _, max_loc1 = cv2.minMaxLoc(res)
            yt_x, yt_y = max_loc1[0] + rx, max_loc1[1] + ry
            print(f"🎯 鎖定 YouTube 遊戲區！座標: ({yt_x}, {yt_y}), 尺寸: {rw}x{rh}")

            cv2.namedWindow("AutoPlatinum Supervisor", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("AutoPlatinum Supervisor", rw, rh)
            
            start_tick, prev_time = 0, time.time()
            chiaki_anchor_pos = None

            with torch.no_grad():
                while True:
                    full_frame = camera.grab()
                    if full_frame is None: continue
                    
                    yt_frame = full_frame[yt_y:yt_y+rh, yt_x:yt_x+rw]
                    
                    curr_time = time.time()
                    ms = ((cv2.getTickCount() - start_tick) / cv2.getTickFrequency()) * 1000 if self.run_state == 2 else 0
                    fps = 1 / (curr_time - prev_time) if curr_time > prev_time else 60
                    delta_time = ms - self.last_time_ms
                    self.last_time_ms = ms
                    prev_time = curr_time
                    
                    key = cv2.waitKey(1) & 0xFF

                    # -------------------------------------------------------------
                    # 🚀 全局狀態機控制與空白鍵邏輯 (無視當前視窗焦點)
                    # -------------------------------------------------------------
                    if keyboard.is_pressed('space') and (time.time() - self.last_space_time > 0.5):
                        self.last_space_time = time.time()
                        
                        if self.run_state == 0:
                            if self.auto_align_chiaki(camera, yt_frame, rw, rh):
                                self.run_state = 1 
                                cv2.setWindowProperty("AutoPlatinum Supervisor", cv2.WND_PROP_TOPMOST, 1)
                                cv2.setWindowProperty("AutoPlatinum Supervisor", cv2.WND_PROP_TOPMOST, 0)
                                
                        elif self.run_state == 1:
                            page.evaluate("document.querySelector('video').play();")
                            start_tick = cv2.getTickCount()
                            self.run_state = 2
                            print("▶️ 自動駕駛啟動！(按鍵已在背景悄悄注入 Chiaki)")
                                
                        elif self.run_state == 2:
                            page.evaluate("document.querySelector('video').pause();")
                            self.release_all_keys()
                            self.run_state = 1 
                            self.is_auto_paused = False       # 🎯 重置自動暫停狀態
                            self.desync_start_time = 0.0      # 🎯 重置不同步計時器
                            print("⏸️ 緊急煞車！影片已暫停。")

                    yt_gray = cv2.cvtColor(yt_frame, cv2.COLOR_BGR2GRAY)
                    
                    # 🎯 預先初始化變數，防止等待狀態下繪圖報錯
                    mask = None
                    roi_box = None
                    chiaki_pos = None
                    
                    # --- 只有在運行狀態才進行畫面辨識 ---
                    if self.run_state == 2:
                        # 👨‍🏫 教練邏輯 (YouTube 視窗)
                        yt_pos, mask, roi_box = self.get_hsv_cursor_pos(yt_frame, rw, rh, self.anchor_pos)
                        
                        if self.waiting_tpl is not None:
                            wait_pos = self.detect_template(yt_gray, self.waiting_tpl, threshold=0.75)
                            if wait_pos:
                                if not self.is_waiting:
                                    self.is_waiting = True
                                    print(f"[YT {int(ms)}ms] Waiting 開始，排入點擊。")
                                    self.click_count += 1
                                    self.ignore_next_scene_click = True
                                    self.pending_clicks.append({"pos": yt_pos if yt_pos else wait_pos})
                            else:
                                self.is_waiting = False
                        
                        curr_hash_np = self.get_sparse_hash(yt_gray) 
                        curr_hash = torch.from_numpy(curr_hash_np).to(self.device)
                        
                        if self.last_global_hash is not None:
                            diff_global = torch.abs(curr_hash - self.last_global_hash)
                            if (torch.count_nonzero(diff_global > 25).item() / curr_hash.numel()) > 0.08:
                                self.stable_frames_count = 0
                                if not self.is_transitioning:
                                    self.is_transitioning = True
                                    self.transition_start_ms = ms
                                    self.scene_change_count += 1
                                    if self.ignore_next_scene_click:
                                        self.ignore_next_scene_click = False
                                    else:
                                        print(f"[YT {int(ms)}ms] 場景切換，排入點擊。")
                                        self.click_count += 1
                                        self.pending_clicks.append({"pos": yt_pos})
                            else:
                                if self.is_transitioning:
                                    self.stable_frames_count += 1
                                    if self.stable_frames_count >= 5 or (ms - self.transition_start_ms) > 1500:
                                        self.is_transitioning = False
                        self.last_global_hash = curr_hash

                        if yt_pos and not self.is_transitioning:
                            dist = math.hypot(yt_pos[0]-self.anchor_pos[0], yt_pos[1]-self.anchor_pos[1]) if self.anchor_pos else 999
                            if dist <= (rw * 0.002): 
                                self.stable_hover_count += 1 
                                if self.stable_hover_count > 5:
                                    if roi_box and self.last_full_gray_np is not None and self.click_cooldown <= 0:
                                        x1, y1, x2, y2 = roi_box
                                        c_patch = cv2.GaussianBlur(yt_gray[y1:y2, x1:x2].copy(), (5, 5), 0)
                                        p_patch = cv2.GaussianBlur(self.last_full_gray_np[y1:y2, x1:x2].copy(), (5, 5), 0)
                                        
                                        ph, pw = c_patch.shape
                                        ix1, iy1 = max(0, pw//2 - 15), max(0, ph//2 - 15)
                                        ix2, iy2 = min(pw, pw//2 + 15), min(ph, ph//2 + 15)
                                        c_patch[iy1:iy2, ix1:ix2] = 0 
                                        p_patch[iy1:iy2, ix1:ix2] = 0 
                                        
                                        c_tsr = torch.from_numpy(c_patch.astype(np.int16)).to(self.device)
                                        p_tsr = torch.from_numpy(p_patch.astype(np.int16)).to(self.device)
                                        
                                        changed_pixels = torch.count_nonzero(torch.abs(c_tsr - p_tsr) > 15).item()
                                        if changed_pixels > 10: 
                                            print(f"[YT {int(ms)}ms] 實體機關作動，排入點擊。")
                                            self.click_count += 1
                                            self.pending_clicks.append({"pos": yt_pos})
                                            self.click_cooldown = 100 
                            else:
                                self.stable_hover_count = 0  
                                
                        self.anchor_pos = yt_pos
                        
                        # 🧑‍🎓 學生邏輯 (Chiaki 背景操作)
                        if self.chiaki_x > 0:
                            chiaki_frame = full_frame[self.chiaki_y:self.chiaki_y+rh, self.chiaki_x:self.chiaki_x+rw]
                            chiaki_gray = cv2.cvtColor(chiaki_frame, cv2.COLOR_BGR2GRAY)
                            
                            chiaki_pos, _, _ = self.get_hsv_cursor_pos(chiaki_frame, rw, rh, chiaki_anchor_pos)
                            chiaki_anchor_pos = chiaki_pos
                            
                            if chiaki_pos:
                                target_pos = self.pending_clicks[0]["pos"] if self.pending_clicks else yt_pos
                                if target_pos:
                                    tx, ty = target_pos
                                    cx, cy = chiaki_pos
                                    x_aligned, y_aligned = False, False
                                    
                                    # 🏎️ 異步 PID 導航
                                    dx = tx - cx
                                    if dx > self.deadzone:
                                        self.update_key_bg('right', True)
                                        self.update_key_bg('left', False)
                                    elif dx < -self.deadzone:
                                        self.update_key_bg('left', True)
                                        self.update_key_bg('right', False)
                                    else:
                                        self.update_key_bg('right', False)
                                        self.update_key_bg('left', False)
                                        x_aligned = True

                                    dy = ty - cy
                                    if dy > self.deadzone:
                                        self.update_key_bg('down', True)
                                        self.update_key_bg('up', False)
                                    elif dy < -self.deadzone:
                                        self.update_key_bg('up', True)
                                        self.update_key_bg('down', False)
                                    else:
                                        self.update_key_bg('down', False)
                                        self.update_key_bg('up', False)
                                        y_aligned = True

                                    # 💥 空間座標點擊
                                    if self.pending_clicks and x_aligned and y_aligned:
                                        self.release_all_keys()
                                        print(f"💥 [Chiaki] 對齊 {target_pos} 執行點擊！")
                                        self.update_key_bg('enter', True)
                                        time.sleep(0.05)
                                        self.update_key_bg('enter', False)
                                        self.pending_clicks.pop(0)
                                else:
                                    self.release_all_keys()
                            else:
                                self.release_all_keys()
                                
                            # ---------------------------------------------------------
                            # 🔗 終極自動同步 (Rubber Band & Scene Sync) 邏輯
                            # ---------------------------------------------------------
                            dist_diff = math.hypot(yt_pos[0]-chiaki_pos[0], yt_pos[1]-chiaki_pos[1]) if yt_pos and chiaki_pos else 0
                            
                            chiaki_hash_np = self.get_sparse_hash(chiaki_gray)
                            c_hash_tsr = torch.from_numpy(chiaki_hash_np).to(self.device)
                            
                            cross_diff = torch.abs(curr_hash - c_hash_tsr)
                            # 🛠️ 修正 2：跨平台色差容忍度必須大於對齊時的基礎誤差 46 (15 -> 55, 0.05 -> 0.15)
                            is_scene_different = (torch.count_nonzero(cross_diff > 55).item() / curr_hash.numel()) > 0.15
                            
                            is_scene_desynced_long = False
                            
                            if is_scene_different:
                                if self.desync_start_time == 0.0:
                                    self.desync_start_time = curr_time
                                elif curr_time - self.desync_start_time > 2.0: 
                                    is_scene_desynced_long = True
                            else:
                                self.desync_start_time = 0.0 
                                
                            # 🛠️ 修正 3：放寬游標跟隨的橡皮筋距離 (80 -> 120)，避免初期稍微落後就死鎖
                            should_pause_video = len(self.pending_clicks) > 0 or dist_diff > 120 or is_scene_desynced_long
                            
                            if should_pause_video and not self.is_auto_paused:
                                page.evaluate("document.querySelector('video').pause();")
                                self.is_auto_paused = True
                                
                                if is_scene_desynced_long:
                                    reason = "場景切換未同步(>2秒)"
                                elif len(self.pending_clicks) > 0:
                                    reason = "處理剩餘點擊"
                                else:
                                    reason = "等待游標跟上"
                                    
                                print(f"⏸️ [自動同步] {reason}，暫停影片等待 Chiaki...")
                                
                            elif not should_pause_video and self.is_auto_paused:
                                page.evaluate("document.querySelector('video').play();")
                                self.is_auto_paused = False
                                self.desync_start_time = 0.0
                                print("▶️ [自動同步] 雙端場景與游標已完美對齊，恢復影片播放！")

                    self.last_full_gray_np = yt_gray.copy()
                    
                    display = self.draw_hud_overlay(yt_frame, mask, roi_box, self.anchor_pos, chiaki_pos, fps)
                    
                    self.click_cooldown = max(0, self.click_cooldown - delta_time)
                    cv2.imshow("AutoPlatinum Supervisor", display)
                    if key == ord('q'): break

            self.release_all_keys()
            browser.close()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    agent = AutoPlatinumHand("https://www.youtube.com/watch?v=7K_NimshHUI")
    agent.process_with_playwright()