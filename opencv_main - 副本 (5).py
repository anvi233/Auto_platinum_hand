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
import keyboard # 保留用於背景注入，但不監聽開關
from playwright.sync_api import sync_playwright

# 強制開啟 DPI 感知，確保抓圖與視窗絕對座標 1:1 對應
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
        self.stable_hover_count = 0 
        
        # 場景控制
        self.last_global_hash = None
        self.is_transitioning = False
        self.is_waiting = False
        self.ignore_next_scene_click = False
        self.transition_start_ms = 0.0
        self.stable_frames_count = 0
        self.last_full_gray_np = None

        # 導航控制器
        self.key_states = {"up": False, "down": False, "left": False, "right": False, "enter": False}
        self.pending_clicks = []
        self.deadzone = 8
        
        self.VK_MAP = {
            'up': win32con.VK_UP, 'down': win32con.VK_DOWN,
            'left': win32con.VK_LEFT, 'right': win32con.VK_RIGHT,
            'enter': win32con.VK_RETURN
        }
        
        # 運行狀態機 (0: 等待對齊, 1: 鎖定待機, 2: 運行中)
        self.run_state = 0 
        self.chiaki_x, self.chiaki_y = 0, 0
        self.chiaki_w, self.chiaki_h = 0, 0
        self.chiaki_hwnd = None
        
        self.last_space_time = 0.0
        self.is_auto_paused = False
        self.desync_start_time = 0.0

    # ---------------------------------------------------------
    # 統一的遊戲區域提取工具 (去黑框)
    # ---------------------------------------------------------
    def _extract_game_roi(self, frame):
        """核心戰力：鎖定彩色遊戲區域，並強制避開 45px 標題列。"""
        h, w = frame.shape[:2]
        # 🎯 避開標題列策略：頂部裁切 45 像素，四周裁切 5 像素
        top_m, side_m = 45, 5
        if h <= top_m: return 0, 0, w, h
        
        # 1. 建立內部觀測區
        roi_view = frame[top_m:h-side_m, side_m:w-side_m]
        gray = cv2.cvtColor(roi_view, cv2.COLOR_BGR2GRAY)
        
        # 2. 二值化：亮度 > 30 視為內容
        _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
        
        # 3. 尋找最大輪廓
        cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            rx, ry, rw, rh = cv2.boundingRect(c)
            # 🎯 還原座標：加上偏移量，這就是相對於視窗左上角的固定起點
            return rx + side_m, ry + top_m, rw, rh

        return 0, 0, w, h

    def update_key_bg(self, key_str, press):
        if not self.chiaki_hwnd: return
        vk = self.VK_MAP.get(key_str)
        if not vk: return
        if self.key_states[key_str] != press:
            if press: win32api.PostMessage(self.chiaki_hwnd, win32con.WM_KEYDOWN, vk, 0)
            else: win32api.PostMessage(self.chiaki_hwnd, win32con.WM_KEYUP, vk, 0)
            self.key_states[key_str] = press

    def release_all_keys(self):
        for k in self.key_states.keys(): self.update_key_bg(k, False)

    def hex_to_hsv_fuzzy(self, hex_code, h_tol, s_tol, v_tol):
        hex_code = hex_code.lstrip('#')
        rgb = tuple(int(hex_code[i:i+2], 16) for i in (0, 2, 4))
        bgr_pixel = np.uint8([[[rgb[2], rgb[1], rgb[0]]]])
        hsv_pixel = cv2.cvtColor(bgr_pixel, cv2.COLOR_BGR2HSV)[0][0]
        h, s, v = int(hsv_pixel[0]), int(hsv_pixel[1]), int(hsv_pixel[2])
        return np.array([max(0, h - h_tol), max(0, s - s_tol), max(0, v - v_tol)]), \
               np.array([min(179, h + h_tol), min(255, s + s_tol), min(255, v + v_tol)])

    def get_sparse_hash(self, gray_frame):
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
            cv2.line(overlay, (cx, cy), (cx, 0), (0, 255, 255), 1); cv2.line(overlay, (cx, cy), (cx, h), (0, 255, 255), 1)
            cv2.line(overlay, (cx, cy), (0, cy), (0, 255, 255), 1); cv2.line(overlay, (cx, cy), (w, cy), (0, 255, 255), 1)
        if chiaki_pos:
            cv2.drawMarker(overlay, chiaki_pos, (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
        display = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        info = f"FPS: {int(fps_val)}  SCENES: {self.scene_change_count}  QUEUE: {len(self.pending_clicks)}"
        cv2.putText(display, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        status_msg = ""
        if self.run_state == 0: status_msg = "PRESS [SPACE] TO ALIGN CHIAKI WINDOW"
        elif self.run_state == 1: status_msg = "ALIGNED! PRESS [SPACE] TO START"
        if status_msg: cv2.putText(display, status_msg, (w//2-250, h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        return display

    def auto_align_chiaki(self, camera, yt_frame, rw, rh):
        """鎖定 Chiaki 視窗位置，並計算一次性的固定偏移量。"""
        self.yt_w, self.yt_h = rw, rh
        hwnds = []
        # 🎯 精確過濾：排除 PowerShell 終端
        def enum_cb(hwnd, param):
            if win32gui.IsWindowVisible(hwnd):
                title = win32gui.GetWindowText(hwnd)
                if "chiaki" in title.lower() and "powershell" not in title.lower():
                    param.append(hwnd)
        win32gui.EnumWindows(enum_cb, hwnds)
        
        if not hwnds:
            print("❌ 找不到有效的 Chiaki 視窗！")
            return False
            
        self.chiaki_hwnd = hwnds[0]
        rect = win32gui.GetWindowRect(self.chiaki_hwnd)
        wx, wy, ww, wh = rect[0], rect[1], rect[2]-rect[0], rect[3]-rect[1]
        
        full_grab = camera.grab()
        if full_grab is None: return False
        window_img = full_grab[wy:wy+wh, wx:wx+ww]
        
        # 🎯 執行一次性 ROI 分析
        cx, cy, cw, ch = self._extract_game_roi(window_img)
        
        # 🎯 存儲固定偏移常數，防止循環中累加導致畫面消失
        self.chiaki_offset_x, self.chiaki_offset_y = cx, cy
        self.chiaki_w, self.chiaki_h = cw, ch
        # 這裡的 chiaki_x/y 用作狀態檢查
        self.chiaki_x, self.chiaki_y = wx + cx, wy + cy 
        
        print(f"✅ 鎖定成功！偏移: x={cx}, y={cy}, 尺寸: {cw}x{ch}")
        return True

    def process_with_playwright(self, start_time_sec=35):
        with sync_playwright() as p:
            browser = p.chromium.connect_over_cdp("http://localhost:9222")
            page = next((pg for pg in browser.contexts[0].pages if "youtube" in pg.url), browser.contexts[0].pages[0])
            video = page.wait_for_selector("video")
            page.evaluate(f"document.querySelector('video').pause(); document.querySelector('video').currentTime = {start_time_sec};")
            
            p_bytes = video.screenshot(type="jpeg", quality=100)
            p_frame = cv2.imdecode(np.frombuffer(p_bytes, np.uint8), cv2.IMREAD_COLOR)
            rx, ry, rw, rh = self._extract_game_roi(p_frame)
            
            camera = dxcam.create(output_idx=0, output_color="BGR") 
            full_init = camera.grab()
            while full_init is None: full_init = camera.grab()
            
            res = cv2.matchTemplate(full_init, p_frame, cv2.TM_CCOEFF_NORMED)
            _, _, _, max_loc1 = cv2.minMaxLoc(res)
            yt_x, yt_y = max_loc1[0] + rx, max_loc1[1] + ry
            print(f"🎯 鎖定 YouTube 遊戲區！座標: ({yt_x}, {yt_y}), 尺寸: {rw}x{rh}")

            cv2.namedWindow("AutoPlatinum Supervisor", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("AutoPlatinum Supervisor", rw, rh)
            
            # --- 迴圈啟動前初始化防崩潰變數 ---
            start_tick, prev_time = 0, time.time()
            chiaki_anchor_pos = None
            self.anchor_pos = None
            self.last_full_gray_np = None
            self._diag_tick = 0

            with torch.no_grad():
                while True:
                    full_frame = camera.grab()
                    if full_frame is None: continue
                    
                    yt_frame = full_frame[yt_y:yt_y+rh, yt_x:yt_x+rw]
                    yt_gray = cv2.cvtColor(yt_frame, cv2.COLOR_BGR2GRAY)
                    curr_time = time.time()
                    
                    # 基礎時間與 FPS 計算
                    fps = 1 / (curr_time - prev_time) if curr_time > prev_time else 60
                    ms = ((cv2.getTickCount() - start_tick) / cv2.getTickFrequency()) * 1000 if self.run_state == 2 else 0
                    delta_time = ms - self.last_time_ms
                    self.last_time_ms, prev_time = ms, curr_time
                    
                    # 空白鍵狀態切換邏輯
                    key = cv2.waitKey(1) & 0xFF
                    if key == 32 and (time.time() - self.last_space_time > 0.5):
                        self.last_space_time = time.time()
                        print(f"⌨️ [Space] 偵測成功，當前狀態: {self.run_state}")
                        if self.run_state == 0:
                            if self.auto_align_chiaki(camera, yt_frame, rw, rh): self.run_state = 1
                        elif self.run_state == 1:
                            page.evaluate("document.querySelector('video').play();")
                            start_tick, self.run_state = cv2.getTickCount(), 2
                        elif self.run_state == 2:
                            page.evaluate("document.querySelector('video').pause();")
                            self.release_all_keys(); self.run_state = 1

                    # 繪圖變數初始化
                    mask, roi_box, chiaki_pos = None, None, None
                    
                    if self.run_state == 2:
                        # 1. 🔍 YouTube 端識別與【預判點擊】邏輯
                        yt_pos, mask, roi_box = self.get_hsv_cursor_pos(yt_frame, rw, rh, self.anchor_pos)
                        
                        # --- [預判: Waiting 指針] ---
                        if self.waiting_tpl is not None:
                            wait_pos = self.detect_template(yt_gray, self.waiting_tpl, threshold=0.75)
                            if wait_pos:
                                if not yt_pos: yt_pos = wait_pos 
                                if not self.is_waiting:
                                    self.is_waiting = True
                                    self.click_count += 1
                                    # 🎯 核心邏輯：如果是 Waiting 觸發，之後的一次場景變化不算點擊
                                    self.ignore_next_scene_click = True
                                    self.pending_clicks.append({"pos": yt_pos})
                                    print(f"🔥 [預判] Waiting 出現，排入點擊。座標: {yt_pos}")
                            else: self.is_waiting = False
                        
                        # 座標繼承 (防止閃爍)
                        if not yt_pos: yt_pos = self.anchor_pos
                        self.anchor_pos = yt_pos

                        # --- [預判: 場景開始變化] (合併過度) ---
                        curr_hash_np = self.get_sparse_hash(yt_gray) 
                        curr_hash = torch.from_numpy(curr_hash_np).to(self.device)
                        if self.last_global_hash is not None:
                            diff_global = torch.abs(curr_hash - self.last_global_hash)
                            # 🎯 判定場景「剛開始」變化
                            if (torch.count_nonzero(diff_global > 25).item() / curr_hash.numel()) > 0.08:
                                if not self.is_transitioning:
                                    self.is_transitioning, self.transition_start_ms, self.scene_change_count = True, ms, self.scene_change_count + 1
                                    # 🎯 檢查是否被 Waiting 屏蔽
                                    if self.ignore_next_scene_click:
                                        self.ignore_next_scene_click = False 
                                        print(f"🛡️ [屏蔽] Waiting 引起的場景變化，忽略點擊。")
                                    else:
                                        self.click_count += 1
                                        self.pending_clicks.append({"pos": yt_pos})
                                        print(f"🔥 [預判] 場景起始變換，排入點擊。座標: {yt_pos}")
                            else:
                                if self.is_transitioning:
                                    self.stable_frames_count += 1
                                    if self.stable_frames_count >= 5 or (ms - self.transition_start_ms) > 1500: self.is_transitioning = False
                        self.last_global_hash = curr_hash

                        # --- [預判: 聚焦區域(ROI)剛開始變化] ---
                        if yt_pos and not self.is_transitioning:
                            dist = math.hypot(yt_pos[0]-self.anchor_pos[0], yt_pos[1]-self.anchor_pos[1]) if self.anchor_pos else 999
                            if dist <= (rw * 0.002): # 指針穩定靜止
                                self.stable_hover_count += 1 
                                if self.stable_hover_count > 5:
                                    if roi_box and self.last_full_gray_np is not None and self.click_cooldown <= 0:
                                        x1, y1, x2, y2 = roi_box
                                        c_patch = cv2.GaussianBlur(yt_gray[y1:y2, x1:x2].copy(), (5, 5), 0)
                                        p_patch = cv2.GaussianBlur(self.last_full_gray_np[y1:y2, x1:x2].copy(), (5, 5), 0)
                                        # 排除中心噪點
                                        ph, pw = c_patch.shape
                                        ix1, iy1 = max(0, pw//2-15), max(0, ph//2-15)
                                        ix2, iy2 = min(pw, pw//2+15), min(ph, ph//2+15)
                                        c_patch[iy1:iy2, ix1:ix2] = 0; p_patch[iy1:iy2, ix1:ix2] = 0 
                                        c_tsr, p_tsr = torch.from_numpy(c_patch.astype(np.int16)).to(self.device), torch.from_numpy(p_patch.astype(np.int16)).to(self.device)
                                        # 🎯 判定 ROI 剛開始變化
                                        if torch.count_nonzero(torch.abs(c_tsr - p_tsr) > 15).item() > 10: 
                                            self.click_count += 1
                                            self.pending_clicks.append({"pos": yt_pos})
                                            self.click_cooldown = 100 
                                            print(f"🔥 [預判] 聚焦區域變化，排入點擊。座標: {yt_pos}")
                            else: self.stable_hover_count = 0 

                        # 2. 🎮 Chiaki 操作與【暫停/自動解除】邏輯
                        if self.chiaki_hwnd:
                            rect = win32gui.GetWindowRect(self.chiaki_hwnd)
                            chiaki_raw = full_frame[rect[1]+self.chiaki_offset_y : rect[1]+self.chiaki_offset_y+self.chiaki_h, 
                                                    rect[0]+self.chiaki_offset_x : rect[0]+self.chiaki_offset_x+self.chiaki_w]
                            chiaki_res = cv2.resize(chiaki_raw, (rw, rh))
                            chiaki_gray = cv2.cvtColor(chiaki_res, cv2.COLOR_BGR2GRAY)

                            # Chiaki 指針識別 (含座標繼承)
                            chiaki_pos, _, _ = self.get_hsv_cursor_pos(chiaki_res, rw, rh, chiaki_anchor_pos)
                            if not chiaki_pos and self.waiting_tpl is not None:
                                chiaki_pos = self.detect_template(chiaki_gray, self.waiting_tpl, threshold=0.75)
                            if not chiaki_pos: chiaki_pos = chiaki_anchor_pos
                            chiaki_anchor_pos = chiaki_pos

                            # 場景比對與計時
                            c_hash_tsr = torch.from_numpy(self.get_sparse_hash(chiaki_gray)).to(self.device)
                            diff_val = torch.count_nonzero(torch.abs(curr_hash - c_hash_tsr) > 55).item() / curr_hash.numel()
                            is_scene_different = diff_val > 0.15
                            
                            # 🎯 暫停計時：場景不同持續 2 秒才觸發
                            is_scene_desynced_long = False
                            if is_scene_different:
                                if self.desync_start_time == 0.0: self.desync_start_time = time.time()
                                elif time.time() - self.desync_start_time > 2.0: is_scene_desynced_long = True
                            else:
                                self.desync_start_time = 0.0 # 畫面一致則重置

                            # 🎯 執行趕路與點擊 (即便暫停中也運行，確保清空任務)
                            current_target = self.pending_clicks[0]["pos"] if self.pending_clicks else yt_pos
                            if chiaki_pos and current_target:
                                dx, dy = current_target[0] - chiaki_pos[0], current_target[1] - chiaki_pos[1]
                                self.update_key_bg('right', dx > self.deadzone); self.update_key_bg('left', dx < -self.deadzone)
                                self.update_key_bg('down', dy > self.deadzone); self.update_key_bg('up', dy < -self.deadzone)
                                
                                if self.pending_clicks and abs(dx) <= self.deadzone and abs(dy) <= self.deadzone:
                                    print(f"🎯 [點擊達成] 目標: {current_target} | 實際: {chiaki_pos}")
                                    self.release_all_keys()
                                    win32api.PostMessage(self.chiaki_hwnd, win32con.WM_KEYDOWN, win32con.VK_RETURN, 0)
                                    time.sleep(0.05); win32api.PostMessage(self.chiaki_hwnd, win32con.WM_KEYUP, win32con.VK_RETURN, 0)
                                    self.pending_clicks.pop(0)

                            # 🎯 自動暫停/解除邏輯
                            # 暫停條件：背景不同步長達2秒 或 距離拉開
                            dist_diff = math.hypot(yt_pos[0]-chiaki_pos[0], yt_pos[1]-chiaki_pos[1]) if yt_pos and chiaki_pos else 999
                            should_pause = len(self.pending_clicks) > 0 or dist_diff > 120 or is_scene_desynced_long

                            if should_pause and not self.is_auto_paused:
                                page.evaluate("var v=document.querySelector('video'); if(v && !v.paused) v.pause();")
                                self.is_auto_paused = True
                                print(f"⏸️ [自動暫停] 原因: {'Queue' if self.pending_clicks else 'Dist' if dist_diff > 120 else 'Scene'}")
                            elif not should_pause and self.is_auto_paused:
                                # 🎯 自動恢復：場景一致且指針對齊
                                page.evaluate("var v=document.querySelector('video'); if(v && v.paused) v.play();")
                                self.is_auto_paused = False
                                print("▶️ [自動恢復] 背景與座標對齊完成")

                    self.last_full_gray_np = yt_gray.copy()
                    display = self.draw_hud_overlay(yt_frame, mask, roi_box, self.anchor_pos, chiaki_pos, fps)
                    self.click_cooldown = max(0, self.click_cooldown - delta_time)
                    cv2.imshow("AutoPlatinum Supervisor", display)
                    if key == ord('q'): break

            self.release_all_keys(); browser.close(); cv2.destroyAllWindows()

if __name__ == "__main__":
    agent = AutoPlatinumHand("https://www.youtube.com/watch?v=7K_NimshHUI")
    agent.process_with_playwright()