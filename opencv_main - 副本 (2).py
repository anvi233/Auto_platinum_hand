import cv2
import numpy as np
import math
import time
import bettercam
from playwright.sync_api import sync_playwright

class AutoPlatinumHand:
    def __init__(self, youtube_url, hex_color="#f2b877", ahk_out_path="AutoPlay_Macro.ahk", 
                 cursor_path='cursor.png', waiting_path='waiting.png'):
        self.youtube_url = youtube_url
        self.ahk_out_path = ahk_out_path
        
        # 1. 資源與模板載入
        self.cursor_tpl = cv2.imread(cursor_path, 0)
        self.waiting_tpl = cv2.imread(waiting_path, 0)
        
        # 2. HSV 定位設定
        self.lower_hsv, self.upper_hsv = self.hex_to_hsv_fuzzy(hex_color, h_tol=10, s_tol=40, v_tol=40)
        
        # 3. 狀態機與計數器
        self.state = "INIT"
        self.is_initialized = False 
        self.anchor_pos = None      
        self.move_start_pos = None  
        self.last_time_ms = 0.0
        self.click_cooldown = 0.0
        self.scene_change_count = 0
        self.click_count = 0
        
        # --- [新增] 場景防抖控制變量 ---
        self.last_global_hash = None
        self.is_transitioning = False
        self.transition_start_ms = 0.0
        self.stable_frames_count = 0
        self.required_stable_frames = 5
        self.sample_step = 20 # 每 20px 採樣
        
        # 數據緩存
        self.action_log = []
        self.last_action_timestamp = 0.0
        self.last_local_patch = None

    def hex_to_hsv_fuzzy(self, hex_code, h_tol, s_tol, v_tol):
        hex_code = hex_code.lstrip('#')
        rgb = tuple(int(hex_code[i:i+2], 16) for i in (0, 2, 4))
        bgr_pixel = np.uint8([[[rgb[2], rgb[1], rgb[0]]]])
        hsv_pixel = cv2.cvtColor(bgr_pixel, cv2.COLOR_BGR2HSV)[0][0]
        h, s, v = int(hsv_pixel[0]), int(hsv_pixel[1]), int(hsv_pixel[2])
        return np.array([max(0, h - h_tol), max(0, s - s_tol), max(0, v - v_tol)]), \
               np.array([min(179, h + h_tol), min(255, s + s_tol), min(255, v + v_tol)])

    def get_sparse_hash(self, gray_frame):
        """[新增] 稀疏採樣：中值濾波後按間隔取樣"""
        denoised = cv2.medianBlur(gray_frame, 5)
        return denoised[::self.sample_step, ::self.sample_step].flatten().astype(np.int16)

    def calculate_acceleration_time(self, pixel_delta, max_pixels):
        ratio = abs(pixel_delta) / max_pixels
        return int(80 + 2800 * (ratio ** 0.75))

    def detect_template(self, gray_frame, template, threshold=0.75):
        if template is None or gray_frame is None or gray_frame.size == 0: return None
        res = cv2.matchTemplate(gray_frame, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        return (max_loc[0] + template.shape[1]//2, max_loc[1] + template.shape[0]//2) if max_val > threshold else None

    def get_hsv_cursor_pos(self, frame, frame_w, frame_h):
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        if self.anchor_pos:
            r = int(frame_h / 6)
            cx, cy = self.anchor_pos
            x1, y1 = max(0, cx - r), max(0, cy - r)
            x2, y2 = min(frame_w, cx + r), min(frame_h, cy + r)
            mask = cv2.inRange(hsv_frame[y1:y2, x1:x2], self.lower_hsv, self.upper_hsv)
            conts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if conts:
                M = cv2.moments(max(conts, key=cv2.contourArea))
                if M["m00"] > 8:
                    return (int(M["m10"]/M["m00"])+x1, int(M["m01"]/M["m00"])+y1), mask, (x1, y1, x2, y2)
        mask = cv2.inRange(hsv_frame, self.lower_hsv, self.upper_hsv)
        conts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if conts:
            M = cv2.moments(max(conts, key=cv2.contourArea))
            if M["m00"] > 8: return (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])), mask, None
        return None, mask, None

    def draw_hud_overlay(self, frame, hsv_mask, roi_box, anchor_pos, fps_val):
        overlay = frame.copy()
        h, w = frame.shape[:2]
        if hsv_mask is not None and roi_box:
            x1, y1, x2, y2 = roi_box
            overlay[y1:y2, x1:x2][hsv_mask > 0] = [0, 255, 0]
        if anchor_pos:
            cx, cy = anchor_pos
            cv2.line(overlay, (cx, 0), (cx, h), (0, 255, 255), 1)
            cv2.line(overlay, (0, cy), (w, cy), (0, 255, 255), 1)
        if roi_box: cv2.rectangle(overlay, (roi_box[0], roi_box[1]), (roi_box[2], roi_box[3]), (0, 165, 255), 2)
        display = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        info = f"FPS: {int(fps_val)}  SCENES: {self.scene_change_count}  CLICKS: {self.click_count}"
        cv2.putText(display, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return display

    def log_click_action(self, ms, reason):
        wait = int(ms - self.last_action_timestamp) if self.last_action_timestamp > 0 else 500
        self.action_log.append({"type": "CLICK", "key": "Enter", "wait_before_next_ms": wait})
        self.click_count += 1
        self.last_action_timestamp = ms
        print(f"[LOG] CLICK ({reason}) | Total: {self.click_count}")

    def process_with_playwright(self, start_time_sec=65, max_duration_sec=30):
        with sync_playwright() as p:
            # 1. 遠端連接與物理校準 (地基穩定性)
            print("🔗 正在連接 Chrome (port 9222)...")
            browser = p.chromium.connect_over_cdp("http://localhost:9222")
            page = next((pg for pg in browser.contexts[0].pages if "youtube" in pg.url), browser.contexts[0].pages[0])
            video = page.wait_for_selector("video")
            
            # 強制定位影片時間
            page.evaluate(f"document.querySelector('video').pause(); document.querySelector('video').currentTime = {start_time_sec};")
            
            # 獲取 Playwright 標準截圖
            p_bytes = video.screenshot(type="jpeg", quality=100)
            p_frame = cv2.imdecode(np.frombuffer(p_bytes, np.uint8), cv2.IMREAD_COLOR)
            
            # GPU 實例與自動對位
            camera = bettercam.create() 
            full_init = camera.grab()
            full_init_bgr = cv2.cvtColor(full_init, cv2.COLOR_RGB2BGR)
            res = cv2.matchTemplate(full_init_bgr, p_frame, cv2.TM_CCOEFF_NORMED)
            _, _, _, max_loc = cv2.minMaxLoc(res)
            
            # 識別彩框邊界 (確保抓到大面積播放器)
            gray_p = cv2.cvtColor(p_frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(cv2.medianBlur(gray_p, 35), 30, 100)
            cnts, _ = cv2.findContours(cv2.dilate(edges, None, iterations=2), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            valid_rects = [cv2.boundingRect(c) for c in cnts if cv2.boundingRect(c)[2] > 300]
            rx, ry, rw, rh = max(valid_rects, key=lambda r: r[2]*r[3]) if valid_rects else (0,0,p_frame.shape[1],p_frame.shape[0])

            real_x, real_y = max_loc[0] + rx, max_loc[1] + ry
            print(f"🎯 鎖定成功！座標: ({real_x}, {real_y}), 尺寸: {rw}x{rh}")

            cv2.namedWindow("Platinum Vision", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Platinum Vision", rw, rh)
            
            # 初始化計數與時間
            recording, start_tick, prev_time = False, 0, time.time()
            self.frame_count = 0
            self.last_roi_hash = None
            self.last_global_hash = None

            while True:
                full_rgb = camera.grab()
                if full_rgb is None: continue
                
                # 內存裁切：100% 同步
                frame_rgb = full_rgb[real_y : real_y + rh, real_x : real_x + rw]
                frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                self.frame_count += 1
                curr_time = time.time()
                ms = ((cv2.getTickCount() - start_tick) / cv2.getTickFrequency()) * 1000 if recording else 0
                fps = 1 / (curr_time - prev_time) if curr_time > prev_time else 60
                delta_time = ms - self.last_time_ms
                self.last_time_ms = ms
                prev_time = curr_time
                
                key = cv2.waitKey(1) & 0xFF

                # --- 1. 場景防抖合併算法 (宏觀) ---
                curr_hash = self.get_sparse_hash(gray) # 每 20px 採樣
                if self.last_global_hash is not None and recording:
                    diff_global = np.abs(curr_hash - self.last_global_hash)
                    if (np.count_nonzero(diff_global > 3) / len(curr_hash)) > 0.01:
                        self.stable_frames_count = 0
                        if not self.is_transitioning:
                            self.is_transitioning = True
                            self.transition_start_ms = ms
                            self.scene_change_count += 1
                            self.log_click_action(0, "場景大變換")
                            self.is_initialized = False 
                    else:
                        if self.is_transitioning:
                            self.stable_frames_count += 1
                            if self.stable_frames_count >= 5 or (ms - self.transition_start_ms) > 1500:
                                self.is_transitioning = False
                self.last_global_hash = curr_hash

                # --- 2. 指針追蹤與行為狀態判定 ---
                pos, mask, roi_box = self.get_hsv_cursor_pos(frame, rw, rh)
                
                if self.is_transitioning:
                    curr_state = "SCENE_CHANGE"
                    self.last_roi_hash = None # 場景切換時必須重置
                elif pos:
                    # 計算位移
                    dist = math.hypot(pos[0]-self.anchor_pos[0], pos[1]-self.anchor_pos[1]) if self.anchor_pos else 999
                    
                    # 判斷是否為「移動中」
                    if dist > (rw * 0.005): 
                        curr_state = "MOVE"
                        self.last_roi_hash = None    # 💥 關鍵：只要在移動，就持續清空監聽基準
                        self.stable_hover_count = 0  # 重置穩定計數
                    else:
                        curr_state = "HOVER"
                        self.stable_hover_count += 1 # 累計靜止幀數
                        
                        # --- 核心：ROI 像素監聽 (僅在穩定靜止 3 幀後才激活) ---
                        if roi_box and self.stable_hover_count > 3 and recording:
                            x1, y1, x2, y2 = roi_box
                            patch = gray[y1:y2, x1:x2].copy()
                            ph, pw = patch.shape
                            
                            # 💥 挖空中心 30x30 指針忽略區 (控制變量)
                            ix1, iy1 = max(0, pw//2 - 15), max(0, ph//2 - 15)
                            ix2, iy2 = min(pw, pw//2 + 15), min(ph, ph//2 + 15)
                            patch[iy1:iy2, ix1:ix2] = 0 
                            
                            # 全像素採樣 (Step=1)
                            curr_roi_hash = patch.astype(np.int16)
                            
                            if self.last_roi_hash is not None and self.click_cooldown <= 0:
                                # 計算絕對差異矩陣
                                diff_matrix = np.abs(curr_roi_hash - self.last_roi_hash)
                                
                                # 💥 優化判定：下調閾值到 4，但要求至少有 5 個像素點同時變化
                                # 這樣可以防止單個像素熱噪點觸發，又能抓到細微動作
                                if np.count_nonzero(diff_matrix > 4) > 2:
                                    self.log_click_action(ms, "微觀環境變化 (ROI)")
                                    self.click_cooldown = 1200 # 略微縮短點擊冷卻
                                    
                            self.last_roi_hash = curr_roi_hash
                    
                    self.anchor_pos = pos
                else:
                    curr_state = "LOST"
                    self.last_roi_hash = None

                # --- 3. HUD 與 GUI 統計面板 ---
                display = self.draw_hud_overlay(frame, mask, roi_box, pos if pos else self.anchor_pos, fps)
                
                # 右側統計面板 (黑底白字)
                panel_x = rw - 180
                cv2.rectangle(display, (panel_x, 0), (rw, 100), (0, 0, 0), -1)
                cv2.putText(display, f"STATE: {curr_state}", (panel_x+5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(display, f"SCENES: {self.scene_change_count}", (panel_x+5, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                cv2.putText(display, f"CLICKS: {self.click_count}", (panel_x+5, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                if not recording:
                    cv2.putText(display, "READY - PRESS SPACE", (rw//2-120, rh//2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                    if key == ord(' '):
                        page.evaluate("document.querySelector('video').play();")
                        start_tick, recording = cv2.getTickCount(), True
                
                self.click_cooldown = max(0, self.click_cooldown - delta_time)
                cv2.imshow("Platinum Vision", display)
                if key == ord('q'): break

            browser.close()
            cv2.destroyAllWindows()
            self.generate_ahk()

    def generate_ahk(self):
        # AHK 邏輯保持 v9.2 穩定版不變...
        with open(self.ahk_out_path, "w", encoding="utf-8") as f:
            f.write("#NoEnv\nSetCapsLockState, AlwaysOff\nWinGet, rid, List, ahk_exe chiaki-ng.exe\ntid := rid1\n$CapsLock::\nPlaying := !Playing\nSetTimer, PlayMacro, -1\nreturn\nPlayMacro:\n")
            for a in self.action_log:
                f.write(f'DllCall("Sleep", "Uint", {max(0, a["wait_before_next_ms"])})\n')
                if a["type"] == "MOVE":
                    d = a["details"]
                    for axis in [("x_dir", "x_hold_ms"), ("y_dir", "y_hold_ms")]:
                        if d[axis[1]] > 0:
                            f.write(f'ControlSend, , {{{d[axis[0]]} down}}, ahk_id %tid%\nDllCall("Sleep", "Uint", {d[axis[1]]})\nControlSend, , {{{d[axis[0]]} up}}, ahk_id %tid%\n')
                else:
                    f.write(f'ControlSend, , {{Enter down}}, ahk_id %tid%\nDllCall("Sleep", "Uint", 50)\nControlSend, , {{Enter up}}, ahk_id %tid%\n')
            f.write("return\n")

if __name__ == "__main__":
    agent = AutoPlatinumHand("https://www.youtube.com/watch?v=7K_NimshHUI")
    agent.process_with_playwright()