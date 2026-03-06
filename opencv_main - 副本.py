import cv2
import numpy as np
import math
import time
from playwright.sync_api import sync_playwright

class AutoPlatinumHand:
    def __init__(self, youtube_url, hex_color="#f2b877", ahk_out_path="AutoPlay_Macro.ahk", 
                 cursor_path='cursor.png', waiting_path='waiting.png'):
        self.youtube_url = youtube_url
        self.ahk_out_path = ahk_out_path
        
        self.cursor_tpl = cv2.imread(cursor_path, 0)
        self.waiting_tpl = cv2.imread(waiting_path, 0)
        if self.cursor_tpl is None or self.waiting_tpl is None:
            print("[警告] 找不到 cursor.png 或 waiting.png，模板匹配將失效！")
            
        self.lower_hsv, self.upper_hsv = self.hex_to_hsv_fuzzy(hex_color, h_tol=10, s_tol=40, v_tol=40)
        
        self.state = "INIT"
        self.is_initialized = False 
        self.anchor_pos = None      # 絕對錨點
        self.move_start_pos = None  
        self.hover_timer = 0.0
        self.last_time_ms = 0.0
        
        self.game_bounds = None     # 儲存純遊戲畫面的絕對坐標 (x, y, w, h)
        
        self.last_global_frame = None
        self.last_local_patch = None
        self.click_cooldown = 0.0
        
        self.action_log = []
        self.last_action_timestamp = 0.0

    def hex_to_hsv_fuzzy(self, hex_code, h_tol, s_tol, v_tol):
        hex_code = hex_code.lstrip('#')
        rgb = tuple(int(hex_code[i:i+2], 16) for i in (0, 2, 4))
        bgr_pixel = np.uint8([[[rgb[2], rgb[1], rgb[0]]]])
        hsv_pixel = cv2.cvtColor(bgr_pixel, cv2.COLOR_BGR2HSV)[0][0]
        h, s, v = int(hsv_pixel[0]), int(hsv_pixel[1]), int(hsv_pixel[2])
        return np.array([max(0, h - h_tol), max(0, s - s_tol), max(0, v - v_tol)]), np.array([min(179, h + h_tol), min(255, s + s_tol), min(255, v + v_tol)])

    def calculate_acceleration_time(self, pixel_delta, max_pixels):
        if pixel_delta == 0: return 0
        ratio = abs(pixel_delta) / max_pixels
        base_ms = 80
        max_ms = 2800 
        power = 0.75 
        return int(base_ms + max_ms * (ratio ** power))

    def detect_template(self, gray_frame, template, threshold=0.75):
        if template is None or gray_frame is None or gray_frame.size == 0: return None
        if gray_frame.shape[0] < template.shape[0] or gray_frame.shape[1] < template.shape[1]: return None
        
        res = cv2.matchTemplate(gray_frame, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        if max_val > threshold:
            return (max_loc[0] + template.shape[1]//2, max_loc[1] + template.shape[0]//2)
        return None

    def get_static_game_bounds(self, frame_raw):
        """【基石邏輯】只在初始執行一次，完美切出遊戲彩框"""
        blurred = cv2.medianBlur(frame_raw, 35)
        edges = cv2.Canny(blurred, 30, 100)
        kernel = np.ones((5,5), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=2)
        
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest)
            if w > 400 and h > 300:
                print(f"✅ 成功鎖定遊戲彩框！坐標: X={x}, Y={y}, W={w}, H={h}")
                return (x, y, w, h)
                
        print("❌ 找不到邊界，退回全圖")
        return (0, 0, frame_raw.shape[1], frame_raw.shape[0])

    def get_safe_roi(self, cx, cy, radius, max_w, max_h):
        x1, y1 = max(0, cx - radius), max(0, cy - radius)
        x2, y2 = min(max_w, cx + radius), min(max_h, cy + radius)
        return x1, y1, x2, y2

    def get_hsv_cursor_pos(self, frame, frame_w, frame_h):
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        search_area = hsv_frame
        x_offset, y_offset = 0, 0
        roi_box = None

        if self.anchor_pos is not None:
            roi_r = int(frame_h / 6)
            cx, cy = self.anchor_pos
            x1, y1, x2, y2 = self.get_safe_roi(cx, cy, roi_r, frame_w, frame_h)
            search_area = hsv_frame[y1:y2, x1:x2]
            x_offset, y_offset = x1, y1
            roi_box = (x1, y1, x2, y2)

        mask = cv2.inRange(search_area, self.lower_hsv, self.upper_hsv)
        full_mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
        if roi_box:
            full_mask[y1:y2, x1:x2] = mask
        else:
            full_mask = mask

        cursor_pos = None
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) > 10: 
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cursor_pos = (int(M["m10"] / M["m00"]) + x_offset, int(M["m01"] / M["m00"]) + y_offset)
                    
        return cursor_pos, full_mask, roi_box

    def draw_hud_overlay(self, frame, hsv_mask, roi_box, anchor_pos):
        overlay = frame.copy()
        frame_h, frame_w = frame.shape[:2]
        
        if hsv_mask is not None:
            overlay[hsv_mask > 0] = [0, 255, 0]
        
        if anchor_pos:
            cx, cy = anchor_pos
            # 8 條動態輔助線：精準打在純遊戲畫面的邊界上
            cv2.line(overlay, (cx, cy), (cx, 0), (0, 255, 255), 1)
            cv2.line(overlay, (cx, cy), (cx, frame_h), (0, 255, 255), 1)
            cv2.line(overlay, (cx, cy), (0, cy), (0, 255, 255), 1)
            cv2.line(overlay, (cx, cy), (frame_w, cy), (0, 255, 255), 1)
            cv2.line(overlay, (cx, cy), (0, 0), (255, 255, 255), 1)
            cv2.line(overlay, (cx, cy), (frame_w, 0), (255, 255, 255), 1)
            cv2.line(overlay, (cx, cy), (0, frame_h), (255, 255, 255), 1)
            cv2.line(overlay, (cx, cy), (frame_w, frame_h), (255, 255, 255), 1)

        if roi_box:
            x1, y1, x2, y2 = roi_box
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 165, 255), 2)
            cv2.putText(overlay, "Core ROI", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)

        return cv2.addWeighted(frame, 0.7, overlay, 0.4, 0)

    def log_move_action(self, start_pos, end_pos, frame_w, frame_h, current_ms):
        if start_pos is None or end_pos is None: return
        dx = end_pos[0] - start_pos[0]
        dy = end_pos[1] - start_pos[1]
        x_ms = self.calculate_acceleration_time(dx, frame_w)
        y_ms = self.calculate_acceleration_time(dy, frame_h)
        wait_time = int(current_ms - self.last_action_timestamp) if self.last_action_timestamp > 0 else 500
        action = {"type": "MOVE", "details": {"x_dir": "Right" if dx > 0 else "Left", "x_hold_ms": x_ms if abs(dx) > (frame_w * 0.01) else 0, "y_dir": "Down" if dy > 0 else "Up", "y_hold_ms": y_ms if abs(dy) > (frame_h * 0.01) else 0}, "wait_before_next_ms": wait_time}
        if action["details"]["x_hold_ms"] > 0 or action["details"]["y_hold_ms"] > 0:
            self.action_log.append(action)
            self.last_action_timestamp = current_ms
            print(f"[LOG] MOVE: {action['details']} | 距離: ({dx}, {dy})")

    def log_click_action(self, current_ms):
        wait_time = int(current_ms - self.last_action_timestamp) if self.last_action_timestamp > 0 else 500
        self.action_log.append({"type": "CLICK", "key": "Enter", "wait_before_next_ms": wait_time})
        self.last_action_timestamp = current_ms
        print(f"[LOG] CLICK")

    def process_with_playwright(self, start_time_sec=0, max_duration_sec=None):
        with sync_playwright() as p:
            try:
                browser = p.chromium.connect_over_cdp("http://localhost:9222")
                context = browser.contexts[0]
                page = context.pages[0] if context.pages else context.new_page()
            except Exception as e:
                print(f"❌ 連接失敗: {e}"); return

            page.goto(self.youtube_url)
            page.wait_for_selector("video")
            
            print("⏸️ 暫停影片於 0:34...")
            page.evaluate("document.querySelector('video').pause(); document.querySelector('video').currentTime = 34;")
            
            print("🔲 請手動將影片設為全螢幕/劇院模式，等待 5 秒讓 UI 消失...")
            time.sleep(5) 
            
            # --- 核心：提取並鎖定真正的遊戲彩框 ---
            screenshot_bytes = page.screenshot(type="jpeg", quality=100)
            frame_raw = cv2.imdecode(np.frombuffer(screenshot_bytes, np.uint8), cv2.IMREAD_COLOR)
            self.game_bounds = self.get_static_game_bounds(frame_raw)
            bx, by, bw, bh = self.game_bounds

            cv2.namedWindow("Platinum Vision", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Platinum Vision", bw, bh)
            
            print("\n==================================================")
            print("【準備就緒】")
            print("1. 遊戲邊界已永久鎖定，監視器只會顯示純淨畫面。")
            print("👉 請點擊 'Platinum Vision' 視窗，按下 [空白鍵 Space] 啟動。")
            print("==================================================")
            
            recording = False
            start_tick = 0
            fps = cv2.getTickFrequency()

            while True:
                screenshot_bytes = page.screenshot(type="jpeg", quality=60)
                frame_raw = cv2.imdecode(np.frombuffer(screenshot_bytes, np.uint8), cv2.IMREAD_COLOR)
                
                # 絕對裁切：只取遊戲畫面
                frame = frame_raw[by : by+bh, bx : bx+bw]
                frame_h, frame_w = frame.shape[:2]
                
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                key = cv2.waitKey(1) & 0xFF

                # 場景切換檢測 (基於裁切後的畫面)
                current_global_snap = cv2.resize(gray_frame, (160, 120))
                if self.last_global_frame is not None and recording:
                    g_diff = cv2.mean(cv2.absdiff(current_global_snap, self.last_global_frame))[0]
                    if g_diff > 25.0: 
                        print("--- 場景大變換 ---")
                        self.is_initialized = False
                        self.anchor_pos = None
                        self.move_start_pos = None
                        self.state = "INIT"
                self.last_global_frame = current_global_snap

                if not recording:
                    cursor_pos, hsv_mask, roi_box = self.get_hsv_cursor_pos(frame, frame_w, frame_h)
                    current_anchor = cursor_pos if cursor_pos else self.anchor_pos
                    display = self.draw_hud_overlay(frame, hsv_mask, roi_box, current_anchor)
                    
                    cv2.putText(display, "[STANDBY] Press SPACE", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    init_pos = self.detect_template(gray_frame, self.cursor_tpl, threshold=0.65)
                    if init_pos:
                        cv2.circle(display, init_pos, 8, (255, 0, 0), -1)
                        
                    cv2.imshow("Platinum Vision", display)
                    if key == ord(' '):
                        page.evaluate("document.querySelector('video').play();")
                        start_tick = cv2.getTickCount()
                        self.last_time_ms = 0
                        recording = True
                    elif key == ord('q'): break
                    continue

                current_ms = ((cv2.getTickCount() - start_tick) / fps) * 1000
                if max_duration_sec is not None and (current_ms / 1000.0) >= max_duration_sec: break

                delta_time = current_ms - self.last_time_ms
                self.last_time_ms = current_ms
                
                cursor_pos, hsv_mask, roi_box = self.get_hsv_cursor_pos(frame, frame_w, frame_h)
                
                wait_pos = self.detect_template(gray_frame, self.waiting_tpl, threshold=0.75)
                if wait_pos:
                    self.state = "WAITING"
                    self.hover_timer = 0
                    self.anchor_pos = wait_pos 
                    display = self.draw_hud_overlay(frame, hsv_mask, roi_box, self.anchor_pos)
                    cv2.imshow("Platinum Vision", display)
                    if key == ord('q'): break
                    continue

                if self.anchor_pos is None and self.state == "INIT":
                    init_pos = self.detect_template(gray_frame, self.cursor_tpl, threshold=0.65)
                    if init_pos:
                        self.anchor_pos = init_pos
                        self.move_start_pos = init_pos
                        self.is_initialized = True
                        self.state = "SEARCHING"
                    display = self.draw_hud_overlay(frame, hsv_mask, roi_box, self.anchor_pos)
                    cv2.putText(display, f"STATE: {self.state}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    cv2.imshow("Platinum Vision", display)
                    if key == ord('q'): break
                    continue

                if cursor_pos:
                    dist = math.hypot(cursor_pos[0] - self.anchor_pos[0], cursor_pos[1] - self.anchor_pos[1]) if self.anchor_pos else 0
                    
                    if dist > (frame_w * 0.005): 
                        self.state = "MOVING"
                        self.hover_timer = 0
                        self.anchor_pos = cursor_pos
                        self.click_cooldown = max(0, self.click_cooldown - delta_time)
                        if self.move_start_pos is None:
                            self.move_start_pos = cursor_pos
                    else:
                        self.hover_timer += delta_time
                        if self.hover_timer > 100:
                            if self.state == "MOVING":
                                self.log_move_action(self.move_start_pos, cursor_pos, frame_w, frame_h, current_ms)
                                self.move_start_pos = cursor_pos
                                self.state = "HOVER"

                            patch_r = int(frame_w * 0.05)
                            cx, cy = cursor_pos
                            tx1, ty1, tx2, ty2 = self.get_safe_roi(cx, cy, patch_r, frame_w, frame_h)
                            local_patch = gray_frame[ty1:ty2, tx1:tx2]
                            
                            if self.last_local_patch is not None and self.click_cooldown <= 0:
                                if local_patch.shape == self.last_local_patch.shape:
                                    l_diff = cv2.mean(cv2.absdiff(local_patch, self.last_local_patch))[0]
                                    if l_diff > 8.0:
                                        self.log_click_action(current_ms)
                                        self.click_cooldown = 2000 
                            self.last_local_patch = local_patch
                else:
                    self.state = "LOST (ANCHOR LOCKED)"

                display = self.draw_hud_overlay(frame, hsv_mask, roi_box, self.anchor_pos)

                if self.anchor_pos and self.is_initialized:
                    color = (0, 0, 255) if self.state.startswith("LOST") else (0, 255, 0)
                    cv2.circle(display, self.anchor_pos, 8, color, -1) 
                    
                if max_duration_sec:
                    remain = max(0, max_duration_sec - (current_ms / 1000.0))
                    cv2.putText(display, f"REC | {remain:.1f}s", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                cv2.putText(display, f"STATE: {self.state}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.imshow("Platinum Vision", display)
                if key == ord('q'): break

            browser.disconnect()
            cv2.destroyAllWindows()
            self.generate_ahk()

    def generate_ahk(self):
        print(f"\n[COMPILE] 開始生成 {self.ahk_out_path} ...")
        header = """#NoEnv\nSetCapsLockState, AlwaysOff\nWinGet, remote_id, List, ahk_exe chiaki-ng.exe\nif (remote_id = 0)\n    WinGet, remote_id, List, ahk_exe chiaki.exe\nif (remote_id = 0) {\n    MsgBox, 16, Error, Chiaki window not found!\n    ExitApp\n}\nglobal targetID := remote_id1\nglobal Playing := 0\nToolTip, Ready! Press CapsLock to Start AutoPlay.\n\n$CapsLock::\n    Playing := !Playing\n    if (Playing) {\n        ToolTip, Auto-Playing...\n        SetTimer, PlayMacro, -1\n    } else {\n        ToolTip\n        Reload\n    }\nreturn\n\nPlayMacro:\n"""
        with open(self.ahk_out_path, "w", encoding="utf-8") as f:
            f.write(header)
            for act in self.action_log:
                wait_time = max(0, act["wait_before_next_ms"])
                f.write(f'DllCall("Sleep", "Uint", {wait_time})\n')
                if act["type"] == "MOVE":
                    x_dir, x_ms = act["details"]["x_dir"], act["details"]["x_hold_ms"]
                    y_dir, y_ms = act["details"]["y_dir"], act["details"]["y_hold_ms"]
                    if x_ms > 0: f.write(f'ControlSend, , {{{x_dir} down}}, ahk_id %targetID%\nDllCall("Sleep", "Uint", {x_ms})\nControlSend, , {{{x_dir} up}}, ahk_id %targetID%\n')
                    if y_ms > 0: f.write(f'ControlSend, , {{{y_dir} down}}, ahk_id %targetID%\nDllCall("Sleep", "Uint", {y_ms})\nControlSend, , {{{y_dir} up}}, ahk_id %targetID%\n')
                elif act["type"] == "CLICK":
                    f.write(f'ControlSend, , {{{act["key"]} down}}, ahk_id %targetID%\nDllCall("Sleep", "Uint", 50)\nControlSend, , {{{act["key"]} up}}, ahk_id %targetID%\n')
            f.write("\nToolTip, Playback Finished!\nPlaying := 0\nreturn\n\n^Esc::ExitApp\n")
        print("[COMPILE] 完成！腳本已就緒。")

if __name__ == "__main__":
    url = "https://www.youtube.com/watch?v=7K_NimshHUI"
    agent = AutoPlatinumHand(url, hex_color="#f2b877", ahk_out_path="AutoPlay_Macro.ahk")
    agent.process_with_playwright(start_time_sec=34, max_duration_sec=30)