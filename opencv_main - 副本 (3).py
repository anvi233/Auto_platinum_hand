import cv2
import numpy as np
import math
import time
import dxcam  # [替換] 改用 dxcam
import torch  # [新增] 引入 PyTorch
from playwright.sync_api import sync_playwright

class AutoPlatinumHand:
    def __init__(self, youtube_url, hex_color="#f2b877", ahk_out_path="AutoPlay_Macro.ahk", 
                 cursor_path='cursor.png', waiting_path='waiting.png'):
        self.youtube_url = youtube_url
        self.ahk_out_path = ahk_out_path
        
        # --- [新增] PyTorch 設備初始化 ---
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🚀 啟動 PyTorch 運算設備: {self.device}")
        
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
        # --- [新增] 場景防抖與等待控制變量 ---
        self.is_transitioning = False
        
        # 🎯 [新增] Waiting 狀態與因果屏蔽
        self.is_waiting = False
        self.ignore_next_scene_click = False
        self.transition_start_ms = 0.0
        self.stable_frames_count = 0
        self.required_stable_frames = 5
        self.sample_step = 20 # 每 20px 採樣
        
        # 數據緩存
        self.action_log = []
        self.last_action_timestamp = 0.0
        self.last_local_patch = None
        self.last_full_gray_np = None

    def hex_to_hsv_fuzzy(self, hex_code, h_tol, s_tol, v_tol):
        hex_code = hex_code.lstrip('#')
        rgb = tuple(int(hex_code[i:i+2], 16) for i in (0, 2, 4))
        bgr_pixel = np.uint8([[[rgb[2], rgb[1], rgb[0]]]])
        hsv_pixel = cv2.cvtColor(bgr_pixel, cv2.COLOR_BGR2HSV)[0][0]
        h, s, v = int(hsv_pixel[0]), int(hsv_pixel[1]), int(hsv_pixel[2])
        return np.array([max(0, h - h_tol), max(0, s - s_tol), max(0, v - v_tol)]), \
               np.array([min(179, h + h_tol), min(255, s + s_tol), min(255, v + v_tol)])

    # def get_sparse_hash(self, gray_frame):
    #     """稀疏採樣：中值濾波後按間隔取樣 (保持原樣，後續在主迴圈轉 Tensor)"""
        # denoised = cv2.medianBlur(gray_frame, 5)
        # return denoised[::self.sample_step, ::self.sample_step].flatten().astype(np.int16)
    def get_sparse_hash(self, gray_frame):
        """邊框高密度採樣：跳過中心動畫，精準提取四周 15% 區域"""
        h, w = gray_frame.shape
        dh, dw = int(h * 0.15), int(w * 0.15)
        step = 4 # 🚀 提高採樣密度 (每 4px 取一點)
        
        # 直接對四個邊進行切片，速度比用 Mask 快很多
        top = gray_frame[0:dh, ::step].flatten()
        bottom = gray_frame[h-dh:h, ::step].flatten()
        left = gray_frame[dh:h-dh, 0:dw:step].flatten()
        right = gray_frame[dh:h-dh, w-dw:w:step].flatten()
        
        # 將四個邊框的像素拼接成一維特徵陣列
        border_pixels = np.concatenate((top, bottom, left, right))
        return border_pixels.astype(np.int16)

    def calculate_acceleration_time(self, pixel_delta, max_pixels):
        ratio = abs(pixel_delta) / max_pixels
        return int(80 + 2800 * (ratio ** 0.75))
    
    def log_action(self, current_ms, action_type, details=None, duration=50):
        """通用的動作日誌處理：計算絕對時間與相對等待時間"""
        abs_time = int(current_ms)
        # 計算與上一次動作「結束後」的相對等待時間
        rel_wait_time = int(current_ms - self.last_action_timestamp) if self.last_action_timestamp > 0 else 500
        
        log_entry = {
            "type": action_type,
            "abs_time_ms": abs_time,
            "wait_before_next_ms": rel_wait_time,
            "duration_ms": duration, # 這裡改為接收動態傳入的時長
            "details": details or {}
        }
        
        self.action_log.append(log_entry)
        # 關鍵：更新時間戳為該動作「結束後」的時間點，確保下一次 wait 計算正確
        self.last_action_timestamp = current_ms + duration 
        
        if action_type == "CLICK":
            self.click_count += 1
            
        print(f"[LOG {abs_time}ms] {action_type} | Wait: {rel_wait_time}ms | Dur: {duration}ms | {details}")

    def log_move_action(self, end_ms, start_pos, end_pos, start_ms):
        """將合併後的連續位移轉換為一個長按操作，以觸發 PS1 的加速曲線"""
        dx = end_pos[0] - start_pos[0]
        dy = end_pos[1] - start_pos[1]
        
        # 計算這段移動持續的總時間
        duration = int(end_ms - start_ms)
        if duration < 50: duration = 50 # 保底時長
        
        x_dir = "Right" if dx > 0 else "Left"
        y_dir = "Down" if dy > 0 else "Up"
        
        details = {
            "x_dir": x_dir,
            "y_dir": y_dir,
            "dx_abs": abs(dx),
            "dy_abs": abs(dy)
        }
        # 使用移動開始的時間點進行 log，並傳入計算出的時長
        self.log_action(start_ms, "MOVE", details, duration=duration)

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
            
            # --- [新增] 1. 顯示絕對座標 ---
            cv2.putText(overlay, f"ABS: ({cx}, {cy})", (cx + 15, cy - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # --- [新增] 2. 計算 8 方向比例尺 (0~100%) ---
            diag = math.hypot(w, h)
            
            # 十字方向比例
            left_pct = int((cx / w) * 100)
            right_pct = int(((w - cx) / w) * 100)
            top_pct = int((cy / h) * 100)
            bottom_pct = int(((h - cy) / h) * 100)
            
            # 對角線方向比例
            tl_pct = int((math.hypot(cx, cy) / diag) * 100)
            tr_pct = int((math.hypot(w - cx, cy) / diag) * 100)
            bl_pct = int((math.hypot(cx, h - cy) / diag) * 100)
            br_pct = int((math.hypot(w - cx, h - cy) / diag) * 100)

            # --- [新增] 3. 繪製輔助線與文字 ---
            # (1) 十字方向 (黃色)
            cv2.line(overlay, (cx, cy), (cx, 0), (0, 255, 255), 1) # 上
            cv2.putText(overlay, f"{top_pct}%", (cx + 5, cy // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            cv2.line(overlay, (cx, cy), (cx, h), (0, 255, 255), 1) # 下
            cv2.putText(overlay, f"{bottom_pct}%", (cx + 5, cy + (h - cy) // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            cv2.line(overlay, (cx, cy), (0, cy), (0, 255, 255), 1) # 左
            cv2.putText(overlay, f"{left_pct}%", (cx // 2, cy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            cv2.line(overlay, (cx, cy), (w, cy), (0, 255, 255), 1) # 右
            cv2.putText(overlay, f"{right_pct}%", (cx + (w - cx) // 2, cy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            
            # (2) 對角線方向 (橘色)
            cv2.line(overlay, (cx, cy), (0, 0), (0, 165, 255), 1) # 左上
            cv2.putText(overlay, f"{tl_pct}%", (cx // 2, cy // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)
            cv2.line(overlay, (cx, cy), (w, 0), (0, 165, 255), 1) # 右上
            cv2.putText(overlay, f"{tr_pct}%", (cx + (w - cx) // 2, cy // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)
            cv2.line(overlay, (cx, cy), (0, h), (0, 165, 255), 1) # 左下
            cv2.putText(overlay, f"{bl_pct}%", (cx // 2, cy + (h - cy) // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)
            cv2.line(overlay, (cx, cy), (w, h), (0, 165, 255), 1) # 右下
            cv2.putText(overlay, f"{br_pct}%", (cx + (w - cx) // 2, cy + (h - cy) // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)

        if roi_box: cv2.rectangle(overlay, (roi_box[0], roi_box[1]), (roi_box[2], roi_box[3]), (0, 165, 255), 2)
        display = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        info = f"FPS: {int(fps_val)}  SCENES: {self.scene_change_count}  CLICKS: {self.click_count}"
        cv2.putText(display, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return display

    def process_with_playwright(self, start_time_sec=35, max_duration_sec=30):
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
            
            # --- [修改] GPU 實例替換為 dxcam ---
            camera = dxcam.create(output_idx=0, output_color="BGR") 
            full_init = camera.grab()
            while full_init is None:  # 確保成功抓取第一幀
                full_init = camera.grab()
                time.sleep(0.01)
            
            # dxcam 已經指定 BGR，所以直接傳入即可
            full_init_bgr = full_init 
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

            with torch.no_grad(): # 防止 PyTorch 記錄梯度佔用顯存
                while True:
                    # --- [修改] 使用 dxcam 的 region 直接在顯存層級裁切，避免後續 CPU 拷貝 ---
                    frame = camera.grab(region=(real_x, real_y, real_x + rw, real_y + rh))
                    if frame is None: continue
                    
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    
                    self.frame_count += 1
                    curr_time = time.time()
                    ms = ((cv2.getTickCount() - start_tick) / cv2.getTickFrequency()) * 1000 if recording else 0
                    fps = 1 / (curr_time - prev_time) if curr_time > prev_time else 60
                    delta_time = ms - self.last_time_ms
                    self.last_time_ms = ms
                    prev_time = curr_time
                    
                    key = cv2.waitKey(1) & 0xFF
                    # --- [新增] 0. Waiting 狀態檢測 ---
                    # 偵測畫面上是否出現 waiting.png，並在出現的第一幀點擊
                    if self.waiting_tpl is not None:
                        wait_pos = self.detect_template(gray, self.waiting_tpl, threshold=0.75)
                        if wait_pos:
                            if not self.is_waiting:
                                self.is_waiting = True
                                self.log_action(ms, "CLICK", {"reason": "Waiting 狀態開始"})
                                # 🎯 發放免責金牌：接下來發生的第一次場景切換，絕對是這個 Waiting 引起的
                                self.ignore_next_scene_click = True
                        else:
                            self.is_waiting = False
                    # --- 1. 場景防抖合併算法 (宏觀) ---
                    curr_hash_np = self.get_sparse_hash(gray) 
                    curr_hash = torch.from_numpy(curr_hash_np).to(self.device)
                    
                    if self.last_global_hash is not None and recording:
                        diff_global = torch.abs(curr_hash - self.last_global_hash)
                        
                        if (torch.count_nonzero(diff_global > 3).item() / curr_hash.numel()) > 0.01:
                            self.stable_frames_count = 0
                            
                            # 🎯 這裡保證了點擊只會發生在場景「剛開始」轉換的第一幀
                            if not self.is_transitioning:
                                self.is_transitioning = True
                                self.transition_start_ms = ms
                                self.scene_change_count += 1
                                
                                # 🛡️ 檢查免責金牌
                                if self.ignore_next_scene_click:
                                    print(f"[LOG {int(ms)}ms] 🚫 忽略場景點擊 (由 Waiting 觸發)")
                                    self.ignore_next_scene_click = False # 沒收金牌
                                else:
                                    self.log_action(ms, "CLICK", {"reason": "場景大變換剛開始"})
                                    
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
                        # 因為 anchor_pos 每幀更新，所以這裡的 dist 其實是「幀與幀之間的速度」
                        dist = math.hypot(pos[0]-self.anchor_pos[0], pos[1]-self.anchor_pos[1]) if self.anchor_pos else 999
                        
                        # 降低一點點速度門檻，避免緩慢滑動時斷掉
                        if dist > (rw * 0.002): 
                            curr_state = "MOVE"
                            self.stable_hover_count = 0  
                            # 🎯 進入 MOVE 的瞬間，紀錄起始點與時間
                            if self.move_start_pos is None:
                                self.move_start_pos = self.anchor_pos if self.anchor_pos else pos
                                self.move_start_time = ms 
                        else:
                            # 速度降下來了，開始累加「靜止幀數」
                            self.stable_hover_count += 1 
                            
                            # 🎯 核心防抖：必須連續靜止超過 5 幀，才確認真正停下了！
                            if self.stable_hover_count > 5:
                                curr_state = "HOVER"
                                
                                # 結算上一段長距離移動
                                if self.move_start_pos is not None:
                                    self.log_move_action(ms, self.move_start_pos, pos, self.move_start_time)
                                    self.move_start_pos = None
                                
                                # --- 以下是你原本的絕對座標 ROI 點擊檢測 ---
                                if roi_box and self.last_full_gray_np is not None and self.click_cooldown <= 0:
                                    x1, y1, x2, y2 = roi_box
                                    
                                    curr_patch_np = gray[y1:y2, x1:x2].copy()
                                    prev_patch_np = self.last_full_gray_np[y1:y2, x1:x2].copy()
                                    
                                    curr_patch_np = cv2.GaussianBlur(curr_patch_np, (5, 5), 0)
                                    prev_patch_np = cv2.GaussianBlur(prev_patch_np, (5, 5), 0)
                                    
                                    ph, pw = curr_patch_np.shape
                                    ix1, iy1 = max(0, pw//2 - 15), max(0, ph//2 - 15)
                                    ix2, iy2 = min(pw, pw//2 + 15), min(ph, ph//2 + 15)
                                    curr_patch_np[iy1:iy2, ix1:ix2] = 0 
                                    prev_patch_np[iy1:iy2, ix1:ix2] = 0 
                                    
                                    curr_roi_tensor = torch.from_numpy(curr_patch_np.astype(np.int16)).to(self.device)
                                    prev_roi_tensor = torch.from_numpy(prev_patch_np.astype(np.int16)).to(self.device)
                                    
                                    diff_matrix = torch.abs(curr_roi_tensor - prev_roi_tensor)
                                    changed_pixels = torch.count_nonzero(diff_matrix > 15).item()
                                    
                                    if changed_pixels > 10: 
                                        self.log_action(ms, "CLICK", {"reason": f"絕對座標比對觸發! (量: {changed_pixels})"})
                                        self.click_cooldown = 100 
                            else:
                                # 雖然速度慢下來了，但還在 5 幀的緩衝期內，維持 MOVE 狀態不斷線
                                pass

                        self.anchor_pos = pos
                    if key == ord(' '):
                        if not recording:
                            page.evaluate("document.querySelector('video').play();")
                            start_tick, recording = cv2.getTickCount(), True
                        else:
                            # 停止錄製並直接跳出循環生成腳本
                            print("🛑 收到停止指令，正在生成腳本...")
                            break
                    else:
                        curr_state = "LOST"

                    # ... HUD 與顯示邏輯維持不變 ...
                    # (顯示邏輯代碼略)

                    # 💥 迴圈的最尾端 (在 cv2.imshow 和 q 鍵退出之前)：
                    # 將當前全域畫面存下來，作為下一幀的「絕對座標地圖」
                    self.last_full_gray_np = gray.copy()

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
        """將 action_log 轉換為針對 chiaki-ng.exe/chiaki.exe 的實機 AHK 腳本"""
        print(f"💾 正在生成 AHK 腳本至: {self.ahk_out_path}")
        
        with open(self.ahk_out_path, "w", encoding="utf-8") as f:
            # --- 1. AHK 檔頭與視窗初始化 (結合你的 record_controller.ahk 邏輯) ---
            f.write("#NoEnv\n")
            f.write("SetCapsLockState, AlwaysOff\n")
            f.write("WinGet, remote_id, List, ahk_exe chiaki-ng.exe\n")
            f.write("if (remote_id = 0)\n")
            f.write("    WinGet, remote_id, List, ahk_exe chiaki.exe\n")
            f.write("if (remote_id = 0) {\n")
            f.write("    MsgBox, 16, Error, Chiaki not find\n")
            f.write("    ExitApp\n")
            f.write("}\n")
            f.write("global targetID := remote_id1\n")
            f.write("global Playing := 0\n")
            f.write("ToolTip, AutoPlatinum: Ready! (Press CapsLock to Play), 10, 10\n\n")
            
            # --- 2. 觸發熱鍵與狀態切換 ---
            f.write("$CapsLock::\n")
            f.write("    Playing := !Playing\n")
            f.write("    if (Playing) {\n")
            f.write("        ToolTip, AutoPlatinum: PLAYING..., 10, 10\n")
            f.write("        SetTimer, PlayMacro, -1\n")
            f.write("    } else {\n")
            f.write("        ToolTip\n")
            f.write("        Reload\n")
            f.write("    }\n")
            f.write("return\n\n")
            
            # --- 3. 巨集主體 ---
            f.write("PlayMacro:\n")
            
            for a in self.action_log:
                # 寫入前置等待時間 (微秒級精準)
                wait = max(0, a["wait_before_next_ms"])
                f.write(f'DllCall("Sleep", "Uint", {wait})\n')
                
                # 執行動作
                if a["type"] == "MOVE":
                    d = a["details"]
                    # 同時按下需要的方向鍵 (完美支援純橫向、純縱向與斜向 D-pad)
                    if d["dx_abs"] > 0: f.write(f'ControlSend, , {{{d["x_dir"]} down}}, ahk_id %targetID%\n')
                    if d["dy_abs"] > 0: f.write(f'ControlSend, , {{{d["y_dir"]} down}}, ahk_id %targetID%\n')
                    
                    # 統一按下時間 (預設 50ms)
                    f.write(f'DllCall("Sleep", "Uint", {a["duration_ms"]})\n')
                    
                    # 同時鬆開方向鍵
                    if d["dx_abs"] > 0: f.write(f'ControlSend, , {{{d["x_dir"]} up}}, ahk_id %targetID%\n')
                    if d["dy_abs"] > 0: f.write(f'ControlSend, , {{{d["y_dir"]} up}}, ahk_id %targetID%\n')
                    
                elif a["type"] == "CLICK":
                    # 點擊 Enter (對應手把 Cross)
                    f.write(f'ControlSend, , {{Enter down}}, ahk_id %targetID%\n')
                    f.write(f'DllCall("Sleep", "Uint", {a["duration_ms"]})\n')
                    f.write(f'ControlSend, , {{Enter up}}, ahk_id %targetID%\n')
            
            # --- 4. 腳本結束收尾 ---
            f.write("ToolTip, AutoPlatinum: FINISHED!, 10, 10\n")
            f.write("Playing := 0\n")
            f.write("return\n\n")
            f.write("^Esc::ExitApp\n")

        print("✅ AHK 腳本生成完成！支援斜向位移與自動尋找 Chiaki。")

if __name__ == "__main__":
    agent = AutoPlatinumHand("https://www.youtube.com/watch?v=7K_NimshHUI")
    agent.process_with_playwright()