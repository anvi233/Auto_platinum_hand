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
        
        if self.key_states.get(key_str) != press:
            msg_name = "KEYDOWN" if press else "KEYUP"
            # print(f"⌨️  [Chiaki Control] {key_str.upper()}: {msg_name}")
            msg = win32con.WM_KEYDOWN if press else win32con.WM_KEYUP
            win32api.PostMessage(self.chiaki_hwnd, msg, vk, self._get_lparam(vk, press))
            self.key_states[key_str] = press

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
        
        def control_axis(delta, pos_key, neg_key):
            abs_d = abs(delta)
            if abs_d > 0.25: 
                self.update_key_bg(pos_key, delta > 0)
                self.update_key_bg(neg_key, delta < 0)
                return 0
            else:
                self.update_key_bg(pos_key, False)
                self.update_key_bg(neg_key, False)
                if abs_d >= 0.005:
                    raw_time = abs_d / 0.85
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
            
            vk_map = {'up': win32con.VK_UP, 'down': win32con.VK_DOWN, 'left': win32con.VK_LEFT, 'right': win32con.VK_RIGHT}
            
            for key in tap_keys:
                vk = vk_map[key]
                win32api.PostMessage(self.chiaki_hwnd, win32con.WM_KEYDOWN, vk, self._get_lparam(vk, True))
            
            time.sleep(max_pulse)
            
            for key in tap_keys:
                vk = vk_map[key]
                win32api.PostMessage(self.chiaki_hwnd, win32con.WM_KEYUP, vk, self._get_lparam(vk, False))
                
            time.sleep(0.1)

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

    def is_scene_changed(self, current_gray, last_gray, cursor_x, cursor_y, is_sync_mode=False):
        if current_gray is None or last_gray is None: return False
        curr_s = cv2.resize(current_gray, (32, 32))
        last_s = cv2.resize(last_gray, (32, 32))
        diff_score = np.mean(cv2.absdiff(curr_s, last_s)) / 255.0
        
        if is_sync_mode:
            return diff_score > 0.08
        else:
            return diff_score > 0.03

    def sync_check(self, yt_gray, chiaki_gray):
        if yt_gray is None or chiaki_gray is None: return True
        ck_res = cv2.resize(chiaki_gray, (self.yt_w, self.yt_h))
        is_different = self.is_scene_changed(yt_gray, ck_res, 0, 0, is_sync_mode=True)
        return not is_different

    # ------------------------------------------
    # 模塊 1: 影片端專屬 (捕捉因)
    # ------------------------------------------
    def detect_and_enqueue_click(self, yt_rel_pos, yt_cls, scene_changed):
        """
        影片端：負責維護動畫結界鎖，並在有效瞬間抓取游標意圖入隊。
        """
        # 1. 更新動畫結界鎖 (狀態機)
        scene_jump_triggered = False
        if scene_changed:
            self.scene_stable_frames = 0
            if not self.is_scene_changing:
                self.is_scene_changing = True
                scene_jump_triggered = True  # 🌟 剛好開始跳變的「第一瞬間」
        else:
            self.scene_stable_frames += 1
            if self.scene_stable_frames >= 10:  # 連續 10 幀穩定，解除動畫鎖定
                self.is_scene_changing = False

        # 2. 觸發判定
        cursor_changed = (yt_rel_pos and self.last_yt_cls and yt_cls != self.last_yt_cls and yt_cls in ('Hold', 'Keep', 'Waiting'))
        is_pointer_stable = (self.prev_intent_pos is not None)

        trigger = False
        trigger_reason = ""

        # 🌟 結界核心邏輯：跳變的瞬間，或是「不在動畫中」的游標變化
        if scene_jump_triggered and is_pointer_stable:
            trigger = True
            trigger_reason = "場景跳變(動畫起點)"
        elif cursor_changed and not self.is_scene_changing:
            trigger = True
            trigger_reason = "游標變化(畫面穩定時)"

        # 3. 執行入隊
        if trigger:
            priority_target = self.prev_intent_pos if self.prev_intent_pos else self.last_known_cursor_pos
            if priority_target:
                new_pos = (float(priority_target[0]), float(priority_target[1]))
                
                # 防呆：距離上次點擊位置大於 2% 才允許入隊
                if not self.queue or math.hypot(new_pos[0]-self.queue[-1][0], new_pos[1]-self.queue[-1][1]) > 0.02:
                    self.queue.append(new_pos)
                    print(f"📥 [New Task] {new_pos} 入隊 ({trigger_reason}) | 總數: {len(self.queue)}")
                    # 入隊後清空快照，避免重複觸發
                    self.prev_intent_pos = None 

        self.last_yt_cls = yt_cls

    # ------------------------------------------
    # 模塊 2: 實機端專屬 (執行果)
    # ------------------------------------------
    def execute_and_dequeue_click(self, current_t):
        """
        實機端：無視畫面，死磕隊列目標。執行極度精確的 0.008 物理到位與雙幀確認。
        """
        if not self.queue:
            if self.chaiki_cursor_pos and self.last_known_cursor_pos:
                dist_roaming = math.hypot(self.chaiki_cursor_pos[0]-self.last_known_cursor_pos[0], 
                                          self.chaiki_cursor_pos[1]-self.last_known_cursor_pos[1])
                if dist_roaming > 0.05:
                    if not hasattr(self, '_last_roaming_log') or current_t - self._last_roaming_log > 2.0:
                        print(f"👣 [Roaming] 跟隨影片指針中... 距離: {dist_roaming:.4f}")
                        self._last_roaming_log = current_t
                self.move_action(self.chaiki_cursor_pos, self.last_known_cursor_pos)
            else:
                self.release_all_keys()
            self.ck_stable_frames = 0
            return

        if not self.chaiki_cursor_pos: return

        target_pos = self.queue[0]
        dist = math.hypot(target_pos[0]-self.chaiki_cursor_pos[0], target_pos[1]-self.chaiki_cursor_pos[1])

        # --- 狀態 A：等待點擊動作完成後出隊 ---
        if getattr(self, 'click_validating', False):
            if current_t - self.val_start_time > 0.8: # 給予 0.8s 完成點擊動作
                print(f"✅ [Task Finished] 彈出任務: {target_pos} | 剩餘: {len(self.queue)-1}")
                self.queue.pop(0) 
                self.click_validating = False
            return

        # --- 狀態 B：執行移動與點擊 ---
        # 🌟 物理精確打擊：容差縮小至 0.008 (不到 1% 屏幕距離)，防止打滑提前開火
        if dist < 0.008:
            self.release_all_keys() # 剎車
            self.ck_stable_frames += 1
            
            # 🌟 雙幀確認：連續兩幀都在 0.008 的靶心內才允許開火
            if self.ck_stable_frames >= 2:
                print(f"🔥 [EXECUTE CLICK] 物理精確到位 (dist={dist:.4f})，開火！")
                self.click_action()
                self.click_validating = True
                self.val_start_time = current_t
                self.ck_stable_frames = 0
        else:
            self.ck_stable_frames = 0
            if not hasattr(self, '_last_move_log') or current_t - self._last_move_log > 1.0:
                print(f"🚚 [Moving] 鎖定任務點: {target_pos} | 當前距離: {dist:.4f}")
                self._last_move_log = current_t
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
                    
                    # 1. 計算場景跳變
                    scene_changed = False
                    try:
                        sx = yt_abs_pos[0] if yt_abs_pos else int((self.last_known_cursor_pos[0] if self.last_known_cursor_pos else 0.5) * self.yt_w)
                        sy = yt_abs_pos[1] if yt_abs_pos else int((self.last_known_cursor_pos[1] if self.last_known_cursor_pos else 0.5) * self.yt_h)
                        scene_changed = self.is_scene_changed(yt_gray, self.reference_gray, sx, sy)
                    except:
                        scene_changed = False
                    
                    # 維護底圖 (果的表現)
                    if scene_changed:
                        self.reference_gray = yt_gray.copy()
                        self.ref_frame_timer = 0
                    else:
                        self.ref_frame_timer += 1
                        if self.ref_frame_timer >= 30:
                            self.reference_gray = yt_gray.copy()
                            self.ref_frame_timer = 0

                    # 2. 影片端專屬：入隊邏輯 (判斷因)
                    if recording and not getattr(self, 'click_validating', False):
                        self.detect_and_enqueue_click(yt_rel_pos, yt_cls, scene_changed)

                    # 3. 實機端專屬：出隊邏輯 (執行果)
                    self.execute_and_dequeue_click(current_t)

                    # --- 定期同步檢查 ---
                    self.frame_counter += 1
                    if self.frame_counter >= 45:
                        self.frame_counter = 0
                        if not getattr(self, 'click_validating', False) and not self.sync_check(yt_gray, chiaki_gray):
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