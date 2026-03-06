import cv2
import numpy as np
import math
import time
import win32gui
import dxcam
from playwright.sync_api import sync_playwright

class ReadyAlignTester:
    def __init__(self, start_time_sec=35, cursor_path='cursor.png', chiaki_cursor_path='cursor2.png'):
        self.start_time_sec = start_time_sec
        self.camera = dxcam.create(output_idx=0, output_color="BGR")
        
        # 🎯 讀取兩組指針模板 (嚴格灰階處理)
        self.cursor_tpl = cv2.imread(cursor_path, 0)
        self.chiaki_cursor_tpl = cv2.imread(chiaki_cursor_path, 0)
        
        if self.cursor_tpl is None: print(f"⚠️ 找不到 {cursor_path}")
        if self.chiaki_cursor_tpl is None: print(f"⚠️ 找不到 {chiaki_cursor_path}")

        self.yt_x, self.yt_y, self.yt_w, self.yt_h = 0, 0, 0, 0
        self.video_cursor_pos = None 
        self.chiaki_hwnd = None
        self.chiaki_x, self.chiaki_y, self.chiaki_w, self.chiaki_h = 0, 0, 0, 0
        self.chiaki_cursor_pos = None 
        self.scale_x, self.scale_y = 1.0, 1.0

    def get_pure_game_scene(self, bgr_image):
        """V10 兩步法：精準鎖定遊戲區域"""
        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        
        # 第一步：找黑框容器
        _, dark_mask = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY_INV)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        dark_mask_closed = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel)
        
        cnts_dark, _ = cv2.findContours(dark_mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts_dark: return 0, 0, bgr_image.shape[1], bgr_image.shape[0]
            
        largest_dark_cnt = max(cnts_dark, key=cv2.contourArea)
        cx, cy, cw, ch = cv2.boundingRect(largest_dark_cnt)
        
        # 第二步：在黑框內找彩色遊戲畫面
        container_roi_gray = gray[cy:cy+ch, cx:cx+cw]
        _, game_mask = cv2.threshold(container_roi_gray, 15, 255, cv2.THRESH_BINARY)
        game_mask = cv2.morphologyEx(game_mask, cv2.MORPH_CLOSE, kernel)
        game_mask = cv2.morphologyEx(game_mask, cv2.MORPH_OPEN, kernel)
        
        cnts_game, _ = cv2.findContours(game_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts_game: return cx, cy, cw, ch
            
        largest_game_cnt = max(cnts_game, key=cv2.contourArea)
        gx, gy, gw, gh = cv2.boundingRect(largest_game_cnt)
        
        return cx + gx, cy + gy, gw, gh

    def _extract_game_roi(self, window_img):
        """獲取 Chiaki 客戶端範圍偏移 [已補回]"""
        cr = win32gui.GetClientRect(self.chiaki_hwnd)
        pt = win32gui.ClientToScreen(self.chiaki_hwnd, (0, 0))
        rect = win32gui.GetWindowRect(self.chiaki_hwnd)
        cx = pt[0] - rect[0]
        cy = pt[1] - rect[1]
        cw = cr[2] - cr[0]
        ch = cr[3] - cr[1]
        return cx, cy, cw, ch

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
        x1, y1, x2, y2 = rect
        
        sh, sw = full_screen_frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(sw, x2), min(sh, y2)
        
        if x1 >= x2 or y1 >= y2: return None
        
        window_img = full_screen_frame[y1:y2, x1:x2]
        ix, iy, iw, ih = self.get_pure_game_scene(window_img)
        
        return x1 + ix, y1 + iy, iw, ih

    def prepare_and_align(self):
        print("🔗 正在連接 Chrome 並準備影片...")
        page = None
        try:
            pw = sync_playwright().start()
            browser = pw.chromium.connect_over_cdp("http://localhost:9222")
            page = next((pg for pg in browser.contexts[0].pages if "youtube" in pg.url), browser.contexts[0].pages[0])
            page.bring_to_front() 
            page.evaluate(f"document.querySelector('video').pause(); document.querySelector('video').currentTime = {self.start_time_sec};")
            time.sleep(0.5) 
        except Exception: pass

        full_grab = self.camera.grab()
        while full_grab is None:
            full_grab = self.camera.grab()
            time.sleep(0.01)

        yt_rect = self.get_absolute_game_rect("youtube", full_grab)
        if yt_rect: self.yt_x, self.yt_y, self.yt_w, self.yt_h = yt_rect
            
        chiaki_rect = self.get_absolute_game_rect("chiaki", full_grab)
        if chiaki_rect: self.chiaki_x, self.chiaki_y, self.chiaki_w, self.chiaki_h = chiaki_rect

        if self.yt_w > 0 and self.chiaki_w > 0:
            self.scale_x, self.scale_y = self.yt_w / self.chiaki_w, self.yt_h / self.chiaki_h
        
        # 🎯 啟動背景幀獲取
        self.camera.start(target_fps=30, video_mode=True)
        return page

    def detect_template(self, gray_frame, template, threshold=0.75):
        if template is None or gray_frame is None or gray_frame.size == 0: return None
        res = cv2.matchTemplate(gray_frame, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        if max_val > threshold:
            return (max_loc[0] + template.shape[1]//2, max_loc[1] + template.shape[0]//2)
        return None

    def draw_8_lines(self, overlay, cx, cy, w, h, color):
        """【補回】完整 V10 百分比輔助線繪製邏輯"""
        diag = math.hypot(w, h)
        left_pct = int((cx / w) * 100) if w > 0 else 0
        right_pct = int(((w - cx) / w) * 100) if w > 0 else 0
        top_pct = int((cy / h) * 100) if h > 0 else 0
        bottom_pct = int(((h - cy) / h) * 100) if h > 0 else 0
        tl_pct = int((math.hypot(cx, cy) / diag) * 100) if diag > 0 else 0
        tr_pct = int((math.hypot(w - cx, cy) / diag) * 100) if diag > 0 else 0
        bl_pct = int((math.hypot(cx, h - cy) / diag) * 100) if diag > 0 else 0
        br_pct = int((math.hypot(w - cx, h - cy) / diag) * 100) if diag > 0 else 0

        # 十字線與文字
        cv2.line(overlay, (cx, cy), (cx, 0), color, 1) 
        cv2.putText(overlay, f"{top_pct}%", (cx + 5, cy // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        cv2.line(overlay, (cx, cy), (cx, h), color, 1) 
        cv2.putText(overlay, f"{bottom_pct}%", (cx + 5, cy + (h - cy) // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        cv2.line(overlay, (cx, cy), (0, cy), color, 1) 
        cv2.putText(overlay, f"{left_pct}%", (cx // 2, cy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        cv2.line(overlay, (cx, cy), (w, cy), color, 1) 
        cv2.putText(overlay, f"{right_pct}%", (cx + (w - cx) // 2, cy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # 對角線與文字
        cv2.line(overlay, (cx, cy), (0, 0), color, 1) 
        cv2.putText(overlay, f"{tl_pct}%", (cx // 2, cy // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        cv2.line(overlay, (cx, cy), (w, 0), color, 1) 
        cv2.putText(overlay, f"{tr_pct}%", (cx + (w - cx) // 2, cy // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        cv2.line(overlay, (cx, cy), (0, h), color, 1) 
        cv2.putText(overlay, f"{bl_pct}%", (cx // 2, cy + (h - cy) // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        cv2.line(overlay, (cx, cy), (w, h), color, 1) 
        cv2.putText(overlay, f"{br_pct}%", (cx + (w - cx) // 2, cy + (h - cy) // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    def run_test(self):
        page = self.prepare_and_align()
        if not page: return
        cv2.namedWindow("GUI - Sync Verification")
        cv2.moveWindow("GUI - Sync Verification", 1920 - self.yt_w - 50, 1080 - self.yt_h - 100)

        while True:
            # 🎯 阻塞等待新幀 (有幀才運行)
            full_frame = self.camera.get_latest_frame()
            if full_frame is None: continue

            yt_frame = full_frame[self.yt_y:self.yt_y+self.yt_h, self.yt_x:self.yt_x+self.yt_w]
            chiaki_frame = full_frame[self.chiaki_y:self.chiaki_y+self.chiaki_h, self.chiaki_x:self.chiaki_x+self.chiaki_w]

            overlay = yt_frame.copy()
            yt_gray = cv2.cvtColor(yt_frame, cv2.COLOR_BGR2GRAY)
            chiaki_gray = cv2.cvtColor(chiaki_frame, cv2.COLOR_BGR2GRAY)

            # 影片指針 (橘線)
            v_pos = self.detect_template(yt_gray, self.cursor_tpl, threshold=0.70)
            if v_pos: self.video_cursor_pos = v_pos
            if self.video_cursor_pos:
                self.draw_8_lines(overlay, self.video_cursor_pos[0], self.video_cursor_pos[1], self.yt_w, self.yt_h, (0, 165, 255))

            # 實機指針 (藍線)
            c_pos = self.detect_template(chiaki_gray, self.chiaki_cursor_tpl, threshold=0.70)
            if c_pos: self.chiaki_cursor_pos = c_pos
            if self.chiaki_cursor_pos:
                mx = int(self.chiaki_cursor_pos[0] * self.scale_x)
                my = int(self.chiaki_cursor_pos[1] * self.scale_y)
                self.draw_8_lines(overlay, mx, my, self.yt_w, self.yt_h, (255, 255, 0))

            display = cv2.addWeighted(yt_frame, 0.7, overlay, 0.3, 0)
            cv2.imshow("GUI - Sync Verification", display)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

        self.camera.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    tester = ReadyAlignTester(cursor_path='cursor.png', chiaki_cursor_path='cursor2.png')
    tester.run_test()