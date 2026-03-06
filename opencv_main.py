import cv2
import numpy as np
import math
import time
import dxcam 
import torch 
import win32gui
from playwright.sync_api import sync_playwright

class AutoPlatinumHand:
    def __init__(self, youtube_url, cursor_path='cursor.png', waiting_path='waiting.png'):
        
        # --- PyTorch 設備初始化 ---
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🚀 啟動 PyTorch 運算設備: {self.device}")
        
        # 1. 資源與模板載入
        self.cursor_tpl = cv2.imread(cursor_path, 0)
        self.waiting_tpl = cv2.imread(waiting_path, 0)
        
        # 2. video狀態機與變量
        self.youtube_url = youtube_url
        self.state = "INIT"  # HOVER,MOVE,WAIT
        self.chaiki_target = None # 判斷是否已得到chaiki窗口
        self.scene_change_count = 0
        self.click_count = 0
        
        self.queue = []  #用於記錄video操作的隊列，每被chaiki執行完一個就刪除一個，記錄內容包括，移動到座標，點擊，場景切換以及背景對齊等，如果影片進行了場景切換，觸發兩邊場景比對，發現chaiki沒有切換，則自動暫停等待同步。如果影片中指針移動，chaiki對比座標也移動，直到影片中指針停止，chaiki開始微調移動座標到影片中指針位置，直到兩者重合，然後繼續執行隊列中的操作。
        # queue中的信息會有，hover（座標），click（到達座標後不一定點擊，所以分開計算）場景change check（觸發檢查），move試試追趕不算queue
        # queue count用於記錄當前隊列中未執行的操作數量，如果count不為0，且queue中出現場景 change check，則暫停執行，直到count為0，然後繼續執行隊列中的操作
        self.last_known_cursor_pos = None #影片的最後指針位置，實時和chaiki對比，如果影片中指針移動，chaiki對比座標也移動，直到影片中指針停止，chaiki開始微調移動座標到影片中指針位置，直到兩者重合，然後繼續執行隊列中的操作。
        self.chaiki_cursor_pos = None #chaiki的指針位置，實時和影片對比，如果影片中指針移動，chaiki對比座標也移動，直到影片中指針停止，chaiki開始微調移動座標到影片中指針位置，直到兩者重合，然後繼續執行隊列中的操作。
        #需要有一個帶時間戳log，記錄影片關鍵幀出現時間，以及chaiki操作記錄和時間（影片和chaiki都是開始結束移動時間，開始結束hover時間，點擊時間，waiting時間（影片獨有，chaiki不關注，因為waiting時chaiki等待場景切換），場景切換完成時間，chaiki出一個座標微調開始結束時間，單獨做一個函數功能，因為長按方向過會讓指針錯過座標（如果到制定目標的px＜一定值觸發微調））

    # ==========================================
    # 待補全與測試的解耦函數 (Empty Stubs)
    # ==========================================

    # 初始化與同步相關函數
    def chaiki_ready(self, frame):
        #(需解耦測試)
        #獲取chaiki-ng窗口，對齊影片和chaiki窗口大小，對齊兩邊座標，在GUI上做影片指針的輔助線,在GUI上做chaiki指針到輔助線,移動chaiki指針到和影片指針同位置,如果八線對齊,則座標對齊.
        #影片八綫橘色,chaiki八綫淺藍色,對齊後準備完畢,按下空格后影片開始播放,queue加載任務隊列

        pass

    def auto_align_chiaki(self, camera, yt_frame, rw, rh):
        #我從之前版本拿過來的一個驗證過的辦法，它本來是在run_live_sync裡面用來鎖定播放器區域的，現在我把它獨立出來，專門用來鎖定chaiki窗口位置，這個功能補齊之前和ready-chaiki合併
        #固定抓Chiaki-ng窗口，計算一次性的偏移量，之後在每一幀中直接使用這個偏移量來對齊座標，避免累加導致畫面消失的問題。
        
        """鎖定 Chiaki 視窗位置，並計算一次性的固定偏移量。"""
        self.yt_w, self.yt_h = rw, rh
        camera = dxcam.create(output_idx=0, output_color="BGR") 
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

    def draw_hud_overlay(self, frame, hsv_mask, roi_box, anchor_pos, fps_val):
        # 主要是要影片和chaiki的八條綫做肉眼對比，所以要加內容
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


    #position and movement related functions
    def Realtime_cursor_position(self):
        """
        (需解耦測試)
        同時獲取影片和chaiki指針位置，實時對比，chaiki指針朝著影片指針移動，直到影片指針停止，chaiki開始微調移動座標到影片指針位置，直到兩者重合，然後繼續執行隊列中的操作。
        位置重合時會自動暫時停，等待下一次指針移動，然後重複上述過程。這個函數會持續運行，直到影片結束或用戶手動停止。
        判斷是使用refine_move_to_target還是move_to_target
        完成queue到達為之後移除queue并更新count
        """
        pass

    def refine_move_to_target(self, current_pos, target_pos):
        """
        (需解耦測試)
        功能：當chaiki指針接近影片指針但未完全重合時，啟動微調機制，精確對齊到目標位置。
        邏輯：
        1. 計算當前位置與目標位置的距離。
        2. 如果距離小於某個閾值（例如20像素），則進入微調模式。
        3. 在微調模式下，每一幀的時間內短促的按下一個會導致幾px（可能是5px或其他，需要寫具體測試）移動的方向鍵（根據當前位置與目標位置的相對位置），持續直到兩者重合或距離小於3像素。
        """
        pass

    def move_to_target(self, target_pos):
        """
        (需解耦測試)
        功能：將chaiki指針移動到指定的目標位置，並在接近目標時啟動微調機制。
        邏輯：
        1. 根據當前chaiki指針位置與目標位置的相對位置，按下相應的方向鍵（上、下、左、右）來移動指針。
        2. 持續監控指針位置，當距離小於微調閾值時，調用refine_move_to_target 進行精確對齊。
        3. 確保在移動過程中不會錯過任何指針變化或場景切換事件。
        """
        pass

    #click and scene change related functions
    def click_check(self, is_scene_about_to_change, is_roi_about_to_change, is_waiting_detected, current_time_ms):
        """
        功能：判斷當前幀是否觸發點擊 (需解耦測試)
        邏輯：
        1. 觸發條件（滿足其一即判定為需要點擊）：
           - 場景切換開始前 (is_scene_about_to_change)
           - ROI 焦點區域變化開始前 (is_roi_about_to_change)
           - waiting.png 出現前 (is_waiting_detected)
        2. 免責/忽略條件（延遲加載處理）：
           - 如果 waiting.png 出現了，記錄當下的時間戳。
           - 在這之後的 4 秒（4000ms）內，如果發生了「場景切換」，該次場景切換 **不算作** 點擊。
        """
        pass

    def is_roi_change(self): 
        """
        功能：判斷當前幀是否觸發 ROI 焦點區域變化 (需解耦測試)
        邏輯：
        1. 定義 ROI 焦點區域變化的判定標準（例如，特定區域內的像素變化超過某個閾值）。
        2. 返回布爾值，指示是否檢測到 ROI 變化。
        """
        pass

    def is_cursor_change(self): 
        """
        功能：hover時啟動，判斷是否觸發指針變化成waiting (需解耦測試)
        """
        pass
        
    def is_scene_change (self): 
        """
        功能：判斷影片是否背景變化 (需解耦測試)
        邏輯：用get_sparse_hash比對
        
        """
        pass

    def execute_click(self, queue_item):
        """
        異步執行隊列中的操作 (需解耦測試)
        根據 queue_item 的類型（hover、click、scene_change_check、move）check是否完成。
         - hover/move: Realtime_cursor_position中回圈。
         - click: 在 hover 狀態下執行點擊操作。
         分開處理
         執行完畢後，從隊列中刪除該項目並更新計數器。
        """
        pass

    def get_sparse_hash(self, gray_frame):
        """邊框高密度採樣：跳過中心動畫，精準提取四周 15% 區域"""
        h, w = gray_frame.shape
        dh, dw = int(h * 0.15), int(w * 0.15)
        step = 4 
        top = gray_frame[0:dh, ::step].flatten()
        bottom = gray_frame[h-dh:h, ::step].flatten()
        left = gray_frame[dh:h-dh, 0:dw:step].flatten()
        right = gray_frame[dh:h-dh, w-dw:w:step].flatten()
        border_pixels = np.concatenate((top, bottom, left, right))
        return border_pixels.astype(np.int16)
    
    def need_stop_for_sync(self):
        """
        功能：判斷是否需要暫停執行以等待同步 (需解耦測試)
        邏輯：
        1. 如果隊列中存在 "scene_change_check" 項目，且當前未完成的操作數量不為零，則返回 True。
        2. 否則返回 False，繼續執行隊列中的操作。
        """
        pass

    
    def run_live_sync(self, start_time_sec=35):
        #這部分之前是ok的，我們把拿到的信息放進隊列，然後在execute_action裡面根據隊列內容執行相應的操作，這樣就解耦了信息獲取和操作執行的邏輯，方便後續測試和優化。
        """核心影片擷取與分析迴圈 (已清理，等待接入新函數)"""
        with sync_playwright() as p:
            browser = p.chromium.connect_over_cdp("http://localhost:9222")
            page = next((pg for pg in browser.contexts[0].pages if "youtube" in pg.url), browser.contexts[0].pages[0])
            video = page.wait_for_selector("video")
            page.evaluate(f"document.querySelector('video').pause(); document.querySelector('video').currentTime = {start_time_sec};")
            
            p_bytes = video.screenshot(type="jpeg", quality=100)
            p_frame = cv2.imdecode(np.frombuffer(p_bytes, np.uint8), cv2.IMREAD_COLOR)
            
            camera = dxcam.create(output_idx=0, output_color="BGR") 
            full_init = camera.grab()
            while full_init is None: full_init = camera.grab()
            
            res = cv2.matchTemplate(full_init, p_frame, cv2.TM_CCOEFF_NORMED)
            _, _, _, max_loc = cv2.minMaxLoc(res)
            
            # 定位播放器區域
            gray_p = cv2.cvtColor(p_frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(cv2.medianBlur(gray_p, 35), 30, 100)
            cnts, _ = cv2.findContours(cv2.dilate(edges, None, iterations=2), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            valid_rects = [cv2.boundingRect(c) for c in cnts if cv2.boundingRect(c)[2] > 300]
            rx, ry, rw, rh = max(valid_rects, key=lambda r: r[2]*r[3]) if valid_rects else (0,0,p_frame.shape[1],p_frame.shape[0])

            real_x, real_y = max_loc[0] + rx, max_loc[1] + ry
            print(f"🎯 鎖定成功！同步座標: ({real_x}, {real_y})")

            cv2.namedWindow("Platinum Vision", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Platinum Vision", rw, rh)
            
            recording, start_tick, prev_time = False, 0, time.time()

            with torch.no_grad():
                while True:
                    frame = camera.grab(region=(real_x, real_y, real_x + rw, real_y + rh))
                    if frame is None: continue
                    
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    curr_time = time.time()
                    ms = ((cv2.getTickCount() - start_tick) / cv2.getTickFrequency()) * 1000 if recording else 0
                    fps = 1 / (curr_time - prev_time) if curr_time > prev_time else 60
                    self.last_time_ms = ms
                    prev_time = curr_time
                    key = cv2.waitKey(1) & 0xFF

                    if recording:
                        # ---------------------------------------------------------
                        # TODO: 這裡之後會依次調用 track_cursor_position 
                        # 和 evaluate_click_condition 來驅動邏輯
                        # ---------------------------------------------------------
                        pass

                    if key == ord(' '):
                        if not recording:
                            page.evaluate("document.querySelector('video').play();")
                            start_tick, recording = cv2.getTickCount(), True
                            print("▶️ 開始即時操作模式")
                        else:
                            break
                    
                    self.last_full_gray_np = gray.copy()
                    display = self.draw_hud_overlay(frame, self.last_known_cursor_pos, fps)
                    cv2.imshow("Platinum Vision", display)
                    if key == ord('q'): break

            browser.close()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    agent = AutoPlatinumHand("https://www.youtube.com/watch?v=7K_NimshHUI")
    agent.run_live_sync()