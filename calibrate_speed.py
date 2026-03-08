import cv2
import time
import math
import dxcam
import win32gui
import win32api
import win32con
import torch
from ultralytics import YOLO

class SpeedCalibrator:
    def __init__(self, model_path=r'E:\Myst_Project\v3_final_1900\weights\best.pt'):
        print("🧠 載入 YOLO V3 視覺引擎進行測速標定...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = YOLO(model_path)
        self.camera = None
        
        # 畫面座標
        self.cx, self.cy, self.cw, self.ch = 0, 0, 0, 0
        
        # 記錄數據 [(timestamp, rel_x, rel_y)]
        self.tracking_data = []

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

    def get_absolute_game_rect(self, full_screen_frame):
        hwnds = []
        def enum_cb(hwnd, param):
            if win32gui.IsWindowVisible(hwnd):
                title = win32gui.GetWindowText(hwnd).lower()
                if "chiaki" in title and "powershell" not in title:
                    param.append(hwnd)
        win32gui.EnumWindows(enum_cb, hwnds)
        if not hwnds: return None
        hwnd = hwnds[0]
        rect = win32gui.GetWindowRect(hwnd)
        sh, sw = full_screen_frame.shape[:2]
        x1, y1, x2, y2 = max(0, rect[0]), max(0, rect[1]), min(sw, rect[2]), min(sh, rect[3])
        if x1 >= x2 or y1 >= y2: return None
        window_img = full_screen_frame[y1:y2, x1:x2]
        ix, iy, iw, ih = self.get_pure_game_scene(window_img)
        return x1 + ix, y1 + iy, iw, ih

    def get_cursor_pos(self, bgr_frame):
        # 💡 使用 conf=0.2 確保能抓到實機指針
        results = self.model.predict(bgr_frame, conf=0.2, verbose=False)
        if len(results[0].boxes) > 0:
            box = sorted(results[0].boxes, key=lambda x: x.conf, reverse=True)[0]
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            h, w = bgr_frame.shape[:2]
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            rel_x = float(cx / w)
            rel_y = float(cy / h)
            return rel_x, rel_y
        return None

    def run(self):
        self.camera = dxcam.create(output_idx=0, output_color="BGR")
        print("啟動攝影機...")
        full_grab = self.camera.grab()
        while full_grab is None:
            full_grab = self.camera.grab()
            time.sleep(0.01)
            
        rect = self.get_absolute_game_rect(full_grab)
        if not rect:
            print("❌ 找不到 Chiaki 視窗！")
            return
        self.cx, self.cy, self.cw, self.ch = rect
        print(f"✅ Chiaki 鎖定: 尺寸 {self.cw}x{self.ch}")
        
        print("\n" + "="*50)
        print("🛠️  準備開始測速標定！")
        print("1. 請先將遊戲內的指針移動到【右下角】。")
        print("2. 準備好後，按下【空白鍵】開始記錄。")
        print("3. 切換到 Chiaki，按死【上】和【左】方向鍵，讓指針移動到左上角。")
        print("4. 到達左上角後，再按一次【空白鍵】結束記錄並計算。")
        print("="*50 + "\n")

        recording = False
        self.camera.start(target_fps=60, video_mode=True)
        
        try:
            while True:
                # 偵測空白鍵啟動
                if not recording and (win32api.GetAsyncKeyState(win32con.VK_SPACE) & 0x8000):
                    print("🔴 開始記錄！請移動指針 (忽略切換視窗的發呆時間)...")
                    recording = True
                    time.sleep(0.5)
                    
                # 偵測空白鍵停止
                if recording and (win32api.GetAsyncKeyState(win32con.VK_SPACE) & 0x8000):
                    print("⏹️ 停止記錄！正在計算數據...")
                    break

                if recording:
                    frame = self.camera.get_latest_frame()
                    if frame is not None:
                        chiaki_frame = frame[self.cy:self.cy+self.ch, self.cx:self.cx+self.cw]
                        pos = self.get_cursor_pos(chiaki_frame)
                        if pos:
                            self.tracking_data.append((time.time(), pos[0], pos[1]))
                            
                # Q 鍵強制退出
                if win32api.GetAsyncKeyState(ord('Q')) & 0x8000:
                    print("🛑 強制退出...")
                    return

        finally:
            self.camera.stop()
            self.analyze_data()

    def analyze_data(self):
        if len(self.tracking_data) < 10:
            print("❌ 收集的數據太少，無法計算。")
            return

        # 1. 尋找真正的「起步點」(過濾掉開頭的發呆時間)
        start_idx = 0
        base_x, base_y = self.tracking_data[0][1], self.tracking_data[0][2]
        for i in range(1, len(self.tracking_data)):
            # 當移動超過 1% 視為開始移動
            if math.hypot(self.tracking_data[i][1] - base_x, self.tracking_data[i][2] - base_y) > 0.01:
                start_idx = i
                break

        # 2. 尋找真正的「停止點」(過濾掉到頂後繼續按著的時間)
        end_idx = len(self.tracking_data) - 1
        last_x, last_y = self.tracking_data[-1][1], self.tracking_data[-1][2]
        for i in range(len(self.tracking_data)-2, start_idx, -1):
            if math.hypot(self.tracking_data[i][1] - last_x, self.tracking_data[i][2] - last_y) > 0.01:
                end_idx = i
                break

        valid_data = self.tracking_data[start_idx:end_idx+1]
        if len(valid_data) < 5:
            print("❌ 有效移動數據太少。")
            return

        t_start, x_start, y_start = valid_data[0]
        t_end, x_end, y_end = valid_data[-1]

        total_time = t_end - t_start
        dist_x = abs(x_end - x_start)
        dist_y = abs(y_end - y_start)

        # 計算每秒移動的歸一化距離 (Velocity)
        vx = dist_x / total_time
        vy = dist_y / total_time
        avg_v = (vx + vy) / 2

        print("\n" + "="*50)
        print("📊 測速標定結果報告")
        print("="*50)
        print(f"有效移動時間: {total_time:.3f} 秒")
        print(f"X軸總位移: {dist_x:.3f} | Y軸總位移: {dist_y:.3f}")
        print(f"X軸速度 (Vx): {vx:.3f} / 秒 | Y軸速度 (Vy): {vy:.3f} / 秒")
        print(f"平均速度 (V): {avg_v:.3f} / 秒")
        print("-" * 50)
        
        print("⏱️ 按下時間 (ms) ↔ 📝 對應位移 (歸一化距離 & 像素)")
        test_ms = [20, 50, 75, 100, 150, 200, 300]
        for ms in test_ms:
            t_sec = ms / 1000.0
            dist_norm = avg_v * t_sec
            dist_px_x = int(dist_norm * self.cw)
            dist_px_y = int(dist_norm * self.ch)
            print(f"[{ms:3d} ms] -> 歸一化距離: {dist_norm:.4f} | X軸像素: ~{dist_px_x}px | Y軸像素: ~{dist_px_y}px")
            
        print("-" * 50)
        print(f"💡 結論推導公式： 所需按下時間 (秒) = 距離差 (Δ) / {avg_v:.3f}")
        print("="*50 + "\n")

if __name__ == "__main__":
    calibrator = SpeedCalibrator()
    calibrator.run()