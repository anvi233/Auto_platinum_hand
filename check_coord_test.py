import cv2
import numpy as np
import time
import win32gui
from ultralytics import YOLO
import dxcam

# 載入 V3 模型
model = YOLO(r'E:\Myst_Project\v3_final_1900\weights\best.pt')
camera = dxcam.create(output_idx=0, output_color="BGR")

def get_window_client_area(keyword):
    """獲取軟體內部渲染區域 (直接借用 data_collector 的邏輯)"""
    hwnds = []
    def enum_cb(hwnd, param):
        if win32gui.IsWindowVisible(hwnd):
            title = win32gui.GetWindowText(hwnd).lower()
            if keyword in title and "powershell" not in title and "code" not in title:
                param.append(hwnd)
    win32gui.EnumWindows(enum_cb, hwnds)
    
    if not hwnds: return None
        
    target_hwnd = hwnds[0]
    rect = win32gui.GetClientRect(target_hwnd)
    left, top = win32gui.ClientToScreen(target_hwnd, (rect[0], rect[1]))
    right, bottom = win32gui.ClientToScreen(target_hwnd, (rect[2], rect[3]))
    
    w = right - left
    h = bottom - top
    if w <= 0 or h <= 0: return None
    return left, top, w, h

def get_pure_game_scene(bgr_image):
    """自動切除瀏覽器UI與影片黑邊 (直接借用 data_collector 的邏輯)"""
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    
    _, dark_mask = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    dark_mask_closed = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel)
    
    cnts_dark, _ = cv2.findContours(dark_mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts_dark: return 0, 0, bgr_image.shape[1], bgr_image.shape[0]
        
    largest_dark_cnt = max(cnts_dark, key=cv2.contourArea)
    cx, cy, cw, ch = cv2.boundingRect(largest_dark_cnt)
    
    container_roi_gray = gray[cy:cy+ch, cx:cx+cw]
    _, game_mask = cv2.threshold(container_roi_gray, 20, 255, cv2.THRESH_BINARY)
    game_mask = cv2.morphologyEx(game_mask, cv2.MORPH_CLOSE, kernel)
    game_mask = cv2.morphologyEx(game_mask, cv2.MORPH_OPEN, kernel)
    
    cnts_game, _ = cv2.findContours(game_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts_game: return cx, cy, cw, ch
        
    largest_game_cnt = max(cnts_game, key=cv2.contourArea)
    gx, gy, gw, gh = cv2.boundingRect(largest_game_cnt)
    
    return cx + gx, cy + gy, gw, gh

def process_target(full_frame, keyword, locked_roi):
    """處理單一目標（YouTube 或 Chiaki），返回相對座標和新的鎖定ROI"""
    rect = get_window_client_area(keyword)
    if rect is None:
        return None, None, None, locked_roi
        
    tx, ty, tw, th = rect
    window_img = full_frame[ty:ty+th, tx:tx+tw]
    
    # 動態鎖定遊戲畫面區域
    if locked_roi is None:
        ix, iy, iw, ih = get_pure_game_scene(window_img)
        if iw > 200 and ih > 200:
            locked_roi = (ix, iy, iw, ih)
            print(f"🔒 [{keyword}] 畫面裁切已鎖定！尺寸: {iw}x{ih}")
        else:
            return None, None, None, locked_roi
    else:
        ix, iy, iw, ih = locked_roi

    # 裁出純淨遊戲畫面
    pure_game_img = window_img[iy:iy+ih, ix:ix+iw]
    
    # 進行 YOLO 預測
    results = model.predict(pure_game_img, conf=0.6, verbose=False)
    
    if len(results[0].boxes) > 0:
        box = sorted(results[0].boxes, key=lambda x: x.conf, reverse=True)[0]
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        
        # 取得純淨畫面內的中心點
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        
        # 計算相對於純淨畫面的歸一化比例
        rel_x = round(cx / iw, 4)
        rel_y = round(cy / ih, 4)
        cls_name = model.names[int(box.cls)]
        
        return rel_x, rel_y, cls_name, locked_roi
        
    return None, None, None, locked_roi

if __name__ == "__main__":
    print("🚀 啟動全自動雙視窗追蹤測試...")
    print("請確保 YouTube 和 Chiaki 視窗可見。")
    print("-" * 60)
    
    camera.start(target_fps=30, video_mode=True)
    
    yt_locked_roi = None
    ck_locked_roi = None
    
    try:
        while True:
            start_time = time.time()
            full_frame = camera.get_latest_frame()
            if full_frame is None: continue

            # 處理 YouTube 端
            y_x, y_y, y_cls, yt_locked_roi = process_target(full_frame, "youtube", yt_locked_roi)
            
            # 處理 Chiaki 端
            c_x, c_y, c_cls, ck_locked_roi = process_target(full_frame, "chiaki", ck_locked_roi)
            
            # 把原本的 y_str 和 c_str 替換成這樣：
            y_str = f"YT Px:({float(y_x):.3f}, {float(y_y):.3f}) [{y_cls}]" if y_x else "YT: 未偵測"
            c_str = f"CK Px:({float(c_x):.3f}, {float(c_y):.3f}) [{c_cls}]" if c_x else "CK: 未偵測"

            # 只有當至少有一邊鎖定成功時才打印
            if yt_locked_roi or ck_locked_roi:
                print(f"{y_str:<30} | {c_str}")
            
            # 控制台輸出頻率約 1 秒
            elapsed = time.time() - start_time
            time.sleep(max(0, 1.0 - elapsed))
            
    except KeyboardInterrupt:
        camera.stop()
        print("\n🛑 測試已手動停止。")