import cv2
import time
import dxcam
import win32gui
import os

# 完全復用主程序中的遊戲畫面精確裁剪邏輯
def get_pure_game_scene(bgr_image):
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

def get_absolute_game_rect(keyword, full_screen_frame):
    hwnds = []
    def enum_cb(hwnd, param):
        if win32gui.IsWindowVisible(hwnd):
            title = win32gui.GetWindowText(hwnd).lower()
            if keyword in title and "powershell" not in title:
                param.append(hwnd)
    win32gui.EnumWindows(enum_cb, hwnds)
    
    if not hwnds: return None
        
    hwnd = hwnds[0]
    rect = win32gui.GetWindowRect(hwnd)
    
    sh, sw = full_screen_frame.shape[:2]
    x1, y1, x2, y2 = max(0, rect[0]), max(0, rect[1]), min(sw, rect[2]), min(sh, rect[3])
    
    if x1 >= x2 or y1 >= y2: return None
    
    window_img = full_screen_frame[y1:y2, x1:x2]
    ix, iy, iw, ih = get_pure_game_scene(window_img)
    
    return x1 + ix, y1 + iy, iw, ih

def auto_capture():
    camera = dxcam.create(output_idx=0, output_color="BGR")
    print("啟動 dxcam 截圖引擎...")
    
    # 建立保存圖片的目錄
    save_dir = "chiaki_samples"
    os.makedirs(save_dir, exist_ok=True)
    
    full_grab = camera.grab()
    while full_grab is None:
        full_grab = camera.grab()
        time.sleep(0.01)
        
    rect = get_absolute_game_rect("chiaki", full_grab)
    if not rect:
        print("❌ 找不到 Chiaki 視窗，請確認遊戲已開啟並顯示在畫面上！")
        return
        
    cx, cy, cw, ch = rect
    print(f"✅ 精確鎖定 Chiaki 遊戲區域: ({cx}, {cy}, {cw}, {ch})")
    print("📸 開始每秒自動截圖... (按 Ctrl+C 終止程序)")
    
    count = 1
    try:
        while True:
            frame = camera.grab()
            if frame is not None:
                # 裁剪出純淨的實機畫面
                chiaki_crop = frame[cy:cy+ch, cx:cx+cw]
                file_path = os.path.join(save_dir, f"chiaki_frame_{count:04d}.jpg")
                cv2.imwrite(file_path, chiaki_crop)
                print(f"💾 已保存: {file_path}")
                count += 1
            
            # 等待 1 秒
            time.sleep(1.0)
            
    except KeyboardInterrupt:
        print("\n🛑 截圖已手動終止，圖片保存在 chiaki_samples 文件夾中。")
    finally:
        camera.stop()

if __name__ == "__main__":
    auto_capture()