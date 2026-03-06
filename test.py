import cv2
import numpy as np
import dxcam
import win32gui
import win32con
import ctypes
import time

# 解決高 DPI 縮放導致的抓取座標偏移問題
ctypes.windll.user32.SetProcessDPIAware()

def run_final_test():
    # 初始化 DXCAM
    camera = dxcam.create(output_color="BGR")
    
    # 🎯 標題過濾關鍵字
    target_keyword = "chiaki-ng" 
    
    # 🎯 存儲鎖定後的固定偏移 (ox, oy, width, height)
    # 一旦鎖定，就不再於迴圈中累加或修改，徹底解決畫面縮小問題
    fixed_roi = None 

    print("🚀 Chiaki 穩定版測試啟動...")
    print("💡 操作說明：")
    print("   1. 確保 Chiaki 視窗在螢幕上可見。")
    print("   2. 按下 's' 鍵：鎖定遊戲區域（去標題列、去黑框）。")
    print("   3. 按下 'q' 鍵：退出測試。")

    while True:
        # 抓取螢幕全圖
        full_frame = camera.grab()
        if full_frame is None: continue

        # 1. 尋找 Chiaki 視窗 (排除掉標題包含 PowerShell 的終端機)
        hwnds = []
        def enum_cb(hwnd, param):
            if win32gui.IsWindowVisible(hwnd):
                title = win32gui.GetWindowText(hwnd)
                if target_keyword in title.lower() and "powershell" not in title.lower():
                    param.append(hwnd)
        win32gui.EnumWindows(enum_cb, hwnds)
        
        if not hwnds:
            cv2.putText(full_frame, "Searching for Chiaki...", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Debug View", cv2.resize(full_frame, (640, 360)))
            if cv2.waitKey(1) & 0xFF == ord('q'): break
            continue

        hwnd = hwnds[0]
        # 獲取視窗當前在螢幕上的即時位置 (wx, wy)
        rect = win32gui.GetWindowRect(hwnd)
        wx, wy, ww, wh = rect[0], rect[1], rect[2]-rect[0], rect[3]-rect[1]

        # 2. 核心識別邏輯：按下 's' 觸發一次性鎖定
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s') or (fixed_roi is None):
            # 截取「整個視窗圖」進行分析
            window_img = full_frame[wy:wy+wh, wx:wx+ww]
            if window_img.size > 0:
                h, w = window_img.shape[:2]
                
                # 🎯 策略：強制跳過頂部 45 像素(標題列)與四周 5 像素(邊框)
                top_m, side_m = 45, 5
                if h > top_m:
                    roi_view = window_img[top_m:h-side_m, side_m:w-side_m]
                    gray = cv2.cvtColor(roi_view, cv2.COLOR_BGR2GRAY)
                    
                    # 二值化過濾非黑色區域
                    _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
                    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    if cnts:
                        c = max(cnts, key=cv2.contourArea)
                        rx, ry, rw, rh = cv2.boundingRect(c)
                        
                        # 🎯 關鍵：計算「固定偏移量」。rx/ry 是相對於視窗左上角的內縮距離
                        # 這個數值一旦算出就存入 fixed_roi，不再變動
                        fixed_roi = (rx + side_m, ry + top_m, rw, rh)
                        print(f"✅ 區域鎖定成功！遊戲區起始點：x={rx+side_m}, y={ry+top_m}, 尺寸: {rw}x{rh}")

        # 3. 使用鎖定的固定偏移從 full_frame 進行「絕對座標」裁切
        if fixed_roi:
            ox, oy, lw, lh = fixed_roi
            # 💡 這裡執行：視窗動態坐標 + 鎖定的固定偏移。絕對不會導致畫面累加縮小。
            pure_game_view = full_frame[wy + oy : wy + oy + lh, wx + ox : wx + ox + lw]
            
            if pure_game_view.size > 0:
                # 唯一的監控窗口
                info = f"Fixed ROI: {lw}x{lh} Offset:({ox}, {oy})"
                cv2.putText(pure_game_view, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow("Chiaki_Pure_Game_View", pure_game_view)
        
        if key == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_final_test()