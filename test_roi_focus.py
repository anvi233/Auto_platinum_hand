import cv2
import numpy as np
import dxcam

class ROITester:
    def __init__(self):
        self.camera = dxcam.create(output_idx=0, output_color="BGR")
        self.prev_patch = None

    def process_roi_focus(self, gray_frame, anchor_pos, roi_size=60, cursor_size=15):
        """核心函數 1：提取 ROI、模糊降噪、挖空指針"""
        if gray_frame is None or anchor_pos is None:
            return None, None

        cx, cy = anchor_pos
        h, w = gray_frame.shape
        
        # 1. 定義大 ROI 矩形
        x1, y1 = max(0, cx - roi_size), max(0, cy - roi_size)
        x2, y2 = min(w, cx + roi_size), min(h, cy + roi_size)
        roi_box = (x1, y1, x2, y2)

        if x2 - x1 <= cursor_size * 2 or y2 - y1 <= cursor_size * 2:
            return None, roi_box

        # 2. 提取 ROI 並模糊降噪
        patch = cv2.GaussianBlur(gray_frame[y1:y2, x1:x2].copy(), (5, 5), 0)
        
        # 3. 挖空中心指針區域 (設為黑色)
        ph, pw = patch.shape
        ix1, iy1 = max(0, pw//2 - cursor_size), max(0, ph//2 - cursor_size)
        ix2, iy2 = min(pw, pw//2 + cursor_size), min(ph, ph//2 + cursor_size)
        patch[iy1:iy2, ix1:ix2] = 0
        
        return patch, roi_box

    def check_roi_change(self, current_patch, prev_patch):
        """核心函數 2：比對兩幀 Patch 的變動像素數量"""
        if current_patch is None or prev_patch is None:
            return 0, None
            
        if current_patch.shape != prev_patch.shape:
            return 0, None

        # 計算差異並二值化 (> 15 的亮度差異才算數)
        diff = cv2.absdiff(current_patch, prev_patch)
        _, diff_thresh = cv2.threshold(diff, 15, 255, cv2.THRESH_BINARY)
        
        # 計算變動的像素總數
        change_count = cv2.countNonZero(diff_thresh)
        
        return change_count, diff_thresh

    def run(self):
        self.camera.start(target_fps=30, video_mode=True)
        print("▶️ ROI 單元測試啟動！")
        print("請在螢幕【正中央】製造畫面變動（如拖動視窗、播放影片）。")
        print("按 'q' 退出測試。")
        
        # 固定錨點為 1920x1080 螢幕的正中央 (若解析度不同請自行微調)
        anchor_pos = (1920 // 2, 1080 // 2)

        while True:
            frame = self.camera.get_latest_frame()
            if frame is None: continue
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # 1. 單幀處理：取得挖空且模糊的 Patch
            curr_patch, roi_box = self.process_roi_focus(gray, anchor_pos)
            
            if curr_patch is not None:
                # 為了方便肉眼觀察，將原本小小的 Patch 放大 3 倍顯示
                patch_display = cv2.resize(curr_patch, (360, 360), interpolation=cv2.INTER_NEAREST)
                cv2.imshow("1. ROI Patch (Blurred & Masked)", patch_display)

                # 2. 比對變動
                if self.prev_patch is not None:
                    change_count, diff_thresh = self.check_roi_change(curr_patch, self.prev_patch)
                    
                    if diff_thresh is not None:
                        # 將變動遮罩放大顯示
                        diff_display = cv2.resize(diff_thresh, (360, 360), interpolation=cv2.INTER_NEAREST)
                        # 在畫面上印出數值
                        cv2.putText(diff_display, f"Change: {change_count} px", (10, 30), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255), 2)
                        cv2.imshow("2. Difference Mask", diff_display)
                        
                        # 模擬 Hover 時的點擊判定條件 (變動數值大於 10)
                        if change_count > 10:
                            print(f"🔥 偵測到背景變動！變動強度: {change_count} 像素")

                self.prev_patch = curr_patch

            # 顯示原圖與測試區域框線
            display_frame = frame.copy()
            if roi_box:
                x1, y1, x2, y2 = roi_box
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2) # 綠色大框 (ROI)
                
                # 畫出中間被挖空的紅色區域示意
                cx, cy = anchor_pos
                cv2.rectangle(display_frame, (cx-15, cy-15), (cx+15, cy+15), (0, 0, 255), 1) 
                
            cv2.imshow("0. Original Screen", cv2.resize(display_frame, (960, 540)))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.camera.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    tester = ROITester()
    tester.run()