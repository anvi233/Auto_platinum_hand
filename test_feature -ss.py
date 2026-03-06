import cv2
import numpy as np

def get_masked_template_with_centroid(path, rotate_code=None):
    """讀取模板，生成遮罩並預計算手指重心"""
    tpl = cv2.imread(path)
    if tpl is None: return None
    if rotate_code is not None:
        tpl = cv2.rotate(tpl, rotate_code)
    
    # 1. 提取手指部分 (排除背景)
    gray = cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
    
    # 2. 計算手指在模板內的質心 (Centroid)
    M = cv2.moments(mask)
    if M["m00"] == 0:
        # 如果無法計算質心，回退到幾何中心
        local_cX, local_cY = tpl.shape[1] // 2, tpl.shape[0] // 2
    else:
        local_cX = int(M["m10"] / M["m00"])
        local_cY = int(M["m01"] / M["m00"])
    
    # 3. 獲取手指像素點數據用於比對
    coords = np.where(mask > 0)
    colors = tpl[coords]
    
    return {
        'coords': coords,
        'colors': colors,
        'local_centroid': (local_cX, local_cY), # 模板內的相對重心
        'w': tpl.shape[1],
        'h': tpl.shape[0],
        'name': f"{path}_rot_{rotate_code}"
    }

def test_centroid_precision_match():
    # 1. 準備 4 方向模板 (帶重心數據)
    templates = [
        get_masked_template_with_centroid('cursor2.png'),
        get_masked_template_with_centroid('cursor2.png', cv2.ROTATE_180),
        get_masked_template_with_centroid('cursor3.png'),
        get_masked_template_with_centroid('cursor3.png', cv2.ROTATE_180)
    ]
    
    scene = cv2.imread('screenshot.png')
    if scene is None: return print("❌ 找不到 screenshot.png")

    # 2. 獲取手指特徵色 (採樣自 cursor2 中心)
    sample_tpl = cv2.imread('cursor2.png')
    h_s, w_s = sample_tpl.shape[:2]
    target_color = np.mean(sample_tpl[h_s//2-2:h_s//2+3, w_s//2-2:w_s//2+3], axis=(0, 1))
    
    # 3. 生成顏色距離地圖定位候選點
    diff = scene.astype(np.float32) - target_color
    dist_map = np.sqrt(np.sum(diff**2, axis=2))
    _, _, min_loc, _ = cv2.minMaxLoc(dist_map)
    
    base_x, base_y = min_loc # 初步定位點 (左上角附近)
    
    best_val = float('inf')
    best_final_loc = None
    best_tpl = None

    # 4. 在定位點周圍微調並驗證 4 方向
    # 微調範圍 +/- 5 像素以應對抗鋸齒偏移
    for t in templates:
        if t is None: continue
        for dy in range(-5, 6):
            for dx in range(-5, 6):
                curr_x, curr_y = base_x + dx, base_y + dy
                try:
                    roi_pixels = scene[curr_y + t['coords'][0], curr_x + t['coords'][1]]
                    error = np.mean(np.abs(roi_pixels.astype(np.float32) - t['colors']))
                    
                    if error < best_val:
                        best_val = error
                        # 最終點 = 左上角坐標 + 模板內相對重心
                        final_cX = curr_x + t['local_centroid'][0]
                        final_cY = curr_y + t['local_centroid'][1]
                        best_final_loc = (final_cX, final_cY, curr_x, curr_y)
                        best_tpl = t
                except:
                    continue

    # 5. 結果顯示
    if best_tpl:
        cX, cY, topX, topY = best_final_loc
        # 畫出手指質心 (紅點)
        cv2.circle(scene, (cX, cY), 5, (0, 0, 255), -1)
        # 畫出匹配框 (綠框)
        cv2.rectangle(scene, (topX, topY), (topX + best_tpl['w'], topY + best_tpl['h']), (0, 255, 0), 2)
        
        cv2.putText(scene, f"CENTER: ({cX}, {cY})", (topX, topY-10), 0, 0.5, (0, 255, 0), 2)
        print(f"✅ 精確中心匹配成功！")
        print(f"🎯 質心坐標: ({cX}, {cY})")
        print(f"📏 像素誤差: {best_val:.2f}")
    else:
        print("❌ 依然找不到精確匹配。")

    cv2.imshow("Centroid Precision Match", scene)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_centroid_precision_match()