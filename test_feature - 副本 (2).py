import cv2
import numpy as np

def get_masked_template_with_centroid(path, rotate_code=None):
    tpl = cv2.imread(path)
    if tpl is None: return None
    if rotate_code is not None:
        tpl = cv2.rotate(tpl, rotate_code)
    
    gray = cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)
    
    M = cv2.moments(mask)
    # 🌟 保持浮點精度，避免 10px 偏移
    local_cX = M["m10"] / M["m00"] if M["m00"] != 0 else tpl.shape[1] / 2.0
    local_cY = M["m01"] / M["m00"] if M["m00"] != 0 else tpl.shape[0] / 2.0
    
    coords = np.where(mask > 0)
    colors = tpl[coords]
    
    return {
        'coords': coords, 'colors': colors,
        'local_centroid': (local_cX, local_cY),
        'w': tpl.shape[1], 'h': tpl.shape[0],
        'name': f"{path}_rot_{rotate_code}"
    }

def test_centroid_precision_match():
    templates = [
        get_masked_template_with_centroid('cursor2.png'),
        get_masked_template_with_centroid('cursor2.png', cv2.ROTATE_180),
        get_masked_template_with_centroid('cursor3.png'),
        get_masked_template_with_centroid('cursor3.png', cv2.ROTATE_180)
    ]
    
    scene = cv2.imread('screenshot.png')
    if scene is None: return print("❌ 找不到 screenshot.png")

    # 1. 精確採樣目標色
    sample_tpl = cv2.imread('cursor2.png')
    h_s, w_s = sample_tpl.shape[:2]
    target_color = np.mean(sample_tpl[h_s//2-1:h_s//2+2, w_s//2-1:w_s//2+2], axis=(0, 1))
    
    # 2. 生成距離地圖
    diff = scene.astype(np.float32) - target_color
    dist_map = np.sqrt(np.sum(diff**2, axis=2))
    dist_map_visual = cv2.normalize(dist_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # 🌟 新增：形狀篩選定位 (解決找錯對象的問題)
    _, mask_for_loc = cv2.threshold(dist_map_visual, 40, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(mask_for_loc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 80 < area < 1000: # 過濾掉背景大塊山石和微小雜點
            x, y, w, h = cv2.boundingRect(cnt)
            # 這是我們真正的「感興趣區域」起點
            candidates.append((x - 5, y - 5)) 

    if not candidates:
        # 如果形狀篩選沒找到，回退到全局最小值
        _, _, min_loc, _ = cv2.minMaxLoc(dist_map)
        candidates = [min_loc]

    best_val = float('inf')
    best_final_loc = None
    best_tpl = None

    # 3. 只在候選對象周圍進行微調
    for base_x, base_y in candidates:
        for t in templates:
            if t is None: continue
            # 擴大微調範圍至 10px，徹底消除偏移
            for dy in range(-10, 11):
                for dx in range(-10, 11):
                    curr_x, curr_y = base_x + dx, base_y + dy
                    try:
                        roi_pixels = scene[curr_y + t['coords'][0], curr_x + t['coords'][1]]
                        error = np.mean(np.abs(roi_pixels.astype(np.float32) - t['colors']))
                        
                        if error < best_val:
                            best_val = error
                            final_cX = curr_x + t['local_centroid'][0]
                            final_cY = curr_y + t['local_centroid'][1]
                            best_final_loc = (final_cX, final_cY, curr_x, curr_y)
                            best_tpl = t
                    except: continue

    # 4. 結果顯示
    if best_tpl and best_val < 70: # 增加誤差門檻，防止強行匹配
        cX, cY, topX, topY = best_final_loc
        cv2.circle(scene, (int(cX), int(cY)), 3, (0, 0, 255), -1)
        cv2.circle(scene, (int(cX), int(cY)), 10, (0, 255, 0), 1) # 10px 綠色範圍圈
        cv2.rectangle(scene, (topX, topY), (topX + best_tpl['w'], topY + best_tpl['h']), (255, 0, 0), 1)
        
        print(f"✅ 精確鎖定！坐標: ({int(cX)}, {int(cY)}), 誤差分值: {best_val:.2f}")
    else:
        print("❌ 未能識別到手指（可能被背景干擾或不在畫面中）")

    cv2.imshow("Shape-Guided Centroid Match", scene)
    cv2.imshow("Distance Map (Radar)", dist_map_visual)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_centroid_precision_match()