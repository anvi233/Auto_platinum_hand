import cv2
import numpy as np

def test_final_perfect_matching():
    # 1. 加載場景與模板
    scene = cv2.imread('screenshot.png')
    tpl = cv2.imread('cursor2.png')
    if scene is None or tpl is None: return print("❌ 讀取檔案失敗")

    # 2. 自動獲取手指核心色 (利用你確認過的顏色距離法)
    h_t, w_t = tpl.shape[:2]
    # 取中心 5x5 區域均值
    core_area = tpl[h_t//2-2:h_t//2+3, w_t//2-2:w_t//2+3]
    target_color = np.mean(core_area, axis=(0, 1)) 
    print(f"🎯 提取手指目標色 (BGR): {target_color}")

    # 3. 【關鍵】高精度距離計算 (還原你要的那張 Distance Map)
    diff = scene.astype(np.float32) - target_color
    dist_map = np.sqrt(np.sum(diff**2, axis=2))
    
    # 🌟 這個是你說你要的那張圖的灰階數據
    # 歸一化到 0-255，指針是黑點，背景是灰/白色
    dist_map_visual = cv2.normalize(dist_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # 4. 生成 Mask (閾值調回你覺得清楚的數值，通常在 25-40 之間)
    # 反轉二值化：距離近（黑）變目標（白）
    _, mask = cv2.threshold(dist_map_visual, 35, 255, cv2.THRESH_BINARY_INV)

    # 5. 形狀幾何匹配 (Contour Analysis)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    best_match_loc = None
    max_score = -1

    for cnt in contours:
        area = cv2.contourArea(cnt)
        # 第一層：面積約束 (過濾雜點和大型山石)
        if area < 80 or area > 1000:
            continue
            
        # 🌟 第二層：旋轉矩形擬合 (比 boundingRect 更精確)
        rect = cv2.minAreaRect(cnt)
        (x, y), (w, h), angle = rect
        if min(w, h) == 0: continue
        
        # 第三層：長寬比判定 (手指通常在 1.4 到 2.8 之間)
        aspect_ratio = max(w, h) / min(w, h)
        
        # 第四層：緊湊度 (Solidity) - 指針形狀是凸的，Solidity 很高；雜亂背景低
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area if hull_area > 0 else 0

        # 綜合評分：手指應該是 Solidity > 0.8 且 AR 在 1.4 - 2.8
        if 1.3 < aspect_ratio < 3.2 and solidity > 0.75:
            # 這裡我們取面積大且最符合比例的作為目標
            score = solidity * area 
            if score > max_score:
                max_score = score
                # 記錄中心點和方向
                direction = "VERTICAL (UP/DOWN)" if h > w else "HORIZONTAL (LEFT/RIGHT)"
                best_match_loc = (rect, aspect_ratio, solidity, direction)

    # 6. 結果判定與輸出 (根據你的要求 180 度補齊)
    if best_match_loc:
        rect, ar, sol, dir_str = best_match_loc
        box = cv2.boxPoints(rect)
        box = box.astype(int)
        
        cv2.drawContours(scene, [box], 0, (0, 255, 0), 2)
        cx, cy = int(rect[0][0]), int(rect[0][1])
        cv2.circle(scene, (cx, cy), 5, (0, 0, 255), -1)
        
        cv2.putText(scene, f"MATCH! {dir_str}", (cx-20, cy-20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        print(f"✅ 成功鎖定！坐標: ({cx}, {cy}), 方向: {dir_str}, 比例: {ar:.2f}")
    else:
        print("❌ 匹配失敗：在 Distance Map 中未發現符合幾何描述的色塊。")

    # 顯示那張「清楚的圖」和結果
    cv2.imshow("Color Distance Map (This is what you want)", dist_map_visual)
    cv2.imshow("Final Perfect Match", scene)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_final_perfect_matching()