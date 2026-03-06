import cv2
import numpy as np

def test_four_direction_masked():
    # 1. 讀取指針並生成 4 方向遮罩模板
    tpl_path = 'cursor2.png'
    tpl_img = cv2.imread(tpl_path, cv2.IMREAD_COLOR)
    if tpl_img is None: return print("❌ 找不到 cursor2.png")

    # --- 自動剝離背景製作遮罩 ---
    # 取左上角顏色，生成「只有手是白色，其餘全黑」的 mask
    bg_color = tpl_img[0, 0]
    lower_bg = np.clip(bg_color.astype(np.int16) - 15, 0, 255).astype(np.uint8)
    upper_bg = np.clip(bg_color.astype(np.int16) + 15, 0, 255).astype(np.uint8)
    bg_mask = cv2.inRange(tpl_img, lower_bg, upper_bg)
    fg_mask = cv2.bitwise_not(bg_mask) 

    hands = []
    for angle in [0, 90, 180, 270]:
        if angle == 0:
            m_img, m_mask = tpl_img, fg_mask
        elif angle == 90:
            m_img = cv2.rotate(tpl_img, cv2.ROTATE_90_CLOCKWISE)
            m_mask = cv2.rotate(fg_mask, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            m_img = cv2.rotate(tpl_img, cv2.ROTATE_180)
            m_mask = cv2.rotate(fg_mask, cv2.ROTATE_180)
        else:
            m_img = cv2.rotate(tpl_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            m_mask = cv2.rotate(fg_mask, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        hands.append({
            'gray': cv2.cvtColor(m_img, cv2.COLOR_BGR2GRAY),
            'mask': m_mask,
            'angle': angle
        })

    # 2. 讀取測試場景
    scene_bgr = cv2.imread('screenshot.png')
    if scene_bgr is None: return print("❌ 找不到 screenshot.png")
    scene_gray = cv2.cvtColor(scene_bgr, cv2.COLOR_BGR2GRAY)

    best_val = -1
    best_loc = None
    best_hand = None

    # 3. 4 方向全螢幕暴力匹配 (帶遮罩)
    print("正在進行 4 方向遮罩匹配...")
    for hand in hands:
        # 🌟 核心：使用遮罩，OpenCV 會完全無視背景像素
        res = cv2.matchTemplate(scene_gray, hand['gray'], cv2.TM_CCORR_NORMED, mask=hand['mask'])
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        print(f"  角度 {hand['angle']}° | 得分: {max_val:.4f}")

        if max_val > best_val:
            best_val = max_val
            best_loc = max_loc
            best_hand = hand

    # 4. 繪製結果
    if best_val > 0.85: # 門檻設高，防止抓到石頭
        h, w = best_hand['gray'].shape
        cv2.rectangle(scene_bgr, best_loc, (best_loc[0]+w, best_loc[1]+h), (0, 255, 0), 2)
        cv2.putText(scene_bgr, f"MATCH! {best_val:.2f}", (best_loc[0], best_loc[1]-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        print(f"✅ 成功鎖定！最佳角度: {best_hand['angle']}°")
    else:
        print("❌ 匹配分數太低，未達成識別。")

    cv2.imshow("Final Brute Force Test", scene_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_four_direction_masked()