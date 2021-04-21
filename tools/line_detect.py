import cv2
import os
import numpy as np
import time

def draw_line(img,lines):
    if lines is not None:
    # if (not type(lines) == type(None)):
    #     if lines[0] is not list:
    #         # line=lines
    #         lines=[lines]
        for line in lines:
            line=np.squeeze(line)
            rho = line[0]
            theta = line[1]
            # for rho, theta in line[0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 1)
    return img
def detect_lines(img,task_select):
    assert task_select in ["screw","insert","pickbolt"]
    if task_select=="screw":

        img = cv2.normalize(img, img, 0, 255, norm_type=cv2.NORM_MINMAX)
        hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        gray_mask = cv2.inRange(hsv_img, (50, 0, 140), (150, 100, 200))
        upper_mask = np.ones([img.shape[0], img.shape[1]]).astype(np.uint8)
        upper_mask[upper_mask.shape[1] - int(upper_mask.shape[1] * 0.5):,:] = 0


        kernel = np.ones((8, 4), np.uint8)
        blue_mask = cv2.inRange(hsv_img, (100, 100, 0), (150, 255, 130))
        blue_mask = cv2.dilate(blue_mask, kernel, iterations=5)

        # black_mask = cv2.inRange(hsv_img, (0, 0, 20), (255, 100, 80))  # screw head black

        # # => screw head black
        # black_mask = cv2.inRange(hsv_img, (40, 0, 0), (160, 150, 80))
        # black_mask = cv2.bitwise_and(black_mask, black_mask, mask=~blue_mask)
        # black_mask = cv2.bitwise_and(black_mask, black_mask, mask=upper_mask)
        # kernel = np.ones((3, 3), np.uint8)
        # black_mask = cv2.dilate(black_mask, kernel, iterations=1)
        # black_mask = cv2.erode(black_mask, kernel, iterations=1)
        # black_mask = cv2.erode(black_mask, kernel, iterations=3)
        # black_mask = cv2.dilate(black_mask, kernel, iterations=3)

        # # => screw head yellow => yellow mask
        black_mask = cv2.inRange(hsv_img, (20, 80, 150), (40, 255, 255))  #
        # black_mask = cv2.bitwise_and(black_mask, black_mask, mask=~
        # blue_mask)
        # black_mask = cv2.bitwise_and(black_mask, black_mask, mask=upper_mask)
        # kernel = np.ones((3, 3), np.uint8)
        # black_mask = cv2.erode(black_mask, kernel, iterations=1)
        # black_mask = cv2.dilate(black_mask, kernel, iterations=1)

        edges = cv2.Canny(black_mask, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 20)

        # lines filter
        res_lines = None
        if lines is not None:
            res_lines=[]
            for l in lines:
                # print(l)
                theta=l.squeeze()[1]
                # if (theta>np.pi/5 and theta<np.pi*3/4) or (theta>np.pi):#and theta<np.pi+np.pi-np.pi/5:
                #     pass
                #
                #     res_lines.append(l)
                # else:
                if l[0][0]>0:
                    l[0][0]=-l[0][0]
                    l[0][1]+=3.1415926
                res_lines.append(l)
                    # if len()
            if len(res_lines):
                # print(res_lines)
                raw_reslines=res_lines

                res_lines=np.array(res_lines)
                res_lines_multi=res_lines.copy()
                # print("res_lines_multi",res_lines_multi)
                res_lines=res_lines.squeeze(axis=1).mean(0)
                res_lines=np.expand_dims(res_lines,0)
                res_lines=np.expand_dims(res_lines,0)

                pass
            else:
                res_lines=None
        try:
            line_img = draw_line(img.copy(), res_lines_multi)
            cv2.imshow("testimg5,line test",
                       cv2.cvtColor(cv2.resize(line_img, (int(960 / 2), int(720 / 2))), cv2.COLOR_RGB2BGR))
        except:
            pass
        cv2.imshow("testimg3",
                   cv2.cvtColor(cv2.resize(black_mask, (int(960 / 2), int(720 / 2))), cv2.COLOR_RGB2BGR))
        cv2.imshow("testimg4",
                   cv2.cvtColor(cv2.resize(edges, (int(960 / 2), int(720 / 2))), cv2.COLOR_RGB2BGR))

    return res_lines
if __name__=="__main__":

    # paras
    task_select="screw"

    imgs_dir = os.path.join(os.getcwd(),"..", "saveimgs", "fakeimgflow")
    imgs_names = os.listdir(imgs_dir)

    for fix_id,_ in enumerate(imgs_names):
        fix_id = 5
        # fix_id = 98
        # fix_id = 60
        # fix_id = -50
        # fix_id = -51
    #     print(fix_id)
    #     fix_id=98#93#86#98
        if "l.png" in _:
            continue
        imgf=os.path.join(imgs_dir,imgs_names[fix_id])


        img=cv2.imread(imgf)
        img = np.array(img).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgraw=img.copy()


        # process and line detect
        lines=detect_lines(img,task_select=task_select)
        if lines is None:
            pass
        else:
            # print("liens=",len(lines))
            pass
        # img=cv2.bitwise_and(img,img,mask=black_mask)
        img=draw_line(img,lines)

        w=2
        cv2.imshow("testimg",
                   cv2.cvtColor(cv2.resize(img, (int(960 / w), int(720 / w))), cv2.COLOR_RGB2BGR))
        cv2.imshow("testimg2",
                   cv2.cvtColor(cv2.resize(imgraw, (int(960 / w), int(720 / w))), cv2.COLOR_RGB2BGR))
        cv2.waitKey(2)
        time.sleep(0.01)

        # key = cv2.waitKey(2)
        #     if key == 27:  # exit on ESC
        #         break
    cv2.destroyAllWindows()
    # print(imgs_name)