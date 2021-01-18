import cv2
from keras.preprocessing import image
import json
import os
import random
import glob
from shutil import copyfile
import imghdr

def add_padding(img, ratio=256, add_bot = True, add_right = True):  #add padding image h and w % 256
    right, bot = 0, 0 #add to right and bot
    if (img.shape[1] % ratio) != 0:
        right = ((img.shape[1]//ratio) + 1) * ratio - img.shape[1]
    if (img.shape[0] % ratio) != 0:
        bot = ((img.shape[0]//ratio) + 1) * ratio - img.shape[0]

    if add_bot:
        imgPadding = cv2.copyMakeBorder(img, 0, bot, 0, 0, cv2.BORDER_CONSTANT)
    if add_right:
        imgPadding = cv2.copyMakeBorder(img, 0, 0, 0, right, cv2.BORDER_CONSTANT)
    return imgPadding

def crop_image_cv2(imgPadding, ratio=256, save_img = False, path_save = '/home/minhhoang/tu/head_detection/cropimage/'): # crop img_original to many img 256x256 return lst_img
    x = 0
    y = 0
    vt = 0
    lst_img = []
    for i in range(int(imgPadding.shape[0]/ratio)):
        for j in range(int(imgPadding.shape[1]/ratio)):
            img_crop = imgPadding[x:x+ratio, y:y+ratio, :]
            if save_img:
                cv2.imwrite(path_save + str(i) + str(j) + ".jpg", img_crop)
            lst_img.append(img_crop)
            # cv2.imwrite("/home/minhhoang/Desktop/test1/abcd/img_crop"+str('{0:04}'.format(vt))+".jpg",img_crop)
            y += ratio
            vt +=1
        y = 0
        x += ratio
    return lst_img

def crop_image_keras(imgPadding, ratio = 256, save_img = False, path_save = '/home/minhhoang/tu/head_detection/cropimage/'):
    x = 0
    y = 0
    vt = 0
    lst_img = []
    w, h = imgPadding.size
    for i in range(int(w/ratio)):
        for j in range(int(h/ratio)):
            img_crop = imgPadding.crop((x, y, x + ratio, y + ratio))
            if save_img:
                img_crop.save(path_save + str(i) + str(j) + ".jpg")
            # img_crop = imgPadding[x:x+ratio, y:y+ratio, :]
            lst_img.append(img_crop)
            y += ratio
            vt +=1
        y = 0
        x += ratio
    return lst_img

def load_json_2para(path_json = None, img = None):
    gts = []
    with open(path_json) as f:
        fh1 = json.load(f)
        '''brainwash'''
        # _ , nameOfImage = nameOfImage.split("-")
        '''end'''
        info_boxes = []
        nameOfImage = path_json.split("/")[-1].split(".json")[0]                          # bao gom 4 thong so cua cac box
        for idx, line in enumerate(fh1['objects']):
            gt = []
            gt.append(nameOfImage)
            # print(line['label'])
            gt.append(line['label'])
            gt.append(1)
            x = int(line['bbox']['x_topleft'])  # x_topleft
            y = int(line['bbox']['y_topleft'])  # y_topleft
            w = int(line['bbox']['w'])
            h = int(line['bbox']['h'])
            # HUY BO SUNG ###########################
            # print("x=", x, "y=", y, "w=", w, "h=", h)
            info_boxes.append([x, y, w, h])
            tup = tuple([x, y, x + w, y + h])
            # print(tup)
            gt.append(tup)
            # print(xnew, ynew, wnew, hnew)
            if img != None:
                image = cv2.putText(img, str(idx + 1), (x - 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255),
                                    1, cv2.LINE_AA)
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            gts.append(gt)
    return gts, img, info_boxes, fh1['width'], fh1['height']

def load_json(directory=None, nameOfImage=None, box_img=None, p_save_img=None, img=None, crop=True, ratio=1):
    """
        box_img[0] # xtopl img
        box_img[1] # ytopl img
        box_img[2] # w img
        box_img[3] # h img
    """
    # path_file = "/home/minhhoang/Desktop/Data_test/head_detec/tu/gt/" + "brainwash_10_27_2014_images-00000000_640x480.json"
    if is_an_json_file(directory):
        path_file = directory
    else:
        path_file = directory + nameOfImage + ".json"
    if os.path.exists(path_file):
        gts = []
        with open(path_file) as f:
            fh1 = json.load(f)
            '''brainwash'''
            # _ , nameOfImage = nameOfImage.split("-")
            '''end'''

            for line in fh1['objects']:
                gt = []
                gt.append(nameOfImage)
                # print(line['label'])
                gt.append(line['label'])
                gt.append(1)
                x = int(line['bbox']['x_topleft']) * ratio #x_topleft
                y = int(line['bbox']['y_topleft'])* ratio #y_topleft
                w = int(line['bbox']['w']) * ratio
                h = int(line['bbox']['h']) * ratio
                if crop:
                    if x > box_img[1] and y > box_img[0] and (x+w)<(box_img[1]+box_img[3]) and (y+h)<(box_img[2]+box_img[0]):
                        xnew = int(x - box_img[1])
                        ynew = int(y - box_img[0])
                        wnew = int(w)
                        hnew = int(h)
                        tup = tuple([xnew, ynew, xnew+w, ynew+h])
                        # print(tup)
                        gt.append(tup)
                        print(xnew, ynew, wnew, hnew)
                        cv2.rectangle(img, (xnew, ynew), (xnew + wnew, ynew + hnew), (0, 255, 0), 2)
                        gts.append(gt)
                else:
                    # if ((w*h/img.shape[1]*img.shape[0])>0.0025):
                    tup = tuple([x, y, x+w, y+h])
                    # print(tup)
                    gt.append(tup)
                    # print(xnew, ynew, wnew, hnew)
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    ratio_w = w/img.shape[1]
                    gts.append(gt)
                    
                
        # cv2.imshow("gts", img)
        # cv2.waitKey()
        # save_img = p_save_img + "/" + nameOfImage + ".png"

        # cv2.imwrite(save_img, img)
        return gts, img

def draw_gt(path_file, img):
    if os.path.exists(path_file):
        with open(path_file) as f:
            fh1 = json.load(f)
            '''brainwash'''
            # _ , nameOfImage = nameOfImage.split("-")
            '''end'''

            for line in fh1['objects']:
                x = int(line['bbox']['x_topleft']) #x_topleft
                y = int(line['bbox']['y_topleft']) #y_topleft
                w = int(line['bbox']['w'])
                h = int(line['bbox']['h'])
                # print(xnew, ynew, wnew, hnew)
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return img
    
def is_an_json_file(filename):
    IMAGE_EXTENSIONS = ['.json']
    for ext in IMAGE_EXTENSIONS:
        if ext in filename:
            return True
    return False

def _resize_image(image, resize):
    height, width = image.shape[0:2]
    if height >= resize[0] and width >= resize[1]:
        ratio = max(resize[0] / height, resize[1] / width)
    elif height <= resize[0] or width <= resize[1]:
        ratio = max(resize[0] / height, resize[1] / width)

    new_height, new_width = round(ratio*height), round(ratio*width)

    image = cv2.resize(image, (new_width, new_height))

    # image = image[0:resize[0], 0:resize[1], :]

    return image, ratio

def _crop_img(new_size, image):
    box_img = []
    h, w = new_size
    w_img = image.shape[0]
    h_img = image.shape[1]
    rdx = random.randint(0, w_img - w)
    rdy = random.randint(0, h_img - h)
    box_img.append(rdx) # xtopl
    box_img.append(rdy) # ytopl
    box_img.append(rdx + w) # w
    box_img.append(rdy + h) # h
    img_crop = image[rdx:rdx + w, rdy:rdy + h, :]
    rd_img = img_crop.copy() #random crop image
    return box_img, rd_img

def random_crop(p_img, folder_json, p_save_img=None, crop=True, new_size = (320,240),resize_crop=False, resize = (1280,720)):
    '''
    parameter: 
        p_img: path image
        folder_json: folder json or path json
        p_save_img: path save image draw ground truth
        crop=True:  random crop follow size input 
        new_size = (320,240) : crop with size input
        resize_crop = resize befoce crop with size input 
        resize = () input resolution fullHD, HD, 2k
    return:
        image: image original
        gts: GT boxs
        img: image draw GT
    '''
    orig_image = cv2.imread(p_img)
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    box_img = []
    name_foder = p_img.rsplit("/", 2)[1]
    nameOfimg = p_img.rsplit("/", 1)[1]
    nameOfimg = nameOfimg.rsplit(".", 1)[0]
    name_json = name_foder+"/"+nameOfimg +"/"
    if crop:
        box_img, rd_img = _crop_img(new_size, image)
        gts, img = load_json(directory=folder_json, nameOfImage=name_json, box_img=box_img, img = rd_img)
        save_img = p_save_img + nameOfimg + "_randomCrop" + str(rd_img.shape[0]) + "x" + str(rd_img.shape[1]) + ".png"
        cv2.imwrite(save_img, img)
        return rd_img, gts, img
    elif resize_crop:
        image_resize, ratio = _resize_image(image, resize)
        box_img, rd_img = _crop_img(new_size, image_resize)
        
        rd_crop_img = rd_img.copy()

        gts, img = load_json(directory=folder_json, nameOfImage=name_json, box_img=box_img, img = rd_img, ratio=ratio)
        # cv2.imshow("abcd", img)
        # cv2.waitKey()
        return rd_crop_img, gts, img
    else:
        gts, img = load_json(directory=folder_json, nameOfImage=nameOfimg, box_img=box_img, img = orig_image, crop=False)
        # load_json(folder_json, nameOfimg, box_img, p_save_img, orig_image, crop=False)
        # '''brainwash'''
        # # gts, img = load_json(folder_json, name_json, box_img, orig_image, crop=False)
        # '''end'''
        return image, gts, img
    
if __name__ == "__main__":
    path_img = "/media/minhhoang/Data/vehicle/Detrac/DETRAC-test-data/Insight-MVT_Annotation_Test/MVI_40714/img00002.jpg"
    folder_img = "/media/minhhoang/Data/vehicle/Detrac/DETRAC-test-data/Insight-MVT_Annotation_Test/"
    folder_json = "/home/minhhoang/Desktop/Data_test/vehic/detrac_test/"
    #size(w ,h)
    path_json = '/home/minhhoang/Desktop/Data_test/head_detec/tu/gt/brainwash_train_to_json/'
    p_save_img = '/home/minhhoang/Desktop/Data_test/vehic/draw_gt_test/'

    img = cv2.imread(path_img)
    names_folder = os.listdir(folder_img)
    count = 0
    #take image test
    path_gt_drawimg_test = "/home/minhhoang/Desktop/Data_test/vehic/draw_gt_test/"
    path_gt_img_test = "/media/minhhoang/Data/vehicle/Detrac/DETRAC-test-data/Insight-MVT_Annotation_Test/"

    path_save_dr = "/home/minhhoang/Desktop/vehicle/gtDatatest/"
    path_save_img = "/home/minhhoang/Desktop/vehicle/datatest/"

    path_json = "/home/minhhoang/Desktop/Data_test/vehic/detrac_test/"

    path_s_json = "/home/minhhoang/Desktop/vehicle/gtjson/"
    name_f_img = os.listdir(path_gt_drawimg_test)
    for name_f in name_f_img:
        for i in range(10):
            value = randint(0, len(os.listdir(path_gt_drawimg_test+name_f)))
            idx = '{0:05}'.format(value+1)
            name_img = "img" + str(idx) + ".png"
            path_dr = os.path.join(path_gt_drawimg_test, name_f+"/"+name_img)

            path_s_dr = os.path.join(path_save_dr, name_img)
            imgdr = cv2.imread(path_dr)
            cv2.imwrite(path_s_dr, imgdr)

            name_ig = "img" + str(idx) + ".jpg"
            path_img = os.path.join(path_gt_img_test, name_f+"/"+name_ig)
            path_s_i = os.path.join(path_save_img, name_ig)
            imgtest = cv2.imread(path_img)
            cv2.imwrite(path_s_i, imgtest)


            name_j = "img" + str(idx) + ".json"
            path_j = os.path.join(path_json, name_f+"/"+name_j)
            path_s_j = os.path.join(path_s_json, name_j)
            copyfile(path_j, path_s_j)


    # end
    for name_f in names_folder:
        print(folder_img+name_f)
        for name_img in os.listdir(folder_img+name_f):
            p_img = folder_img+name_f + "/" + name_img
            name_j, _ = name_img.split(".")
            name_json = folder_json + name_f + "/"
            p_s_img = p_save_img + name_f 
            if not os.path.exists(p_s_img):
                os.makedirs(p_s_img)
            random_crop(p_img, name_json, p_s_img, crop=False, new_size=(320, 240))

            print(folder_img+name_f + "/"+name_img)


    # load_json(path_json, "PartB_00005")

    # img = image.load_img(path_img)
    # # img_crop = img.crop((0, 0, 0 + 300, 200 + 300))
    # # img_crop.show("output.jpg")
    # #print(img_crop.size)
    # crop_image_keras(img)
    # cv2.waitKey()
    # print(img.shape)
    # img_padd = add_padding(img, 512, add_right = False)
    # print(img_padd.shape)
    # cv2.imshow("img_padd", img_padd)
    # cv2.waitKey()
    # for img_c in crop_image_cv2(img_padd, 512):
    #     cv2.imshow("crop_image_cv2", img_c)
    #     cv2.waitKey()