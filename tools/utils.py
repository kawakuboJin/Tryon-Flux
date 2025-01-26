import numpy as np
from PIL import Image, ImageDraw
from torchvision import transforms
import torch
import cv2
    
def background_remove(image, model):
    device = model.device
    # Data settings
    image_size = (1024, 1024)
    transform_image = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    input_images = transform_image(image).unsqueeze(0).to(device)

    # Prediction
    with torch.no_grad():
        preds = model(input_images)[-1].sigmoid().cpu()
    pred = preds[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred)
    mask = pred_pil.resize(image.size)
    
    image_white = Image.new("RGB", image.size, "white")
    image = Image.composite(image, image_white, mask)
    #image.putalpha(mask)
    return image, mask

def value_to_16scale(val):
    if val % 16 == 0:
        return val
    return int(val // 16 + 1) * 16
    
def resize_only(img:Image, dsize:tuple = (1024, 1024)):
    h, w = img.size
    t_w, t_h = dsize
    if h == t_h and w == t_w:
        return img

    # resize
    scale = min(t_h/h, t_w/w)

    scale_height = value_to_16scale(int(scale * h))
    scale_width = value_to_16scale(int(scale * w))

    scaled_image = img.resize((scale_height, scale_width))

    resize_scaled = scale_height / float(h)
    return scaled_image, resize_scaled

def concatImage(imagelist, mode = 0):

    if len(imagelist) < 2:
        return imagelist[0]
    
    refImage = imagelist[0]
    rw,rh = refImage.size
    
    process_imagelist = [refImage]
    shift = [0]

    if mode == 0:
        w_total = rw
        pre_w = rw
        
        for i in range(1, len(imagelist)):
            concImage = imagelist[i]
            w,h = concImage.size
            w_new = int(rh * w/h)
            concImage = concImage.resize((w_new,rh))

            process_imagelist.append(concImage)
            shift.append(w_total)
            w_total += w_new
            pre_w = w_new
            
        img_conc = Image.new("RGB", (w_total, rh), "white")
        for s, i in zip(shift, process_imagelist):
            img_conc.paste(i,(s, 0))
    else:
        h_total = rh
        pre_h = rh
        
        for i in range(1, len(imagelist)):
            concImage = imagelist[i]
            w,h = concImage.size
            h_new = int(rw * h/w)
            concImage = concImage.resize((rw,h_new))

            process_imagelist.append(concImage)
            shift.append(h_total)
            h_total += h_new
            pre_h = h_new
            
        img_conc = Image.new("RGB", (rw, h_total), "white")
        for s, i in zip(shift, process_imagelist):
            img_conc.paste(i,(0, s))

    return img_conc

def concat_2_img(img, ref, mask):
    ref_mask = Image.new('RGB', ref.size, 'black')
    w_r, h_r = ref.size
    if h_r / float(w_r) > 1.2:
        concImg = concatImage([img,ref], 0)
        concMask = concatImage([mask,ref_mask], 0)
    else:
        concImg = concatImage([img,ref], 1)
        concMask = concatImage([mask,ref_mask], 1)

    return concImg, concMask, (img.size[0],img.size[1])

def draw_kps(canvas_ori, kps):
    print('type(canvas_ori): ', type(canvas_ori))
    
    if isinstance(canvas_ori, Image.Image):
        canvas = np.array(canvas_ori).astype('uint8')
    else:
        canvas = canvas_ori

    text_border = canvas.shape[0] / 1024
    text_thickness = int(canvas.shape[0] / 512)
    circle_border = int(canvas.shape[0] / 256)

    for i,k in enumerate(kps):
            if k is not None:
                cv2.putText(canvas, str(i), (int(k[0]), int(k[1])), cv2.FONT_HERSHEY_SIMPLEX, text_border, (255, 0, 0), text_thickness)  
                cv2.circle(canvas, (int(k[0]), int(k[1])), circle_border, (0, 255, 0), 2)

    if isinstance(canvas_ori, Image.Image):
        return Image.fromarray(canvas)
    else:        
        return canvas

def extend_arm_mask(wrist, elbow, scale):
  wrist = elbow + scale * (wrist - elbow)
  return wrist

def hole_fill(img):
    img = np.pad(img[1:-1, 1:-1], pad_width = 1, mode = 'constant', constant_values=0)
    img_copy = img.copy()
    mask = np.zeros((img.shape[0] + 2, img.shape[1] + 2), dtype=np.uint8)

    cv2.floodFill(img, mask, (0, 0), 255)
    img_inverse = cv2.bitwise_not(img)
    dst = cv2.bitwise_or(img_copy, img_inverse)
    return dst

def refine_mask(mask):
    contours, hierarchy = cv2.findContours(mask.astype(np.uint8),
                                           cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)
    area = []
    for j in range(len(contours)):
        a_d = cv2.contourArea(contours[j], True)
        area.append(abs(a_d))
    refine_mask = np.zeros_like(mask).astype(np.uint8)
    if len(area) != 0:
        i = area.index(max(area))
        cv2.drawContours(refine_mask, contours, i, color=255, thickness=-1)

    return refine_mask

def get_mask_location(category, model_parse: Image.Image, keypoint: dict, cloth_length = -1,width=384,height=512):
    im_parse = model_parse.resize((width, height), Image.NEAREST)
    parse_array = np.array(im_parse)

    label_map = {
        "background": 0,
        "hat": 1,
        "hair": 2,
        "sunglasses": 3,
        "upper_clothes": 4,
        "skirt": 5,
        "pants": 6,
        "dress": 7,
        "belt": 8,
        "left_shoe": 9,
        "right_shoe": 10,
        "head": 11,
        "left_leg": 12,
        "right_leg": 13,
        "left_arm": 14,
        "right_arm": 15,
        "bag": 16,
        "scarf": 17,
        "neck": 18
        }
    ### check the human parsing defination
    # for i in range(25):
    #     parse_array = np.array(im_parse)
    #     mask = (parse_array == i).astype(np.float32)
    #     mask = Image.fromarray(mask.astype(np.uint8) * 255)
    #     mask.save(str(i) + '.png')
    # exit(-1)

    arm_width = 45
    thigh_width = 60

    parse_head = (parse_array == label_map["hat"]).astype(np.float32) + \
                 (parse_array == label_map["sunglasses"]).astype(np.float32) + \
                 (parse_array == label_map["head"]).astype(np.float32)

    parser_mask_fixed = (parse_array == label_map["left_shoe"]).astype(np.float32) + \
                        (parse_array == label_map["right_shoe"]).astype(np.float32) + \
                        (parse_array == label_map["hat"]).astype(np.float32) + \
                        (parse_array == label_map["sunglasses"]).astype(np.float32) + \
                        (parse_array == label_map["bag"]).astype(np.float32)


    parser_mask_changeable = (parse_array == label_map["background"]).astype(np.float32)

    arms_left = (parse_array == label_map["left_arm"]).astype(np.float32)
    arms_right = (parse_array == label_map["right_arm"]).astype(np.float32)
    # arms = arms_left + arms_right

    if category == 'whole_body' or category == 'short_dresses' or category == 'long_dresses':

        parse_mask = (parse_array == label_map["upper_clothes"]).astype(np.float32) + \
                     (parse_array == label_map["skirt"]).astype(np.float32) + \
                     (parse_array == label_map["pants"]).astype(np.float32) + \
                     (parse_array == label_map["dress"]).astype(np.float32)
        
        parser_mask_changeable += np.logical_and(parse_array, np.logical_not(parser_mask_fixed))
    elif category == 'upper_body':

        parse_mask = (parse_array == label_map["upper_clothes"]).astype(np.float32) + \
                     (parse_array == label_map["dress"]).astype(np.float32)
        
        parser_mask_fixed_lower_cloth = (parse_array == label_map["skirt"]).astype(np.float32) + \
                                        (parse_array == label_map["pants"]).astype(np.float32) + \
                                        (parse_array == label_map["left_leg"]).astype(np.float32) + \
                                        (parse_array == label_map["right_leg"]).astype(np.float32)
        
        parser_mask_fixed += parser_mask_fixed_lower_cloth
        parser_mask_changeable += np.logical_and(parse_array, np.logical_not(parser_mask_fixed))
    elif category == 'lower_body':

        parse_mask = (parse_array == label_map["skirt"]).astype(np.float32) + \
                     (parse_array == label_map["pants"]).astype(np.float32) + \
                     (parse_array == label_map["left_leg"]).astype(np.float32) + \
                     (parse_array == label_map["right_leg"]).astype(np.float32)

        parser_mask_fixed += (parse_array == label_map["upper_clothes"]).astype(np.float32) + \
                             (parse_array == label_map["left_arm"]).astype(np.float32) + \
                             (parse_array == label_map["right_arm"]).astype(np.float32)
        parser_mask_changeable += np.logical_and(parse_array, np.logical_not(parser_mask_fixed))
    elif category == 'short_skirt' or category == 'long_skirt':

        parse_mask = (parse_array == label_map["skirt"]).astype(np.float32) + \
                     (parse_array == label_map["pants"]).astype(np.float32) + \
                     (parse_array == label_map["dress"]).astype(np.float32)

        parser_mask_fixed += (parse_array == label_map["upper_clothes"]).astype(np.float32) + \
                             (parse_array == label_map["left_arm"]).astype(np.float32) + \
                             (parse_array == label_map["right_arm"]).astype(np.float32) + \
                             (parse_array == label_map["neck"]).astype(np.float32)
        parser_mask_changeable += np.logical_and(parse_array, np.logical_not(parser_mask_fixed))
    else:
        raise NotImplementedError

    # Load pose points
    pose_data = keypoint["pose_keypoints_2d"]
    pose_data = np.array(pose_data)
    pose_data = pose_data.reshape((-1, 2))

    im_arms_left = Image.new('L', (width, height))
    im_arms_right = Image.new('L', (width, height))
    arms_draw_left = ImageDraw.Draw(im_arms_left)
    arms_draw_right = ImageDraw.Draw(im_arms_right)

    # if category == 'whole_body' or category == 'upper_body' or category == 'short_dresses' or category == 'long_dresses':
    shoulder_right = np.multiply(tuple(pose_data[2][:2]), height / 512.0)
    shoulder_left = np.multiply(tuple(pose_data[5][:2]), height / 512.0)
    elbow_right = np.multiply(tuple(pose_data[3][:2]), height / 512.0)
    elbow_left = np.multiply(tuple(pose_data[6][:2]), height / 512.0)
    wrist_right = np.multiply(tuple(pose_data[4][:2]), height / 512.0)
    wrist_left = np.multiply(tuple(pose_data[7][:2]), height / 512.0)
    
    ARM_LINE_WIDTH = int(arm_width / 512 * height)
    size_left = [shoulder_left[0] - ARM_LINE_WIDTH // 2, shoulder_left[1] - ARM_LINE_WIDTH // 2, shoulder_left[0] + ARM_LINE_WIDTH // 2, shoulder_left[1] + ARM_LINE_WIDTH // 2]
    size_right = [shoulder_right[0] - ARM_LINE_WIDTH // 2, shoulder_right[1] - ARM_LINE_WIDTH // 2, shoulder_right[0] + ARM_LINE_WIDTH // 2,shoulder_right[1] + ARM_LINE_WIDTH // 2]
    
    if wrist_right[0] <= 1. and wrist_right[1] <= 1.:
        im_arms_right = arms_right
    else:
        wrist_right = extend_arm_mask(wrist_right, elbow_right, 1.2)
        arms_draw_right.line(np.concatenate((shoulder_right, elbow_right, wrist_right)).astype(np.uint16).tolist(), 'white', ARM_LINE_WIDTH, 'curve')
        arms_draw_right.arc(size_right, 0, 360, 'white', ARM_LINE_WIDTH // 2)

    if wrist_left[0] <= 1. and wrist_left[1] <= 1.:
        im_arms_left = arms_left
    else:
        wrist_left = extend_arm_mask(wrist_left, elbow_left, 1.2)
        arms_draw_left.line(np.concatenate((wrist_left, elbow_left, shoulder_left)).astype(np.uint16).tolist(), 'white', ARM_LINE_WIDTH, 'curve')
        arms_draw_left.arc(size_left, 0, 360, 'white', ARM_LINE_WIDTH // 2)


    hands_left = np.logical_and(np.logical_not(im_arms_left), arms_left)
    hands_right = np.logical_and(np.logical_not(im_arms_right), arms_right)
    parser_mask_fixed += hands_left + hands_right
        
    parser_mask_fixed = np.logical_or(parser_mask_fixed, parse_head)

    parse_mask = cv2.dilate(parse_mask, np.ones((5, 5), np.uint16), iterations=5)

    if category == 'whole_body' or category == 'upper_body' or category == 'short_dresses' or category == 'long_dresses':
        neck_mask = (parse_array == label_map["neck"]).astype(np.float32)
        neck_mask = cv2.dilate(neck_mask, np.ones((5, 5), np.uint16), iterations=1)
        neck_mask = np.logical_and(neck_mask, np.logical_not(parse_head))

        parse_mask = np.logical_or(parse_mask, neck_mask)
        arm_mask = cv2.dilate(np.logical_or(im_arms_left, im_arms_right).astype('float32'), np.ones((5, 5), np.uint16), iterations=4)

        parse_mask += np.logical_or(parse_mask, arm_mask)

    if category == 'whole_body' or category == 'short_dresses' or category == 'long_dresses' or category == 'short_skirt' or category == 'long_skirt' :
        im_thigh_left = Image.new('L', (width, height))
        im_thigh_right = Image.new('L', (width, height))
        thigh_draw_left = ImageDraw.Draw(im_thigh_left)
        thigh_draw_right = ImageDraw.Draw(im_thigh_right)

        waist_left = np.multiply(tuple(pose_data[8][:2]), height / 512.0)
        waist_right = np.multiply(tuple(pose_data[11][:2]), height / 512.0)
        knee_left = np.multiply(tuple(pose_data[9][:2]), height / 512.0)
        knee_right = np.multiply(tuple(pose_data[12][:2]), height / 512.0)
        ankle_left = np.multiply(tuple(pose_data[10][:2]), height / 512.0)
        ankle_right = np.multiply(tuple(pose_data[13][:2]), height / 512.0)

        im_thigh_mid = Image.new('L', (width, height))
        thigh_draw_mid = ImageDraw.Draw(im_thigh_mid)
        waist_mid = 0.5 * (waist_left + waist_right)
        knee_mid = 0.5 * (knee_left + knee_right)
        ankle_mid = 0.5 * (ankle_left + ankle_right)

        if category == 'short_dresses' or category == 'short_skirt':
            wk_rate = 0.3
            if cloth_length > -1:
                min_length,max_length = 20.0, 60.0
                cloth_length = max(cloth_length - 40, min_length)
                min_rate, max_rate = wk_rate, 1.0
                wk_rate = min_rate + (max_rate - min_rate) * (cloth_length - min_length) / (max_length - min_length)
                print('wk_rate: ', wk_rate)

            THIGH_LINE_WIDTH = int(thigh_width / 512 * height)
            wk_left = (1 - wk_rate) * waist_left + (wk_rate) * knee_left
            thigh_draw_left.line(np.concatenate((waist_left, wk_left)).astype(np.uint16).tolist(), 'white', THIGH_LINE_WIDTH, 'curve')
            wk_right = (1 - wk_rate) * waist_right + (wk_rate) * knee_right
            thigh_draw_right.line(np.concatenate((waist_right, wk_right)).astype(np.uint16).tolist(), 'white', THIGH_LINE_WIDTH, 'curve')
        else:
            ka_rate = 0.5
            if cloth_length > -1:
                min_length,max_length = 10.0, 50.0
                cloth_length = max(cloth_length - 100, min_length)
                min_rate, max_rate = wk_rate, 1.0
                ka_rate = min_rate + (max_rate - min_rate) * (cloth_length - min_length) / (max_length - min_length)

            THIGH_LINE_WIDTH = int(thigh_width / 512 * height)
            ka_left = (1 - ka_rate) * knee_left + (ka_rate) * ankle_left
            thigh_draw_left.line(np.concatenate((waist_left, knee_left, ka_left)).astype(np.uint16).tolist(), 'white', THIGH_LINE_WIDTH, 'curve')
            ka_right = (1 - ka_rate) * knee_right + (ka_rate) * ankle_right
            thigh_draw_right.line(np.concatenate((waist_right, knee_right, ka_right)).astype(np.uint16).tolist(), 'white', THIGH_LINE_WIDTH, 'curve')
            ka_mid = (1 - ka_rate) * knee_mid + (ka_rate) * ankle_mid
            thigh_draw_mid.line(np.concatenate((waist_mid, knee_mid, ka_mid)).astype(np.uint16).tolist(), 'white', THIGH_LINE_WIDTH, 'curve')
            
        left_leg_mask = (parse_array == label_map["right_leg"]).astype(np.float32)
        right_leg_mask = (parse_array == label_map["left_leg"]).astype(np.float32) 

        left_thigh_mask = np.logical_and(left_leg_mask, im_thigh_left)
        right_thigh_mask = np.logical_and(right_leg_mask, im_thigh_right)
        mid_thigh_mask = np.array(im_thigh_mid)

        parse_mask = np.logical_or(parse_mask, cv2.dilate(left_thigh_mask.astype('float32'), np.ones((5, 5), np.uint16), iterations=8))
        parse_mask = np.logical_or(parse_mask, cv2.dilate(right_thigh_mask.astype('float32'), np.ones((5, 5), np.uint16), iterations=8))        
        parse_mask = np.logical_or(parse_mask, cv2.dilate(mid_thigh_mask.astype('float32'), np.ones((5, 5), np.uint16), iterations=8)) 

    no_parse_mask = np.logical_and(parser_mask_changeable, np.logical_not(parse_mask))
    no_parse_mask_total = np.logical_or(no_parse_mask, parser_mask_fixed)
    
    inpaint_mask = 1 - no_parse_mask_total
    img = np.where(inpaint_mask, 255, 0)
    dst = hole_fill(img.astype(np.uint8))
    dst = refine_mask(dst)
    inpaint_mask = dst / 255 * 1
    mask = Image.fromarray(inpaint_mask.astype(np.uint8) * 255)
    mask_gray = Image.fromarray(inpaint_mask.astype(np.uint8) * 127)

    return mask, mask_gray
