from pprint import pprint
import glob, os
import base64
import cv2
import numpy as np



# a = 1,2,3
# b= x,y,z
# zip(a,b) =(1,x)(2,y)(3,z)


def assign_to_line(list_boxes, list_class, list_conf):
    zipped_list = zip(list_boxes, list_class, list_conf)
    if len(list_boxes):
        list_boxes = sorted(list_boxes, key=(lambda box: box[1]))

        zipped_list = sorted(zipped_list, key=(lambda x: x[0][1]))

        y1 = list_boxes[0][1] + list_boxes[0][3] // 2
        list1 = []
        list2 = []
        list3 = []
        index = 0
        for i, box in enumerate(list_boxes):
            if box[1] < y1:
                list1.append(zipped_list[i])
            else:
                index = i
                break

        idx = 0
        if index:
            list_boxes2 = list_boxes[index:]
            zipped_list2 = zipped_list[index:]
            y2 = list_boxes2[0][1] + list_boxes2[0][3] // 2
            for i, box in enumerate(list_boxes2):
                if box[1] < y2:
                    list2.append(zipped_list2[i])
                else:
                    idx = i
                    break

        if idx:
            y3 = list_boxes2[idx][1] + list_boxes2[idx][3] // 2
            zipped_list3 = zipped_list2[idx:]
            for i, box in enumerate(list_boxes2[idx:]):
                if box[1] < y3:
                    list3.append(zipped_list3[i])

        list1 = sorted(list1, key=(lambda x: x[0][0]))
        list2 = sorted(list2, key=(lambda x: x[0][0]))
        list3 = sorted(list3, key=(lambda x: x[0][0]))
        return list1 + list2 + list3
    else:
        return list_boxes


class PlateModel:
    def __init__(self, model_config, model_path, label_path):
        self.conf_threshold = 0.2
        self.nms_threshold = 0.2
        self.inp_width = 320
        self.inp_height = 320
        with open(label_path, 'r') as f:
            self.classes = f.read().rstrip('\n').split('\n')
        self.net = cv2.dnn_DetectionModel(model_config, model_path)
        self.net.setInputSize(self.inp_width, self.inp_height)
        self.net.setInputScale(1.0 / 255)

    def test(self, img_path):
        frame = cv2.imread(img_path)
        # if frame is None:
        #     print("Frame is None")
        #     return

        classes, confidences, boxes = self.net.detect(frame, confThreshold=0.1, nmsThreshold=0.4)
        temp = zip(classes.flatten(), confidences.flatten(), boxes)
        list_box = []
        label = []
        for classId, confidence, box in temp:
            # print(classId, confidence)
            # label = '%.2f' % confidence
            # label = '%s: %s' % (self.classes[classId], label)
            label_ = self.classes[classId]
            label.append(label_)

            labelSize, baseLine = cv2.getTextSize(label_, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            left, top, width, height = box
            list_box.append(box)

            top = max(top, labelSize[1])
            cv2.rectangle(frame, box, color=(0, 255, 0), thickness=1)
            cv2.rectangle(frame, (left, top - labelSize[1]), (left + labelSize[0], top + baseLine),
                          (255, 255, 255), cv2.FILLED)
            cv2.putText(frame, label_, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
            # cv2.imshow("show " + str(label_), frame)
            # cv2.waitKey(0)

        return frame, list_box, label

    def map_char(self, class_ids):
        if not isinstance(class_ids, list):
            class_ids = list(class_ids)
        classes = []
        for i in range(len(class_ids)):
            classes.append(self.classes[class_ids[i]])
        return classes

    def predict(self, img):
        class_ids, confidences, boxes = self.net.detect(img, confThreshold=0.1, nmsThreshold=0.4)
        # clss = np.array(class_ids)
        classes = self.map_char(class_ids.flatten())
        # cf = np.array(confidences)
        list_char = assign_to_line(boxes, classes, confidences.flatten())
        # print(boxes)
        return list_char


import json


def write_coordinate(file_name, list_box, label, width, height, base64):
    file_id, ext = os.path.splitext(file_name)
    with open(os.path.join("./file_json/" + file_id + ".json"), "w") as f:
        a = {"shapes": [],
             "imagePath": "",
             "flags": {},
             "version": "4.5.6",
             "imageData": "",
             "imageWidth": 0,
             "imageHeight": 0
             }
        for i in range(len(label)):
            label_ = {"shapetype": "rectangle",
                      "points": [[int(list_box[i][0]), int(list_box[i][1])],
                                 [int(list_box[i][0] + list_box[i][2]),
                                  int(list_box[i][1] + list_box[i][3])]],
                      "flags": {},
                      "group_id": None,
                      "label": label[i]
                      }
            a["shapes"].append(label_)

        a["imagePath"] += file_name
        a["imageData"] += base64
        a["imageWidth"] = width
        a["imageHeight"] = height
        string_ = json.dumps(a, ensure_ascii=False, indent=2)
        f.write(string_)


def save_img(data_dir):
    model = PlateModel("./plate-yolov3.cfg", "./plate-yolov3_best.weights", "./plate.names")
    list_file = glob.glob(data_dir + "/*")
    for file in list_file:
        file_name = os.path.basename(file)
        img, list_box, label = model.test(file)
        height, width = img.shape[:2]

        with open(file, "rb") as img_file:
            my_string = base64.b64encode(img_file.read())
            my_string = my_string.decode("utf-8")
            write_coordinate(file_name, list_box, label, width, height, my_string)


def is_feature(file_name, width=1, height=1):
    with open(file_name) as f:
        data_ = json.load(f)
        data = data_['shapes']
        label, x1, y1, x2, y2 = [], [], [], [], []
        for i in range(len(data)):
            label.append(data[i]['label'])
            x1.append(data[i]['points'][0][0] / width)
            y1.append(data[i]['points'][0][1] / height)
            x2.append(data[i]['points'][1][0] / width)
            y2.append(data[i]['points'][1][1] / height)
        return label, x1, y1, x2, y2


def to_file_text(data_dir, input_image, file_text):
    list_file_json = glob.glob(data_dir + "*")

    for file in list_file_json:
        file_name = os.path.basename(file)
        file_id, ext = os.path.splitext(file_name)
        file_img = os.path.join(input_image, "{}.jpg".format(file_id))
        img = cv2.imread(file_img)
        h, w = img.shape[:2]
        label, x1, y1, x2, y2 = is_feature(file)
        temp_label = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E",
                      "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
                      "U", "V", "W", "X", "Y", "Z"]
        # print(temp_label.shape)
        with open(os.path.join(file_text, file_id + ".txt"), "w") as f:
            label_ = []
            for j in range(len(label)):
                for k in range(len(temp_label)):
                    if label[j] == temp_label[k]:
                        label_.append(k)
                        break
                # Input của 1 bounding box là label, x_center, y_center, width, height
                x_center = (x1[j] + x2[j]) / 2 / w
                y_center = (y1[j] + y2[j]) / 2 / h
                width_box = (x2[j] - x1[j]) / w
                height_box = (y2[j] - y1[j]) / h
                f.write("%s %.8s %.8s %.8s %.8s " % (
                    label_[j], x_center, y_center, width_box, height_box))
                f.write("\n")


from detect_output import save_a_img


def load_img(name, input):
    img = cv2.imread(input)
    # img1=cv2.resize(img, (800, 600))
    cv2.imshow(name, img)
    cv2.waitKey(0)


def result(output):
    lst = []
    for i in range(len(output)):
        lst.append(output[i][1])
    print("Plate need to read:")
    print(lst)
    return lst

if __name__ == '__main__':

    # save_img to create boundary box around character --> repair again coordinate
    # save_img("./plate_detected")
    # after repair coordinate, transpose coordinate to file .txt
    # to_file_text('./file_new/json_new_file/', './file_new/image_new/', './file_text_new/')
    # # print("Label is done")

    model = PlateModel("./weights/plate-yolov3-tiny.cfg",
                       "./weights/plate-yolov3-tiny_last.weights",
                       "./weights/plate.names")

    input_img = 'xemay9.jpg'
    load_img('Original Image', './input_images/' + input_img)
    save_a_img('./input_images/' + input_img, './test_detected')

    load_img('Image Detected', './test_detected/' + input_img)
    fram, box, label = model.test('./test_detected/' + input_img)
    frame = cv2.resize(fram, (400, 300))
    cv2.imshow("show", frame)
    cv2.waitKey(0)

    img = cv2.imread("./test_detected/" + input_img)
    output = model.predict(img)
    pprint((output))
    result(output)

