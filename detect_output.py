import cv2
from data_utils import order_points
from imutils import perspective
import glob, os
from detect import detectNumberPlate



#Function cut automatic nunber
def save_img(data_dir, output_dir):
    model = detectNumberPlate() #initizial class to detect image
    list_file = glob.glob(data_dir + "/*.jpg") #take list file
    for file in list_file: #take each file in list file
        file_name = os.path.basename(file) #name file match with list file original
        img = cv2.imread(file) #read file

        coordinates = model.detect(img) #detect file
        for coordinate in coordinates:  # detect license plate by yolov3
            # candidates = []
            # convert (x_min, y_min, width, height) to coordinate(top left, top right, bottom left, bottom right)
            pts = order_points(coordinate)
            # crop number plate used by bird's eyes view transformation
            lp_region = perspective.four_point_transform(img, pts)
            cv2.imwrite(os.path.join(output_dir, file_name), lp_region)
            # cv2.imshow(os.path.join(output_dir, file_name), lp_region)
            # cv2.waitKey(0)
        # break


def save_a_img(link_img, output):
    model = detectNumberPlate()
    file_name = os.path.basename(link_img)
    img = cv2.imread(link_img)
    coordinates = model.detect(img)
    for coordinate in coordinates:
        pts = order_points(coordinate)
        ls_region = perspective.four_point_transform(img, pts)
        cv2.imwrite(os.path.join(output, file_name), ls_region)
        return ls_region
        # cv2.imshow("Original Image", img)
        # cv2.waitKey(0)
        # cv2.imshow(os.path.join(output,file_name),ls_region)
        # cv2.waitKey(0)
# save_a_img('./input_images/0000_00532_b.jpg', './test_detected')
