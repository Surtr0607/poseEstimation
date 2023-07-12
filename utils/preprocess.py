import os
import shutil
import glob
import cv2


def sort_training_img(path):
    old_path = r'./processed'
    old_file_list1 = os.listdir(old_path + r'\side_image')
    old_file_list2 = os.listdir(old_path + r'\front_image')

    for i in old_file_list1:
        os.remove(os.path.join(old_path + r'\side_image', i))

    for i in old_file_list2:
        os.remove(os.path.join(old_path + r'\front_image', i))

    print('Remove old files successfully')

    file_list = os.listdir(path)

    side_image = glob.glob(path + r'/*s.jpg')
    for i in range(len(file_list)):
        new_old_path = side_image[i//2].replace('\\', '/')
        prefix, suffix = file_list[i].split('.')
        if prefix.endswith("s"):
            shutil.copy(new_old_path, './processed/side_image/' + file_list[i])

    front_image = glob.glob(path + r'/*f.jpg')
    for i in range(len(file_list)):
        new_old_path = front_image[i//2].replace('\\', '/')
        prefix, suffix = file_list[i].split('.')
        if prefix.endswith("f"):
            shutil.copy(new_old_path, './processed/front_image/' + file_list[i])


def resize_image(path):
    for root, dirs, files in os.walk(path):
        for img in files:
            img_path = os.path.join(root, img)
            try:
                image = cv2.imread(img_path)
                dim = (500, 500)
                resized = cv2.resize(image, dim)
                cv2.imwrite(img_path, resized)
            except:
                print(img_path)


if __name__ == '__main__':
    sort_training_img(r"C:\Users\34779\PycharmProjects\COMP6211CW\res\training")
    resize_image(r'C:\Users\34779\PycharmProjects\COMP6211CW\processed')
