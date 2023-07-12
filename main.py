from model.keypointsClassifier import *
from utils.preprocess import *
from sklearn.metrics import precision_score, recall_score

if __name__ == "__main__":
    img_mapping = {
        "DSC00165.JPG": "021z001ps.jpg",
        "DSC00166.JPG": "021z001pf.jpg",
        "DSC00167.JPG": "021z002ps.jpg",
        "DSC00168.JPG": "021z002pf.jpg",
        "DSC00169.JPG": "021z003ps.jpg",
        "DSC00170.JPG": "021z003pf.jpg",
        "DSC00171.JPG": "021z004ps.jpg",
        "DSC00172.JPG": "021z004pf.jpg",
        "DSC00173.JPG": "021z005ps.jpg",
        "DSC00174.JPG": "021z005pf.jpg",
        "DSC00175.JPG": "021z006ps.jpg",
        "DSC00176.JPG": "021z006pf.jpg",
        "DSC00177.JPG": "021z007ps.jpg",
        "DSC00178.JPG": "021z007pf.jpg",
        "DSC00179.JPG": "021z008ps.jpg",
        "DSC00180.JPG": "021z008pf.jpg",
        "DSC00181.JPG": "021z009ps.jpg",
        "DSC00182.JPG": "021z009pf.jpg",
        "DSC00183.JPG": "021z010ps.jpg",
        "DSC00184.JPG": "021z010pf.jpg",
        "DSC00185.JPG": "024z011pf.jpg",
        "DSC00186.JPG": "024z011ps.jpg",
    }

    # Sort the training image directory
    sort_training_img('./res/training')

    res = []
    test_dir = r"C:\Users\34779\PycharmProjects\COMP6211CW\res\test"
    test_images = glob.glob(os.path.join(test_dir, "*.jpg"))
    side_test_img, front_test_img = split_test_images(test_images, img_mapping)

    new = KeypointsClassifier(r'C:\Users\34779\PycharmProjects\COMP6211CW\processed')
    new.getShapeContextDict()

    predict_res, distance_list = new.test(img_mapping)

    for i in range(len(side_test_img)):
        target_name = img_mapping[os.path.basename(side_test_img[i] + '.JPG')[:-4]][:-6]
        # print(i)
        # print(target_name)
        # print(predict_res[i])
        if target_name in predict_res[i]:
            res.append(1)
        else:
            res.append(0)
    print(res)
    for index in distance_list.keys():
        print(index, distance_list[index])
