# USAGE
# python main.py train --style ./path/to/style/image.jpg(video.mp4)   \
#                      --dataset ./path/to/dataset \
#                      --weights ./path/to/weights  \
#                      --batch 2

# python main.py evaluate --content ./path/to/content/image.jpg   \
#                         --weights ./path/to/weights \
#                         --result ./path/to/save/results/image.jpg

import os
import argparse
from train import trainer
from evaluate import transfer
import os
import tensorflow as tf


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
# tf.config.experimental.set_memory_growth(physical_devices[0], True)
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#   # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
#   try:
#     tf.config.experimental.set_virtual_device_configuration(
#         gpus[0],
#         [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1048)])
#     logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#     print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#   except RuntimeError as e:
#     # Virtual devices must be set before GPUs have been initialized
#     print(e)

CONTENT_WEIGHT = 6e0
STYLE_WEIGHT = 2e-3
TV_WEIGHT = 6e2

# CONTENT_WEIGHT = 7.5e0
# STYLE_WEIGHT = 1e2
# TV_WEIGHT = 2e2


LEARNING_RATE = 1e-3
NUM_EPOCHS = 6
BATCH_SIZE = 2

STYLE_IMAGE = './images/style/style_6.jpg'
#STYLE_IMAGE = './images/style/wave.jpg'
CONTENT_IMAGE = './images/content/1-5-1airplane_train_139.jpg'
#CONTENT_IMAGE = "E:\kg\FastStyle-master\images\content/2.mp4"
#DATASET_PATH = '../datasets/train2014'
#DATASET_PATH = "E:\kg\data\VOCdevkit\VOC2007\JPEGImages"
#DATASET_PATH = "E:\kg\data\VOCdevkit\VOC2007/1"
DATASET_PATH = "E:\kg\data\coco/val2014"

#DATASET_PATH = "E:\kg\data\VOCdevkit\CropVOC2007_3"
#DATASET_PATH = "E:\kg/fast-neural-style/fast-neural-style-master\images\output-images"
WEIGHTS_PATH = './weights/sonar_style_3/weights'
#WEIGHTS_PATH = "E:\kg\FastStyle-master\weights\wave"

RESULT_NAME = './images/results/Resul22.jpg'
#RESULT_NAME = './images/results/Resul5.mp4'
#CONTENT_IMAGE_FOLDER = "E:\kg\data\SONAR_VOC_MulitScale_styletransfer/target_1/person/"

names = ['aeroplane','bicycle','car','person','ship']

#CONTENT_IMAGE_FOLDER = "E:\kg\data\SONAR_VOC_MulitScale_styletransfer/target_augmentation\SaltAndPepper/ship/"
#car

CONTENT_IMAGE_FOLDER = r"E:\kg\data\SONAR_VOC_MulitScale_styletransfer\target_noVOC/aeroplane/"

#RESULT_NAME_FOLDER = "E:\kg\data\SONAR_VOC_MulitScale_styletransfer\VOC2007_Style_1/noise_SaltPeper\content_styled/ship/"
RESULT_NAME_FOLDER = r"E:\kg\data\SONAR_VOC_MulitScale_styletransfer\VOC2007_Style_6\content_styled/aeroplane/"

subfoler = ['origin','line','rectangle','circle']

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Fast Style Transfer')
    parser.add_argument('command',
                         metavar='<command>',
                         help="'train' or 'evaluate'")
    parser.add_argument('--debug', required=False, type=bool,
                         metavar=False,
                         help='Whether to print the loss',
                         default=False)
    parser.add_argument('--dataset', required=False,
                         metavar=DATASET_PATH,
                         default=DATASET_PATH)
    parser.add_argument('--style', required=False,
                         metavar=STYLE_IMAGE,
                         help='Style image to train the specific style',
                         default=STYLE_IMAGE) 
    parser.add_argument('--content', required=False,
                         metavar=CONTENT_IMAGE,
                         help='Content image/video to evaluate with',
                         default=CONTENT_IMAGE)  
    parser.add_argument('--weights', required=False,
                         metavar=WEIGHTS_PATH,
                         help='Checkpoints directory',
                         default=WEIGHTS_PATH)
    parser.add_argument('--result', required=False,
                         metavar=RESULT_NAME,
                         help='Path to the transfer results',
                         default=RESULT_NAME)
    parser.add_argument('--batch', required=False, type=int,
                         metavar=BATCH_SIZE,
                         help='Training batch size',
                         default=BATCH_SIZE)
    parser.add_argument('--max_dim', required=False, type=int,
                         metavar=None,
                         help='Resize the result image to desired size or remain as the original',
                         default=None)

    args = parser.parse_args()


    # Validate arguments
    if args.command == "train":
        assert os.path.exists(args.dataset), 'dataset path not found !'
        assert os.path.exists(args.style), 'style image not found !'
        assert args.batch > 0
        assert NUM_EPOCHS > 0
        assert CONTENT_WEIGHT >= 0
        assert STYLE_WEIGHT >= 0
        assert TV_WEIGHT >= 0
        assert LEARNING_RATE >= 0

        parameters = {
                'style_file' : args.style,
                'dataset_path' : args.dataset,
                'weights_path' : args.weights,
                'debug' : args.debug,
                'content_weight' : CONTENT_WEIGHT,
                'style_weight' : STYLE_WEIGHT,
                'tv_weight' : TV_WEIGHT,
                'learning_rate' : LEARNING_RATE,
                'batch_size' : args.batch,
                'epochs' : NUM_EPOCHS,
            }

        trainer(**parameters)


    elif args.command == "evaluate":
        assert args.content, 'content image/video not found !'
        assert args.weights, 'weights path not found !'
        index = 0
        #CONTENT_IMAGE_FOLDER_A, RESULT_NAME_FOLDER_A
        #CONTENT_IMAGE_FOLDER = CONTENT_IMAGE_FOLDER_A
        #RESULT_NAME_FOLDER = RESULT_NAME_FOLDER_A
        # for root, dirs, files in os.walk(CONTENT_IMAGE_FOLDER):
        #     for f in files:
        #         index = index + 1
        #         print(str(index) + "/" + str(len(files))+ " " + str(index/len(files)))
        #         con = CONTENT_IMAGE_FOLDER+f
        #         res = RESULT_NAME_FOLDER + f
        #         parameters = {
        #                 'content' : con,
        #                 'weights' : args.weights,
        #                 'max_dim' : args.max_dim,
        #                 'result' : res,
        #             }
        #
        #         transfer(**parameters)
        parameters = {
                'content' : args.content,
                'weights' : args.weights,
                'max_dim' : args.max_dim,
                'result' : args.result,
            }

        transfer(**parameters)


    else:
        print('Example usage : python main.py evaluate --content ./path/to/content/image.jpg')
        
def path_combine():

    names_1 = ['aeroplane', 'bicycle', 'car', 'person', 'ship']

    CONTENT_IMAGE_FOLDER_1 = "E:\kg\data\SONAR_VOC_MulitScale_styletransfer/target_augmentation/"

    RESULT_NAME_FOLDER_1 = "E:\kg\data\SONAR_VOC_MulitScale_styletransfer\VOC2007_Style_1\content_styled/"

    subfoler_1 = ['line', 'rectangle', 'circle']
    for name in names_1:
        for shape in subfoler_1:
            CONTENT_IMAGE_FOLDER_A = CONTENT_IMAGE_FOLDER_1 + name+"/"+shape +"/8/"
            RESULT_NAME_FOLDER_A = RESULT_NAME_FOLDER_1 + name+"/"+shape +"/8/"
            main(CONTENT_IMAGE_FOLDER_A,RESULT_NAME_FOLDER_A)
if __name__ == '__main__':
    #path_combine()
    main()