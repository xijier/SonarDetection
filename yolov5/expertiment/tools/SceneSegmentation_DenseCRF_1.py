import pixellib
from pixellib.semantic import semantic_segmentation

segment_image = semantic_segmentation()

segment_image.load_pascalvoc_model("../model/deeplabv3_xception_tf_dim_ordering_tf_kernels.h5")

segment_image.segmentAsPascalvoc("../data/1-airplane_train_50.jpg", output_image_name = "../data/9_deeplabv3.jpg")
