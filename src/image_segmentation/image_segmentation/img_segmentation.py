#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from PIL import Image as PILImage

import numpy as np
import cv2
import torch
import torchvision.transforms as T
from torchvision.models.segmentation import deeplabv3_resnet101
from torchvision.models.segmentation import DeepLabV3_ResNet101_Weights


LABEL_COLORS_LIST = {
    0:  (128,  64, 128),
    1:  (244,  35, 232),
    2:  ( 70,  70,  70),
    3:  (102, 102, 156),
    4:  (190, 153, 153),
    5:  (153, 153, 153),
    6:  (250, 170,  30),
    7:  (220, 220,   0),
    8:  (107, 142,  35),
    9:  (152, 251, 152),
    10: ( 70, 130, 180),
    11: (220,  20,  60),
    12: (255,   0,   0),
    13: (  0,   0, 142),
    14: (  0,   0,  70),
    15: (  0,  60, 100),
    16: (  0,  80, 100),
    17: (  0,   0, 230),
    18: (119,  11,  32),
    255: (0, 0, 0),
}


class ImageSegmentation(Node):

    def __init__(self):
        super().__init__('segmentation_node')

        self.bridge = CvBridge()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load pretrained DeepLabV3
        weights = DeepLabV3_ResNet101_Weights.DEFAULT
        self.model = deeplabv3_resnet101(weights=weights)
        self.model.to(self.device)
        self.model.eval()

        self.transforms = weights.transforms()

        # Convert color dictionary to numpy palette
        max_class = max(LABEL_COLORS_LIST.keys())
        self.color_palette = np.zeros((max_class + 1, 3), dtype=np.uint8)

        for k, v in LABEL_COLORS_LIST.items():
            if k <= max_class:
                self.color_palette[k] = v

        self.img_subscriber = self.create_subscription(Image, '/sensors/camera/left/image_raw', self.image_callback, 10)

        self.seg_publisher = self.create_publisher(Image, '/segmented_img', 10)

        self.get_logger().info("Segmentation Node Started")

    def image_callback(self, img_msg):

        # Convert ROS → OpenCV
        image = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')

        # BGR → RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = PILImage.fromarray(image_rgb)

        input_tensor = self.transforms(pil_image).unsqueeze(0).to(self.device)

        # Inference
        with torch.no_grad():
            output = self.model(input_tensor)['out'][0]

        segmentation_map = output.argmax(0).cpu().numpy().astype(np.uint8)

        # Resize segmentation map to original image size
        segmentation_map = cv2.resize(
            segmentation_map,
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )

        color_mask = self.segmentation_map_to_rgb(segmentation_map)

        overlay = cv2.addWeighted(image, 0.5, color_mask, 0.5, 0)

        # Publish result
        seg_msg = self.bridge.cv2_to_imgmsg(overlay, encoding='bgr8')
        seg_msg.header = img_msg.header
        self.seg_publisher.publish(seg_msg)

    def segmentation_map_to_rgb(self, segmentation_map):
        segmentation_map = np.clip(segmentation_map, 0, len(self.color_palette)-1)
        return self.color_palette[segmentation_map]


def main(args=None):
    rclpy.init(args=args)
    node = ImageSegmentation()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()