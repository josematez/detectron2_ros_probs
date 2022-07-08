#!/usr/bin/env python
import sys
import threading
import time

import cv2 as cv
import numpy as np
import rospy
import torch
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from cv_bridge import CvBridge, CvBridgeError
# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer
from detectron2_ros_probs.msg import ResultWithWalls
from sensor_msgs.msg import Image, RegionOfInterest


class Detectron2node(object):
    def __init__(self):
        rospy.logwarn("Initializing Panoptic Detectron2")
        setup_logger()

        self._bridge = CvBridge()
        self._last_msg = None
        self._msg_lock = threading.Lock()
        self._image_counter = 0

        self.cfg = get_cfg()
        self.cfg.merge_from_file(self.load_param('~config'))
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.load_param(
            '~detection_threshold')  # set threshold for this model
        self.cfg.MODEL.WEIGHTS = self.load_param('~model')
        self.predictor = DefaultPredictor(self.cfg)
        self._class_names = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]).get("thing_classes", None)
        self.wall_classes = [30, 31, 32, 33, 52]  # [wall-brick, wall-stone, wall-tile, wall-wood, wall]

        self._visualization = self.load_param('~visualization', True)
        self._result_pub = rospy.Publisher('~result', ResultWithWalls, queue_size=1)
        self._vis_pub = rospy.Publisher('~visualization', Image, queue_size=1)
        self._sub = rospy.Subscriber(self.load_param('~input'), Image, self.callback_image, queue_size=1)
        self.start_time = time.time()

        # for i in range(len(self._class_names)):
        #    print(str(i)+" "+self._class_names[i]+"\n")
        # self._class_names[46] = "food"

        self.interest_classes = [39, 56, 58, 59, 60, 61, 62, 68, 69, 70, 71, 72]
        rospy.logwarn("Initialized")

    def run(self):

        rate = rospy.Rate(100)
        while not rospy.is_shutdown():
            if self._msg_lock.acquire(False):
                img_msg = self._last_msg
                self._last_msg = None
                self._msg_lock.release()
            else:
                rate.sleep()
                continue

            if img_msg is not None:
                rospy.loginfo("Time since last image=%.2f s", (time.time() - self.start_time))
                self.start_time = time.time()
                np_image = self.convert_to_cv_image(img_msg)

                outputs = self.predictor(np_image)

                walls = [item for item in outputs['panoptic_seg'][1] if item['category_id'] in self.wall_classes]
                walls_ids = [wall_id['id'] for wall_id in walls]

                walls_mask = (np.isin(outputs['panoptic_seg'][0].to("cpu").numpy(), walls_ids))

                result = outputs["instances"].to("cpu")
                result_msg = self.getResult(result)

                mask = np.zeros(walls_mask.shape, dtype="uint8")
                mask[walls_mask] = 255

                result_msg.walls = self._bridge.cv2_to_imgmsg(mask.copy())


                self._result_pub.publish(result_msg)

                #img_msg = None

                # Visualize results
                if self._visualization:

                    v = Visualizer(np_image[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=1.2)
                    # Show objects
                    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
                    # Show walls
                    out = v.draw_panoptic_seg_predictions(outputs['panoptic_seg'][0].to("cpu"), walls)
                    img = out.get_image()[:, :, ::-1]

                    image_msg_a = self._bridge.cv2_to_imgmsg(img)
                    self._vis_pub.publish(image_msg_a)

            rate.sleep()

    def getResult(self, predictions):
        boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None

        if predictions.has("pred_masks"):
            masks = np.asarray(predictions.pred_masks)
        else:
            return

        result_msg = ResultWithWalls()
        result_msg.header = self._header
        result_msg.class_ids = predictions.pred_classes if predictions.has("pred_classes") else None

        # ToDo: Inicio Cambios JL
        # result_msg.class_names = np.array(self._class_names)[self.interest_classes]
        result_msg.class_names = np.array(self._class_names)

        # result_msg.scores = torch.flatten( self.normalize( predictions.all_scores[:, self.interest_classes] )  ) if predictions.has("all_scores") else []
        result_msg.scores = torch.flatten(
            self.normalize(predictions.all_scores[:, :])) if predictions.has("all_scores") else []

        # ToDo: Fin Cambios JL

        for i, (x1, y1, x2, y2) in enumerate(boxes):
            mask = np.zeros(masks[i].shape, dtype="uint8")
            mask[masks[i, :, :]] = 255
            mask = self._bridge.cv2_to_imgmsg(mask)
            result_msg.masks.append(mask)

            box = RegionOfInterest()
            box.x_offset = np.uint32(x1)
            box.y_offset = np.uint32(y1)
            box.height = np.uint32(y2 - y1)
            box.width = np.uint32(x2 - x1)
            result_msg.boxes.append(box)

        return result_msg

    def normalize(self, scores):
        for i in range(list(scores.shape)[0]):
            scores[i] /= sum(scores[i])
        return scores

    def convert_to_cv_image(self, image_msg):

        if image_msg is None:
            return None

        self._width = image_msg.width
        self._height = image_msg.height
        channels = int(len(image_msg.data) / (self._width * self._height))

        encoding = None
        if image_msg.encoding.lower() in ['rgb8', 'bgr8']:
            encoding = np.uint8
        elif image_msg.encoding.lower() == 'mono8':
            encoding = np.uint8
        elif image_msg.encoding.lower() == '32fc1':
            encoding = np.float32
            channels = 1

        cv_img = np.ndarray(shape=(image_msg.height, image_msg.width, channels),
                            dtype=encoding, buffer=image_msg.data)

        if image_msg.encoding.lower() == 'mono8':
            cv_img = cv.cvtColor(cv_img, cv.COLOR_RGB2GRAY)
        else:
            cv_img = cv.cvtColor(cv_img, cv.COLOR_RGB2BGR)

        return cv_img

    def callback_image(self, msg):
        if self._msg_lock.acquire(False):
            self._last_msg = msg
            self._header = msg.header
            self._msg_lock.release()

    @staticmethod
    def load_param(param, default=None):
        new_param = rospy.get_param(param, default)
        rospy.loginfo("[Detectron2] %s: %s", param, new_param)
        return new_param


def main(argv):
    rospy.init_node('detectron2_ros')
    node = Detectron2node()
    node.run()


if __name__ == '__main__':
    main(sys.argv)
