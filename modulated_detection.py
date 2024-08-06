import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T

torch.set_grad_enabled(False)


def nms(dets, scores, thresh):
    """
    dets is a numpy array : num_dets, 4
    scores ia  nump array : num_dets,
    """
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]  # get boxes with more ious first

    keep = []
    while order.size > 0:
        i = order[0]  # pick maxmum iou box
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)  # maximum width
        h = np.maximum(0.0, yy2 - yy1 + 1)  # maxiumum height
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return dets[keep], scores[keep]


def class_agnostic_nms(boxes, scores, iou=0.5):
    if len(boxes) > 1:
        boxes, scores = nms(np.array(boxes), np.array(scores), iou)
        return list(boxes), list(scores)
    else:
        return boxes, scores


def generate_image_crops(img, num_crops=8):
    """
    Note: num_crops must be greater than 2 and of multiple of 2
    """
    assert num_crops > 2
    assert num_crops % 2 == 0
    # Get the image width and height
    img_w, img_h = img.size
    crops = []
    coordinates = []
    crops.append(img)
    coordinates.append((0, 0, img_w, img_h))
    crop_chunks_x = int(num_crops / 2)
    crop_chunks_y = int(num_crops / crop_chunks_x)
    x_inc = int(img_w / crop_chunks_y)
    y_inc = int(img_h / crop_chunks_y)
    x_space = np.linspace(0, img_w - x_inc, crop_chunks_y)
    y_spcae = np.linspace(0, img_h - y_inc, int(num_crops / crop_chunks_y))
    if num_crops > 1:
        for x in x_space:
            for y in y_spcae:
                x1, y1 = x, y
                x2, y2 = x1 + x_inc, y1 + y_inc
                crops.append((img.crop((x1, y1, x2, y2))).resize((img_w, img_h)))
                coordinates.append((x1, y1, x2, y2))
    return crops, coordinates, (img_w, img_h)


def scale_boxes(boxes, coordinates, img_dims):
    x1, y1, x2, y2 = coordinates
    img_w, img_h = img_dims
    w = x2 - x1
    h = y2 - y1
    for b in boxes:
        b[0], b[1], b[2], b[3] = (
            int((b[0] / img_w) * w) + x1,
            int((b[1] / img_h) * h) + y1,
            int((b[2] / img_w) * w) + x1,
            int((b[3] / img_h) * h) + y1,
        )
    return boxes


class Inference:
    def __init__(self, model):
        self.model = model
        self.model = self.model.cuda()
        self.model.eval()

    def infer_image(self, image_path, **kwargs):
        raise NotImplementedError


class ModulatedDetection(Inference):
    """
    The class supports the inference using both MDETR & MDef-DETR models.
    """

    def __init__(self, model, confidence_thresh=0.0):
        Inference.__init__(self, model)
        self.conf_thresh = confidence_thresh
        self.transform = T.Compose(
            [T.Resize(800), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        )

    @staticmethod
    def box_cxcywh_to_xyxy(x):
        x_c, y_c, w, h = x.unbind(1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=1)

    def rescale_bboxes(self, out_bbox, size):
        img_w, img_h = size
        b = self.box_cxcywh_to_xyxy(out_bbox)
        b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
        return b

    def infer_image(self, image_path, **kwargs):
        caption = kwargs["caption"]

        # Read the image
        im = Image.open(image_path)
        imq = np.array(im)
        if len(imq.shape) != 3:
            im = im.convert("RGB")
        img = self.transform(im).unsqueeze(0).cuda()

        # propagate through the models
        memory_cache = self.model(img, [caption], encode_and_save=True)
        outputs = self.model(img, [caption], encode_and_save=False, memory_cache=memory_cache)

        # keep only predictions with self.conf_thresh+ confidence
        probas = 1 - outputs["pred_logits"].softmax(-1)[0, :, -1].cpu()
        keep = (probas > self.conf_thresh).cpu()

        # convert boxes from [0; 1] to image scales
        bboxes_scaled = self.rescale_bboxes(outputs["pred_boxes"].cpu()[0, keep], im.size)
        kept_probs = probas[keep]

        # Convert outputs to the required format
        bboxes = list(bboxes_scaled.numpy())
        probs = list(kept_probs.numpy())
        boxes, scores = [], []
        for b, conf in zip(bboxes, probs):
            boxes.append([int(b[0]), int(b[1]), int(b[2]), int(b[3])])
            scores.append(conf)

        return boxes, scores
