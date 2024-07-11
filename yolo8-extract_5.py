import torch
import json
from ultralytics import YOLO


class CustomDetect(YOLO):
    def __init__(self, model_path):
        super().__init__(model_path)

    def _inference(self, x):
        """Decode predicted bounding boxes and class probabilities based on multiple-level feature maps."""
        # Inference path
        shape = x[0].shape  # BCHW
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        self.save_as_json(x_cat, 'x_cat.json')

        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        if self.export and self.format in {"saved_model", "pb", "tflite", "edgetpu", "tfjs"}:  # avoid TF FlexSplitV ops
            box = x_cat[:, :self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4:]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)

        self.save_as_json(box, 'box.json')
        self.save_as_json(cls, 'cls.json')

        if self.export and self.format in {"tflite", "edgetpu"}:
            # Precompute normalization factor to increase numerical stability
            grid_h = shape[2]
            grid_w = shape[3]
            grid_size = torch.tensor([grid_w, grid_h, grid_w, grid_h], device=box.device).reshape(1, 4, 1)
            norm = self.strides / (self.stride[0] * grid_size)
            dbox = self.decode_bboxes(self.dfl(box) * norm, self.anchors.unsqueeze(0) * norm[:, :2])
        else:
            dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides

        print("_inference print is here!")
        return torch.cat((dbox, cls.sigmoid()), 1)

    def save_as_json(self, tensor, filename):
        """Save tensor as a JSON file."""
        tensor_list = tensor.cpu().numpy().tolist()  # Convert tensor to list
        with open(filename, 'w') as f:
            json.dump(tensor_list, f)


# Load the custom model
model = CustomDetect("yolov8n.pt")

# Perform inference
results = model("bus.jpg")

# Check for saved JSON files: 'x_cat.json', 'box.json', 'cls.json'
