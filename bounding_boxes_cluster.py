from evalutils import DetectionAlgorithm
import SimpleITK as sitk
import numpy as np
import torch
import json
from typing import Dict
import os
from pathlib import Path
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
import itertools
from pandas import DataFrame
from postprocessing import get_NonMaxSup_boxes

class Hip_detection(DetectionAlgorithm):
    def __init__(self, input_dir, output_dir):
        super().__init__(
            input_path = Path(input_dir),
            output_file = Path(os.path.join(output_dir,'output_mha4.json'))
        )
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_path, self.output_path = input_dir, output_dir
        print('using the device ', self.device)

        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights)
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        num_classes = 5
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        
        print('loading the model.pth file :')
        self.model.load_state_dict(
          torch.load(
              Path("/mnt/netcache/bodyct/experiments/hip_oa_grading/docker/bounding_boxes/fasterrcnn/model_retrained_307.pth"),
              map_location=self.device,
              )
          ) 
        self.model.to(self.device)
        
    def save(self):
        with open(str(self._output_file), "w") as f:
            json.dump(self._case_results, f)
            
    def process_case(self, *, idx, case):
        '''
        Read the input, perform model prediction and return the results.         
        '''
        # Load and test the image for this case
        input_image, input_image_file_path = self._load_input_image(case=case)
        
        # Detect and score candidates
        scored_candidates = self.predict(input_image=input_image,input_image_file_path=input_image_file_path)
        # Write resulting candidates to nodules.json for this case
        return scored_candidates
    
    def draw_best_bb(self,label):
        proba = []
        for box in label[0]:
          proba.append(box['probability'])
        done = False
        for box in label[0]:
          if box['probability'] == np.max(proba) and not done and box['probability']>0.83:
            done = True
            return box
        if not done: return 0

    def extract_results(self,input_image_file_path,out):
        boxes = []
        left = []
        right = []
        L = []
        R = []
        faux = []
        if out['name']==str(input_image_file_path):
            for box in out['boxes']:
                if box['label'] == 4.0: 
                    size = sitk.ReadImage(str(input_image_file_path)).GetSize()
                    if box['corners'][2][0] > size[0]/2: left.append(box)
                    else: right.append(box)
                elif box['label'] == 1.0: L.append(box)
                elif box['label'] == 2.0: R.append(box)
                elif box['label'] == 3.0: faux.append(box)
            labels = ([left, 'draw left hip'],[right, 'draw right hip'],[L,'L'],[R,'R'],[faux, 'faux profile'])
            for l in labels:
                if len(l[0])!=0:
                    box = self.draw_best_bb(l)
                    if box != 0: 
                        boxes.append(box)
        final_dict={"type": "Multiple 2D bounding boxes", 'boxes': boxes, "name": out['name']} 
        return final_dict
          
    def format_to_GC(self, np_prediction, spacing, name) -> Dict:
        '''
        Convenient function returns detection prediction in required grand-challenge format.
        See:
        https://comic.github.io/grandchallenge.org/components.html#grandchallenge.components.models.InterfaceKind.interface_type_annotation
        
        
        np_prediction: dictionary with keys boxes and scores.
        np_prediction[boxes] holds coordinates in the format as x1,y1,x2,y2
        spacing :  pixel spacing for x and y coordinates.
        
        return:
        a Dict in line with grand-challenge.org format.
        '''
        # For the test set, we expect the coordinates in millimeters. 
        # This transformation ensures that the pixel coordinates are transformed to mm.
        # And boxes coordinates saved according to grand challenge ordering.
        x_y_spacing = [spacing[0], spacing[1], spacing[0], spacing[1]]
        boxes = []
        for i, bb in enumerate(np_prediction['boxes']):
            box = {}   
            box['corners']=[]
            x_min, y_min, x_max, y_max = bb*x_y_spacing
            x_min, y_min, x_max, y_max  = round(x_min, 2), round(y_min, 2), round(x_max, 2), round(y_max, 2)
            bottom_left = [x_min, y_min,  np_prediction['slice'][i]] 
            bottom_right = [x_max, y_min,  np_prediction['slice'][i]]
            top_left = [x_min, y_max,  np_prediction['slice'][i]]
            top_right = [x_max, y_max,  np_prediction['slice'][i]]
            box['corners'].extend([top_right, top_left, bottom_left, bottom_right])
            box['probability'] = round(float(np_prediction['scores'][i]), 2)
            box['label'] = float(np_prediction['labels'][i])
            boxes.append(box)
        return dict(type="Multiple 2D bounding boxes", boxes=boxes, version={ "major": 1, "minor": 0 }, name=str(name))
        
    def merge_dict(self, results):
        merged_d = {}
        for k in results[0].keys():
            merged_d[k] = list(itertools.chain(*[d[k] for d in results]))
        return merged_d
        
    def predict(self, *, input_image: sitk.Image, input_image_file_path) -> DataFrame:
        self.model.eval() 
        image_data = sitk.GetArrayFromImage(input_image)
        spacing = input_image.GetSpacing()
        image_data = np.array(image_data)
        
        if len(image_data.shape)==2:
            image_data = np.expand_dims(image_data, 0)
            
        results = []
        # Operate on 3D image (CXRs are stacked together)
        for j in range(len(image_data)):
            # Pre-process the image
            image = image_data[j,:,:]
            # The range should be from 0 to 1
            image = image.astype(np.float32) / np.max(image)  # normalize
            image = np.expand_dims(image, axis=0)
            tensor_image = torch.from_numpy(image).to(self.device)
            with torch.no_grad():
                prediction = self.model([tensor_image.to(self.device)])
            
            prediction = [get_NonMaxSup_boxes(prediction[0])]
            # Convert predictions from tensor to numpy array
            np_prediction = {str(key):[i.cpu().numpy() for i in val]
                   for key, val in prediction[0].items()}
            np_prediction['slice'] = len(np_prediction['boxes'])*[j]
            results.append(np_prediction)
        
        predictions = self.merge_dict(results)
        data = self.format_to_GC(predictions, spacing, input_image_file_path)
        data = self.extract_results(input_image_file_path,data)
        return data

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        prog='process.py',
        description=
            'Reads all images from an input directory and produces '
            'results in an output directory')

    parser.add_argument('input_dir', help = "input directory to process")
    parser.add_argument('output_dir', help = "output directory generate result files in")
    
    parsed_args = parser.parse_args() 
    Hip_detection(parsed_args.input_dir, parsed_args.output_dir).process()