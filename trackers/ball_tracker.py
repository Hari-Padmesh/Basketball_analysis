from ultralytics import YOLO
import numpy as np
import supervision as sv
import sys
sys.path.append("../")
from utils import read_stub, save_stub

class BallTracker:

    def __init__(self,model_path):
        self.model = YOLO(model_path)
    
    def detect_frames(self, frame):
        batch_size=20
        detections=[]
        for i in range(0, len(frame), batch_size):
            batch_frames=frame[i:i+batch_size]
            results=self.model.predict(batch_frames, conf=0.5)
            detections+=results
        return detections
    
    def get_object_tracks(self, frames,read_from_stub=False ,stub_path=None):
        
        tracks=read_stub(read_from_stub,stub_path)
        if tracks is not None:
            if len(tracks)==len(frames):
                return tracks
        
        detections=self.detect_frames(frames)
        tracks=[]
        for frame_num, detection in enumerate(detections):
            cls_names=detection.names
            cls_names_inv={v:k for k,v in cls_names.items()}
            detection_supervision=sv.Detections.from_ultralytics(detection)
            tracks.append({})
            chosen_bbox=None
            max_confidence=0

            for frame_detection in detection_supervision:
                bbox=frame_detection[0].tolist()
                cls_id=frame_detection[3]
                confidence=frame_detection[2]
                if cls_id==cls_names_inv["Ball"] and confidence>max_confidence:
                    chosen_bbox=bbox
                    max_confidence=confidence
            
            if chosen_bbox is not None:
                tracks[frame_num][1]={"bbox":chosen_bbox}


           

        save_stub(stub_path,tracks) 
        return tracks
    
    def remove_wrong_detections(self,ball_positions):
        maximum_allowed_distance=25
        last_good_frame_index=-1
        for i in range(len(ball_positions)):
            current_bbox=ball_positions[i].get(1,{}).get("bbox",[])
            if len(current_bbox)==0:
                continue
            if last_good_frame_index==-1:
                last_good_frame_index=i
                continue
            last_good_bbox=ball_positions[last_good_frame_index].get(1,{}).get("bbox",[])
            frame_gap=i-last_good_frame_index
            adjusted_max_distance=maximum_allowed_distance*frame_gap

            #calculate the distance between the current bbox and the last good bbox
            if np.linalg.norm(np.array(last_good_bbox[:2])-np.array(current_bbox[:2])) > adjusted_max_distance:
                ball_positions[i]={}
            else:
                last_good_frame_index=i

        return ball_positions
      
