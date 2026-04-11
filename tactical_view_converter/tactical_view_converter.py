import os
import sys
import pathlib
import numpy as np
import cv2
from copy import deepcopy
from .homography import Homography

folder_path = pathlib.Path(__file__).parent.resolve()
sys.path.append(os.path.join(folder_path,"../"))
from utils import get_foot_position,measure_distance

class TacticalViewConverter:
    def __init__(self, court_image_path):
        self.court_image_path = court_image_path
        self.width = 300
        self.height= 161

        self.actual_width_in_meters=28
        self.actual_height_in_meters=15 

        self.key_points = [
            # left edge
            (0,0),
            (0,int((0.91/self.actual_height_in_meters)*self.height)),
            (0,int((5.18/self.actual_height_in_meters)*self.height)),
            (0,int((10/self.actual_height_in_meters)*self.height)),
            (0,int((14.1/self.actual_height_in_meters)*self.height)),
            (0,int(self.height)),

            # Middle line
            (int(self.width/2),self.height),
            (int(self.width/2),0),
            
            # Left Free throw line
            (int((5.79/self.actual_width_in_meters)*self.width),int((5.18/self.actual_height_in_meters)*self.height)),
            (int((5.79/self.actual_width_in_meters)*self.width),int((10/self.actual_height_in_meters)*self.height)),

            # right edge
            (self.width,int(self.height)),
            (self.width,int((14.1/self.actual_height_in_meters)*self.height)),
            (self.width,int((10/self.actual_height_in_meters)*self.height)),
            (self.width,int((5.18/self.actual_height_in_meters)*self.height)),
            (self.width,int((0.91/self.actual_height_in_meters)*self.height)),
            (self.width,0),

            # Right Free throw line
            (int(((self.actual_width_in_meters-5.79)/self.actual_width_in_meters)*self.width),int((5.18/self.actual_height_in_meters)*self.height)),
            (int(((self.actual_width_in_meters-5.79)/self.actual_width_in_meters)*self.width),int((10/self.actual_height_in_meters)*self.height)),
        ]
        self.previous_homography = None
        self.previous_tactical_positions = {}

    def _extract_keypoints_xy(self, frame_keypoints):
        if not hasattr(frame_keypoints, "xy") or frame_keypoints.xy is None:
            return None

        xy_list = frame_keypoints.xy.tolist()
        if len(xy_list) == 0:
            return None

        return xy_list[0]

    def _compute_homography_matrix(self, detected_keypoints):
        valid_indices = [i for i, kp in enumerate(detected_keypoints) if kp[0] > 0 and kp[1] > 0]

        if len(valid_indices) < 4:
            return None

        source_points = np.array([detected_keypoints[i] for i in valid_indices], dtype=np.float32)
        target_points = np.array([self.key_points[i] for i in valid_indices], dtype=np.float32)

        matrix, inlier_mask = cv2.findHomography(
            source_points,
            target_points,
            method=cv2.RANSAC,
            ransacReprojThreshold=5.0,
        )

        if matrix is None or inlier_mask is None:
            return None

        inliers = int(inlier_mask.sum())
        if inliers < 4:
            return None

        return matrix

    def _transform_frame_players(self, frame_tracks, homography_matrix):
        tactical_positions = {}

        for player_id, player_data in frame_tracks.items():
            bbox = player_data["bbox"]
            player_position = np.array([get_foot_position(bbox)], dtype=np.float32).reshape(-1, 1, 2)
            tactical_position = cv2.perspectiveTransform(player_position, homography_matrix).reshape(-1, 2)[0]

            x = float(tactical_position[0])
            y = float(tactical_position[1])

            # Ignore only clearly invalid projections; keep near-border points by clamping.
            if x < -40 or x > self.width + 40 or y < -40 or y > self.height + 40:
                continue

            x = min(max(x, 0.0), float(self.width - 1))
            y = min(max(y, 0.0), float(self.height - 1))
            tactical_positions[player_id] = [x, y]

        return tactical_positions

    def _median_displacement(self, current_positions, previous_positions):
        common_ids = [pid for pid in current_positions.keys() if pid in previous_positions]
        if len(common_ids) < 2:
            return None

        displacements = []
        for pid in common_ids:
            current_point = current_positions[pid]
            previous_point = previous_positions[pid]
            displacements.append(measure_distance(current_point, previous_point))

        if len(displacements) == 0:
            return None
        return float(np.median(displacements))

    def validate_keypoints(self, keypoints_list):
        """
        Validates detected keypoints by comparing their proportional distances
        to the tactical view keypoints.
        
        Args:
            keypoints_list (List[List[Tuple[float, float]]]): A list containing keypoints for each frame.
                Each outer list represents a frame.
                Each inner list contains keypoints as (x, y) tuples.
                A keypoint of (0, 0) indicates that the keypoint is not detected for that frame.
        
        Returns:
            List[bool]: A list indicating whether each frame's keypoints are valid.
        """

        keypoints_list = deepcopy(keypoints_list)

        for frame_idx, frame_keypoints in enumerate(keypoints_list):
            # SAFE extraction (DO NOT SKIP ANY LINE)
            if not hasattr(frame_keypoints, "xy") or frame_keypoints.xy is None:
                continue

            xy_list = frame_keypoints.xy.tolist()

            if len(xy_list) == 0:
                continue

            frame_keypoints = xy_list[0]
            
            # Get indices of detected keypoints (not (0, 0))
            detected_indices = [i for i, kp in enumerate(frame_keypoints) if kp[0] >0 and kp[1]>0]
            
            # Need at least 3 detected keypoints to validate proportions
            if len(detected_indices) < 3:
                continue
            
            invalid_keypoints = []
            # Validate each detected keypoint
            for i in detected_indices:
                # Skip if this is (0, 0)
                if frame_keypoints[i][0] == 0 and frame_keypoints[i][1] == 0:
                    continue

                # Choose two other random detected keypoints
                other_indices = [idx for idx in detected_indices if idx != i and idx not in invalid_keypoints]
                if len(other_indices) < 2:
                    continue

                # Take first two other indices for simplicity
                j, k = other_indices[0], other_indices[1]

                # Calculate distances between detected keypoints
                d_ij = measure_distance(frame_keypoints[i], frame_keypoints[j])
                d_ik = measure_distance(frame_keypoints[i], frame_keypoints[k])
                
                # Calculate distances between corresponding tactical keypoints
                t_ij = measure_distance(self.key_points[i], self.key_points[j])
                t_ik = measure_distance(self.key_points[i], self.key_points[k])

                # Calculate and compare proportions with 50% error margin
                if t_ij > 0 and t_ik > 0:
                    prop_detected = d_ij / d_ik if d_ik > 0 else float('inf')
                    prop_tactical = t_ij / t_ik if t_ik > 0 else float('inf')

                    error = (prop_detected - prop_tactical) / prop_tactical
                    error = abs(error)

                    if error >0.8:  # 80% error margin                        
                        keypoints_list[frame_idx].xy[0][i] *= 0
                        keypoints_list[frame_idx].xyn[0][i] *= 0
                        invalid_keypoints.append(i)
            
        return keypoints_list

    def transform_players_to_tactical_view(self, keypoints_list, player_tracks):
        """
        Transform player positions from video frame coordinates to tactical view coordinates.
        
        Args:
            keypoints_list (list): List of detected court keypoints for each frame.
            player_tracks (list): List of dictionaries containing player tracking information for each frame,
                where each dictionary maps player IDs to their bounding box coordinates.
        
        Returns:
            list: List of dictionaries where each dictionary maps player IDs to their (x, y) positions
                in the tactical view coordinate system. The list index corresponds to the frame number.
        """
        tactical_player_positions = []
        self.previous_homography = None
        self.previous_tactical_positions = {}
        
        for frame_idx, (frame_keypoints, frame_tracks) in enumerate(zip(keypoints_list, player_tracks)):
            # Initialize empty dictionary for this frame
            tactical_positions = {}

            detected_keypoints = self._extract_keypoints_xy(frame_keypoints)
            if detected_keypoints is None:
                tactical_player_positions.append(tactical_positions)
                continue

            candidate_homography = self._compute_homography_matrix(detected_keypoints)
            previous_homography = self.previous_homography

            selected_homography = None
            if candidate_homography is None and previous_homography is None:
                tactical_player_positions.append(tactical_positions)
                continue

            if candidate_homography is None:
                selected_homography = previous_homography
            elif previous_homography is None:
                selected_homography = candidate_homography
            else:
                candidate_positions = self._transform_frame_players(frame_tracks, candidate_homography)
                previous_positions = self._transform_frame_players(frame_tracks, previous_homography)

                candidate_disp = self._median_displacement(candidate_positions, self.previous_tactical_positions)
                previous_disp = self._median_displacement(previous_positions, self.previous_tactical_positions)

                # Reject sudden mirrored/unstable flips if previous homography is clearly more consistent.
                if candidate_disp is not None and previous_disp is not None:
                    if candidate_disp > max(35.0, previous_disp * 2.2):
                        selected_homography = previous_homography
                    else:
                        selected_homography = candidate_homography
                else:
                    selected_homography = candidate_homography

            if selected_homography is None:
                tactical_player_positions.append(tactical_positions)
                continue

            self.previous_homography = selected_homography
            
            try:
                tactical_positions = self._transform_frame_players(frame_tracks, selected_homography)
                self.previous_tactical_positions = tactical_positions.copy()
                    
            except (ValueError, cv2.error) as e:
                # If homography fails, continue with empty dictionary
                pass
            
            tactical_player_positions.append(tactical_positions)
        
        return tactical_player_positions
