from utils import read_video, save_video    
from trackers import playerTracker,BallTracker
from drawers import(PlayerTracksDrawer, BallTracksDrawer, TeamBallControlDrawer, PassInterceptionDrawer, CourtKeypointDrawer)
from team_assigner import TeamAssigner
from ball_acquisition import BallAcquisitionDetector
from pass_and_interception_detector import PassAndInterceptionDetector
from court_key_point_detector import CourtKeypointDetector

def main():
    video_frame=read_video("input_videos/video_3.mp4")
    #initialise Tracker
    player_tracker= playerTracker("models/player_detector.pt")
    ball_tracker= BallTracker("models/ball_detector_model.pt")

    #Initailise court key point detector
    court_key_point_detector = CourtKeypointDetector("models/court_keypoint_detector.pt")

    #Run Trackers
    player_tracks= player_tracker.get_object_tracks(video_frame,
                                                    read_from_stub=True,
                                                stub_path="stubs/player_track_stubs.pkl")
    
    ball_tracks= ball_tracker.get_object_tracks(video_frame,
                                                read_from_stub=True,
                                                stub_path="stubs/ball_track_stubs.pkl")
    #Get court key points
    court_key_points = court_key_point_detector.get_court_keypoints(video_frame,
                                                                    read_from_stub=True,
                                                                    stub_path="stubs/court_key_points_stubs.pkl")
    print(court_key_points)


    #remove wrong ball detections
    ball_tracks=ball_tracker.remove_wrong_detections(ball_tracks)   

    #Interpolate ball positions
    ball_tracks=ball_tracker.interpolate_ball_positions(ball_tracks)
    #Assign teams to players
    team_assigner= TeamAssigner()
    player_assignment= team_assigner.get_player_teams_across_frames(video_frame,
                                                                player_tracks,
                                                                read_from_stubs=True,
                                                                stub_path="stubs/player_team_assignment_stubs.pkl")
    #Detect ball acquisition
    ball_acquisition_detector= BallAcquisitionDetector()
    ball_acquisition= ball_acquisition_detector.detect_ball_possession(player_tracks, ball_tracks)
  

    #detect passes and interceptions
    pass_and_interception_detector = PassAndInterceptionDetector()
    passes = pass_and_interception_detector.detect_passes(ball_acquisition,player_assignment)
    interceptions = pass_and_interception_detector.detect_interceptions(ball_acquisition,player_assignment)
   

    #Draw output
    #initailise drawers
    player_tracks_drawer=PlayerTracksDrawer()
    ball_tracks_drawer=BallTracksDrawer()
    team_ball_control_drawer=TeamBallControlDrawer()
    pass_and_interceptions_drawer = PassInterceptionDrawer()
    court_keypoint_drawer = CourtKeypointDrawer()

    #Draw Object Tracks
    output_video_frames=player_tracks_drawer.draw(video_frame, player_tracks,player_assignment, ball_acquisition)   
    output_video_frames=ball_tracks_drawer.draw(output_video_frames, ball_tracks)   

    #Draw Team Ball Control
    output_video_frames=team_ball_control_drawer.draw(output_video_frames,player_assignment,ball_acquisition)

    # Draw Passes and Interceptions
    output_video_frames = pass_and_interceptions_drawer.draw(output_video_frames,
                                                             passes,
                                                             interceptions)
    #Draw Court Keypoints
    output_video_frames = court_keypoint_drawer.draw(output_video_frames, court_key_points)

    save_video(output_video_frames,"output_videos/output_video.avi")   

if __name__ == "__main__":
    main()