from utils import read_video, save_video    
from trackers import playerTracker,BallTracker
from drawers import(PlayerTracksDrawer, BallTracksDrawer, TeamBallControlDrawer)
from team_assigner import TeamAssigner
from ball_acquisition import BallAcquisitionDetector

def main():
    video_frame=read_video("input_videos/video_1.mp4")
    #initialise Tracker
    player_tracker= playerTracker("models/player_detector.pt")
    ball_tracker= BallTracker("models/ball_detector_model.pt")
    #Run Trackers
    player_tracks= player_tracker.get_object_tracks(video_frame,
                                                    read_from_stub=True,
                                                stub_path="stubs/player_track_stubs.pkl")
    
    ball_tracks= ball_tracker.get_object_tracks(video_frame,
                                                read_from_stub=True,
                                                stub_path="stubs/ball_track_stubs.pkl")


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
    print(ball_acquisition)

    #Draw output
    #initailise drawers
    player_tracks_drawer=PlayerTracksDrawer()
    ball_tracks_drawer=BallTracksDrawer()
    team_ball_control_drawer=TeamBallControlDrawer()


    #Draw Object Tracks
    output_video_frames=player_tracks_drawer.draw(video_frame, player_tracks,player_assignment, ball_acquisition)   
    output_video_frames=ball_tracks_drawer.draw(output_video_frames, ball_tracks)   

    #Draw Team Ball Control
    output_video_frames=team_ball_control_drawer.draw(output_video_frames,player_assignment,ball_acquisition)

    save_video(output_video_frames,"output_videos/output_video.avi")   

if __name__ == "__main__":
    main()