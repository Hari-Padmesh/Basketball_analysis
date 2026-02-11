from utils import read_video, save_video    
from trackers import playerTracker,BallTracker
from drawers import(PlayerTracksDrawer, BallTracksDrawer)

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

    
    #Draw output
    #initailise drawers
    player_tracks_drawer=PlayerTracksDrawer()
    ball_tracks_drawer=BallTracksDrawer()

    #Draw Object Tracks
    output_video_frames=player_tracks_drawer.draw(video_frame, player_tracks)   
    output_video_frames=ball_tracks_drawer.draw(output_video_frames, ball_tracks)   

    save_video(output_video_frames,"output_videos/output_video.avi")   

    

if __name__ == "__main__":
    main()