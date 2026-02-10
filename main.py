from utils import read_video, save_video    
from trackers import playerTracker
def main():
    video_frame=read_video("input_videos/video_1.mp4")
    #initialise Tracker
    player_tracker= playerTracker("models/player_detector.pt")

    #Run Trackers
    player_tracks= player_tracker.get_object_tracks(video_frame,
                                                    read_from_stub=True,
                                                stub_path="stubs/player_track_stubs.pkl")

    print(player_tracks) 

    save_video(video_frame,"output_videos/output_video.avi")   

    

if __name__ == "__main__":
    main()