import cv2
import numpy as np
from pytube import YouTube
import io
import os
import tempfile
import sys

def download_video(youtube_url):
    yt = YouTube(youtube_url)
    stream = yt.streams.filter(file_extension='mp4', res='720p').first()

    # Specify the temporary file path
    temp_dir = tempfile.gettempdir()
    print(f'Temporary directory: {temp_dir}')
    temp_file_path = os.path.join(temp_dir, 'temp_video.mp4')

    # Download the video to the specified temporary file
    stream.download(output_path=temp_dir, filename='temp_video.mp4')

    # Check if the file exists
    if not os.path.exists(temp_file_path):
        print(f"Error: File not found - {temp_file_path}")
        sys.exit(1)

    return temp_file_path

def detect_humans_from_youtube(youtube_url):
    video_file_path = download_video(youtube_url)
    cap = cv2.VideoCapture(video_file_path)

    # Load pre-trained HOG detector
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize the frame for better visibility of detections
        frame = cv2.resize(frame, (800, 600))

        # Detect humans in the frame
        humans, _ = hog.detectMultiScale(frame, winStride=(8, 8), padding=(4, 4), scale=1.05)

        # Draw rectangles around detected humans
        for (x, y, w, h) in humans:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.imshow('Human Detection', frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Use the function with your YouTube video URL
detect_humans_from_youtube('https://www.youtube.com/watch?v=xjtZYFOcjBs')
