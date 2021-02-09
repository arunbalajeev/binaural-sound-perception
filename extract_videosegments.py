"""
Author: Arun Balajee Vasudevan
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""

import glob,os
import numpy as np

from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy.editor import VideoFileClip

############################### Extract video segments ############################

for sc in range(1,166):
    fdir="./dataset_public/scene%04d/"%sc
    videonum = int(glob.glob(fdir+"/*cropped.mp4")[0].split('/')[-1].split('_')[1]);
    videofile="VIDEO_"+"%04d"%videonum+"_cropped.mp4"
    save_dir = fdir +"split_videos/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    clip = VideoFileClip(fdir+videofile);
    for t in np.arange(0,clip.duration,2):
        start=t;end = t+2;
        ffmpeg_extract_subclip(fdir+videofile, start, end, targetname=save_dir+videofile[:-4]+"_%06d"%t+".mp4")

############################### Extract video frames ###############################

import moviepy.editor as mpe

for sc in range(1,166):
    fdir="./dataset_public/scene%04d/"%sc
    videonum = int(glob.glob(fdir+"/*cropped.mp4")[0].split('/')[-1].split('_')[1]);
    videofile="VIDEO_"+"%04d"%videonum+"_cropped.mp4"
    save_dir = fdir +"split_videoframes/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    video = mpe.VideoFileClip(fdir+videofile)
    clip = VideoFileClip(fdir+videofile);
    for it in np.arange(1,clip.duration,2):
        #np_frame = video.get_frame(it) # get the frame at t=2 seconds
        #np_frame = video.get_frame(frame_number * video_fps) # get frame by index
        video.save_frame(save_dir+videofile[:-4]+"_%06d"%(it-1)+".png", t=it) # save frame at t=2 as PNG
