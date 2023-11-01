import cv2
import numpy as np
import os
import json
from IPython.display import HTML, Video
from ipywidgets import Image
from typing import Union
import time

def get_font_words_size_dictionary(font:int, font_scale:float, thickness:int):
    '''
    return {'a': (7, 8), 'b': (7, 8), 'c': (7, 8), ... ,  '~': (7, 8), ' ': (7, 8)}, max_height
    check your font's every words pixel size
    '''
    alphabet_lower = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    alphabet_upper = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    special_characters = ['!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~']
    space = ' '
    
    every_words = alphabet_upper+alphabet_lower+number+special_characters
    every_words.append(space)
    every_words_dict={}

    max_height = 0
    for word in every_words:
        text_size, _ = cv2.getTextSize(word, font, font_scale, thickness)
        every_words_dict[word] = text_size
        if max_height <= text_size[1]:
            max_height = text_size[1]
    return every_words_dict, max_height



def putSubtitle(frame:np.array, text:str, text_position:tuple, font:int, font_scale:float, font_color:Union[int, tuple], thickness:int, index_list:list, word_height:int):
    
    '''
    return X
    
    Prevents long subtitles from stuttering in the video
    function arguments are as same as cv2.puttext()

    frame = np.array
    text = "hey how are you"
    text_position = (10, 30)  #  (x, y)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (255, 255, 255)
    font_thickness = 2
    '''
    # write subtitles
    text_loc = text_position[1]
    if len(index_list)>=2:
        sub_start = 0
        for k, sub_end in enumerate(index_list):
            text_loc += word_height+2  # 2 for easy watch
            cv2.putText(frame, text[sub_start:sub_end], (0, text_loc), font, font_scale, (255, 255, 255), thickness)
            sub_start = sub_end
    else:
        text_loc += word_height+2
        cv2.putText(frame, text, (0, text_loc), font, font_scale, (255, 255, 255), thickness)

    # write number of words
    text_loc += word_height+2

    cv2.putText(frame, 'num of words: '+str(text.count(' ')), (0, text_loc), font, font_scale, (255, 255, 255), thickness)
    #cv2.putText(frame, text[:text.find(' ')] + str(text.count(' ')), (0, text_loc), font, font_scale, (255, 255, 255), thickness)


def check_subtitle_pad(timestamps_np, sentences, every_words_dict,resize_size, text_height):
    '''
    To ensure that subtitles don't get truncated to fit the width of the video
    Added a sentence separator and a Create padding to add subtitles to

    sent_index : indexes that separate a sentence
    padding : where to add subtitle
    '''
    max_index = 0

    sent_index_list = []
    for i, (start_frame, end_frame) in enumerate(timestamps_np):
        sent_index = []
        sentence = 0  # subtitle length for one sentence 
        word = 0 # check length for one word
        for k, w in enumerate(sentences[i]):
            size = every_words_dict[w]
            word += size[0]
            if w == ' ':
                sentence += word
                if resize_size < sentence:
                    sent_index.append(idx)
                    sentence = 0
                idx = k
                word = 0
        if len(sentences[i]) not in sent_index:
            sent_index.append(len(sentences[i]))
        if max_index < len(sent_index):
            max_index = len(sent_index)
        sent_index_list.append(sent_index)
    padding = np.full(((text_height)*(max_index+3), resize_size, 3), 0, dtype=np.uint8) # max_index + 3 for write sentences + number of words

    return sent_index_list, padding



def video_open_in_jupyter(video_path: str = None, resize_size: int=250, timestamps: list = None, label_sentences: list = None, pred_sentences: list = None):
    '''
    return X
    
    read your video at jupyter notebook 
    
    if timestamps exist:
        play the corresponding timestamped portion of the video
    if timestamps exist and label_sentences exist:
        play the corresponding timestamped portion of the video with sentences below with padding
    '''
    
    font, font_scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1
    
    video = Image()
    display(video)
    if not os.path.exists(video_path):
        print(f"There is no \"{video_path}\" file")
        return None
    else:
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        img_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        img_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        file_size = os.path.getsize(video_path)
    
    timestamps_np = np.array(timestamps) # for multiply fps ( time -> frame_num )
    timestamps_np = (timestamps_np * fps).astype(int)
    every_words_dict, text_height = get_font_words_size_dictionary(font, font_scale, thickness)


    # 1. check pixel location subtitle to write
    if timestamps:
        if label_sentences:
            label_sentences = ['label: '+s for s in label_sentences]
            label_index_list, label_padding = check_subtitle_pad(timestamps_np, label_sentences, every_words_dict, resize_size, text_height)
        if pred_sentences:
            pred_sentences = ['pred: '+s for s in pred_sentences]
            pred_index_list, pred_padding = check_subtitle_pad(timestamps_np, pred_sentences, every_words_dict, resize_size, text_height)
        # 2. Play the corresponding timestamped portion of the video
        for i, (start_frame, end_frame) in enumerate(timestamps_np):
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            for frame_num in range(start_frame, end_frame + 1):
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.resize(frame, (resize_size, resize_size))
                if label_sentences and pred_sentences:
                    frame = np.vstack((frame, label_padding, pred_padding))
                    putSubtitle(frame, label_sentences[i], (0, resize_size), font, font_scale, (255, 255, 255), thickness, label_index_list[i], text_height)
                    putSubtitle(frame, pred_sentences[i], (0, resize_size+label_padding.shape[0]), font, font_scale, (255, 255, 255), thickness, pred_index_list[i], text_height)
                elif label_sentences:
                    frame = np.vstack((frame, label_padding))
                    putSubtitle(frame, label_sentences[i], (0, resize_size), font, font_scale, (255, 255, 255), thickness, label_index_list[i], text_height)
                elif pred_sentences:
                    frame = np.vstack((frame, pred_padding))
                    putSubtitle(frame, pred_sentences[i], (0, resize_size), font, font_scale, (255, 255, 255), thickness, pred_index_list[i], text_height)
                else: # play without subtitle
                    pass
                frame = cv2.imencode('.jpg', frame)[1].tobytes()
                video.value = frame
                
                if (end_frame + 1 - start_frame)<500:
                    time.sleep(5/(end_frame + 1 - start_frame))
                else:
                    pass


def main():
  '''
  Examples for ActivityNet 
  '''
  dirPath = "/workspace/llm_dataset/ActivityNet"
  json_list = [file for file in os.listdir(os.path.join(dirPath, 'captions')) if file.endswith('json')]
  with open(os.path.join(dirPath, "captions", json_list[1]), "r") as json_file:
      data = json.load(json_file)
  
  vid_ext = ['.mp4', '.mkv', '.webm']
  
  for vid_name, info in data.items():
      video_path = os.path.join("/workspace/llm_dataset/ActivityNet/videos/train_video", vid_name)
      for ext in vid_ext:
          vid_check = video_path+ext
          if os.path.isfile(vid_check):
              video_path = vid_check
              break
      if not os.path.isfile(video_path):
          continue
  
      info = data[vid_name]
      timestamps, sentences_la = info["timestamps"], info["sentences"]
      video_open_in_jupyter(video_path, 400, timestamps, sentences_la)
      #video_open_in_jupyter(video_path, 400, timestamps, sentences_la, sentences_pr) # you can include predicts to compare your outputs



if __name__ == "__main__":
    main()
