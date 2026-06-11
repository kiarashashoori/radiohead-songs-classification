from parameters import Parameters as p
from pydub import AudioSegment
import os
import logging
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import time
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)


class ExtractDataset():
    def __init__(self):
        self.music_path = p.music_path
        self.dataset_path = p.dataset_path

        self.each_clip_dur = 5000
        self.between_clip_dur = 5000
        self.skip_dur = 2000

        self.music_classes = os.listdir(self.music_path)

    def record_sound(self,name,DURATION):

        SAMPLE_RATE = 16000  
        print(1)
        time.sleep(1)
        print(2)
        time.sleep(1)
        print(3)
        time.sleep(1)

        print("Recording...")
        audio = sd.rec(
            int(DURATION * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype=np.int16
        )
        sd.wait()

        write(f"{name}", SAMPLE_RATE, audio)
        print(f"Saved {name}")
    def create_classes_folder(self,ds = 'train'):
        self.music_classes = os.listdir(self.music_path)
        for music in self.music_classes:
            if not music.endswith(p.music_formats):
                self.music_classes.remove(music)
                logging.info(f"unkown file '{music}'")
                continue
            os.mkdir(p.dataset_path+ds+'/'+music[:len(music)-4])
            logging.info(f"directory for '{music[:len(music)-4]}' class created ---- {ds}")
    
    def clip_train_audios(self,mode = 0):
        if mode == 0:
            for file in self.music_classes:
                audio = AudioSegment.from_file(self.music_path+file)
                song_duration = len(audio)
                for i in range(0,int(song_duration//(self.between_clip_dur+self.each_clip_dur))):
                    start_time = i*(self.between_clip_dur+self.each_clip_dur)
                    end_time = start_time+self.each_clip_dur
                    clipped = audio[start_time:end_time]
                    output_path = self.dataset_path+'train/'+file[:len(file)-4]+'/'
                    clipped.export(output_path+str(i)+".mp3", format='mp3')
                logging.info(f"{i} labels created for '{file[:len(file)-4]}' class ------ train")
        if mode == 1:
            for file in self.music_classes:
                audio = AudioSegment.from_file(self.music_path+file)
                song_duration = len(audio)
                for i in range(0,int(song_duration//(self.skip_dur))):
                    start_time = i*self.skip_dur
                    end_time = start_time+self.each_clip_dur
                    if end_time >= song_duration-5000:break
                    clipped = audio[start_time:end_time]
                    output_path = self.dataset_path+'train/'+file[:len(file)-4]+'/'
                    clipped.export(output_path+str(i)+".mp3", format='mp3')
                logging.info(f"{i} labels created for '{file[:len(file)-4]}' class ------ train")
        if mode == 2:
            for file in self.music_classes:
                audio = AudioSegment.from_file(self.music_path+file)
                song_duration = len(audio)
                for i in range(0,int(song_duration//(self.skip_dur))):
                    start_time = i*self.skip_dur
                    end_time = start_time+self.each_clip_dur
                    clipped = audio[start_time:end_time]
                    output_path = self.dataset_path+'train/'+file[:len(file)-4]+'/'
                    clipped.export(output_path+str(i)+".mp3", format='mp3')
                logging.info(f"{i} labels created for '{file[:len(file)-4]}' class ------ train")

    def clip_val_audios(self):
        for file in self.music_classes:
            audio = AudioSegment.from_file(self.music_path+file)
            song_duration = len(audio)
            for i in range(0,int(song_duration//(self.between_clip_dur+self.each_clip_dur))):
                start_time = i*(self.between_clip_dur+self.each_clip_dur)+self.each_clip_dur
                end_time = start_time+self.between_clip_dur
                clipped = audio[start_time:end_time]
                output_path = self.dataset_path+'val/'+file[:len(file)-4]+'/'
                clipped.export(output_path+str(i)+".mp3", format='mp3')
            logging.info(f"{i} labels created for '{file[:len(file)-4]}' class ----- val")

dataset = ExtractDataset()
# dataset.create_classes_folder(mode='train')
# dataset.create_classes_folder(mode='val')
# dataset.clip_train_audios()
# dataset.clip_val_audios()

dataset.record_sound('Where I End and You Begin.wav',269)

