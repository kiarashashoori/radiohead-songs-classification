from parameters import Parameters as p
from pydub import AudioSegment
import os
import logging
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

        self.music_classes = os.listdir(self.music_path)
    
    def create_classes_folder(self,mode = 'train'):
        self.music_classes = os.listdir(self.music_path)
        for music in self.music_classes:
            if not music.endswith(p.music_formats):
                self.music_classes.remove(music)
                logging.info(f"unkown file '{music}'")
                continue
            os.mkdir(p.dataset_path+mode+'/'+music[:len(music)-4])
            logging.info(f"directory for '{music[:len(music)-4]}' class created ---- {mode}")
    
    def clip_train_audios(self):

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
dataset.create_classes_folder(mode='train')
dataset.create_classes_folder(mode='val')
dataset.clip_train_audios()
dataset.clip_val_audios()
