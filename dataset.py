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
    
    def create_music_folder(self):
        self.music_classes = os.listdir(self.music_path)
        for music in self.music_classes:
            if not music.endswith(p.music_formats):
                self.music_classes.remove(music)
                logging.info(f"unkown file '{music}'")
                continue
            os.mkdir(p.dataset_path+music[:len(music)-4])
            logging.info(f"directory for '{music[:len(music)-4]}' class created")
    
    def clip_audios(self):

        for file in self.music_classes:
            audio = AudioSegment.from_file(self.music_path+file)
            song_duration = len(audio)
            for i in range(0,int(song_duration//(self.between_clip_dur+self.each_clip_dur))):
                start_time = i*(self.between_clip_dur+self.each_clip_dur)
                end_time = start_time+self.each_clip_dur
                clipped = audio[start_time:end_time]
                output_path = self.dataset_path+file[:len(file)-4]+'/'
                clipped.export(output_path+str(i), format='mp3')
            logging.info(f"{i} labels created for '{file[:len(file)-4]}' class")

dataset = ExtractDataset()
# dataset.create_music_folder()
dataset.clip_audios()
