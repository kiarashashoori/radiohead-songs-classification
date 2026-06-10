class Parameters():

    music_path = 'music/'
    dataset_path = 'dataset/'
    music_formats = ('mp3','m4a')
    model_path = 'ast-radiohead/checkpoint-1196'


    classes_num = 17
    label2id = {'Knives Out': 0, 'All I Need': 1, 'I Can t': 2, 'Everything In Its Right Place': 3,
                'Nude': 4, 'Videotape': 5, 'Bullet Proof ... I Wish I Was': 6, 'Jigsaw Falling into Place': 7,
                'No Surprises': 8, 'Creep': 9, 'Codex': 10, 'Black Star': 11, 'Exit Music (For a Film)': 12,
                'Street Spirit (Fade Out)': 13, 'Man of War': 14, 'Where I End and You Begin': 15, 'Killer Cars': 16}
    

    id2label = {0: 'Knives Out', 1: 'All I Need', 2: 'I Can t', 3: 'Everything In Its Right Place', 4: 'Nude',
                5: 'Videotape', 6: 'Bullet Proof ... I Wish I Was', 7: 'Jigsaw Falling into Place',
                8: 'No Surprises', 9: 'Creep', 10: 'Codex', 11: 'Black Star', 12: 'Exit Music (For a Film)',
                13: 'Street Spirit (Fade Out)', 14: 'Man of War', 15: 'Where I End and You Begin', 16: 'Killer Cars'}