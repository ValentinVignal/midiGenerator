import music21

music21_instruments_dict = dict([(cls().instrumentName, cls) for name, cls in music21.instrument.__dict__.items() if
                                 (isinstance(cls, type) and hasattr(cls(), 'instrumentName'))])

similar_music21_instruments = [
    ['Keyboard', 'Piano', 'Harpsichord', 'Clavichord', 'Celesta', 'Vibraphone', 'Marimba', 'Xylophone', 'Glockenspiel',
     'Contrabass'],
    ['Organ', 'Pipe Organ', 'Electric Organ', 'Reed Organ', 'Accordion', 'Harmonica'],
    ['StringInstrument', 'Violin', 'Viola', 'Violoncello', 'Contrabass'],
    ['Harp', 'Guitar', 'Acoustic Guitar', 'Electric Guitar'],
    ['Acoustic Bass', 'Electric Bass', 'Fretless Bass', 'Contrabass', 'Bass clarinet', 'Bass'],
    ['Mandolin', 'Ukulele', 'Banjo', 'Lute', 'Sitar', 'Shamisen'],
    ['Koto', 'Woodwind', 'Flute', 'Piccolo', 'Recorder', 'Pan Flute', 'Shakuhachi', 'Whistle', 'Ocarina', 'Oboe',
     'English Horn', 'Clarinet', 'Bass clarinet', 'Bassoon'],
    ['Saxophone', 'Soprano Saxophone', 'Alto Saxophone', 'Tenor Saxophone', 'Baritone Saxophone', 'Bagpipes',
     'Shehnai'],
    ['Brass', 'Horn', 'Trumpet', 'Trombone', 'Bass Trombone', 'Tuba'],
    ['Percussion', 'Vibraphone', 'Marimba', 'Xylophone', 'Glockenspiel', 'Church Bells', 'Tubular Bells', 'Gong',
     'Handbells', 'Dulcimer', 'Steel Drum', 'Timpani', 'Kalimba', 'Woodblock'],
    ['Temple Block', 'Castanets', 'Maracas', 'Vibraslap', 'Cymbals', 'Finger Cymbals', 'Crash Cymbals',
     'Suspended Cymbal', 'Sizzle Cymbal', 'Splash Cymbals', 'Ride Cymbals', 'Hi-Hat Cymbal', 'Triangle', 'Cowbell',
     'Agogo', 'Tam-Tam', 'Sleigh Bells', 'Snare Drum', 'Tenor Drum', 'Bongo Drums', 'Tom-Tom', 'Timbales', 'Conga Drum',
     'Bass Drum', 'Taiko', 'Tambourine', 'Whip', 'Ratchet', 'Siren', 'Sandpaper Blocks', 'Wind Machine'],
    ['Voice', 'Soprano', 'Mezzo-Soprano', 'Alto', 'Tenor', 'Baritone', 'Bass']
]

bach_instruments = ['Flute', 'Ocarina', 'Tuba', 'Contrabass']


def string2instrument(name):
    """

    :param name: Name of the instrument
    :return: The music21 corresponding instrument class to instenciate
    """
    return music21_instruments_dict[name]


def return_similar_instruments(name):
    """

    :param name: The name of the instrument
    :return: All the possible instruments to train on which are similar
    """
    sim_inst = []
    for l in similar_music21_instruments:
        if name in l:
            sim_inst += l
    return sim_inst


all_midi_instruments = [
    # Piano
    'Acoustic-Piano',
    'BrtAcou-Piano',
    'ElecGrand-Piano',
    'Honky-Tonk-Piano',
    'Elec.Piano-1',
    'Elec.Piano-2',
    'Harsichord',
    'Clavichord',
    # Chromatic Percussion
    'Celesta',
    'Glockenspiel',
    'Music-Box',
    'Vibraphone',
    'Marimba',
    'Xylophone',
    'Tubular-Bells',
    'Dulcimer',
    # Organ
    'Drawbar Organ',
    'Perc.Organ',
    'Rock-Organ',
    'Church-Organ',
    'Reed-Organ',
    'Accordian',
    'Harmonica',
    'Tango-Accordian',
    # Guitar
    'Acoustic-Guitar',
    'SteelAcous.Guitar',
    'El.Jazz-Guitar',
    'Electric-Guitar',
    'El.Muted-Guitar',
    'Overdriven-Guitar',
    'Distortion-Guitar',
    'Guitar-Harmonic',
    # Bass
    'Acoustic-Bass',
    'El.Bass-Finger',
    'El.Bass-Pick',
    'Fretless-Bass',
    'Slap Bass 1',
    'Slap Bass 2',
    'Synth Bass 1',
    'Synth Bass 2',
    # Strings
    'Violin',
    'Viola',
    'Cello',
    'Contra-Bass',
    'Tremelo-Strings',
    'Pizz.Strings',
    'Orch.Strings',
    'Timpani',
    # Ensemble
    'String-Ens.1',
    'String-Ens.2',
    'Synth.Strings-1',
    'Synth.Strings-2',
    'Choir-Aahs',
    'Voice-Oohs',
    'Synth-Voice',
    'Orchestra-Hit',
    # Brass
    'Trumpet',
    'Trombone',
    'Tuba',
    'Muted-Trumpet',
    'French-Horn',
    'Brass-Section',
    'Synth-Brass-1',
    'Synth-Brass-2',
    # Reed
    'Soprano-Sax',
    'Alto-Sax',
    'Tenor-Sax',
    'Baritone-Sax',
    'Oboe',
    'English-Horn',
    'Bassoon',
    'Clarinet',
    # Pipe
    'Piccolo',
    'Flute',
    'Recorder',
    'Pan-Flute',
    'Blown-Bottle',
    'Shakuhachi',
    'Whistle',
    'Ocarina',
    # Synth Lead
    'Lead1-Square',
    'Lead2-Sawtooth',
    'Lead3-Calliope',
    'Lead4-Chiff',
    'Lead5-Charang',
    'Lead6-Voice',
    'Lead7-Fifths',
    'Lead8-Bass-Ld',
    # Synth Pad
    '9-Pad-1',
    '0-Pad-2',
    '1-Pad-3',
    '2-Pad-4',
    '3-Pad-5',
    '4-Pad-6',
    '5-Pad-7',
    '6-Pad-8',
    # Synth F / X
    'FX1-Rain',
    'FX2-Soundtrack',
    'FX3-Crystal',
    'FX4-Atmosphere',
    'FX5-Brightness',
    'FX6-Goblins',
    'FX7-Echoes',
    'FX8-Sci-Fi',
    # Ethnic
    'Sitar',
    'Banjo',
    'Shamisen',
    'Koto',
    'Kalimba',
    'Bagpipe',
    'Fiddle',
    'Shanai',
    # Percussive
    'TinkerBell',
    'Agogo',
    'SteelDrums',
    'Woodblock',
    'TaikoDrum',
    'Melodic-Tom',
    'SynthDrum',
    'Reverse-Cymbal',
    # Sound F / X
    'Guitar-Fret-Noise',
    'Breath-Noise',
    'Seashore',
    'BirdTweet',
    'Telephone',
    'Helicopter',
    'Applause',
    'Gunshot',
]


