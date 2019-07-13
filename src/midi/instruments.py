import music21

music21_instruments_dict = dict([(cls().instrumentName, cls) for name, cls in music21.instrument.__dict__.items() if
                                 (isinstance(cls, type) and hasattr(cls(), 'instrumentName'))])

similar_music21_instruments = [
    ['Keyboard', 'Piano', 'Harpsichord', 'Clavichord', 'Celesta', 'Vibraphone', 'Marimba', 'Xylophone', 'Glockenspiel'],
    ['Organ', 'Pipe Organ', 'Electric Organ', 'Reed Organ', 'Accordion', 'Harmonica'],
    ['StringInstrument', 'Violin', 'Viola', 'Violoncello', 'Contrabass'],
    ['Harp', 'Guitar', 'Acoustic Guitar', 'Electric Guitar'],
    ['Acoustic Bass', 'Electric Bass', 'Fretless Bass', 'Contrabass', 'Bass Clarinet'],
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
