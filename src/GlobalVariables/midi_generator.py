from . import midi

work_on = 'measure'
split_model_id = ','


def work_on2nb(wo):
    if wo == 'note':
        return 1
    elif wo == 'beat':
        return midi.step_per_beat
    elif wo == 'measure':
        return 4 * midi.step_per_beat
    else:
        raise Exception('Unknow work_on type :{0}'.format(wo))


def work_on2letter(wo):
    if wo == 'note':
        return 'n'
    elif wo == 'beat':
        return 'b'
    elif wo == 'measure':
        return 'm'
    elif wo is None:
        return ''
    else:
        raise Exception('Unknow work_on type :{0}'.format(wo))


def letter2work_on(letter):
    if letter == 'n':
        return 'note'
    elif letter == 'b':
        return 'beat'
    elif letter == 'm':
        return 'measure'
    elif letter == '':
        return None
    else:
        raise Exception('Unknow work_on letter :{0}'.format(letter))
