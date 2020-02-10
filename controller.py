import pynput
from pynput import keyboard
import pygame.midi
import time
import sys
from string import ascii_lowercase

pygame.midi.init()

player = pygame.midi.Output(0)
player.set_instrument(0)
player.note_on(64, 127)
time.sleep(1)
player.note_off(64, 127)

key_to_note = {
    'q': 72,
    'w': 74,
    'e': 76,
    'r': 77,
    't': 79,
    'y': 81,
    'u': 83
}


class Controller:
    def __init__(self, player):
        self.player = player
        self.accepted_keys = ascii_lowercase + "01234567890-=m,./[]'\\"
        self.already_pressed = {k: False for k in self.accepted_keys}

    def on_press(self, key):
        # print('self', self, 'key', key)
        try:
            # print(f'alphanumeric key {key.char} pressed')
            if not self.already_pressed[key.char]:
                self.player.note_on(key_to_note[key.char], 127)
                self.already_pressed[key.char] = True
        except KeyError:
            pass
        except AttributeError:
            # print(f'special key {key} pressed')
            pass

    def on_release(self, key):
        # print(f'{key} released')
        try:
            self.player.note_off(key_to_note[key.char], 127)
            self.already_pressed[key.char] = False
        except KeyError:
            pass
        except AttributeError:
            pass
        if key == keyboard.Key.esc:
            # Stop listener
            return False


controller = Controller(player)

# Collect events until released
with keyboard.Listener(
        on_press=controller.on_press,
        on_release=controller.on_release) as listener:
    listener.join()

'''
# ...or, in a non-blocking fashion:
listener = keyboard.Listener(
    on_press=controller.on_press,
    on_release=controller.on_release)
listener.start()
'''

del player
pygame.midi.quit()
