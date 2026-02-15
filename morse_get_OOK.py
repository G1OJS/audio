import numpy as np
import time
import threading

SPEED = {'MAX':45, 'MIN':12, 'ALPHA':0.3}
TICKER = {'MORSE':10, 'TEXT':30}
TIMESPEC = {'DOT_DASH':np.array([0.8, 2, 4]), 'CHARSEP_WORDSEP':np.array([1.5, 4, 10])}
MORSE = {
".-": "A",    "-...": "B",  "-.-.": "C",  "-..": "D",
".": "E",     "..-.": "F",  "--.": "G",   "....": "H",
"..": "I",    ".---": "J",  "-.-": "K",   ".-..": "L",
"--": "M",    "-.": "N",    "---": "O",   ".--.": "P",
"--.-": "Q",  ".-.": "R",   "...": "S",   "-": "T",
"..-": "U",   "...-": "V",  ".--": "W",   "-..-": "X",
"-.--": "Y",  "--..": "Z",

"-----": "0", ".----": "1", "..---": "2", "...--": "3",
"....-": "4", ".....": "5", "-....": "6", "--...": "7",
"---..": "8", "----.": "9",

"-..-.": "/", "..--": "Ü",

".-.-.": "_AR_", "..--..": "?", "-...-": "_BK_", "...-.-": "_SK_", "..-.-.": "_UR_"

}

class TimingDecoder:

    def __init__(self, axs, audio, fbin):
        
        self.audio = audio
        self.keymoves = {'fall_t': None, 'lift_t': time.time()}
        self.morse_elements = ''
        self.last_lift_t = 0
        self.ticker = {'ticker': axs[1].text(0, fbin, '*'), 'wpm':16, 'morse':' ' * TICKER['MORSE'], 'text':' ' * TICKER['TEXT']}
        self.update_speed(1.2/16)
        threading.Thread(target = self.run, args = (fbin,) ).start()

    def update_speed(self, mark_dur):
        if(1.2/SPEED['MAX'] < mark_dur < 3*1.2/SPEED['MIN']):
            wpm_new = 1.2/mark_dur if mark_dur < 1.2/SPEED['MIN'] else 3 * 1.2/mark_dur
            wpm_new = np.clip(wpm_new, SPEED['MIN'], SPEED['MAX'])
            self.ticker['wpm'] = SPEED['ALPHA'] * wpm_new + (1-SPEED['ALPHA']) * self.ticker['wpm']
            tu = 1.2/self.ticker['wpm']
            ts = TIMESPEC
            self.timespec = {'dot_dash':ts['DOT_DASH']*tu, 'charsep_wordsep':ts['CHARSEP_WORDSEP']*tu}

    def detect_transition(self, level):
        t = time.time()
        if(self.keymoves['fall_t'] and level <0.4): # key -> up
            mark_dur = t - self.keymoves['fall_t']
            self.keymoves = {'fall_t': False, 'lift_t': t}
            self.last_lift_t = t
            return mark_dur, False
        idle = (t - self.last_lift_t > self.timespec['charsep_wordsep'][2]) and self.morse_elements
        if(idle):
            return False, self.timespec['charsep_wordsep'][1]+0.1
        if(self.keymoves['lift_t'] and level >0.6): # key -> down
            space_dur = t - self.keymoves['lift_t']
            self.keymoves = {'fall_t': t, 'lift_t': False}
            return False, space_dur
        return False, False
    
    def classify_duration(self, mark_dur, space_dur):
        ts = self.timespec
        if(mark_dur):
            self.update_speed(mark_dur)
            if ts['dot_dash'][0] < mark_dur < ts['dot_dash'][1]:
                return '.'
            if ts['dot_dash'][1] < mark_dur < ts['dot_dash'][2]:
                return '-'
        elif(space_dur):
            if ts['charsep_wordsep'][0] < space_dur < ts['charsep_wordsep'][1]:
                return ' '
            if ts['charsep_wordsep'][1] < space_dur < ts['charsep_wordsep'][2]:
                return '/'
        return ''

    def process_element(self, el):
        self.ticker['morse'] = (self.ticker['morse'] + el)[-TICKER['MORSE']:]
        if(el in ['/', ' ']):
            char = MORSE.get(self.morse_elements, '')
            self.morse_elements = ''
            self.ticker['text'] = ( self.ticker['text'] + char + (' ' if el == '/' else '') )[-TICKER['TEXT']:]
        else:
            self.morse_elements = self.morse_elements + el

    def run(self, fbin):
        self.morse_elements = ''
        sigstate = {'up':time.time(), 'down':False, 'new':False}
        while True:
            time.sleep(self.audio.params['dt'])
            level = float(self.audio.snr[fbin])
            mark, space = self.detect_transition(level)
            if mark or space:
                el = self.classify_duration(mark, space)
                self.process_element(el)            


def run():
    import matplotlib.pyplot as plt
    from audio import Audio_in
    import time

    audio = Audio_in(df = 50, dt = 0.005, fRng = [400, 600], snr_clip = [18,80])

    refresh_dt = 0.025
    nf = audio.params['nf']
 
    fig, axs = plt.subplots(1,2, figsize = (14,5))
    spec_plot = axs[0].imshow(audio.display_grid, origin = 'lower', aspect='auto', interpolation = 'none')
    axs[1].set_ylim(axs[0].get_ylim())
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    axs[1].set_axis_off()

    decoders = []
    for i in range(len(audio.snr)):
        d = TimingDecoder(axs, audio, i)
        decoders.append(d)

    while True:
        time.sleep(refresh_dt/2)
        spec_plot.set_data(audio.display_grid)
        spec_plot.autoscale()
        for d in decoders:  
            d.ticker['ticker'].set_text(f"{d.ticker['wpm']:4.1f} {d.ticker['morse']} {d.ticker['text']}") 
        plt.pause(refresh_dt)

run()
