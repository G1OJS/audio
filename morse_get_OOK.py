import numpy as np
import time
import threading

NDECODERS = 3
DECODER_MAX_AGE = 30
THRESHOLD_SNR = 30
SPEED = {'MAX':45, 'MIN':12, 'ALPHA':0.3}
TICKER = {'MORSE':30, 'TEXT':30}
TIMESPEC = {'DOT_DASH':np.array([0.8, 2, 4]), 'CHARSEP_WORDSEP':np.array([2, 4, 10])}
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
        self.ticker = {'ticker': axs[1].text(0, fbin, '*'), 'wpm':16, 'morse':' ' * TICKER['MORSE'], 'text':' ' * TICKER['TEXT'], 'rendered_text':''}
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
            return mark_dur, False, False 
        if (t - self.last_lift_t > self.timespec['charsep_wordsep'][2]) and self.morse_elements:
            return False, False, True
        if(self.keymoves['lift_t'] and level >0.6): # key -> down
            space_dur = t - self.keymoves['lift_t']
            self.keymoves = {'fall_t': t, 'lift_t': False}
            return False, space_dur, False
        return False, False, False
    
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

    def render_ticker(self, text = None, color = 'blue'):
        if(text is None):
            text = f"{self.ticker['wpm']:4.1f} {self.ticker['morse']} {self.ticker['text']}"
        if(self.ticker['rendered_text'] != text):
            self.ticker['ticker'].set_color(color)
            self.ticker['ticker'].set_text(text) 
            self.ticker['rendered_text'] = text
            return time.time()

    def run(self, fbin):
        self.morse_elements = ''
        sigstate = {'up':time.time(), 'down':False, 'new':False}
        while True:
            time.sleep(self.audio.params['dt'])
            level = float(self.audio.snr[fbin])
            mark, space, idle = self.detect_transition(level)
            if mark or space:
                el = self.classify_duration(mark, space)
                self.process_element(el)
            if(idle):
                self.process_element('/')
                            
def run():
    import matplotlib.pyplot as plt
    from audio import Audio_in
    import time

    audio = Audio_in(df = 50, dt = 0.005, fRng = [200, 1800], snr_clip = [THRESHOLD_SNR,80])

    refresh_dt = 0.05
    nf = audio.params['nf']
 
    fig, axs = plt.subplots(1,2, figsize = (14,5))
    spec_plot = axs[0].imshow(audio.display_grid, origin = 'lower', aspect='auto', interpolation = 'none')
    axs[1].set_ylim(axs[0].get_ylim())
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    axs[1].set_axis_off()

    decoder_slots = np.array([{'decoder':None, 'last_heard':0} for i in range(nf)])
    last_decoder_assignment = 0
    snr_avgs = audio.snr_raw
    while True:
        time.sleep(refresh_dt/2)
        snr_avgs = snr_avgs*0.9 + 0.1*audio.snr_raw
        
        if(time.time() - last_decoder_assignment > 1):
            last_decoder_assignment = time.time()
            best_snrs = np.argsort(-snr_avgs)
            for i in best_snrs[:NDECODERS]:
                if(snr_avgs[i]>THRESHOLD_SNR):
                    s = decoder_slots[i]
                    if s['decoder'] is None:
                        s['decoder'] = TimingDecoder(axs, audio, i)
                        s['last_heard'] = time.time()

        spec_plot.set_data(audio.display_grid)
        spec_plot.autoscale()

        for s in decoder_slots:
            if s['decoder'] is not None:
                color = 'blue' if s['last_heard'] > time.time() - DECODER_MAX_AGE/2 else 'red'
                last_change = s['decoder'].render_ticker(color = color)
                if(last_change):
                    s['last_heard'] = last_change
                if s['last_heard'] < time.time() - DECODER_MAX_AGE:
                    s['decoder'].render_ticker(text='')
                    s['decoder'] = None

        plt.pause(refresh_dt/2)

run()
