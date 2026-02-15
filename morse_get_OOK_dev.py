SPEED = {'MAX':45, 'MIN':12, 'ALPHA':0.3}

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

import numpy as np
        
class TimingDecoder:

    def __init__(self, axs, audio, fbin):
        import threading
        self.audio = audio
        self.ticker = {'ticker': axs[1].text(0, fbin, '*'), 'wpm':16, 'morse':'', 'text':''}
        self.update_speed(1.2/16)
        threading.Thread(target = self.run, args = (fbin,) ).start()

    def update_speed(self, mark_dur):
        if(1.2/SPEED['MAX'] < mark_dur < 3*1.2/SPEED['MIN']):
            wpm_new = 1.2/mark_dur if mark_dur < 1.2/SPEED['MIN'] else 3 * 1.2/mark_dur
            wpm_new = np.clip(wpm_new, SPEED['MIN'], SPEED['MAX'])
            self.ticker['wpm'] = SPEED['ALPHA'] * wpm_new + (1-SPEED['ALPHA']) * self.ticker['wpm']
            tu = 1.2/self.ticker['wpm']
            self.timespec = {'dot':(0.5*tu, 2*tu), 'dash':(2*tu, 4*tu), 'charsep':(1.2*tu, 4.1*tu), 'wordsep':(4.1*tu, 100*tu)}

    def match_element(self, time_el):
        ts = self.timespec
        if(time_el['mark']):
            dur = time_el['mark']
            self.update_speed(dur)
            if ts['dot'][0] < dur < ts['dot'][1]:
                return '.'
            if ts['dash'][0] < dur < ts['dash'][1]:
                return '-'
        elif(time_el['space']):
            dur = time_el['space']
            if ts['charsep'][0] < dur < ts['charsep'][1]:
                return ' '
            if ts['wordsep'][0] < dur < ts['wordsep'][1]:
                return '/'
        return ''
        

    def render_ticker(self):
        self.ticker['ticker'].set_text(f"{self.ticker['wpm']:4.1f} {self.ticker['morse']} {self.ticker['text']}") 

    def run(self, fbin):
        import time
        morse_buffer = ''
        sigstate = {'up':time.time(), 'down':False, 'new':False}
        last_el = {'mark':0, 'space':0}
        
        while(True):
            time.sleep(self.audio.params['dt'])

            # get most recent mark or space duration from signal level transitions (with hysteresis)
            level = float(self.audio.snr[fbin])
            if(sigstate['up'] and level <0.4): # signal -> down (key -> up)
                last_el = {'mark':time.time() - sigstate['up'], 'space':0}
                sigstate = {'up':False, 'down':time.time(), 'new':True}
            if(sigstate['down'] and level >0.6): # signal -> up (key -> down)
                last_el = {'mark':0, 'space':time.time() - sigstate['down']}
                sigstate = {'up':time.time(), 'down':False, 'new':True}

            if(sigstate['new']):
                sigstate.update({'new':False})
                el = self.match_element(last_el)
                # always add morse element to ticker
                if(el):
                    self.ticker['morse'] = self.ticker['morse'][-10:] + el
                # if space duration matches interchar or interword, add character to ticker
                if(el in [' ','/']):
                    char = ch = MORSE.get(morse_buffer, '')
                    if(el == '/'):
                        char = char + " "
                    if(char):
                        self.ticker['text'] = self.ticker['text'][-30:] + char
                    morse_buffer = ''
                else:
                    morse_buffer = morse_buffer + el
                self.render_ticker()

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
        plt.pause(refresh_dt)

run()
