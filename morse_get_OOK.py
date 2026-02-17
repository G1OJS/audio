import numpy as np
import time
import threading
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from audio import Audio_in

NDECODERS = 3
SHOW_KEYLINES = True
SPEED = {'MAX':45, 'MIN':12, 'ALPHA':0.1}
TICKER_FIELD_LENGTHS = {'MORSE':30, 'TEXT':30}
TIMESPEC = {'DOT_SHORT':0.65, 'DOT_LONG':2, 'CHARSEP_SHORT':1.5, 'CHARSEP_LONG':4, 'WORDSEP':6.5}
DISPLAY_DT = 0.01
DISPLAY_DUR = 2
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

".-.-.": "_AR_", "..--..": "?", "-...-": "_BK_",
"...-.-": "_SK_", "..-.-.": "_UR_", "-.--.": "_KN_"

}



class TimingDecoder:

    def __init__(self, fbin):
        self.keypos = 0
        self.fbin = fbin
        self.level_hist = np.zeros(10)
        self.keymoves = {'press_t': None, 'lift_t': time.time()}
        self.morse_elements = ''
        self.last_lift_t = 0
        self.ticker_dict = {'wpm':16, 'morse':' ' * TICKER_FIELD_LENGTHS['MORSE'], 'text':' ' * TICKER_FIELD_LENGTHS['TEXT'], 'rendered_text':''}
        self.update_speed(1.2/16)
        self.noise = 0
        self.decaying_min = 0
        self.decaying_max = 1
        
    def update_speed(self, mark_dur):
        if(1.2/SPEED['MAX'] < mark_dur < 3*1.2/SPEED['MIN']):
            wpm_new = 1.2/mark_dur if mark_dur < 1.2/SPEED['MIN'] else 3 * 1.2/mark_dur
            wpm_new = np.clip(wpm_new, SPEED['MIN'], SPEED['MAX'])
            self.ticker_dict['wpm'] = SPEED['ALPHA'] * wpm_new + (1-SPEED['ALPHA']) * self.ticker_dict['wpm']
            tu = 1.2/self.ticker_dict['wpm']
            ts = TIMESPEC
            self.timespec = {'dot_short':ts['DOT_SHORT']*tu, 'dot_long':ts['DOT_LONG']*tu,
                             'charsep_short':ts['CHARSEP_SHORT']*tu, 'charsep_long':ts['CHARSEP_LONG']*tu, 'wordsep':ts['WORDSEP']*tu, }

    def detect_transition(self, sig):
        t = time.time()
        
        if(self.keymoves['press_t'] and sig < 0.4): # key -> up
            mark_dur = t - self.keymoves['press_t']
            self.keymoves = {'press_t': False, 'lift_t': t}
            self.last_lift_t = t
            self.keypos = 0
            return mark_dur, False, False
        
        if (t - self.last_lift_t > self.timespec['wordsep']) and self.morse_elements:
            return False, False, True
        
        if(self.keymoves['lift_t'] and sig > 0.6): # key -> down
            space_dur = t - self.keymoves['lift_t']
            self.keymoves = {'press_t': t, 'lift_t': False}
            self.keypos = 1
            return False, space_dur, False
        
        return False, False, False
    
    def classify_duration(self, mark_dur, space_dur, idle):
        ts = self.timespec
        wordsep_char = ''
        if(self.morse_elements):
            if self.morse_elements[-1] != '/':
                wordsep_char = '/'
        if(idle):
            return wordsep_char
        elif(mark_dur > ts['dot_short']):
            return '.' if mark_dur < ts['dot_long'] else '-'
        elif(space_dur > ts['charsep_short']):
            return ' ' if space_dur < ts['charsep_long'] else wordsep_char
        return ''

    def process_element(self, el):
        self.ticker_dict['morse'] = (self.ticker_dict['morse'] + el)[-TICKER_FIELD_LENGTHS['MORSE']:]
        if(el in ['/', ' ']):
            char = MORSE.get(self.morse_elements, '')
            self.morse_elements = ''
            self.ticker_dict['text'] = ( self.ticker_dict['text'] + char + (' ' if el == '/' else '') )[-TICKER_FIELD_LENGTHS['TEXT']:]
        else:
            self.morse_elements = self.morse_elements + el
                
    def step(self, pwr):
       # self.decaying_min = max(self.noise, 0.95*self.decaying_min)
        self.decaying_max = max(pwr, 0.95*self.decaying_max)
        sig = pwr / self.decaying_max
        self.noise = min(self.noise, pwr)
        mark_dur, space_dur, is_idle = self.detect_transition(sig)
        if any([mark_dur, space_dur, is_idle]):
            el = self.classify_duration(mark_dur, space_dur, is_idle)
            if(mark_dur):
                self.update_speed(mark_dur)
            self.process_element(el)

class UI_decoder:
    def __init__(self, axs, fbin, timevals):
        self.fbin = fbin
        self.decoder = TimingDecoder(fbin)
        self.ticker = axs[1].text(-0.15, fbin, '*')
        kld = np.zeros_like(timevals)
        self.keyline = {'data':kld, 'line':axs[0].plot(timevals, kld, color = 'black')[0]}

    def remove_elements(self):
        self.ticker.set_text(' ' * len(self.decoder.ticker_dict['rendered_text']))
        try:
            self.ticker.remove()
        except:
            pass
        try:
            self.keyline['line'].remove()
        except:
            pass
    

class App:
    def __init__(self):
        self.decoders = {}
        self.audio = Audio_in(df = 80, dt = 0.02,  fRng = [200, 1500])
        self.siglevels = np.zeros(self.audio.params['nf'])
        self.display_nt = int(DISPLAY_DUR / DISPLAY_DT)
        self.timevals = np.linspace(0, DISPLAY_DUR, self.display_nt)
        self.animate(DISPLAY_DT*1000, DISPLAY_DUR)

    def animate(self, frame_ms, display_duration):
        fig, axs = plt.subplots(1,2, figsize = (14,2))
        axs[1].set_ylim(0, self.audio.params['nf'])
        axs[0].set_xticks([])
        axs[0].set_yticks([])
        axs[1].set_axis_off()
        
        waterfall = np.zeros((self.audio.params['nf'], self.display_nt))
        spec_plot = axs[0].imshow(waterfall, origin = 'lower', aspect='auto', alpha = 1, 
                                      interpolation = 'none', extent=[0, display_duration, 0, self.audio.params['nf']])
        def refresh(i):
            nonlocal waterfall, spec_plot, axs
            if(i % 10 == 0):
                self.siglevels = np.maximum(self.siglevels * 0.9, self.audio.pwr)
                fbins_to_cover = np.argsort(-self.siglevels)[:NDECODERS]
                for fbin in fbins_to_cover:
                    if fbin not in self.decoders:
                        self.decoders[fbin] = UI_decoder(axs, fbin, self.timevals)
                decoders_to_remove = [d for d in self.decoders.values() if d.fbin not in fbins_to_cover]
                for d in decoders_to_remove:
                    d.remove_elements()
                    
            for fbin in self.decoders:
                self.decoders[fbin].decoder.step(self.audio.pwr[fbin])

            waterfall = np.roll(waterfall, -1, axis =1)
            waterfall[:, -1]  = self.audio.pwr/(1+np.max(self.audio.pwr))
            spec_plot.set_array(waterfall)
            spec_plot.autoscale()
            
            if(SHOW_KEYLINES):
                for fbin in self.decoders:
                    d = self.decoders[fbin]                
                    d.keyline['data'][-1] = 0.2 + 0.6 * d.decoder.keypos + fbin
                    d.keyline['data'][:-1] = d.keyline['data'][1:]
                    d.keyline['line'].set_ydata(d.keyline['data'])

            for fbin in self.decoders:
                d = self.decoders[fbin]
                if(d is not None):
                    td = d.decoder.ticker_dict
                    text = f"{td['wpm']:4.1f}   {td['morse']}  {td['text'].strip()}"
                    if(td['rendered_text'] != text):
                        d.ticker.set_text(text) 
                        td['rendered_text'] = text
 

            return None
        
        ani = FuncAnimation(plt.gcf(), refresh, interval = frame_ms, frames=range(100000), blit=False)
        plt.show()
        
app = App()




        


