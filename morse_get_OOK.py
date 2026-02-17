import numpy as np
import time
import threading
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from audio import Audio_in

F_RANGE = [200, 1500]
NDECODERS = 3
SHOW_KEYLINES = True
SPEED = {'MAX':45, 'MIN':12, 'ALPHA':0.05}
TICKER_FIELD_LENGTHS = {'MORSE':30, 'TEXT':30}
TIMESPEC = {'DOT_SHORT':0.65, 'DOT_LONG':2, 'CHARSEP_SHORT':1.5, 'CHARSEP_LONG':4, 'WORDSEP':6.5}
AUDIO_REFRESH_DT = 0.02
AUDIO_RES= 80
DISPLAY_REFRESH_DT = .08
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
        self.set_fbin(fbin)

    def set_fbin(self, fbin):
        self.keypos = 0
        self.fbin = fbin
        self.keymoves = {'press_t': None, 'lift_t': time.time()}
        self.morse_elements = ''
        self.last_lift_t = 0
        self.ticker_dict = {'wpm':16, 'morse':' ' * TICKER_FIELD_LENGTHS['MORSE'], 'text':' ' * TICKER_FIELD_LENGTHS['TEXT'], 'rendered_text':''}
        self.update_speed(1.2/16)

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
                
    def step(self, sig):
        mark_dur, space_dur, is_idle = self.detect_transition(sig)
        if any([mark_dur, space_dur, is_idle]):
            el = self.classify_duration(mark_dur, space_dur, is_idle)
            if(mark_dur):
                self.update_speed(mark_dur)
            self.process_element(el)

class UI_decoder:
    def __init__(self, axs, fbin, timevals):
        self.timevals = timevals
        self.axs = axs
        self.decoder = TimingDecoder(fbin)
        self.ticker = None
        self.keyline = None
        self.s_meter = 0
        self.set_fbin(fbin)

    def set_fbin(self, fbin):
        self.fbin = fbin
        if(self.ticker is not None):
            self.ticker.set_text(' ' * len(self.decoder.ticker_dict['rendered_text']))
            self.ticker.remove()
        if(self.keyline is not None):
            self.keyline['line'].remove()
        self.decoder.set_fbin(fbin)
        self.ticker = self.axs[1].text(-0.15, fbin, '*')
        kld = np.zeros_like(self.timevals)
        self.keyline = {'data':kld, 'line':self.axs[0].plot(self.timevals, kld, color = 'white', drawstyle='steps-post')[0]}

class App:
    def __init__(self):
        self.audio = Audio_in(df = AUDIO_RES, dt = AUDIO_REFRESH_DT,  fRng = F_RANGE)
        self.display_refresh_dt = DISPLAY_REFRESH_DT if DISPLAY_REFRESH_DT > 0 else self.audio.params['dt']
        self.display_nt = int(DISPLAY_DUR / self.audio.params['dt'])
        self.waterfall = np.zeros((self.audio.params['nf'], self.display_nt))
        self.decoders = []
        self.s_meter = np.zeros(self.audio.params['nf'])
        self.timevals = np.linspace(0, DISPLAY_DUR, self.display_nt)
        threading.Thread(target = self.dsp).start()
        self.animate()

    def dsp(self):
        nf = self.audio.params['nf']
        dt = self.audio.params['dt']
        noise = np.ones(nf)
        noise_decaying = np.zeros(nf)
        self.sig_unnorm = np.ones(nf)
        self.sig = np.ones(nf)
        decaying_max = np.ones(nf)
        
        while True:
            time.sleep(dt)
            noise = np.minimum(noise, self.audio.pwr)
            noise_decaying = np.maximum(noise_decaying*0.9, noise)
            self.sig_unnorm = self.audio.pwr / noise_decaying
            decaying_max = np.maximum(decaying_max*0.95, self.sig_unnorm)
            self.sig = self.sig_unnorm / decaying_max
            for d in self.decoders:
                d.decoder.step(self.sig[d.decoder.fbin])
                d.keyline['data'][-1] = 0.2 + 0.6 * d.decoder.keypos + d.fbin
                d.keyline['data'][:-1] = d.keyline['data'][1:]
            self.waterfall = np.roll(self.waterfall, -1, axis =1)
            self.waterfall[:, -1]  = self.sig_unnorm

    def animate(self):
        fig, axs = plt.subplots(1,2, figsize = (14,2))
        axs[1].set_ylim(0, self.audio.params['nf'])
        axs[0].set_xticks([])
        axs[0].set_yticks([])
        axs[1].set_axis_off()
        self.decoders = [UI_decoder(axs, fb, self.timevals) for fb in range(NDECODERS)]
        
        spec_plot = axs[0].imshow(self.waterfall, origin = 'lower', aspect='auto', alpha = 1, 
                                      interpolation = 'none', extent=[0, DISPLAY_DUR, 0, self.audio.params['nf']])
        def refresh(i):
            nonlocal spec_plot, axs
            snr_db = 10 * np.log10(self.sig_unnorm)
            self.s_meter = np.maximum(self.s_meter * 0.999, snr_db)
            
            if(i % 100 == 0):
                fbins_to_decode = np.argsort(-self.s_meter)[:NDECODERS]
                decoders_sorted = sorted(self.decoders, key=lambda d: self.s_meter[d.fbin])
                current_bins_with_decoders = [d.fbin for d in self.decoders]
                for fb in fbins_to_decode:
                    if fb not in current_bins_with_decoders:
                        weakest_decoder = decoders_sorted[0]
                        weakest_decoder.set_fbin(fb)
                        break

            spec_plot.set_array(self.waterfall)
            spec_plot.autoscale()
            
            if(SHOW_KEYLINES):
                for d in self.decoders:

                    d.keyline['line'].set_ydata(d.keyline['data'])

            for d in self.decoders:
                if(d is not None):
                    td = d.decoder.ticker_dict
                    s = self.s_meter[d.fbin]
                    text = f"{s - 80:+03.0f}dB {td['wpm']:3.0f}wpm   {td['morse']}  {td['text'].strip()}"
                    if(td['rendered_text'] != text):
                        d.ticker.set_text(text) 
                        td['rendered_text'] = text

            return None
        
        ani = FuncAnimation(plt.gcf(), refresh, interval = self.display_refresh_dt * 1000, frames=range(100000), blit=False)
        plt.show()
        
app = App()




        


