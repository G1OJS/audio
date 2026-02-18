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
TICKER_FIELD_LENGTHS = {'MORSE':40, 'TEXT':10}
TIMESPEC = {'DOT_SHORT':0.65, 'DOT_LONG':2, 'CHARSEP_SHORT':2, 'CHARSEP_WORDSEP':6, 'TIMEOUT':7.5}
AUDIO_REFRESH_DT = 0.02
AUDIO_RES= 80
DISPLAY_REFRESH_DT = -1
DISPLAY_DUR = 3
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
        self.timeactual = {'.':1, '`':1, '-':3, '_':3} # dot, intra-char silence, dash, inter-char silence

    def set_fbin(self, fbin):
        self.keypos = 0
        self.fbin = fbin
        self.keymoves = {'press_t': 0, 'lift_t': time.time()}
        self.morse_elements = ''
        self.last_lift_t = 0
        self.info_dict = {'wpm':16, 'morse':' ' * TICKER_FIELD_LENGTHS['MORSE'], 'text':' ' * TICKER_FIELD_LENGTHS['TEXT'], 'rendered_text':''}
        self.update_speed(1.2/16)

    def update_speed(self, mark_dur):
        if(1.2/SPEED['MAX'] < mark_dur < 3*1.2/SPEED['MIN']):
            wpm_new = 1.2/mark_dur if mark_dur < 1.2/SPEED['MIN'] else 3 * 1.2/mark_dur
            wpm_new = np.clip(wpm_new, SPEED['MIN'], SPEED['MAX'])
            self.info_dict['wpm'] = SPEED['ALPHA'] * wpm_new + (1-SPEED['ALPHA']) * self.info_dict['wpm']
            tu = 1.2/self.info_dict['wpm']
            ts = TIMESPEC
            self.timespec = {'dot_short':ts['DOT_SHORT']*tu, 'dot_long':ts['DOT_LONG']*tu,
                             'charsep_short':ts['CHARSEP_SHORT']*tu, 'charsep_wordsep':ts['CHARSEP_WORDSEP']*tu, 'timeout':ts['TIMEOUT']*tu, }

    def detect_transition(self, sig):
        t = time.time()
        
        if(0 < self.keymoves['press_t'] < t - self.timespec['dot_short'] and sig < 0.4): # key -> up
            mark_dur = t - self.keymoves['press_t']
            self.keymoves = {'press_t': 0, 'lift_t': t}
            self.last_lift_t = t
            self.keypos = 0
            return mark_dur, False, False
        
        if (t - self.last_lift_t > self.timespec['timeout']) and self.morse_elements:
            return False, False, True
        
        if(0 < self.keymoves['lift_t'] and sig > 0.6): # key -> down
            space_dur = t - self.keymoves['lift_t']
            self.keymoves = {'press_t': t, 'lift_t': 0}
            self.keypos = 1
            return False, space_dur, False
        
        return False, False, False
    
    def classify_duration(self, mark_dur, space_dur, is_idle):
        ts = self.timespec
        if(is_idle):
            return '/'
        elif(mark_dur > ts['dot_short']):
            return '.' if mark_dur < ts['dot_long'] else '-'
        elif(space_dur > ts['charsep_short']):
            return '_' if space_dur < ts['charsep_wordsep'] else '/'
        return '`' if space_dur else ''

    def update_durations(self, mark_dur, space_dur, el):
        if el in self.timeactual:
            self.timeactual[el] = 0.99*self.timeactual[el] + 0.01* (mark_dur if mark_dur else space_dur) / (1.2/self.info_dict['wpm'])

    def process_element(self, el):
        if(el =='`'):
            return
        if(el not in ['_', '/'] or self.info_dict['morse'][-1] not in [' ', '/']):
            self.info_dict['morse'] = (self.info_dict['morse'] + el.replace('_',' '))[-TICKER_FIELD_LENGTHS['MORSE']:]
        if el in ['.','-']:
            self.morse_elements = self.morse_elements + el
        elif(el in ['_', '/']):
            char = MORSE.get(self.morse_elements, '')
            self.morse_elements = ''
            wdspace = ' ' if el == '/' and not self.info_dict['text'].endswith(' ') else ''
            self.info_dict['text'] = ( self.info_dict['text'] + char + wdspace)[-TICKER_FIELD_LENGTHS['TEXT']:]
                
    def step(self, sig):
        mark_dur, space_dur, is_idle = self.detect_transition(sig)
        el = self.classify_duration(mark_dur, space_dur, is_idle)
        self.update_durations(mark_dur, space_dur, el)
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
            self.ticker.set_text(' ' * len(self.decoder.info_dict['rendered_text']))
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

    def squelch(self, x, a, b):
        f = np.where(x>a, x, a + b*(x-a))
        return np.clip(f, 0, None)
    
    def dsp(self):
        nf = self.audio.params['nf']
        dt = self.audio.params['dt']

        time.sleep(0.1)
        sig_max = self.audio.calc_spectrum()
        noise = sig_max
        while True:
            time.sleep(dt)
            pwr = self.audio.calc_spectrum()
            noise = 0.9 * noise + 0.1 * np.minimum(noise*1.05, pwr)
            self.snr_lin = pwr / noise
            sig = self.squelch(self.snr_lin, 100, 10)
            sig_max = np.maximum(sig_max * 0.85, sig)
            sig_norm = sig / sig_max
            for d in self.decoders:
                s = sig_norm[d.fbin]
                d.decoder.step(s)
                d.keyline['data'][-1] = 0.2 + 0.6 * d.decoder.keypos + d.fbin
                d.keyline['data'][:-1] = d.keyline['data'][1:]
            self.waterfall = np.roll(self.waterfall, -1, axis =1)
            self.waterfall[:, -1]  = 10*np.log10(self.snr_lin)

    def animate(self):
        fig, axs = plt.subplots(1,2, figsize = (14,2))
        axs[1].set_ylim(0, self.audio.params['nf'])
        axs[0].set_xticks([])
        axs[0].set_yticks([])
        axs[1].set_axis_off()
        self.decoders = [UI_decoder(axs, fb, self.timevals) for fb in range(NDECODERS)]
        
        spec_plot = axs[0].imshow(self.waterfall, origin = 'lower', aspect='auto',
                                alpha = 1, vmin = 5,  vmax=25, interpolation = 'bilinear',
                                extent=[0, DISPLAY_DUR, 0, self.audio.params['nf']])
        def refresh(i):
            nonlocal spec_plot, axs
            snr_db = 10 * np.log10(self.snr_lin)
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
            
            if(SHOW_KEYLINES):
                for d in self.decoders:
                    d.keyline['line'].set_ydata(d.keyline['data'])

            for d in self.decoders:
                if(d is not None):
                    td = d.decoder.info_dict
                    s = self.s_meter[d.fbin]
                    speed_info = ' '.join([f"{k}{v:5.3f}" for k,v in d.decoder.timeactual.items()])
                    text = f"{s:+03.0f}dB {td['wpm']:3.0f}wpm  {speed_info} {td['morse']}  {td['text'].strip()}"
                    if(td['rendered_text'] != text):
                        d.ticker.set_text(text) 
                        td['rendered_text'] = text

            return None
        
        ani = FuncAnimation(plt.gcf(), refresh, interval = self.display_refresh_dt * 1000, frames=range(100000), blit=False)
        plt.show()
        
app = App()




        


