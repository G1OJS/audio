import numpy as np
import time
import threading
import pyqtgraph as pg
import sys
import pyaudio
import argparse

SHOW_KEYLINES = True
SPEED = {'MAX':45, 'MIN':12, 'ALPHA':0.05}
TICKER_FIELD_LENGTHS = {'MORSE':40, 'TEXT':30}
TIMINGS_INITIAL = {'DOT_SHORT':0.65, 'DOT_LONG':2, 'CHARSEP_SHORT':2, 'CHARSEP_WORDSEP':6, 'TIMEOUT':7.5}
DISPLAY_DUR = .5
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
    
class Audio_in:
    
    def __init__(self, input_device_keywords = ['Mic', 'CODEC'], df = 10, fft_len = 256, fRng = [300,800]):
        fft_out_len = fft_len //2 + 1
        fmax = fft_out_len * df
        fRng = np.clip(fRng, None, fmax)
        sample_rate = int(fft_len * df)
        dt1 = fft_len/sample_rate
        hops_per_fft = 1
        dt = dt1/hops_per_fft
        self.frames_perbuff = fft_len // hops_per_fft
        self.audiobuff = np.zeros(fft_len, dtype=np.float32)

        self.fBins = range(int(fRng[0]/df), int(fRng[1]/df) - 1)
        fRng = [self.fBins[0] * df, self.fBins[-1] * df]
        nf = len(self.fBins)
        self.params = {'dt':dt, 'nf':nf, 'dt_wpm': int(12/dt)/10, 'hpf': hops_per_fft,
                       'df':df, 'sr':sample_rate, 'fmax':fmax, 'fRng':fRng}

        self.pya = pyaudio.PyAudio()
        self.input_device_idx = self.find_device(input_device_keywords)
        self.window = np.hanning(fft_len)
        self.start_audio_in(sample_rate)
        print(self.params)
        
    def find_device(self, device_str_contains):
        if(not device_str_contains): #(this check probably shouldn't be needed - check calling code)
            return
        print(f"[Audio] Looking for audio device matching {device_str_contains}")
        for dev_idx in range(self.pya.get_device_count()):
            name = self.pya.get_device_info_by_index(dev_idx)['name']
            match = True
            for pattern in device_str_contains:
                if (not pattern in name): match = False
            if(match):
                print(f"[Audio] Found device {name} index {dev_idx}")
                return dev_idx
        print(f"[Audio] No audio device found matching {device_str_contains}")
    
    def start_audio_in(self, sample_rate):
        stream = self.pya.open(
            format = pyaudio.paInt16, channels=1, rate = sample_rate,
            input = True, input_device_index = self.input_device_idx,
            frames_per_buffer = self.frames_perbuff, stream_callback=self._pya_callback,)
        stream.start_stream()

    def _pya_callback(self, in_data, frame_count, time_info, status_flags):
        samples = np.frombuffer(in_data, dtype=np.int16).astype(np.float32)
        ns = len(samples)
        self.audiobuff[:-ns] = self.audiobuff[ns:]
        self.audiobuff[-ns:] = samples
        return (None, pyaudio.paContinue)

    def calc_spectrum(self):
        buff = self.audiobuff
        z = np.fft.rfft(buff * self.window)[self.fBins]
        return z.real*z.real + z.imag*z.imag


class TimingDecoder:

    def __init__(self, fbin):
        self.set_fbin(fbin)
        self.timings_measured = {'.':1, '`':1, '-':3, '_':3} # dot, intra-char silence, dash, inter-char silence

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
            ts = TIMINGS_INITIAL
            self.TIMINGS_INITIAL = {'dot_short':ts['DOT_SHORT']*tu, 'dot_long':ts['DOT_LONG']*tu,
                             'charsep_short':ts['CHARSEP_SHORT']*tu, 'charsep_wordsep':ts['CHARSEP_WORDSEP']*tu, 'timeout':ts['TIMEOUT']*tu, }

    def update_durations(self, mark_dur, space_dur, el):
        if el in self.timings_measured:
            self.timings_measured[el] = 0.99*self.timings_measured[el] + 0.01* (mark_dur if mark_dur else space_dur) / (1.2/self.info_dict['wpm'])
            
    def detect_transition(self, sig):
        t = time.time()
        
        if(0 < self.keymoves['press_t'] < t - self.TIMINGS_INITIAL['dot_short'] and sig < 0.4): # key -> up
            mark_dur = t - self.keymoves['press_t']
            self.keymoves = {'press_t': 0, 'lift_t': t}
            self.last_lift_t = t
            self.keypos = 0
            return mark_dur, False, False
        
        if (t - self.last_lift_t > self.TIMINGS_INITIAL['timeout']) and self.morse_elements:
            return False, False, True
        
        if(0 < self.keymoves['lift_t'] and sig > 0.6): # key -> down
            space_dur = t - self.keymoves['lift_t']
            self.keymoves = {'press_t': t, 'lift_t': 0}
            self.keypos = 1
            return False, space_dur, False
        
        return False, False, False
    
    def classify_duration(self, mark_dur, space_dur, is_idle):
        ts = self.TIMINGS_INITIAL
        if(is_idle):
            return '/'
        elif(mark_dur > ts['dot_short']):
            return '.' if mark_dur < ts['dot_long'] else '-'
        elif(space_dur > ts['charsep_short']):
            return '_' if space_dur < ts['charsep_wordsep'] else '/'
        return '`' if space_dur else ''

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
    def __init__(self, fbin, timevals):
        self.timevals = timevals
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
       # self.ticker = self.axs[1].text(-0.15, fbin, '*')
        kld = np.zeros_like(self.timevals)
        self.keyline = {'data':kld, 'line':None}

class App:
    def __init__(self, input_device_keywords, freq_range, df, n_decoders, show_processing):
        self.n_decoders = n_decoders
        print(input_device_keywords, freq_range, df, n_decoders, show_processing)
        self.audio = Audio_in(input_device_keywords = input_device_keywords, df = 80,  fRng = freq_range)
        self.display_refresh_dt = self.audio.params['dt']
        self.display_nt = int(DISPLAY_DUR / self.audio.params['dt'])
        self.run()

    def squelch(self, x, a, b):
        f = np.where(x>a, x, a + b*(x-a))
        return np.clip(f, 0, None)
    
    def run(self):
        timevals = np.linspace(0, DISPLAY_DUR, self.display_nt)

        time.sleep(0.1)
        sig_max = self.audio.calc_spectrum()
        snr_lin = sig_max /1e12
        s_meter = sig_max /1e12
        noise = sig_max
        sig_norm = sig_max /1e12
        
        waterfall = np.zeros((self.audio.params['nf'], self.display_nt))
        wf_idx = 0
        decoders = [UI_decoder(fb, timevals) for fb in range(self.n_decoders)]

        widg = pg.GraphicsLayoutWidget()

        plot = widg.addPlot()
        plot.setLabel('left', 'Frequency bin')
        plot.setLabel('bottom', 'Time')
        spec_plot = pg.ImageItem()
        plot.addItem(spec_plot)
        nf, nt = self.audio.params['nf'], self.display_nt
        spec_plot.setColorMap(pg.colormap.get('viridis'))
        spec_plot.setImage(waterfall, autoLevels=False)
        plot.setLimits(xMin=0, xMax=nt, yMin=0, yMax=nf)
        plot.setRange(xRange=(0, nt), yRange=(0, nf), padding=0)
        plot.setAspectLocked(False)
        widg.show()
        sys.exit(self.run_loop.exec_())

    def run_loop(self):
        while True:
            time.sleep(self.audio.params['dt'])

            pwr = self.audio.calc_spectrum()
            noise = 0.9 * noise + 0.1 * np.minimum(noise*1.05, pwr)
            snr_lin = pwr / noise + 0.01
            sig = self.squelch(snr_lin, 100, 10)
            sig_max = np.maximum(sig_max * 0.85, sig)
            sig_norm = sig / sig_max
            wf_idx = (wf_idx +1) % waterfall.shape[1]

            waterfall[:, wf_idx]  = 10*np.log10(snr_lin)

            snr_db = 10 * np.log10(snr_lin)
            s_meter = np.maximum(s_meter * 0.999, snr_db)

            for d in decoders:
                d.decoder.step(sig_norm[d.fbin])
                if(SHOW_KEYLINES):
                    d.keyline['data'][wf_idx] = 0.2 + 0.6 * d.decoder.keypos + d.fbin

            spec_plot.setImage(waterfall, autoLevels=False)
            """
            if(SHOW_KEYLINES):
                for d in decoders:
                    d.keyline['line'].set_ydata(d.keyline['data'])
                    
            if(i % 10 == 0):
                show_speed_info = False
                for d in decoders:
                    if(d is not None):
                        td = d.decoder.info_dict
                        s = s_meter[d.fbin]
                        speed_info = ' '.join([f"{k}{v:5.3f}" for k,v in d.decoder.timings_measured.items()]) if show_speed_info else ''
                        text = f"{s:+03.0f}dB {td['wpm']:3.0f}wpm  {speed_info} {td['morse']}  {td['text'].strip()}"
                        if(td['rendered_text'] != text):
                            d.ticker.set_text(text) 
                            td['rendered_text'] = text
            
            if(i % 100 == 0):
                fbins_to_decode = np.argsort(-s_meter)[:self.n_decoders]
                decoders_sorted = sorted(decoders, key=lambda d: s_meter[d.fbin])
                current_bins_with_decoders = [d.fbin for d in decoders]
                for fb in fbins_to_decode:
                    if fb not in current_bins_with_decoders:
                        weakest_decoder = decoders_sorted[0]
                        if(s_meter[fb] > 2 + s_meter[weakest_decoder.fbin]):
                            weakest_decoder.set_fbin(fb)
                            break
            """

def cli():
    parser = argparse.ArgumentParser(prog='PyMorseRx', description = 'Command Line Morse decoder')
    parser.add_argument('-i', '--inputcard_keywords', help = 'Comma-separated keywords to identify the input sound device', default = "Mic, CODEC") 
    parser.add_argument('-df', '--df', help = 'Frequency step, Hz') 
    parser.add_argument('-fr', '--freq_range', help = 'Frequency range Hz e.g. [600,800]') 
    parser.add_argument('-n', '--n_decoders', help = 'Number of decoders') 
  #  parser.add_argument('-p','--show_processing', action='store_true', help = 'Show processing') 
    
    args = parser.parse_args()
    input_device_keywords = args.inputcard_keywords.replace(' ','').split(',') if args.inputcard_keywords is not None else None
    df = args.df if args.df is not None else 50
    freq_range = np.array(args.freq_range) if args.freq_range is not None else [200, 800]
    n_decoders = args.n_decoders if args.n_decoders is not None else 3
    
    input_device_keywords = args.inputcard_keywords.replace(' ','').split(',') if args.inputcard_keywords is not None else None
   # show_processing = args.show_processing if args.show_processing is not None else False
    show_processing = False
    
    app = App(input_device_keywords, freq_range, df, n_decoders, show_processing)


cli()

        


