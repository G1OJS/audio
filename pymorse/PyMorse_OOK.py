import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import threading
import pyaudio
import argparse

SHOW_KEYLINES = True
SPEED = {'MAX':45, 'MIN':18, 'ALPHA':0.1}
TICKER_FIELD_LENGTHS = {'MORSE':30, 'TEXT':30}
TIMESPEC = {'DOT_SHORT':0.65, 'DOT_LONG':2, 'CHARSEP_SHORT':2, 'CHARSEP_WORDSEP':6, 'TIMEOUT':7.5}


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
    
class Audio_in:
    
    def __init__(self, input_device_keywords, sample_rate, bufflen, frames_perbuff):
        self.audiobuff = np.zeros(bufflen, dtype=np.float32)
        self.pya = pyaudio.PyAudio()
        self.input_device_idx = self.find_device(input_device_keywords)
        self.start_audio_in(sample_rate, frames_perbuff)
        
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
    
    def start_audio_in(self, sample_rate, frames_perbuff):
        stream = self.pya.open(
            format = pyaudio.paInt16, channels=1, rate = sample_rate,
            input = True, input_device_index = self.input_device_idx,
            frames_per_buffer = frames_perbuff, stream_callback=self._pya_callback,)
        stream.start_stream()

    def _pya_callback(self, in_data, frame_count, time_info, status_flags):
        samples = np.frombuffer(in_data, dtype=np.int16).astype(np.float32)
        ns = len(samples)
        self.audiobuff[:-ns] = self.audiobuff[ns:]
        self.audiobuff[-ns:] = samples
        return (None, pyaudio.paContinue)

class Spectrum:
    def __init__(self, input_device_keywords, df,  freq_range, fft_len = 256):
        fft_out_len = fft_len //2 + 1
        fmax = fft_out_len * df
        freq_range = np.clip(freq_range, None, fmax)
        sample_rate = int(fft_len * df)
        self.fBins = range(int(freq_range[0]/df), int(freq_range[1]/df) - 1)
        freq_range = [self.fBins[0] * df, self.fBins[-1] * df]
        self.nf = len(self.fBins)
        self.params = {'nf':self.nf, 'df':df, 'sr':sample_rate, 'fmax':fmax, 'fRng':freq_range}
        print(self.params)
        self.window = np.hanning(fft_len)
        self.audio = Audio_in(input_device_keywords, sample_rate, fft_len, int(fft_len/8))
        time.sleep(0.5)
        self.noise = np.ones(self.nf) * 1e9
        self.snr_lin = np.zeros(self.nf)

    def calc_spectrum(self):
        z = np.fft.rfft(self.audio.audiobuff * self.window)[self.fBins]
        self.pwr = z.real*z.real + z.imag*z.imag

class TimingDecoder:

    def __init__(self):
        self.reset()
        self.timeactual = {'.':1, '`':1, '-':3, '_':3} # dot, intra-char silence, dash, inter-char silence

    def reset(self):
        self.sig_max = None
        self.noise = None
        self.keypos = 0
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
        if(self.sig_max is None): self.sig_max = sig
        if(self.noise is None): self.noise = sig/10
        self.noise = 0.99 * self.noise + 0.01 * np.minimum(self.noise*1.05, sig)
        self.sig_max = np.maximum(self.sig_max * 0.99, sig)
        sig = (sig - self.noise) / (self.sig_max - self.noise)

#        keypos = 'sig'
        keypos = 'key'
        if(0 < self.keymoves['press_t'] < t - self.timespec['dot_short'] and sig < 0.1): # key -> up
            mark_dur = t - self.keymoves['press_t']
            self.keymoves = {'press_t': 0, 'lift_t': t}
            self.last_lift_t = t
            self.keypos = sig if keypos == 'sig' else 0
            return mark_dur, False, False
        
        if (t - self.last_lift_t > self.timespec['timeout']) and self.morse_elements:
            return False, False, True
        
        if(0 < self.keymoves['lift_t'] and sig > .6): # key -> down
            space_dur = t - self.keymoves['lift_t']
            self.keymoves = {'press_t': t, 'lift_t': 0}
            self.keypos = sig if keypos == 'sig' else 1
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
        self.decoder = TimingDecoder()
        self.keyline_data = np.zeros_like(self.timevals)
        self.keyline = self.axs[0].plot(self.timevals, self.keyline_data, color = 'white', drawstyle='steps-post')[0]
        self.fbin = fbin

    def set_fbin(self, fbin):
        self.fbin = fbin

    def step(self, sig):
        self.decoder.step(sig[self.fbin])
        self.keyline_data[:-1] = self.keyline_data[1:]
        self.keyline_data[-1] = 0.2 + 0.6 * self.decoder.keypos + self.fbin

    def display(self, tickers):
        self.keyline.set_ydata(self.keyline_data)
        td = self.decoder.info_dict
        new_text = f" {td['wpm']:3.0f} wpm {td['morse']}  {td['text'].strip()}"
        ticker_obj = tickers[self.fbin]['ticker']
        if(ticker_obj.get_text() != new_text):
            ticker_obj.set_text(new_text)
            ticker_obj.set_color('green')
            tickers[self.fbin]['last_updated'] = time.time()
            

class UI_waterfall:
    def __init__(self, axs, nf,  timevals):
        self.waterfall = np.zeros((nf, len(timevals)))
        self.s_meter = np.zeros(nf)
        self.spec_plot = axs[0].imshow(self.waterfall, origin = 'lower', aspect='auto', alpha = 1,
                                  vmin = 5,  vmax=25, interpolation = 'none', extent=[0, DISPLAY_DUR, 0, nf])
    def step(self, newvals):
        self.waterfall[:, :-1] = self.waterfall[:, 1:]
        self.waterfall[:, -1]  = newvals

    def display(self):
        self.s_meter = np.maximum(self.s_meter * 0.995, self.waterfall[:, -1])
        self.spec_plot.set_array(self.waterfall)
        vmax = np.max(self.waterfall)
        self.spec_plot.set_clim(vmax = vmax, vmin = vmax - 20)
        return self.spec_plot

def define_figure(nf):
    fig, axs = plt.subplots(1,2, width_ratios=[1, 1], figsize = (12,3))
    fig.set_facecolor("lightgrey")
    ax_wf, ax_tx = axs
    box_wf = ax_wf.get_position()
    box_wf.x0 = 0.1
    box_wf.x1 = 0.45
    box_tx = ax_tx.get_position()
    box_tx.x0 = box_wf.x1
    ax_wf.set_position(box_wf)
    ax_tx.set_position(box_tx)
    
    fig.suptitle("PyMorse by G1OJS", horizontalalignment = 'left', x = 0.1)
    ax_tx.set_ylim(0, nf)
    ax_wf.set_xticks([])
    ax_wf.set_yticks([])
    ax_tx.set_axis_off()
    
    return fig, axs

class Fast_loop:
    def __init__(self, spectrum, hop_ms, decoders, waterfall):
        self.data_counter = 0
        self.abort = False
        self.last_hop = time.time()
        threading.Thread(target = self.loop, args = (spectrum, hop_ms, decoders, waterfall,) ).start()
        
    def loop(self, spectrum, hop_ms, decoders, waterfall):
        while True:
            delay = hop_ms/1000 - (time.time() - self.last_hop)
            if(delay > 0):
                time.sleep(delay)
            else:
                self.abort = True
            self.last_hop = time.time()
            spectrum.calc_spectrum()
            pwr_dB = 10*np.log10(spectrum.pwr)
            for d in decoders:
                d.step(spectrum.pwr)
            waterfall.step(pwr_dB)
            self.data_counter += 1

class Slow_manager:
    def __init__(self, decoders, waterfall, tickers):
        self.data_counter = 0
        threading.Thread(target = self.loop, args = (decoders, waterfall, tickers, ) ).start()
        
    def loop(self, decoders, waterfall, tickers): 
        while True:
            time.sleep(1)
            source_decoder = decoders[np.argmin([waterfall.s_meter[d.fbin] for d in decoders])]
            source_decoder_s = waterfall.s_meter[source_decoder.fbin]
            for d in decoders:
                waterfall.s_meter[d.fbin] = 0
            target = np.argmax(waterfall.s_meter)
            if(waterfall.s_meter[target] > 3 + source_decoder_s):
                source_decoder.set_fbin(target)
                source_decoder.decoder.reset()
            for ticker in tickers:
                ticker_obj = ticker['ticker']
                age = time.time() - ticker['last_updated']
                if age > 5:
                    ticker_obj.set_color('blue')
                if age > 15:
                    ticker_obj.set_color('lightgrey')
                    
def run(input_device_keywords, freq_range, df, hop_ms, display_decimate, n_decoders, show_processing):
    
        display_nt = int(DISPLAY_DUR * 1000 / hop_ms)
        timevals = np.linspace(0, DISPLAY_DUR, display_nt)
        spectrum = Spectrum(input_device_keywords, df,  freq_range)
        fig, axs = define_figure(spectrum.nf)
        waterfall = UI_waterfall(axs, spectrum.nf, timevals)
        decoders = [UI_decoder(axs, fb, timevals) for fb in range(n_decoders)]
        spec_plot = waterfall.display()
        keylines = [d.keyline for d in decoders]
        tickers  = [{'ticker': axs[1].text(0, fbin, ''), 'last_updated': time.time()} for fbin in range(spectrum.nf)]
        animation_artists_list = spec_plot, *keylines, *[ticker['ticker'] for ticker in tickers]
        
        slow_mgr = Slow_manager(decoders, waterfall, tickers)
        fast_loop = Fast_loop(spectrum, hop_ms, decoders, waterfall)

        def animation_callback(frame):
            while fast_loop.data_counter < display_decimate :
                time.sleep(0)
            fast_loop.data_counter = 0
            spec_plot = waterfall.display()
            for d in decoders:
                d.display(tickers)                
            if fast_loop.abort:
                print("Hop duration too short for cpu")
            return animation_artists_list

        ani = FuncAnimation(plt.gcf(), animation_callback, interval = 30, frames = 100000,  blit = True)
        plt.show()

def cli():
    parser = argparse.ArgumentParser(prog='PyMorseRx', description = 'Command Line Morse decoder')
    parser.add_argument('-i', '--inputcard_keywords', help = 'Comma-separated keywords to identify the input sound device', default = "Mic, CODEC") 
    parser.add_argument('-df', '--df', help = 'Frequency step, Hz') 
    parser.add_argument('-fr', '--freq_range', help = 'Frequency range Hz e.g. [600,800]') 
    parser.add_argument('-n', '--n_decoders', help = 'Number of decoders') 
    parser.add_argument('-p','--show_processing', action='store_true', help = 'Show processing') 
    
    args = parser.parse_args()
    input_device_keywords = args.inputcard_keywords.replace(' ','').split(',') if args.inputcard_keywords is not None else None
    df = args.df if args.df is not None else 40
    freq_range = np.array(args.freq_range) if args.freq_range is not None else [200, 800]
    n_decoders = args.n_decoders if args.n_decoders is not None else 3
    
    input_device_keywords = args.inputcard_keywords.replace(' ','').split(',') if args.inputcard_keywords is not None else None
    show_processing = args.show_processing if args.show_processing is not None else False
    show_processing = False

    hop_ms = 10
    display_decimate = 2
    run(input_device_keywords, freq_range, df, hop_ms, display_decimate, n_decoders, show_processing)


if __name__ == '__main__':
    run(['Mic', 'CODEC'], [200,800],  df = 40, hop_ms = 8, display_decimate = 2, n_decoders = 2, show_processing = True)

        


