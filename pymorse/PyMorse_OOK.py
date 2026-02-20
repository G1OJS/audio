import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import threading
import pyaudio
import argparse

SHOW_KEYLINES = True
SPEED = {'MAX':45, 'MIN':12, 'ALPHA':0.05}
TICKER_FIELD_LENGTHS = {'MORSE':40, 'TEXT':30}
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
        self.initialise_spectrum_vars()

    def initialise_spectrum_vars(self):
        self.sig_max = self.calc_spectrum()
        self.noise = self.sig_max
        self.snr_lin = np.zeros(self.nf)
        self.sig_norm = np.zeros(self.nf)

    def squelch(self, x, a, b):
        f = np.where(x>a, x, a + b*(x-a))
        return np.clip(f, 0, None)

    def calc_spectrum(self):
        z = np.fft.rfft(self.audio.audiobuff * self.window)[self.fBins]
        return z.real*z.real + z.imag*z.imag
        
    def update_spectrum_vars(self):
        pwr = self.calc_spectrum()
        self.noise = 0.9 * self.noise + 0.1 * np.minimum(self.noise*1.05, pwr)
        self.snr_lin = pwr / (self.noise + 0.01)
        self.sig = self.squelch(self.snr_lin, 100, 10)
        self.sig_max = np.maximum(self.sig_max * 0.85, self.sig)
        self.sig_norm = self.sig / self.sig_max


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
        self.set_fbin(fbin)

    def set_fbin(self, fbin):
        self.fbin = fbin
        self.decoder.set_fbin(fbin)
        kld = np.zeros_like(self.timevals)
        self.keyline = {'data':kld, 'line':self.axs[0].plot(self.timevals, kld, color = 'white', drawstyle='steps-post')[0]}

    def remove(self, fbin):
        self.fbin = None
        if(self.keyline is not None):
            self.keyline['line'].remove()
        self.decoder = None

def run(input_device_keywords, freq_range, df, hop_ms, display_decimate, n_decoders, show_processing):

        spectrum = Spectrum(input_device_keywords, df,  freq_range)
        nf = spectrum.nf
        display_nt = int(DISPLAY_DUR * 1000 / hop_ms)
        waterfall = np.zeros((nf, display_nt))
        timevals = np.linspace(0, DISPLAY_DUR, display_nt)
        s_meter = np.zeros(nf)
        show_speed_info = False

        fig, axs = plt.subplots(1,2, width_ratios=[1, 1], figsize = (15,3))
        fig.suptitle("PyMorse by G1OJS", horizontalalignment = 'left', x = 0.1)
        axs[1].set_ylim(0, nf)
        axs[0].set_xticks([])
        axs[0].set_yticks([])
        axs[1].set_axis_off()
        decoders = [UI_decoder(axs, fb, timevals) for fb in range(n_decoders)]
        
        spec_plot = axs[0].imshow(waterfall, origin = 'lower', aspect='auto', alpha = 1,
                                  vmin = 5,  vmax=25, interpolation = 'bilinear', extent=[0, DISPLAY_DUR, 0, nf])

        blank_text = ' ' * (TICKER_FIELD_LENGTHS['MORSE'] + TICKER_FIELD_LENGTHS['TEXT'])
        last_updated = [0] * nf
        tickers = []
        for fbin in range(nf):
            tickers.append(axs[1].text(-0.15, fbin, '*'))

        last_hop = time.time()
        data_counter = 0
        def processing_loop(hop_ms):
            nonlocal waterfall, last_hop, data_counter
            while True:
                delay = hop_ms/1000 - (time.time() - last_hop)
                time.sleep(delay if delay > 0 else 0)
                last_hop = time.time()
                spectrum.update_spectrum_vars()
                sig_norm = spectrum.sig_norm
                for d in decoders:
                    fbin = d.fbin
                    d.decoder.step(sig_norm[fbin])
                    d.keyline['data'] = np.roll(d.keyline['data'], -1)
                    d.keyline['data'][-1] = 0.2 + 0.6 * d.decoder.keypos + fbin
                inst_dB = 10*np.log10(spectrum.snr_lin)
                waterfall = np.roll(waterfall, -1, axis = 1)
                waterfall[:, -1]  = inst_dB
                data_counter += 1
    
        def display_loop(display_idx):
            nonlocal data_counter, display_nt, display_decimate, spec_plot, s_meter, waterfall, decoders, show_speed_info

            while data_counter < display_decimate :
                time.sleep(0)
            data_counter = 0
 
            spec_plot.set_data(waterfall)            
            s_meter = np.maximum(s_meter * 0.995, waterfall[:, -1])
            spec_plot.set_array(waterfall)
            if(SHOW_KEYLINES):
                for d in decoders:
                    d.keyline['line'].set_ydata(d.keyline['data'])

            if((display_idx % 10) == 0):
                current_bins_with_decoders = [d.fbin for d in decoders]
                fbins_to_decode = np.argsort(-s_meter)[:n_decoders]
                decoders_sorted = sorted(decoders, key=lambda d: s_meter[d.fbin])
                for fb in fbins_to_decode:
                    if fb not in current_bins_with_decoders:
                        weakest_decoder = decoders_sorted[0]
                        if(s_meter[fb] > 6 + s_meter[weakest_decoder.fbin]):
                            weakest_decoder.set_fbin(fb)
                            break
     
            if((display_idx % 10) == 0):
                current_bins_with_decoders = [d.fbin for d in decoders]
                for fbin in range(nf):
                    if fbin not in current_bins_with_decoders:
                        if (time.time() - last_updated[fbin] > 15):
                            new_text = f"{s_meter[fbin]:+03.0f}dB {blank_text}"
                            if(tickers[fbin].get_text != new_text):
                                tickers[fbin].set_text(new_text)
                for d in decoders:
                    td = d.decoder.info_dict
                    speed_info = ' '.join([f"{k}{v:5.3f}" for k,v in d.decoder.timeactual.items()]) if show_speed_info else ''
                    decoder_text = f" {td['wpm']:3.0f}wpm  {speed_info} {td['morse']}  {td['text'].strip()}"
                    new_text = f"{s_meter[fbin]:+03.0f}dB {decoder_text}"
                    fbin = d.fbin
                    if(tickers[fbin].get_text != new_text):
                        tickers[fbin].set_text(new_text)


            return spec_plot, *[d.keyline['line'] for d in decoders], *[ticker for ticker in tickers],
   
        threading.Thread(target = processing_loop, args = (hop_ms,) ).start()
        ani = FuncAnimation(plt.gcf(), display_loop, interval = 0, frames = 100000,  blit=True)
        plt.show()

def cli():
    parser = argparse.ArgumentParser(prog='PyMorseRx', description = 'Command Line Morse decoder')
    parser.add_argument('-i', '--inputcard_keywords', help = 'Comma-separated keywords to identify the input sound device', default = "Mic, CODEC") 
    parser.add_argument('-df', '--df', help = 'Frequency step, Hz') 
    parser.add_argument('-fr', '--freq_range', help = 'Frequency range Hz e.g. [600,800]') 
    parser.add_argument('-n', '--n_decoders', help = 'Number of decoders') 
  #  parser.add_argument('-p','--show_processing', action='store_true', help = 'Show processing') 
    
    args = parser.parse_args()
    input_device_keywords = args.inputcard_keywords.replace(' ','').split(',') if args.inputcard_keywords is not None else None
    df = args.df if args.df is not None else 40
    freq_range = np.array(args.freq_range) if args.freq_range is not None else [200, 800]
    n_decoders = args.n_decoders if args.n_decoders is not None else 3
    
    input_device_keywords = args.inputcard_keywords.replace(' ','').split(',') if args.inputcard_keywords is not None else None
   # show_processing = args.show_processing if args.show_processing is not None else False
    show_processing = False

    hop_ms = 5
    display_decimate = 5
    run(input_device_keywords, freq_range, df, hop_ms, display_decimate, n_decoders, show_processing)


cli()

        


