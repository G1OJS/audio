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
        self.keypos = 'up'
        self.key_last_moved = time.time()
        self.element_buffer = ''
        self.wpm = 16
        self.update_speed(1.2/16)
        self.morse = ''
        self.text = ''

    def update_speed(self, mark_dur):
        if(1.2/SPEED['MAX'] < mark_dur < 3*1.2/SPEED['MIN']):
            wpm_new = 1.2/mark_dur if mark_dur < 1.2/SPEED['MIN'] else 3 * 1.2/mark_dur
            wpm_new = np.clip(wpm_new, SPEED['MIN'], SPEED['MAX'])
            self.wpm = SPEED['ALPHA'] * wpm_new + (1-SPEED['ALPHA']) * self.wpm
            tu = 1.2/self.wpm
            ts = TIMESPEC
            self.timespec = {'dot_short':ts['DOT_SHORT']*tu, 'dot_long':ts['DOT_LONG']*tu,
                             'charsep_short':ts['CHARSEP_SHORT']*tu, 'charsep_wordsep':ts['CHARSEP_WORDSEP']*tu, 'timeout':ts['TIMEOUT']*tu, }

    def clockstep(self, keypos_new):
        t = time.time()
        ts = self.timespec
        timeout = t - self.key_last_moved > ts['timeout']
        if(keypos_new != self.keypos or timeout):
            dur = t - self.key_last_moved
            if dur > ts['dot_short']:
                self.keypos = keypos_new
                self.key_last_moved = t
                if keypos_new == 'down' or (timeout and self.element_buffer):
                    el = None if dur < ts['charsep_short'] else (' ' if dur < ts['charsep_wordsep'] else '/')
                else:
                    el = '.' if dur < ts['dot_long'] else '-'
                    self.update_speed(dur)
                if(el is not None):
                    if(el in [' ', '/']):
                        char = MORSE.get(self.element_buffer, '')
                        self.element_buffer = ''
                        space = '' if el == ' ' else ' '
                        self.text = (self.text + char + space)[-TICKER_FIELD_LENGTHS['TEXT']:]
                    else:
                        self.element_buffer = self.element_buffer + el 
                    self.morse = (self.morse + el)[-TICKER_FIELD_LENGTHS['TEXT']:]
                if(timeout):
                    self.element_buffer = ''

class UI_channel:
    def __init__(self, axs, fbin, timevals):
        self.timevals = timevals
        self.axs = axs
        self.decoder = TimingDecoder()
        self.active = False
        self.keyline_data = np.zeros_like(self.timevals)
        self.keyline = self.axs[0].plot(self.timevals, self.keyline_data, color = 'white', drawstyle='steps-post')[0]
        self.fbin = fbin
        self.quality_fast = 0
        self.sig_max = None
        self.noise = None

    def clockstep(self, sig):
        if self.quality_fast > 10:
            if(self.sig_max is None): self.sig_max = sig
            if(self.noise is None): self.noise = sig/10
            self.noise = 0.99 * self.noise + 0.01 * np.minimum(self.noise*1.05, sig)
            self.sig_max = np.maximum(self.sig_max * 0.99, sig)
            sig = (sig - self.noise) / (self.sig_max - self.noise)
            keypos = 'up' if sig <0.15 else 'down'
            self.decoder.clockstep(keypos)
        self.keyline_data[:-1] = self.keyline_data[1:]
        self.keyline_data[-1] = self.fbin
        self.keyline_data[-1] += 0.2 if self.decoder.keypos == 'up' else 0.8            

    def display(self, tickers):
        ticker_obj = tickers[self.fbin]['ticker']
        if(self.active):
            self.keyline.set_ydata(self.keyline_data)
            self.keyline.set_linestyle('solid')
            d = self.decoder
            new_text = f" {d.wpm:3.0f} wpm {d.morse}  {d.text}"
            if(ticker_obj.get_text() != new_text):
                ticker_obj.set_text(new_text)
                ticker_obj.set_color('black')
                tickers[self.fbin]['last_updated'] = time.time()
        else:
            ticker_obj.set_color('blue')
            self.keyline.set_linestyle('none')
            
class UI_waterfall:
    def __init__(self, axs, nf,  timevals):
        self.data = np.zeros((nf, len(timevals)))
        self.spec_plot = axs[0].imshow(self.data, origin = 'lower', aspect='auto', alpha = 1,
                                  vmin = 5,  vmax=25, interpolation = 'bilinear', extent=[0, DISPLAY_DUR, 0, nf])
    def clockstep(self, newvals):
        self.data[:, :-1] = self.data[:, 1:]
        self.data[:, -1]  = newvals

    def display(self):
        self.spec_plot.set_array(self.data)
        vmax = np.max(self.data)
        self.spec_plot.set_clim(vmax = vmax, vmin = vmax - 20)
        return self.spec_plot

class Hot_loop:
    def __init__(self, spectrum, hop_ms, channels, waterfall):
        self.data_counter = 0
        self.abort = False
        self.last_hop = time.time()
        threading.Thread(target = self.loop, args = (spectrum, hop_ms, channels, waterfall,) ).start()
        
    def loop(self, spectrum, hop_ms, channels, waterfall):
        while True:
            delay = hop_ms/1000 - (time.time() - self.last_hop)
            if(delay > 0):
                time.sleep(delay)
            else:
                self.abort = True
            self.last_hop = time.time()
            spectrum.calc_spectrum()
            pwr_dB = 10*np.log10(spectrum.pwr)
            waterfall.clockstep(pwr_dB)
            for ch in channels:
                if(ch.active):
                    ch.clockstep(spectrum.pwr[ch.fbin])
            self.data_counter += 1

class Channel_manager:
    def __init__(self, channels, waterfall, n_decoders):
        for ch in channels[:n_decoders]:
            ch.active = True
        threading.Thread(target = self.loop, args = (channels, waterfall, ) ).start()
        
    def loop(self, channels, waterfall): 
        while True:
            time.sleep(1)
            quality = np.std(waterfall.data, axis = 1)
            weakest_decoder = [-1,1e6]
            for fbin, ch in enumerate(channels):
                if ch.active:
                    ch.quality_fast = np.std(waterfall.data[fbin][-75:])
                    if quality[fbin] < weakest_decoder[1]:
                        weakest_decoder[1] = quality[fbin]
                        weakest_decoder[0] = fbin
                    quality[fbin] = 0
            best_fbin = np.argmax(quality)
            if not channels[best_fbin].active:
                channels[best_fbin].active = True
                channels[weakest_decoder[0]].active = False

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
                    
def run(input_device_keywords, freq_range, df, hop_ms, display_decimate, n_decoders, show_processing):
    
    display_nt = int(DISPLAY_DUR * 1000 / hop_ms)
    timevals = np.linspace(0, DISPLAY_DUR, display_nt)
    spectrum = Spectrum(input_device_keywords, df,  freq_range)
    fig, axs = define_figure(spectrum.nf)
    waterfall = UI_waterfall(axs, spectrum.nf, timevals)
    channels = [UI_channel(axs, fb, timevals) for fb in range(spectrum.nf)]
    spec_plot = waterfall.display()
    keylines = [ch.keyline for ch in channels]
    tickers  = [{'ticker': axs[1].text(0, ch.fbin, ''), 'last_updated': time.time()} for ch in channels]
    animation_artists_list = spec_plot, *keylines, *[ticker['ticker'] for ticker in tickers]

    ch_mgr = Channel_manager(channels, waterfall, n_decoders)
    hot_loop = Hot_loop(spectrum, hop_ms, channels, waterfall)

    def animation_callback(frame):
        while hot_loop.data_counter < display_decimate :
            time.sleep(0)
        hot_loop.data_counter = 0
        spec_plot = waterfall.display()
        for ch in channels:
            ch.display(tickers)                
        if hot_loop.abort:
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
    run(['Mic', 'CODEC'], [200,800],  df = 40, hop_ms = 8, display_decimate = 2, n_decoders = 3, show_processing = True)

        


