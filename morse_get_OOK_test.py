
import numpy as np
import wave
import pyaudio
import time
import threading
import matplotlib.pyplot as plt
    
global waterfall

class AudioFrontEnd:
    
    def __init__(self, device_keywords = ['Mic', 'CODEC']):
        global waterfall
        waterfall = {'dur':4, 'dt':0.03, 'df':0, 'fRng':[200,1000], 'binRng':[0,0], 'waterfall': None, 'idx':0, 'tlast':0}
        waterfall.update({'df': 1 / waterfall['dt']})
        self.sample_rate = 16000
        self.fft_len = int(self.sample_rate * waterfall['dt'])
        self.audio_buff = np.zeros(self.fft_len, dtype=np.float32)
        df = waterfall['df']
        fRng = waterfall['fRng']
        binRng = [int(fRng[0]/df), int(fRng[1]/df)]
        waterfall.update({'binRng': binRng})
        
        waterfall.update({'waterfall': np.zeros((1+binRng[1]-binRng[0], int(waterfall['dur'] / waterfall['dt'])))})             
        self.pya = pyaudio.PyAudio()
        self.input_device_idx = self.find_device(device_keywords)
        self.speclev = 1
        
        self.start_audio_in()

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
    
    def start_audio_in(self):
        stream = self.pya.open(
            format = pyaudio.paInt16, channels=1, rate = self.sample_rate,
            input = True, input_device_index = self.input_device_idx,
            frames_per_buffer = len(self.audio_buff), stream_callback=self._pya_callback,)
        stream.start_stream()

    def _pya_callback(self, in_data, frame_count, time_info, status_flags):
        samples = np.frombuffer(in_data, dtype=np.int16).astype(np.float32)
        ns = len(samples)
        self.audio_buff[:] = samples
        self.calc_spectrum()
        return (None, pyaudio.paContinue)

    def calc_spectrum(self):
        z = np.fft.rfft(self.audio_buff)[waterfall['binRng'][0]:waterfall['binRng'][1]+1]
        p = (z.real*z.real + z.imag*z.imag)
        self.speclev = max(self.speclev, np.max(p))
        p /= self.speclev
        i = waterfall['idx']
        waterfall['waterfall'][:, i] = np.clip(10*p, 0, 1)
        waterfall['idx'] = (i + 1) % waterfall['waterfall'].shape[1]

class TimingDecoder:

    def __init__(self, ax):
        self.ax = ax
        self.key_is_down = False
        self.n_fbins = waterfall['waterfall'].shape[0]
        self.fbin = 0
        self.wpm = 16
        tu = 1.2 / self.wpm
        self.speed_elements = {'dot':1*tu, 'dash':3*tu, 'intra':1*tu, 'inter':3*tu, 'word':7*tu}
        self.ticker = False
        self.set_fbin(10)
        self.symbols = ""
        threading.Thread(target = self.get_symbols).start()
        threading.Thread(target = self.decoder).start()

    def set_fbin(self, fbin):
        if(fbin == self.fbin):
            return
        if(self.ticker):
            self.ticker.set_text(" " * 20)
        self.fbin = fbin
        self.ticker = self.ax.text(0, (0.5 + self.fbin) / self.n_fbins,'')
        self.ticker_text = []

    def follow_speed(self, dur, speed_element):
        alpha = 0.3
        se = self.speed_elements
        se[speed_element] = alpha * dur + (1-alpha) * se[speed_element]
        tu_new = np.mean([se['dot'], se['dash']/3.0, se['intra'], se['inter']/3.0])
        self.wpm = 1.2/tu_new

    def acceptable(self, down_dur, unit):
        timing_tolerance = 0.2
        se = self.speed_elements
        t = se[unit]
        return ((1-timing_tolerance)*t < down_dur < (1+timing_tolerance)*t)

    def get_symbols(self):
        t_key_down = False
        t_key_up = time.time()
        s = ""
        speclev = 1
        
        while(True):
            time.sleep(0.002)
            
            level = waterfall['waterfall'][self.fbin, waterfall['idx']]
            if not self.key_is_down and level > 0.6:
                self.key_is_down = True
            elif self.key_is_down and level < 0.3:
                self.key_is_down = False

            if(not self.key_is_down and t_key_down):
                t_key_up = time.time()
                down_duration = t_key_up - t_key_down
                t_key_down = False
                if(self.acceptable(down_duration, 'dot')):
                    s = s + "."
                    self.follow_speed(down_duration, 'dot')
                elif(self.acceptable(down_duration, 'dash')):
                    s = s + "-"
                    self.follow_speed(down_duration, 'dash')
             #   else:
                    
            if(t_key_up):
                if(time.time() - t_key_up > 1.5*1.2/self.wpm and len(s)):
                    self.symbols = s
                    s = ""
            if(self.key_is_down and t_key_up):
                t_key_down = time.time()
                t_key_up = False

    def decoder(self):
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
        "---..": "8", "----.": "9"
        }

        last_symbols = time.time()
        while(True):
            time.sleep(0.1)
            if(len(self.symbols)):
                last_symbols = time.time()
                self.ticker_text.append(MORSE.get(self.symbols, "?"))
                self.ticker_text = self.ticker_text[-20:]
                self.symbols = ""
            self.ticker.set_text(f"{self.wpm:4.1f} {''.join(self.ticker_text)}")
            if(time.time() - last_symbols > 14*1.2/self.wpm and len(self.ticker_text)):
                if(self.ticker_text[-1] != " "):
                    self.ticker_text.append(" ")

        
def run():
    global waterfall
    fig, axs = plt.subplots(1,2, figsize = (8,8))
    audio = AudioFrontEnd()
    waterfall_plot = axs[0].imshow(waterfall['waterfall'], origin = 'lower', aspect='auto')
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    axs[1].set_axis_off()

    decoders = []
    for i in range(waterfall['waterfall'].shape[0]):
        d = TimingDecoder(axs[1])
        d.set_fbin(i)
        decoders.append(d)

    while True:
        time.sleep(0.01)
        idx = waterfall['idx']
        wf = waterfall['waterfall']
        display = np.hstack((wf[:, idx:], wf[:, :idx]))
        waterfall_plot.set_data(display)
        waterfall_plot.autoscale()
        plt.pause(0.1)

run()
