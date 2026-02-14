MAX_WPM = 45
MIN_WPM = 12
DOT_TOL_FACTS = (0.6, 2)
DASH_TOL_FACTS = (0.5, 1.4)
CHARSEP_THRESHOLD = 0.7
WORDSEP_THRESHOLD = 0.7

import numpy as np
        
class TimingDecoder:

    def __init__(self, ax, spec):
        import threading
        self.ax = ax
        self.spec = spec
        self.key_is_down = False
        self.n_fbins = spec['buff'].shape[0]
        self.fbin = 0
        self.wpm = 16
        self.keydown_history = {'buffer':[1.2/self.wpm]*10, 'idx':0}
        self.check_speed(1.2/self.wpm)
        self.ticker = False
        self.ticker_text = []
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

    def check_element(self, dur):
        se = self.speed_elements
        if DOT_TOL_FACTS[0]*se['dot'] < dur < DOT_TOL_FACTS[1]*se['dot']:
            return '.'
        if DASH_TOL_FACTS[0]*se['dash'] < dur < DASH_TOL_FACTS[1]*se['dash']:
            return '-'
        return ''

    def _check_speed(self, dd):
        buffer = self.keydown_history['buffer']
        idx = (self.keydown_history['idx'] + 1) % len(buffer)
        self.keydown_history['idx'] = idx
        buffer[idx] = dd
        dot = np.percentile(buffer,20)
        self.wpm = 1.2/dot
        tu = 1.2/self.wpm
        self.speed_elements = {'dot':1*tu, 'dash':3*tu, 'charsep':3*tu, 'wordsep':7*tu}

    def check_speed(self, dd):
        alpha = 0.3
        if(dd < 1.2/MAX_WPM or dd > 3*1.2/MIN_WPM):
            return
        if(dd < 1.2/MIN_WPM):
            wpm = 1.2/dd
        else:
            wpm = 3*1.2/dd
        if(wpm > MAX_WPM): wpm = MAX_WPM
        if(wpm < MIN_WPM): wpm = MIN_WPM
        self.wpm = alpha * wpm + (1-alpha) * self.wpm
        tu = 1.2/self.wpm
        self.speed_elements = {'dot':1*tu, 'dash':3*tu, 'charsep':3*tu, 'wordsep':7*tu}
            
    def get_symbols(self):
        import time
        t_key_down = False
        keydown_level = 0
        self.t_key_up = time.time()
        s = ""
        
        while(True):
            time.sleep(self.spec['dt'])

            # hysteresis
            level = self.spec['buff'][self.fbin, -1]
            if not self.key_is_down and level > 0.6 * keydown_level:
                self.key_is_down = True
                keydown_level = level
            elif self.key_is_down and level < 0.4 * keydown_level:
                self.key_is_down = False

        #    if(self.key_is_down):
        
            #idx = spec['idx']
            #wf = spec['pgrid']

            # key_down to key_up transition
            if t_key_down and not self.key_is_down:
                self.t_key_up = time.time()
                down_duration = self.t_key_up - t_key_down
                t_key_down = False
                self.check_speed(down_duration)
                s = s + self.check_element(down_duration)

            # watch key_up duration for inter-character and inter-word gaps
            if self.t_key_up:
                key_up_dur = time.time() - self.t_key_up
                if key_up_dur > CHARSEP_THRESHOLD * self.speed_elements['charsep']:
                    if(len(s)):
                        self.symbols = s
                        s = ""
                if key_up_dur > WORDSEP_THRESHOLD * self.speed_elements['wordsep']:
                    if(len(self.ticker_text)):
                        if(self.ticker_text[-1] != " "):
                            self.ticker_text.append(" ")

            # key_up to key_down transition
            if(self.t_key_up and self.key_is_down):
                t_key_down = time.time()
                self.t_key_up = False

    def decoder(self):
        import time
        
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

        while(True):
            time.sleep(0.2)
            if(not self.ticker):
                continue
            
            # decode and print single character
            if len(self.symbols):
                ch = MORSE.get(self.symbols, "_")
                self.symbols = ""
                prevchars = ''.join(self.ticker_text).replace(' ','')[-2:]
                skip = False
                skip = skip or (prevchars == "EE" and ch == "E" or ch == "T")
                skip = skip or (prevchars == "TT" and ch == "T")
                if(not skip):
                    self.ticker_text.append(ch)
                    self.ticker_text = self.ticker_text[-20:]
                    self.ticker.set_text(f"{self.wpm:4.1f} {''.join(self.ticker_text)}")
            
def run():
    import matplotlib.pyplot as plt
    from audio import Audio_in
    import time
        
    fig, axs = plt.subplots(1,2, figsize = (14,5))
    audio = Audio_in(dur = 2, df = 50, dt = 0.01, fRng = [400, 700])
    spec = audio.specbuff
    spec_plot = axs[0].imshow(spec['buff'], origin = 'lower', aspect='auto', interpolation = 'none')
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    axs[1].set_axis_off()

    decoders = []
    for i in range(spec['buff'].shape[0]):
        d = TimingDecoder(axs[1], spec)
        d.set_fbin(i)
        decoders.append(d)

    while True:
        time.sleep(0.05)
        spec_plot.set_data(spec['buff'])
        spec_plot.autoscale()
        plt.pause(0.03)

run()
