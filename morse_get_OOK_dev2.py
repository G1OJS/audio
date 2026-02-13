

class TimingDecoder:

    def __init__(self, ax, spec):
        import threading
        self.ax = ax
        self.spec = spec
        self.key_is_down = False
        self.n_fbins = spec['pgrid'].shape[0]
        self.fbin = 0
        self.wpm = 16
        tu = 1.2 / self.wpm
        self.speed_elements = {'dot':1*tu, 'dash':3*tu, 'interchar':3*tu, 'interword':7*tu}
        self.ticker = False
        self.set_fbin(10)
        self.symbols = ""
        self.locked = False
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

    def check_element(self, dur, element_type, no_update = False, no_max = False):
        import numpy as np
        timing_tolerance = 0.3 
        alpha = 0.3
        se = self.speed_elements
        t = se[element_type]
        if(not no_max):
            good_element = ((1-timing_tolerance)*t < dur < (1+timing_tolerance)*t)
        else:
            good_element = ((1-timing_tolerance)*t < dur)
        if(good_element and not no_update):
            se[element_type] = alpha * dur + (1-alpha) * se[element_type]
            self.wpm = 1.2/np.mean([se['dot'], se['dash']/3.0])
            tu_new = 1.2/self.wpm
            se['interword'] = 7*tu_new
            self.locked = True
        return good_element

    def get_symbols(self):
        import time
        
        t_key_down = False
        self.t_key_up = time.time()
        s = ""
        speclev = 1
        
        while(True):
            time.sleep(0.002)

            # hysteresis
            level = self.spec['pgrid'][self.fbin, self.spec['idx']]
            if not self.key_is_down and level > 0.6:
                self.key_is_down = True
            elif self.key_is_down and level < 0.3:
                self.key_is_down = False

            # key_down to key_up transition
            if t_key_down and not self.key_is_down:
                self.t_key_up = time.time()
                down_duration = self.t_key_up - t_key_down
                t_key_down = False
                if self.check_element(down_duration, 'dot'):
                    s = s + "."
                elif self.check_element(down_duration, 'dash'):
                    s = s + "-"

            # watch key_up duration for inter-character gap
            if self.t_key_up:
                key_up_dur = time.time() - self.t_key_up
                if self.check_element(key_up_dur, 'interchar', no_update = True, no_max = True):
                    if(len(s)):
                        self.symbols = s
                        s = ""

            # key_up to key_down transition
            if(self.t_key_up and self.key_is_down):
                key_up_dur = time.time() - self.t_key_up
                if self.check_element(key_up_dur, 'interword', no_update = True):
                    if(len(self.ticker_text)):
                        if(self.ticker_text[-1] != " "):
                            self.ticker_text.append(" ")
                t_key_down = time.time()
                self.check_element(key_up_dur, 'interchar')
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
            time.sleep(0.05)
            
            # decode and print single character
            if(len(self.symbols) and self.locked):
                self.ticker_text.append(MORSE.get(self.symbols, "?"))
                self.ticker_text = self.ticker_text[-20:]
                self.symbols = ""
                self.ticker.set_text(f"{self.wpm:4.1f} {''.join(self.ticker_text)}")

def run():
    import matplotlib.pyplot as plt
    from audio import Audio_in
    import time
    import numpy as np
        
    fig, axs = plt.subplots(1,2, figsize = (8,8))
    audio = Audio_in(df = 50, dt = 0.01, fRng = [450, 550])
    spec = audio.specbuff
    spec_plot = axs[0].imshow(spec['pgrid'], origin = 'lower', aspect='auto', interpolation = 'none')
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    axs[1].set_axis_off()

    #currently working quite well but laggy
    decoders = []
    for i in range(spec['pgrid'].shape[0]):
        d = TimingDecoder(axs[1], spec)
        d.set_fbin(i)
        decoders.append(d)

    while True:
        time.sleep(0.01)
        idx = spec['idx']
        wf = spec['pgrid']
        display = np.hstack((wf[:, idx:], wf[:, :idx]))
        spec_plot.set_data(display)
        spec_plot.autoscale()
        plt.pause(0.1)

run()
