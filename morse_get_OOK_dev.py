

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
        self.speed_elements = {'dot':1*tu, 'dash':3*tu, 'intra':1*tu, 'inter':3*tu, 'word':7*tu}
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

    def follow_speed(self, dur, speed_element):
        import numpy as np
        alpha = 0.3
        se = self.speed_elements
        se[speed_element] = alpha * dur + (1-alpha) * se[speed_element]
        tu_new = np.mean([se['dot'], se['dash']/3.0, se['intra'], se['inter']/3.0])
        self.wpm = 1.2/tu_new

    def check_element(self, down_dur, unit):
        timing_tolerance = 0.5
        se = self.speed_elements
        t = se[unit]
        good_element = ((1-timing_tolerance)*t < down_dur < (1+timing_tolerance)*t)
        self.locked = good_element or self.locked
        return good_element

    def get_symbols(self):
        import time
        
        t_key_down = False
        t_key_up = time.time()
        s = ""
        speclev = 1
        
        while(True):
            time.sleep(0.002)
            
            level = self.spec['pgrid'][self.fbin, self.spec['idx']]
            if not self.key_is_down and level > 0.6:
                self.key_is_down = True
            elif self.key_is_down and level < 0.3:
                self.key_is_down = False

            if(not self.key_is_down and t_key_down):
                t_key_up = time.time()
                down_duration = t_key_up - t_key_down
                t_key_down = False
                if(self.check_element(down_duration, 'dot')):
                    s = s + "."
                    self.follow_speed(down_duration, 'dot')
                elif(self.check_element(down_duration, 'dash')):
                    s = s + "-"
                    self.follow_speed(down_duration, 'dash')

            if(t_key_up):
                if(time.time() - t_key_up > 1.5*1.2/self.wpm and len(s)):
                    self.symbols = s
                    s = ""
            if(self.key_is_down and t_key_up):
                t_key_down = time.time()
                t_key_up = False

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

        last_symbols = time.time()
        while(True):
            time.sleep(0.1)
            if(len(self.symbols) and self.locked):
                last_symbols = time.time()
                self.ticker_text.append(MORSE.get(self.symbols, "?"))
                self.ticker_text = self.ticker_text[-20:]
                self.symbols = ""
                self.ticker.set_text(f"{self.wpm:4.1f} {''.join(self.ticker_text)}")
            if(time.time() - last_symbols > 14*1.2/self.wpm and len(self.ticker_text)):
                if(self.ticker_text[-1] != " "):
                    self.ticker_text.append(" ")

        
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
