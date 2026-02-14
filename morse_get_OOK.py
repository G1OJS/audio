MAX_WPM = 45
MIN_WPM = 12
DOT_TOL_FACTS = (0.6, 2)
DASH_TOL_FACTS = (0.6, 1.4)
CHARSEP_THRESHOLD = 0.6
WORDSEP_THRESHOLD = 0.7

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

".-.-.": "_AR_", "..--..": "?", "-...-": "_BK_", "...-.-": "_SK_", "..-.-.": "_UR_"

}

import numpy as np
        
class TimingDecoder:

    def __init__(self, axs, audio, fbin):
        import threading
        self.text_ax = axs[1]
        self.audio = audio
        self.fbin = fbin
        self.key_is_now_down = False
        self.dt = audio.params['dt']
        self.wpm = 16
        self.keydown_history = {'buffer':[1.2/self.wpm]*10, 'idx':0}
        self.check_speed(1.2/self.wpm)
        self.ticker = False
        self.ticker_text_blank = [' ']*20
        self.ticker_text = self.ticker_text_blank
        self.symbols = ""
        threading.Thread(target = self.get_symbols).start()
        self.ticker = self.text_ax.text(0, fbin,'*')

    def check_element(self, dur):
        se = self.speed_elements
        if DOT_TOL_FACTS[0]*se['dot'] < dur < DOT_TOL_FACTS[1]*se['dot']:
            return '.'
        if DASH_TOL_FACTS[0]*se['dash'] < dur < DASH_TOL_FACTS[1]*se['dash']:
            return '-'
        return ''

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
        self.t_key_up = time.time()
        s = ""
        
        while(True):
            time.sleep(self.dt)

            # hysteresis
            level = float(self.audio.snr[self.fbin])
            if not self.key_is_now_down and level > 0.6:
                self.key_is_now_down = True
            elif self.key_is_now_down and level < 0.4:
                self.key_is_now_down = False

            # key_down to key_up transition
            if t_key_down and not self.key_is_now_down:
                self.t_key_up = time.time()
                down_dur = self.t_key_up - t_key_down
                t_key_down = False
                self.check_speed(down_dur)
                s = s + self.check_element(down_dur)

            # watch key_up dur for inter-character and inter-word gaps
            if self.t_key_up:
                key_up_dur = time.time() - self.t_key_up
                if key_up_dur > CHARSEP_THRESHOLD * self.speed_elements['charsep']:
                    if(len(s)):
                        self.ticker_text = self.ticker_text[-20:]
                        ch = MORSE.get(s, "*")
                        s = ""
                        self.ticker_text.append(ch)
                        tkrstr = ''.join(self.ticker_text)
                        for pat in [' E E', 'E E ', ' E ', ' T ', ' IE ', 'EEE', 'EIE']:
                            tkrstr = tkrstr.replace(pat,'')
                        self.ticker.set_text(f"{self.wpm:4.1f} {tkrstr}")                        
                if key_up_dur > WORDSEP_THRESHOLD * self.speed_elements['wordsep']:
                    if(len(self.ticker_text)):
                        if(self.ticker_text[-1] != " "):
                            self.ticker_text.append(" ")

            # key_up to key_down transition
            if(self.t_key_up and self.key_is_now_down):
                t_key_down = time.time()
                self.t_key_up = False

def run():
    import matplotlib.pyplot as plt
    from audio import Audio_in
    import time

    audio = Audio_in(df = 50, dt = 0.005, fRng = [200, 1400], snr_clip = [18,80])

    refresh_dt = 0.025
    nf = audio.params['nf']
 
    fig, axs = plt.subplots(1,2, figsize = (14,5))
    spec_plot = axs[0].imshow(audio.display_grid, origin = 'lower', aspect='auto', interpolation = 'none')
    axs[1].set_ylim(axs[0].get_ylim())
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    axs[1].set_axis_off()

    decoders = []
    for i in range(len(audio.snr)):
        d = TimingDecoder(axs, audio, i)
        decoders.append(d)

    while True:
        time.sleep(refresh_dt/2)
        spec_plot.set_data(audio.display_grid)
        spec_plot.autoscale()
        plt.pause(refresh_dt)

run()
