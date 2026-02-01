import numpy as np
import wave
import pyaudio
import time
import threading
import matplotlib.pyplot as plt

        
class AudioFrontEnd:
    
    def __init__(self, device_keywords = ['Mic', 'CODEC']):
        import queue
        self.sample_rate = 32000
        self.wf_df = 40
        self.fft_len = int(self.sample_rate / self.wf_df)
        self.n_fft_bins = self.fft_len//2 + 1
        self.wf_cols = int(self.n_fft_bins / self.wf_df)
        self.max_freq = 1000
        self.wf_cols = int(self.max_freq / self.wf_df)
        self.wf_rows = 1000
        self.waterfall = np.zeros((self.wf_rows, self.wf_cols))
        self.fft_window = np.kaiser(self.fft_len, 10)

        self.wf_lock = threading.Lock()
        self.spectrum_queue = queue.Queue(maxsize=10)
        self.pya = pyaudio.PyAudio()
        self.input_device_idx = self.find_device(device_keywords)
        self.speclev = 1
        self.audio_buff = np.zeros(self.fft_len, dtype=np.float32)
        threading.Thread(target=self.calc_spectrum, daemon=True).start()
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
        return (None, pyaudio.paContinue)

    def calc_spectrum(self):
        
        while True:
            time.sleep(self.fft_len / self.sample_rate / 4)
            z = np.fft.rfft(self.audio_buff * self.fft_window)[:self.wf_cols]
            p = (z.real*z.real + z.imag*z.imag)
            self.speclev = max(self.speclev, np.max(p))
            p /= self.speclev
            # update waterfall
            with self.wf_lock:
                self.waterfall[:-1] = self.waterfall[1:]
                self.waterfall[-1] = p[:self.wf_cols]
            try:
                self.spectrum_queue.put_nowait(p)
            except:
                pass

    def get_waterfall(self):
        with self.wf_lock:
            return self.waterfall.copy()

    def get_queue(self):
        return self.spectrum_queue, self.wf_df


class CWChannelizer:
    def __init__(self, spectrum_queue, wf_df, bin_idx):
        self.queue = spectrum_queue
        self.channel = CWChannel(bin_idx)
        threading.Thread(target=self.run, daemon=True).start()

    def run(self):
        while True:
            p = self.queue.get()
            power = p[self.channel.bin_idx]
            self.channel.process_power(power)
           
class CWChannel:
    def __init__(self, bin_idx):
        self.last_wpm = 0
        self.bin_idx = bin_idx
        self.dot = 0.1
        self.max_dot = 0.15
        self.min_dot = 0.01
        self.env = 0
        self.noise = 0
        self.symbols = ""
        self.key_down = False
        self.last_edge = time.time()
        self.down_times = []
        self.decoder = MorseDecoder()

        self.hist_len = 400
        self.hist_env = np.zeros(self.hist_len)
        self.hist_noise = np.zeros(self.hist_len)
        self.hist_key = np.zeros(self.hist_len)


    def process_power(self, power):
        # smooth envelope (low-pass)
        attack = 0.95
        decay = 0.9

        if power > self.env:
            self.env = (1 - attack) * self.env + attack * power
        else:
            self.env = (1 - decay) * self.env + decay * power

        # track noise floor
        noise_alpha = 0.01
        if not self.key_down:  # only update noise in gaps
            self.noise = (1 - noise_alpha) * self.noise + noise_alpha * self.env
        
        key = self.env > (self.noise + 0.08)


        now = time.time()

        # falling edge → end of dot/dash
        if self.key_down and not key:
            dur = now - self.last_edge
            self.down_times.append(dur)
            self.down_times = self.down_times[-20:]
            self.dot = np.clip(np.percentile(self.down_times, 30),
                               self.min_dot, self.max_dot)
            self.last_wpm = 60 / (50 * self.dot)

            if dur < 2*self.dot:
                self.symbols += "."
            else:
                self.symbols += "-"

            self.last_edge = now

        # rising edge
        if not self.key_down and key:
            self.last_edge = now

        # gap → end of character
        if not key and self.symbols and (now - self.last_edge) > 1.5*self.dot:
            letter = self.decoder.decode(self.symbols)
            print(letter, end="", flush=True)
            self.symbols = ""

        self.key_down = key
        self.hist_env[:-1] = self.hist_env[1:]
        self.hist_noise[:-1] = self.hist_noise[1:]
        self.hist_key[:-1] = self.hist_key[1:]
        self.hist_env[-1] = self.env
        self.hist_noise[-1] = self.noise
        self.hist_key[-1] = 1 if key else 0

    def get_wpm(self):
        return self.last_wpm

    def get_debug(self):
        return (self.hist_env.copy(),
                self.hist_noise.copy(),
                self.hist_key.copy())



class MorseDecoder:
    MORSE = {
        ".-": "A", "-...": "B", "-.-.": "C", "-..": "D",
        ".": "E", "..-.": "F", "--.": "G", "....": "H",
        "..": "I", ".---": "J", "-.-": "K", ".-..": "L",
        "--": "M", "-.": "N", "---": "O", ".--.": "P",
        "--.-": "Q", ".-.": "R", "...": "S", "-": "T",
        "..-": "U", "...-": "V", ".--": "W", "-..-": "X",
        "-.--": "Y", "--..": "Z",
        "-----": "0", ".----": "1", "..---": "2", "...--": "3",
        "....-": "4", ".....": "5", "-....": "6", "--...": "7",
        "---..": "8", "----.": "9"
    }

    def decode(self, symbols):
        return self.MORSE.get(symbols, "?")




def init_GUI():
    import matplotlib.pyplot as plt
    plt.ion()
    fig, (ax_wf, ax_key) = plt.subplots(2,1, figsize=(8,6))
    wf_img = ax_wf.imshow([[0,0]], aspect='auto', origin='lower',
                    extent=[0, 1000, 0, 1000])
    wpm_text = ax_wf.text(0.02, 0.95, "", transform=ax_wf.transAxes, color="white")

    line_env, = ax_key.plot([], label="env")
    line_noise, = ax_key.plot([], label="noise")
    line_key, = ax_key.plot([], label="key")
    ax_key.set_ylim(0,1.2)
    ax_key.legend()
    return plt, fig, ax_wf, ax_key, wf_img, wpm_text, line_env, line_noise, line_key


def show_key(ax_key, env, noise, key_hist, line_env, line_noise, line_key):
    line_env.set_ydata(env)
    line_env.set_xdata(range(len(env)))
    line_noise.set_ydata(noise)
    line_noise.set_xdata(range(len(noise)))
    line_key.set_ydata(key_hist)
    line_key.set_xdata(range(len(key_hist)))
    ax_key.set_xlim(0, len(env))

def run():
    debug = False
    audio = AudioFrontEnd()
    spec_q, wf_df = audio.get_queue()

    tone_freq = 600
    bin_idx = int(tone_freq / wf_df)
    channelizer = CWChannelizer(spec_q, wf_df, bin_idx)

    plt, fig, ax_wf, ax_key, wf_img, wpm_text, line_env, line_noise, line_key = init_GUI()
    ax_wf.axhline(channelizer.channel.bin_idx * wf_df, color='r')
    while True:
        wf = audio.get_waterfall().T
        wf_img.set_data(wf)
        wf_img.set_clim(0, np.max(wf) + 1e-6)
        wpm_text.set_text(f"{channelizer.channel.get_wpm():.1f} WPM")
        env, noise, key_hist = channelizer.channel.get_debug()
        show_key(ax_key, env, noise, key_hist, line_env, line_noise, line_key)
        plt.pause(0.05)




run()




