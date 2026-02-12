import numpy as np
import wave
import pyaudio
import time
import threading

global audio_buff, waterfall, symbols, dot, wpm, ticker

waterfall_duration = 1
waterfall_dt = 0.0005
waterfall_update_dt = 0.1
sample_rate = 8000
max_freq = 1000
waterfall_df = 50
fft_len = int(sample_rate / waterfall_df)
nFreqs = int(max_freq / waterfall_df)

audio_buff = np.zeros(fft_len, dtype=np.float32)
pya = pyaudio.PyAudio()

def find_device(device_str_contains):
    if(not device_str_contains): #(this check probably shouldn't be needed - check calling code)
        return
    print(f"[Audio] Looking for audio device matching {device_str_contains}")
    for dev_idx in range(pya.get_device_count()):
        name = pya.get_device_info_by_index(dev_idx)['name']
        match = True
        for pattern in device_str_contains:
            if (not pattern in name): match = False
        if(match):
            print(f"[Audio] Found device {name} index {dev_idx}")
            return dev_idx
    print(f"[Audio] No audio device found matching {device_str_contains}")

def audio_in():
    stream = pya.open(
        format = pyaudio.paInt16, channels=1, rate = sample_rate,
        input = True, input_device_index = input_device_idx,
        frames_per_buffer = len(audio_buff), stream_callback=_pya_callback,)
    stream.start_stream()

def audio_out():
    out_stream = pya.open(format=pyaudio.paInt16, channels=1, rate=sample_rate,
                          output=True,
                          output_device_index = output_device_idx)
    while(True):
        time.sleep(0)
        wf = np.int16(audio_buff * 32767)
        out_stream.write(wf.tobytes())

def _pya_callback(in_data, frame_count, time_info, status_flags):
    global audio_buff
    samples = np.frombuffer(in_data, dtype=np.int16).astype(np.float32)
    ns = len(samples)
    audio_buff[:] = samples
    return (None, pyaudio.paContinue)
    
def get_symbols():
    global symbols, dot, waterfall, key
    
    def hysteresis(signal, lo=0.3, hi=0.6):
        out = np.zeros_like(signal, dtype=bool)
        state = False
        for i, v in enumerate(signal):
            if not state and v > hi:
                state = True
            elif state and v < lo:
                state = False
            out[i] = state
        return out

    t_key_down = False
    t_key_up = time.time()
    s = ""
    speclev = 1
    down_durations = []
    while(True):
        time.sleep(waterfall_dt)
        z = np.fft.rfft(audio_buff)[:nFreqs]
        p = z.real*z.real + z.imag*z.imag
        speclev = np.max([speclev,np.max(p)])
        p /= speclev
        waterfall[:-1,:] = waterfall[1:,:] 
        waterfall[-1,:] = np.clip(10*p, 0, 1)
        key = hysteresis(waterfall[:,int(waterfall.shape[1]/2)])

        key_is_down = key[-1]
        if(not key_is_down and t_key_down):
            t_key_up = time.time()
            down_duration = t_key_up - t_key_down
            t_key_down = False
            down_durations.append(down_duration)
            down_durations = down_durations[-20:]
            dot = np.clip(np.percentile(down_durations, 20), min_dot, max_dot)
            if(down_duration < dot * 2):
                s = s + "."
            elif(down_duration > dot * 2):
                s = s + "-"
        if(t_key_up):
            if(time.time() - t_key_up > 1.5*dot and len(s)):
                symbols = s
                s = ""
        if(key_is_down and t_key_up):
            t_key_down = time.time()
            t_key_up = False


def decoder():

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

    global symbols, wpm, ticker
    ticker_text = []
    last_symbols = time.time()
    while(True):
        time.sleep(0.05)
        if(len(symbols)):
            last_symbols = time.time()
            wpm.set_text(f"{int(60 / (50 * dot))} wpm")
            ticker_text.append( MORSE.get(symbols, "?"))
            ticker_text = ticker_text[-20:]
            ticker.set_text(''.join(ticker_text))
            symbols = ""
        if(time.time() - last_symbols > 14*dot and len(ticker_text)):
            if(ticker_text[-1] != " "):
                ticker_text.append(" ")

def run():
    import matplotlib.pyplot as plt
    global wpm, ticker, waterfall, key
    waterfall = np.zeros((int(waterfall_duration / waterfall_dt), nFreqs))
    key = np.zeros(waterfall.shape[0])
    
    fig, axs = plt.subplots(2,1, figsize = (8,3))
    waterfall_plot = axs[0].imshow(waterfall, extent = (0, waterfall.shape[1]*50, 0, waterfall.shape[0]))
    key_plot, = axs[1].plot(key)
    axs[1].set_ylim(0,2)
    wpm = fig.text(0.1,0.6,"WPM")
    ticker = fig.text(0.1,0.8,"TEXT")

    threading.Thread(target = get_symbols).start()
    threading.Thread(target = decoder).start()
    threading.Thread(target = audio_out).start()
    while(True):
        waterfall_plot.set_data(waterfall.copy())
        waterfall_plot.autoscale()
        key_plot.set_ydata(key)
        plt.pause(waterfall_update_dt)
        time.sleep(0.001)

input_device_idx =  find_device(['Mic', 'CODEC'])
output_device_idx =  find_device(['Spea', 'High'])
audio_in()

symbols = ""
dot = 0.4
min_dot = 0.04
max_dot = 0.4

run()




