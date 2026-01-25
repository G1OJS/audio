import numpy as np
import wave
import pyaudio
import time
import threading

global audio_buff, out_stream

waterfall_duration = 1
waterfall_dt = 0.0005
waterfall_update_dt = 0.1
sample_rate = 8000
max_freq = 1000
waterfall_df = 50
fft_len = int(sample_rate / waterfall_df)
nFreqs = int(max_freq / waterfall_df)

audio_buff = np.zeros(fft_len, dtype=np.float32)
out_stream = None

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


def start_audio(input_device_idx, output_device_idx):
    global out_stream
    stream = pya.open(
        format = pyaudio.paInt16, channels=1, rate = sample_rate,
        input = True, input_device_index = input_device_idx,
        frames_per_buffer = len(audio_buff), stream_callback=_callback,)
    stream.start_stream()
    out_stream = pya.open(format=pyaudio.paInt16, channels=1, rate=sample_rate,
                          output=True,
                          output_device_index = output_device_idx)
    threading.Thread(target = threaded_output).start()


def threaded_output():
    global audio_buff
    while(True):
        time.sleep(0)
        wf = np.int16(audio_buff * 32767)
        out_stream.write(wf.tobytes())

def _callback(in_data, frame_count, time_info, status_flags):
    global audio_buff
    samples = np.frombuffer(in_data, dtype=np.int16).astype(np.float32)
    ns = len(samples)
    audio_buff = samples
    time.sleep(0)
    return (None, pyaudio.paContinue)

global waterfall, to_print, dot
waterfall = np.zeros((int(waterfall_duration / waterfall_dt), nFreqs))
to_print = ""
dot = 0.4
min_dot = 0.04
max_dot = 0.4
key = None
    
def threaded_get_key():
    global key, to_print, dot

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
            if(down_duration < dot*2):
                s = s + "."
                new_dot = dot * 0.85 + 0.15 * down_duration
                if(max_dot> new_dot > min_dot):
                    dot = new_dot
            elif(down_duration > dot * 2):
                s = s + "-"
                new_dot = dot * 0.85 + 0.15 * down_duration /3
                if(max_dot> new_dot > min_dot):
                    dot = new_dot

        if(t_key_up):
            if(time.time() - t_key_up > 1.5*dot and len(s)):
                to_print = s
                s = ""

        if(key_is_down and t_key_up):
            t_key_down = time.time()
            t_key_up = False


def time_plot():
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2,1, figsize = (8,3))
    waterfall_plot = axs[0].imshow(waterfall, extent = (0, waterfall.shape[1]*50, 0, waterfall.shape[0]))
    key_plot, = axs[1].plot(key)
    axs[1].set_ylim(0,2)

    while(True):
        waterfall_plot.set_data(waterfall)
        key_plot.set_ydata(key)
        plt.pause(waterfall_update_dt)
        time.sleep(0.001)

def threaded_printer():
    global to_print, dot
    while(True):
        time.sleep(0.05)
        if(len(to_print)):
            wpm = 20 * dot / 0.06
            print(f"{wpm:5.0f} {to_print}")
            to_print = ""
    
input_device_idx =  find_device(['Mic', 'CODEC'])
output_device_idx =  find_device(['Spea', 'High'])
start_audio(input_device_idx, output_device_idx)
threading.Thread(target = threaded_get_key).start()
threading.Thread(target = threaded_printer).start()
time_plot()



