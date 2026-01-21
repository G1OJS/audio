
import numpy as np
import wave
import pyaudio
import time
import threading

global audio_buff, out_stream
audio_buff = np.zeros(2000, dtype=np.float32)
out_stream = None
sample_rate = 24000

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


def start_live(input_device_idx, output_device_idx):
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
        time.sleep(0.001)
        wf = np.int16(audio_buff * 32767)
        out_stream.write(wf.tobytes())

def _callback(in_data, frame_count, time_info, status_flags):
    global audio_buff
    samples = np.frombuffer(in_data, dtype=np.int16).astype(np.float32)
    ns = len(samples)
    audio_buff = samples
    time.sleep(0.001)
    return (None, pyaudio.paContinue)


def time_plot():
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    bufflen = 150
    levbuff = np.zeros(bufflen)
    line, = ax.plot(levbuff)
    ax.set_ylim(0,1)
    while(True):
        time.sleep(0.01)
        lev = np.mean(np.abs(audio_buff)) / 2000
        levbuff[:-1] = levbuff[1:]
        levbuff[-1] = lev
        line.set_ydata(levbuff)
        plt.pause(0.04)

    
input_device_idx =  find_device(['Mic', 'CODEC'])
output_device_idx =  find_device(['Spea', 'High'])
start_live(input_device_idx, output_device_idx)
time_plot()



