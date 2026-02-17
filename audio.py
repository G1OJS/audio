import numpy as np
import pyaudio
import time
import threading

CABLE = ['CABLE', 'Out']
RIG = ['Min', 'CODEC']
    
class Audio_in:
    
    def __init__(self, device_keywords = RIG, df = 10, dt = 0.01, fft_len = 256, fRng = [300,800], snr_clip_limits = [0,70]):
        dt_req = dt
        fft_out_len = fft_len //2 + 1
        fmax = fft_out_len * df
        fRng = np.clip(fRng, None, fmax)
        sample_rate = int(fft_len * df)
        dt1 = fft_len/sample_rate
        hops_per_fft = np.max([int(dt1/dt_req),1])
        dt = dt1/hops_per_fft
        self.frames_perbuff = fft_len // hops_per_fft
        self.audiobuff = np.zeros(fft_len, dtype=np.float32)

        self.snr_clip_limits = snr_clip_limits
        self.fBins = range(int(fRng[0]/df), int(fRng[1]/df) - 1)
        fRng = [self.fBins[0] * df, self.fBins[-1] * df]
        nf = len(self.fBins)
        self.params = {'dt':dt, 'nf':nf, 'dt_wpm': int(12/dt)/10, 'hpf': hops_per_fft,
                       'df':df, 'sr':sample_rate, 'fmax':fmax, 'fRng':fRng}

        self.pwr = np.ones(nf)
        self.pya = pyaudio.PyAudio()
        self.input_device_idx = self.find_device(device_keywords)
        self.window = np.hanning(fft_len)
        self.start_audio_in(sample_rate)
        
        print(self.params)
        
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
    
    def start_audio_in(self, sample_rate):
        stream = self.pya.open(
            format = pyaudio.paInt16, channels=1, rate = sample_rate,
            input = True, input_device_index = self.input_device_idx,
            frames_per_buffer = self.frames_perbuff, stream_callback=self._pya_callback,)
        stream.start_stream()

    def _pya_callback(self, in_data, frame_count, time_info, status_flags):
        samples = np.frombuffer(in_data, dtype=np.int16).astype(np.float32)
        ns = len(samples)
        self.audiobuff[:-ns] = self.audiobuff[ns:]
        self.audiobuff[-ns:] = samples
        self.calc_spectrum()
        return (None, pyaudio.paContinue)

    def calc_spectrum(self):
        z = np.fft.rfft(self.audiobuff * self.window)[self.fBins]
        self.pwr = (z.real*z.real + z.imag*z.imag)
            

# testing code
if __name__ == "__main__":
    pass
