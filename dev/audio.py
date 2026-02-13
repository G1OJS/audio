import numpy as np
import pyaudio
import time

CABLE = ['CABLE', 'Out']
RIG = ['Min', 'CODEC']
    
class Audio_in:
    
    def __init__(self, device_keywords = RIG, dur = 4, df = 10, dt = 0.01, fft_len = 128, fRng = [300,800]):
        dt_req = dt
        fft_out_len = fft_len //2 + 1
        fmax = fft_out_len * df
        if(fmax < fRng[0]): fRng[0] = fmax
        if(fmax < fRng[1]): fRng[1] = fmax
        sample_rate = int(fft_len * df)
        dt1 = fft_len/sample_rate
        hops_per_fft = int(dt1/dt_req)
        dt = dt1/hops_per_fft
        self.frames_perbuff = fft_len // hops_per_fft
        self.audiobuff = np.zeros(fft_len, dtype=np.float32)

        binRng = [int(fRng[0]/df), int(fRng[1]/df) - 1]
        fRng = [binRng[0] * df, binRng[1] * df]
        self.params = {'dur':dur, 'dt':dt, 'dt_wpm': int(12/dt)/10, 'hpf': hops_per_fft, 'df':df, 'sr':sample_rate, 'fmax':fmax, 'fRng':fRng, 'binRng': binRng}
        self.specbuff = {'pgrid': np.zeros((1+binRng[1]-binRng[0], int(self.params['dur'] / self.params['dt']))), 'idx':0}

        
        self.pya = pyaudio.PyAudio()
        self.input_device_idx = self.find_device(device_keywords)
        self.speclev = 1
        print(self.params)
        self.window = np.hanning(fft_len)
        self.start_audio_in(sample_rate)

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
        z = np.fft.rfft(self.audiobuff * self.window)[self.params['binRng'][0]:self.params['binRng'][1]+1]
        p = (z.real*z.real + z.imag*z.imag)
        self.speclev = max(self.speclev, np.max(p))
        p /= self.speclev
        i = self.specbuff['idx']
        self.specbuff['pgrid'][:, i] = np.clip(10*p, 0, 1)
        self.specbuff['idx'] = (i + 1) % self.specbuff['pgrid'].shape[1]

# testing code
if __name__ == "__main__":

    def test():
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1,2, figsize = (8,8))
        audio = Audio_in()
        print(audio.specbuff['pgrid'].shape)
        spec_plot = axs[0].imshow(audio.specbuff['pgrid'], origin = 'lower', aspect='auto', interpolation = 'none')
        axs[0].set_xticks([])
        axs[0].set_yticks([])
        axs[1].set_axis_off()

        while True:
            time.sleep(0.01)
            idx = audio.specbuff['idx']
            p = audio.specbuff['pgrid']
            display = np.hstack((p[:, idx:], p[:, :idx]))
            spec_plot.set_data(display)
            spec_plot.autoscale()
            plt.pause(0.1)

    test()
