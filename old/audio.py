import numpy as np
import pyaudio
import time

CABLE = ['CABLE', 'Out']
RIG = ['Min', 'CODEC']
    
class Audio_in:
    
    def __init__(self, device_keywords = RIG, dur = 0.5, df = 10, dt = 0.01, fft_len = 256, fRng = [300,800], snr_clip = [0,70]):
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

        self.snr_clip = snr_clip
        binRng = [int(fRng[0]/df), int(fRng[1]/df) - 1]
        fRng = [binRng[0] * df, binRng[1] * df]
        nf, nt = 1+binRng[1]-binRng[0], int(dur / dt)
        self.params = {'dur':dur, 'dt':dt, 'nf':nf, 'nt':nt, 'dt_wpm': int(12/dt)/10, 'hpf': hops_per_fft, 'df':df, 'sr':sample_rate, 'fmax':fmax, 'fRng':fRng, 'binRng': binRng}

        self._pgrid = np.ones((nf, nt))
        self.display_grid = np.zeros_like(self._pgrid)
        self.snr = np.zeros(nf)
        self.grid_idx = 0
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
        z = np.fft.rfft(self.audiobuff * self.window)[self.params['binRng'][0]:self.params['binRng'][1]+1]
        pwr = (z.real*z.real + z.imag*z.imag)
        noise = np.percentile(self._pgrid, 20,  axis = 1)
        snr = pwr / noise
        
        i = self.grid_idx
        snr_clip = self.snr_clip
        self._pgrid[:, i] = pwr
        self.snr = np.clip(snr, snr_clip[0],snr_clip[1]) / snr_clip[1]
        self.display_grid = np.roll(self._pgrid, -i, axis = 1)        
        self.grid_idx = (i + 1) % self._pgrid.shape[1]

# testing code
if __name__ == "__main__":

    def test():
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1,2, figsize = (14,5))
        audio = Audio_in(dur = 1, dt = 0.005, df = 50, snr_clip = [25,100])
        refresh_dt = 0.025
        dur = 2
        binRng = audio.params['binRng']
        waterfall = np.zeros((1 + binRng[1] - binRng[0], int(dur/refresh_dt)))
        spec_plot = axs[0].imshow(waterfall, origin = 'lower', aspect='auto', interpolation = 'none')
        axs[0].set_xticks([])
        axs[0].set_yticks([])
        axs[1].set_axis_off()

        while True:
            time.sleep(refresh_dt/2)
            spec_plot.set_data(audio.display_grid)
            spec_plot.autoscale()
            plt.pause(refresh_dt)

    test()
