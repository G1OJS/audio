import matplotlib.pyplot as plt
import numpy as np

class Encoder:
    def __init__(self):
        self.C2M = {
        'A':'.-',   'B':'-...',  'C':'-.-.',  'D':'-..',  'E':'.',     'F':'..-.',  'G':'--.',  'H':'....',
        'I':'..',   'J':'.---',  'K':'-.-',   'L':'.-..', 'M':'--',    'N':'-.',    'O':'---',  'P':'.--.',
        'Q':'--.-', 'R':'.-.',   'S':'...',   'T':'-',    'U':'..-',   'V':'...-',  'W':'.--',  'X':'-..-',
        'Y':'-.--',  'Z':'--..',

        '0':'-----', '1':'.----', '2':'..---', '3':'...--',
        '4':'....-', '5':'.....', '6':'-....', '7':'--...',
        '8':'---..', '9':'----.'
        }
    def encode_syms(self, sig, bits_per_dit = 5):
        syms = ['-','.',' ','/']
        tbl = [[1,1,1,0],[1,0],[0,0,0],[0,0,0,0,0,0,0]]
        bits = [tbl[syms.index(sym)] for sym in sig]
        bits = [b for bb in bits for b in bb]
        return [b for b in bits for i in range(bits_per_dit)]
        
    def encode_chars(self, text):
        syms = []
        words = text.split(' ')
        for wd in words:
            syms.append(' '.join([self.C2M.get(c) for c in wd]))
        return '/'.join(syms)

class Channel:
    def fuzzy(self, code):
        code = np.array(code)
        a = 0.5
        for i in range(2):
            k = (1-a/2)+ a* np.random.rand(3)
            code = np.convolve(code, k)
        return code

class Decoder:
    def __init__(self):
        self.M2C = {
        '.-':'A',   '-...':'B',  '-.-.':'C',  '-..':'D',   '.':'E',    '..-.':'F',  '--.':'G', '....':'H',
        '..':'I',   '.---':'J',  '-.-':'K',   '.-..':'L',  '--':'M',   '-.':'N',    '---':'O', '.--.':'P',
        '--.-':'Q', '.-.':'R',   '...':'S',   '-': 'T',    '..-':'U',  '...-':'V',  '.--':'W', '-..-':'X',
        '-.--':'Y', '--..':'Z',
        '-----': '0', '.----': '1', '..---': '2', '...--': '3',
        '....-': '4', '.....': '5', '-....': '6', '--...': '7',
        '---..': '8', '----.': '9'
        }
        self.tests = []
        pA = [1,-1,1,1,1]
        for scale in range(1, 10):
            pA_test = [b for b in pA for i in range(scale) ]
            self.tests.append(pA_test)
        pN = [1,1,1,-1,1]
        for scale in range(1, 10):
            pN_test = [b for b in pN for i in range(scale) ]
            self.tests.append(pN_test)     

    def pattern_test(self, x):
        x = np.array(x)
        x_bp = 2*x - 1
        self.max_test_len = 50
        for t in range(0, len(x) - self.max_test_len):
            res = []
            for tst in self.tests:
                res.append(int(np.dot(x_bp[t:t+len(tst)], tst)))
            print(t, np.max(res), self.tests[np.argmax(res)])

    def vectorised_test(self, x):
        fig, ax = plt.subplots(figsize=(7,2))
        x = np.array(x)
        diffs = x[1:]-x[:-1]
        rising_edges = np.where(diffs>0.2, 1, 0)
        falling_edges = np.where(diffs<-0.2, 1, 0)    
        idxs_r = np.nonzero(rising_edges)[0]
        idxs_f = np.nonzero(falling_edges)[0]
        ls_on = idxs_f - idxs_r
        ls_off = idxs_r[1:] - idxs_f[:-1]
        dot_on = np.percentile(ls_on,20)
        dot_off = np.percentile(ls_off,20)
        print(ls_on/dot_on)
        print(ls_off/dot_off)
        ax.plot(x)
        ax.plot(rising_edges)
        ax.plot(falling_edges)
        plt.show()  


test_sig = ' CQ '
encoder = Encoder()
channel = Channel()
decoder = Decoder()
syms = encoder.encode_chars(test_sig)
code = encoder.encode_syms(syms)
#code = channel.fuzzy(code)
decoder.pattern_test(code)


