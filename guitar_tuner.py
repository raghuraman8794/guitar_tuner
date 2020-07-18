import numpy as np
import struct
import matplotlib.pyplot as plt
from scipy.fftpack import fft as fft_2
import pyaudio
import time
from tkinter import TclError

# to display in separate Tk window
get_ipython().run_line_magic('matplotlib', 'tk')

#####################################################################
NOTE_MIN = 40       # E2
NOTE_MAX = 64       # E4
FSAMP = 22050       # Sampling frequency in Hz
FRAME_SIZE = 2048   # How many samples per frame?
FRAMES_PER_FFT = 16 # FFT takes average across how many frames?
######################################################################
SAMPLES_PER_FFT = FRAME_SIZE*FRAMES_PER_FFT
FREQ_STEP = float(FSAMP)/SAMPLES_PER_FFT

######################################################################
# For printing out notes

NOTE_NAMES = 'C C# D D# E F F# G G# A A# B'.split()

######################################################################
def freq_to_number(f): return 69 + 12*np.log2(f/440.0)
def number_to_freq(n): return 440 * 2.0**((n-69)/12.0)
def note_name(n): return NOTE_NAMES[n % 12] + str(n/12 - 1)

######################################################################
# Ok, ready to go now.

# Get min/max index within FFT of notes we care about.
# See docs for numpy.rfftfreq()
def note_to_fftbin(n): return number_to_freq(n)/FREQ_STEP
imin = max(0, int(np.floor(note_to_fftbin(NOTE_MIN-1))))
imax = min(SAMPLES_PER_FFT, int(np.ceil(note_to_fftbin(NOTE_MAX+1))))

# Allocate space to run an FFT. 
buf = np.zeros(SAMPLES_PER_FFT, dtype=np.float32)
num_frames = 0

#Subplots for plotting the input audio and frequency waveforms
fig, (ax1, ax2) = plt.subplots(2, figsize=(15, 7))

# Initialize audio
stream = pyaudio.PyAudio().open(format=pyaudio.paInt16,
                                channels=1,
                                rate=FSAMP,
                                input=True,
                                frames_per_buffer=FRAME_SIZE)

stream.start_stream()

# variable for plotting
x = np.arange(0, 2 * FRAME_SIZE, 2)       # samples (waveform)
xf = np.linspace(0, FSAMP, FRAME_SIZE)    # frequencies (spectrum)

# create a line object with random data
line, = ax1.plot(x, np.random.rand(FRAME_SIZE), '-', lw=2)

# create semilogx line for spectrum
line_fft, = ax2.plot(xf, np.random.rand(FRAME_SIZE), '-', lw=2)

# format waveform axes
ax1.set_title('AUDIO WAVEFORM')
ax1.set_xlabel('samples')
ax1.set_ylabel('volume')
ax1.set_ylim(0, 255)
ax1.set_xlim(0, 2 * FRAME_SIZE)
plt.setp(ax1, xticks=[0, FRAME_SIZE, 2 * FRAME_SIZE], yticks=[0, 128, 255])

# format spectrum axes
ax2.set_ylim(0,1)
ax2.set_xlim(20,500)

# Create Hanning window function
window = 0.5 * (1 - np.cos(np.linspace(0, 2*np.pi, SAMPLES_PER_FFT, False)))

# Print initial text
print ('sampling at', FSAMP, 'Hz with max resolution of', FREQ_STEP, 'Hz')
# As long as we are getting data:

# for measuring frame rate
frame_count = 0
start_time = time.time()
i=0;
while stream.is_active():
    

    # binary data
    i=i+1
    data = stream.read(FRAME_SIZE)  
    
    # convert data to integers, make np array, then offset it by 127
    data_int = struct.unpack(str(2 * FRAME_SIZE) + 'B', data)
    data_int = np.asarray(data_int)
    # create np array and offset by 128
    data_np = np.array(data_int, dtype='b')[::2] + 128
    
    line.set_ydata(data_np)
    
    # compute FFT and update line
    yf = fft_2(data_int)
    line_fft.set_ydata(np.abs(yf[0:FRAME_SIZE])  / (128 * FRAME_SIZE))

    # Shift the buffer down and new data in
    buf[:-FRAME_SIZE] = buf[FRAME_SIZE:]
    buf[-FRAME_SIZE:] = np.fromstring(stream.read(FRAME_SIZE), np.int16)

    # Run the FFT on the windowed buffer
    fft = np.fft.rfft(buf * window)
#    line_fft.set_ydata(np.abs(fft[0:FRAME_SIZE])  / (128 * FRAME_SIZE))

    # Get frequency of maximum response in range
    freq = (np.abs(fft[imin:imax]).argmax() + imin) * FREQ_STEP

    # Get note number and nearest note
    n = freq_to_number(freq)
    n0 = int(round(n))

    # Console output once we have a full buffer
    num_frames += 1

    if num_frames >= FRAMES_PER_FFT:
        print ('freq: {:7.2f} Hz     note: {:>3s} {:+.2f}'.format(
            freq, note_name(n0), n-n0))
        
    try:
#        if(i%15==0):

#            text = ax2.text(400,0.8,'freq: {:7.2f} Hz     note: {:>3s} {:+.2f}'.format(
#            freq, note_name(n0), n-n0))
            
        text = ax2.text(400,0.8,"")
        text.set_position((400, 0.8))
        fig.canvas.draw()
        fig.canvas.flush_events()
        frame_count += 1
        
    except TclError:
        
        # calculate average frame rate
        frame_rate = frame_count / (time.time() - start_time)
        
        print('stream stopped')
        print('average frame rate = {:.0f} FPS'.format(frame_rate))
        break

