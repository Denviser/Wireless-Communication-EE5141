#Charan Srikkanth - EE23B127
"""This code contains my workings for assignment 2 for the course EE5141-Fundamentals of Wireless and Cellular Communication"""
"""For Q1 uncomment the function call in main plot_mse_vs_SNR_freq_offset()"""
"""For Q2 uncomment the function call in main plot_sc_time_sync()"""
"""For Q3 uncomment the function call in main plot_channel_estimation_performance()"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy.signal import find_peaks
#np.random.seed(42) # For reproducibility


class Parameters:
    def __init__(self):
        self.N= 512
        self.subcarrier_bandwidth = 10000 # 10 Khz
        self.entire_bandwidth = 5.12e6 # 5.12 Mhz
        self.sampling_rate = self.entire_bandwidth
        self.cyclic_prefix_duration = 6.25e-6 # 6.25 microseconds
        self.N_cp = int(self.cyclic_prefix_duration*self.sampling_rate)
        self.num_blocks_per_frame = 5
        self.blk_duration_w_o_cyclic_prefix = 1/self.subcarrier_bandwidth
        self.block_duration = self.blk_duration_w_o_cyclic_prefix + self.cyclic_prefix_duration
        self.frame_duration = self.num_blocks_per_frame * self.block_duration
        #0th index represents -255 and 512 is 255 subcarrier
        self.valid_frequency_indexes = np.array([i for i in range(15,255)]+[i for i in range(256,496)])
        #self.all_frequency_indexes = np.array([i for i in range(-255,257)])

class OFDM_Channel(Parameters):
    def __init__(self):
        super().__init__()
        self.channel_response = np.zeros(27,dtype=complex)
        tap_delays = np.array([0,7,13,18,21,26])
        tap_gains_dB = np.array([-3,0,-1,-4,-9,-17])
        tap_gains = 10**(tap_gains_dB/10)
        normalised_tap_gains = tap_gains/np.sum(tap_gains)
        for i in range(len(tap_delays)):
            gauss_var_per_axis = normalised_tap_gains[i]/2
            #The channel response is a complex circular gaussian rv
            self.channel_response[tap_delays[i]] = np.sqrt(gauss_var_per_axis)*(np.random.randn()+1j*np.random.randn())

        #Take a N size fft in order to get the channel frequency response
        self.channel_freq_response_ortho = np.fft.fft(self.channel_response,self.N,norm='ortho')
        self.channel_freq_response = np.fft.fft(self.channel_response,self.N)

    def act(self,transmitted_symbols,frequency_offset=0.0,SNR=100)->np.ndarray:
        #The output symbols are just convolution with the channel response
        transmitted_symbols = np.squeeze(transmitted_symbols)
        output_symbols = np.convolve(transmitted_symbols,self.channel_response)[:len(transmitted_symbols)]
        output_symbols = output_symbols*np.exp(1j*2*np.pi*(frequency_offset/self.sampling_rate)*np.arange(len(output_symbols)))
        #By default the SNR is 100
        noise_var = 1/(10**(SNR/10))
        noise = np.sqrt(noise_var/2)*(np.random.randn(len(output_symbols))+1j*np.random.randn(len(output_symbols)))
        output_symbols = output_symbols + noise                  
        return output_symbols



class Transmitter(Parameters):
    def __init__(self):
        super().__init__()
        #Create a IDFT matrix of (N+N_cp)*N
        IDFT_matrix_wo_cyclic_prefix = np.zeros((self.N,self.N),dtype=complex)
        for n in range(self.N):
            for k in range(self.N):
                IDFT_matrix_wo_cyclic_prefix[n,k] = np.exp(1j*2*np.pi*k*n/self.N)
        IDFT_matrix_wo_cyclic_prefix = IDFT_matrix_wo_cyclic_prefix / np.sqrt(self.N)

        cyclic_prefix = IDFT_matrix_wo_cyclic_prefix[-self.N_cp:, :]
    
        # Vertically stack the prefix on top of the original matrix
        self.IDFT_matrix = np.vstack((cyclic_prefix, IDFT_matrix_wo_cyclic_prefix))

    def create_random_transmit_block(self)->np.ndarray:
        transmit_symbols = np.zeros((self.N,1),dtype=complex)
        #print(transmit_symbols.shape)
        for index in self.valid_frequency_indexes:
            transmit_symbols[index,0] = 2*np.random.randint(2)-1 + 1j*(2*np.random.randint(2)-1) # Generate QPSK symbols
            transmit_symbols[index,0] *= 1/np.sqrt(2) #Normalise to unit energy 
        #print(transmit_symbols)
        
        transmit_block = np.matmul(self.IDFT_matrix,transmit_symbols)
        return transmit_block
        #return transmit_symbols

    def create_pilot_block(self)->tuple[np.ndarray,np.ndarray]:
        pilot_symbols = np.zeros((self.N,1),dtype=complex)
        only_pilots_arr = []
        for freq_index in range(15,496):
            if freq_index%8 == 7: # Pilot is placed every 8 subcarriers
                if freq_index == 255: # Pilot is placed every 8 subcarriers
                    continue
                pilot = 2*np.random.randint(2)-1 + 1j*(2*np.random.randint(2)-1) # Generate QPSK symbols
                pilot *= 1/np.sqrt(2) #Normalise to unit energy 
                only_pilots_arr.append(pilot)
                pilot_symbols[freq_index,0] = pilot
            else:
                pilot_symbols[freq_index,0] = 2*np.random.randint(2)-1 + 1j*(2*np.random.randint(2)-1) # Generate QPSK symbols
                pilot_symbols[freq_index,0] *= 1/np.sqrt(2) #Normalise to unit energy

        only_pilots_arr = np.array(only_pilots_arr)
        pilot_block = np.matmul(self.IDFT_matrix,pilot_symbols)
        return pilot_block,only_pilots_arr
       
    def plot_preamble(self,preamble_block, preamble_symbols):
        """Plots the Frequency and Time domain representations of the Schmidl-Cox Preamble."""
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        # --- 1. Frequency Domain Plot ---
        # Flatten arrays to 1D for plotting
        freq_data = np.abs(preamble_symbols.flatten())
        
        axes[0].stem(freq_data, basefmt=" ", markerfmt="b.", linefmt="b-")
        axes[0].set_title("Frequency Domain: Schmidl-Cox Preamble Symbols")
        axes[0].set_xlabel("Subcarrier Index")
        axes[0].set_ylabel("Magnitude")
        axes[0].grid(True, alpha=0.3)
        # Zooming in on the first 50 subcarriers to clearly see the alternating pattern
        axes[0].set_xlim(-1, 50) 

        # --- 2. Time Domain Plot ---
        time_data = np.abs(preamble_block.flatten())
        
        axes[1].plot(time_data, 'r-')
        axes[1].set_title("Time Domain: Preamble Block (Magnitude)")
        axes[1].set_xlabel("Sample Index")
        axes[1].set_ylabel("Magnitude")
        
        # Adding vertical lines to highlight the Cyclic Prefix and the N/2 symmetry
        axes[1].axvline(x=self.N_cp, color='k', linestyle='--', label="Start of OFDM Symbol (End of CP)")
        axes[1].axvline(x=self.N_cp + (self.N // 2), color='g', linestyle=':', label="Midpoint of OFDM Symbol")
        
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("./plots/schmidl_cox_preamble.png")

    def create_preamble_block(self)->tuple[np.ndarray,np.ndarray]:
        #This block is used to create the schmidl cox preamble block which is used to estimate frequency offset.
        preamble_symbols = np.zeros((self.N,1),dtype=complex)
        #For the schmidl cox we need the time domain to be symmetric for this the frequency domain should have alternate symbols.
        for index in range(0,self.N,2):
            preamble_symbols[index,0] = 2*np.random.randint(2)-1 + 1j*(2*np.random.randint(2)-1) # Generate QPSK symbols
            preamble_symbols[index,0] *= 1/np.sqrt(2) #Normalise to unit energy 
        
        preamble_block = np.matmul(self.IDFT_matrix,preamble_symbols)
        
        return preamble_block,preamble_symbols
    
    def create_transmit_frame(self):
        transmit_frame = np.zeros((self.num_blocks_per_frame*(self.N+self.N_cp),1),dtype=complex)
        first_preamble_frame,_ = self.create_preamble_block()
        transmit_frame[0:self.N+self.N_cp] = first_preamble_frame
        for i in range(1,self.num_blocks_per_frame):
            random_block = self.create_random_transmit_block()
            transmit_frame[i*(self.N+self.N_cp):(i+1)*(self.N+self.N_cp)] = random_block

        return transmit_frame


class Receiver(Parameters):
    def __init__(self):
        super().__init__()
    
    def schmidl_cox_correlation(self,recieved_symbols):
        recieved_symbols = np.squeeze(recieved_symbols)
        corr_output=np.zeros(len(recieved_symbols),dtype=complex)
        #print(corr_output.shape)
        curr_window_sum = 0
        for index in range(len(recieved_symbols)):
            if index < int(self.N/2):
                continue
            else:
                curr_window_sum+=recieved_symbols[index]*np.conj(recieved_symbols[index-int(self.N/2)])
            
            #Basically a sliding window where you remove the sum of the last most element.
            if index>=self.N:
                curr_window_sum -= recieved_symbols[index-int(self.N/2)]*np.conj(recieved_symbols[index-int(self.N)])
            
            corr_output[index] = curr_window_sum

        return corr_output


    def get_frequency_offset_est(self, corr_output):
        peak_idx = np.argmax(np.abs(corr_output))
        # Average phase over a small window around the peak to reduce noise impact
        window = 2
        avg_phase = np.angle(np.mean(corr_output[peak_idx - window : peak_idx + window + 1]))
        
        freq_offset_est = (avg_phase * self.subcarrier_bandwidth) / np.pi
        return freq_offset_est
    
    def plot_correlation(self, corr_output, label="Timing Metric", threshold_factor=0.7):
        magnitude = np.abs(corr_output)
        
        # 1. Detect multiple peaks
        # 'height' ensures we ignore noise. 
        # 'distance' ensures we don't pick multiple points on the same plateau (set to ~75% of N)
        height_threshold = np.max(magnitude) * threshold_factor
        peaks, _ = find_peaks(magnitude, height=height_threshold, distance=int(self.N * 0.8))

        # 2. Find the "End of Peak" for every detected peak
        # We walk forward from each peak until the value drops significantly
        boundary_indices = []
        for p in peaks:
            idx = p
            # Walk forward until the value drops below 95% of the local peak value
            # This identifies the "falling edge" of the timing plateau
            while idx < len(magnitude) - 1 and magnitude[idx+1] > magnitude[p] * 0.99:
                idx += 1
            boundary_indices.append(idx)

        # 3. Handle Figure Persistence (same as before)
        if not plt.get_fignums():
            plt.figure(figsize=(14, 7))
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.title("Multi-Frame Schmidl-Cox Synchronization", fontsize=14, fontweight='bold')
            plt.xlabel("Sample Index ($d$)", fontsize=12)
            plt.ylabel("Metric Magnitude $|P(d)|$", fontsize=12)

        # 4. Plot the correlation metric
        line, = plt.plot(magnitude, linewidth=1.5, label=label)
        current_color = line.get_color()
        
        # 5. Mark and Label all boundaries
        for i, b_idx in enumerate(boundary_indices):
            plt.scatter(b_idx, magnitude[b_idx], color=current_color, s=60, edgecolors='black', zorder=5)
            
            # Label only the first few to avoid clutter, or all if preferred
            plt.annotate(f'Start Frame {i+1}\nIdx: {b_idx}',
                         xy=(b_idx, magnitude[b_idx]),
                         xytext=(0, 20), # 20pts above the point
                         textcoords='offset points',
                         ha='center',
                         arrowprops=dict(arrowstyle="->", color=current_color),
                         fontsize=8, 
                         bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=current_color, alpha=0.7))

        plt.legend(loc='upper right')
        # Reminder: Call plt.show() outside this function to see the result

class ChannelEstimation(Parameters):
    def __init__(self):
        super().__init__()
        self.pilot_spacing = 8
        self.pilot_indexes = list(range(15,496,self.pilot_spacing))
        self.pilot_indexes.remove(255)
        self.N_p = len(self.pilot_indexes)
        self.valid_indexes = list(range(15,496))
        self.N_u = len(self.valid_indexes)
        self.non_pilot_indexes = [i for i in self.valid_indexes if i not in self.pilot_indexes]
        self.channel_indexes = np.array([0,7,13,18,21,26])
        #self.channel_indexes = tap_delays
    def create_A_matrix_for_mlsq_estimate(self, pilot_symbols):
        # 1. Create the N_p x N_p DFT matrix
        row_indices = np.array(self.pilot_indexes)[:, np.newaxis]
        col_indices = np.arange(self.N_cp)
        DFT_matrix_Np = np.exp(-1j * 2 * np.pi * row_indices * col_indices / self.N)
        
        # 2. Extract only the pilot symbols (Shape: N_p,)
        extracted_pilots = pilot_symbols[self.pilot_indexes]
        
        # 3. Create the N_p x N_p diagonal pilot matrix
        P_matrix = np.diag(extracted_pilots)
        
        # 4. Multiply the diagonal matrix by the DFT matrix
        A_matrix = P_matrix @ DFT_matrix_Np
        
        # return the pseudo inverse
        return np.linalg.pinv(A_matrix)   

    def create_A_matrix_known_tap_locations(self, pilot_symbols):
        # 1. Create the N_p x N_p DFT matrix
        row_indices = np.array(self.pilot_indexes)[:, np.newaxis]
        col_indices = self.channel_indexes
        DFT_matrix_Np = np.exp(-1j * 2 * np.pi * row_indices * col_indices / self.N)
        
        # 2. Extract only the pilot symbols (Shape: N_p,)
        extracted_pilots = pilot_symbols[self.pilot_indexes]
        
        # 3. Create the N_p x N_p diagonal pilot matrix
        P_matrix = np.diag(extracted_pilots)
        
        # 4. Multiply the diagonal matrix by the DFT matrix
        A_matrix = P_matrix @ DFT_matrix_Np
        
        # return the pseudo inverse
        return np.linalg.pinv(A_matrix)
    
    def create_pilot_symbols(self)->np.ndarray:
        pilot_symbols = np.zeros(self.N,dtype=complex)
        only_pilots_arr = []
        for freq_index in range(15,496):
            if freq_index%8 == 7: # Pilot is placed every 8 subcarriers
                if freq_index == 255: # Pilot is placed every 8 subcarriers
                    continue
                pilot = 2*np.random.randint(2)-1 + 1j*(2*np.random.randint(2)-1) # Generate QPSK symbols
                pilot *= 1/np.sqrt(2) #Normalise to unit energy 
                only_pilots_arr.append(pilot)
                pilot_symbols[freq_index] = pilot
            else:
                pilot_symbols[freq_index] = 2*np.random.randint(2)-1 + 1j*(2*np.random.randint(2)-1) # Generate QPSK symbols
                pilot_symbols[freq_index] *= 1/np.sqrt(2) #Normalise to unit energy
        return pilot_symbols

    def get_mse(self,channel_estimate,true_channel):
        mse = 0
        for index in self.valid_frequency_indexes:
            mse += abs(channel_estimate[index]-true_channel[index])**2
        return mse/len(self.valid_frequency_indexes)

    
    def get_zf_estimate(self,pilot_symbols,recieved_symbols):
        H = np.zeros(self.N,dtype=complex)
        for index in self.pilot_indexes:
            H[index] = recieved_symbols[index]/pilot_symbols[index]
        return H

    def get_zf_estimate_linear_interp(self,pilot_symbols,recieved_symbols):
        H = np.zeros(self.N,dtype=complex)
        H_known = []
        for index in self.pilot_indexes:
            if pilot_symbols[index] == 0:
                ValueError("Pilot symbol is zero")
            H[index] = recieved_symbols[index]/pilot_symbols[index]
            H_known.append(H[index])
        H_known = np.array(H_known)
        interp_H = np.interp(range(self.N),self.pilot_indexes,H_known)
        return interp_H

    def get_fft_interp_estimate(self,pilot_symbols,recieved_symbols):
        H_zf = self.get_zf_estimate(pilot_symbols,recieved_symbols)
        channel_response = np.fft.ifft(H_zf)
        # plt.plot(abs(channel_response),'r-')
        # plt.show()

        #Multiply with pilot spacing so as to get back the original energy as truncating reduces the energy
        channel_response_truncated = channel_response[:self.N_cp]*self.pilot_spacing
        channel_freq_response = np.fft.fft(channel_response_truncated,self.N)
        return channel_freq_response
    
    def get_fft_interp_hann_window_estimate(self, pilot_symbols, recieved_symbols):
        H_zf = self.get_zf_estimate(pilot_symbols, recieved_symbols)
        channel_response = np.fft.ifft(H_zf)
        
        # Truncate to N_cp (32) to discard obvious noise
        channel_response_truncated = channel_response[:self.N_cp] * self.pilot_spacing
        
        # Create a custom window: 
        # Flat 1.0 for the first Ncp-10 taps (to protect the true channel)
        # Taper smoothly to 0 for the remaining 5 taps of the Cyclic Prefix
        window = np.ones(self.N_cp)
        # Use the falling half of a hanning window for the last 5 indices
        window[self.N_cp-10:] = np.hamming(20)[10:] 
        
        channel_response_windowed = window * channel_response_truncated
        channel_freq_response = np.fft.fft(channel_response_windowed, self.N)
        
        return channel_freq_response

    def get_least_squares_estimate(self,recieved_symbols,A_matrix):
        recieved_symbols_pilot_indexed = recieved_symbols[self.pilot_indexes]
        recieved_symbols_pilot_indexed = recieved_symbols_pilot_indexed.reshape((self.N_p,1))
        #print(A_matrix.shape)
        #print(recieved_symbols_pilot_indexed.shape)
        channel_resp = np.matmul(A_matrix,(recieved_symbols_pilot_indexed))
        channel_resp=channel_resp.squeeze()
        channel_freq_response = np.fft.fft(channel_resp,self.N)
        #print(channel_freq_response.shape)
        return channel_freq_response

    def get_least_squares_known_taps_estimate(self, recieved_symbols, A_matrix):
        rec_pilots = recieved_symbols[self.pilot_indexes].reshape((self.N_p, 1))
        
        # Get the 6 tap amplitudes
        tap_estimates = (A_matrix @ rec_pilots).squeeze()
        
        # Create a full time-domain array and place the taps at the CORRECT indices
        channel_resp_full = np.zeros(self.N, dtype=complex)
        channel_resp_full[self.channel_indexes] = tap_estimates
        
        # Now take the FFT
        channel_freq_response = np.fft.fft(channel_resp_full, self.N)
        return channel_freq_response

def plot_channel_estimation_performance():
    num_iterations = 100
    channelest = ChannelEstimation()
    pilot_symbols = channelest.create_pilot_symbols()
    Mse_zf_arr = []
    Mse_zf_linear_arr = []
    Mse_fft_interp_arr = []
    Mse_least_squares_arr = []
    Mse_fft_hann_window_arr = []
    Mse_least_sq_known_indexes_arr=[]
    pseudo_inv = channelest.create_A_matrix_for_mlsq_estimate(pilot_symbols)
    pseudo_inv_known_indexes = channelest.create_A_matrix_known_tap_locations(pilot_symbols)
    SNR_arr = np.arange(0,31,3)
    for SNR in SNR_arr:
        noise_var = 1/(10**(SNR/10)) 
        Mse_zf = 0
        Mse_zf_linear = 0
        Mse_fft_interp = 0
        Mse_least_squares = 0
        Mse_fft_hann = 0
        Mse_known_indexes = 0
        for i in range(num_iterations):
            channel = OFDM_Channel()
            noise = np.sqrt(noise_var/2)*(np.random.randn(len(pilot_symbols))+1j*np.random.randn(len(pilot_symbols)))
            recieved_symbols =  channel.channel_freq_response*pilot_symbols + noise
            H_zf = channelest.get_zf_estimate(pilot_symbols,recieved_symbols)
            H_zf_linear = channelest.get_zf_estimate_linear_interp(pilot_symbols,recieved_symbols)
            H_fft_interp = channelest.get_fft_interp_estimate(pilot_symbols,recieved_symbols)  
            H_least_squares = channelest.get_least_squares_estimate(recieved_symbols,pseudo_inv)
            H_fft_hann = channelest.get_fft_interp_hann_window_estimate(pilot_symbols,recieved_symbols)
            H_known_indexes = channelest.get_least_squares_known_taps_estimate(recieved_symbols,pseudo_inv_known_indexes)
            #print(np.mean(np.abs(channel.channel_freq_response)**2))
            Mse_zf_linear += channelest.get_mse(H_zf_linear,channel.channel_freq_response)
            Mse_fft_interp += channelest.get_mse(H_fft_interp,channel.channel_freq_response)
            Mse_least_squares += channelest.get_mse(H_least_squares,channel.channel_freq_response)
            Mse_fft_hann += channelest.get_mse(H_fft_hann,channel.channel_freq_response)
            Mse_known_indexes += channelest.get_mse(H_known_indexes,channel.channel_freq_response)
            # For zf at pilot locations
            temp_mse = 0
            for index in channelest.pilot_indexes:
                temp_mse += abs(H_zf[index]-channel.channel_freq_response[index])**2
            temp_mse/=len(channelest.pilot_indexes)
            Mse_zf += temp_mse            

        Mse_zf /= num_iterations
        Mse_zf_linear /= num_iterations
        Mse_fft_interp /= num_iterations
        Mse_least_squares /= num_iterations
        Mse_fft_hann /= num_iterations
        Mse_known_indexes /= num_iterations


        Mse_zf_arr.append(Mse_zf)
        Mse_zf_linear_arr.append(Mse_zf_linear)
        Mse_fft_interp_arr.append(Mse_fft_interp)
        Mse_least_squares_arr.append(Mse_least_squares)
        Mse_fft_hann_window_arr.append(Mse_fft_hann)
        Mse_least_sq_known_indexes_arr.append(Mse_known_indexes)
    
    MSE_zf_dB = 10*np.log10(np.array(Mse_zf_arr))
    MSE_zf_linear_dB = 10*np.log10(np.array(Mse_zf_linear_arr))
    MSE_fft_interp_dB = 10*np.log10(np.array(Mse_fft_interp_arr))
    MSE_least_squares_dB = 10*np.log10(np.array(Mse_least_squares_arr))
    MSE_fft_hann_window_dB = 10*np.log10(np.array(Mse_fft_hann_window_arr))
    Mse_least_sq_known_indexes_arr = 10*np.log10(np.array(Mse_least_sq_known_indexes_arr))

    plt.figure(figsize=(12, 6))
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.plot(SNR_arr,MSE_zf_dB,label="Zero Forcing")
    plt.plot(SNR_arr,MSE_zf_linear_dB,label="Linear Interpolation")
    plt.plot(SNR_arr,MSE_fft_interp_dB,label="FFT Interpolation")
    plt.plot(SNR_arr,MSE_least_squares_dB,label="Least Squares")
    plt.plot(SNR_arr,MSE_fft_hann_window_dB,label="FFT Interpolation with Hann Window")
    plt.plot(SNR_arr,Mse_least_sq_known_indexes_arr,label="Least Squares with known indexes")
    plt.legend(fontsize=12)
    plt.xlabel("SNR(dB)",fontsize=12)
    plt.ylabel("MSE(dB)",fontsize=12)
    plt.title("MSE vs SNR for channel estimation",fontsize=14,fontweight='bold')
    plt.savefig("./plots/MSE_vs_SNR_channel_estimation.png")
    plt.close()

def plot_mse_vs_SNR_freq_offset():
    transmitter = Transmitter()
    channel = OFDM_Channel()
    reciever = Receiver()

    J=100
    SNR_arr = np.arange(0,21,2)
    MSE_arr = []
    
    frequency_off = 0.5*channel.subcarrier_bandwidth # Set some constant frequency offset which is less than subcarrier bandwidth
    for SNR in SNR_arr:
        mse_sum = 0
        for j in range(J):
            transmit_frame = transmitter.create_transmit_frame()
            channel_output = channel.act(transmit_frame,frequency_off,SNR)
            recieved_symbols = reciever.schmidl_cox_correlation(channel_output)
            freq_offset_est = reciever.get_frequency_offset_est(recieved_symbols)
            mse_sum += (freq_offset_est-frequency_off)**2
            #print("Frequency offset is",frequency_off,"Estimate is",freq_offset_est)
        
        MSE_arr.append(mse_sum/J)
    
    MSE_arr = np.array(MSE_arr)
    MSE_dB_arr = 10*np.log10(MSE_arr)

    plt.figure(figsize=(12, 6))
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.plot(SNR_arr,MSE_dB_arr)
    plt.xlabel("SNR(dB)",fontsize=12)
    plt.ylabel("MSE(dB)",fontsize=12)
    plt.title("MSE vs SNR for frequency offset estimation",fontsize=14,fontweight='bold')
    plt.savefig("./plots/MSE_vs_SNR.png")
    plt.close()   

def plot_sc_time_sync():
    transmitter = Transmitter()
    channel = OFDM_Channel()
    reciever = Receiver()
    SNR = 6 # 6dB given in question

    frame1 = transmitter.create_transmit_frame()
    frame2 = transmitter.create_transmit_frame()
    transmitted_symbols = np.concatenate((frame1,frame2),axis=0,dtype=complex)
    noise_var = 1/(10**(SNR/10))
    #Channel 1 is such h=1 which is the transmitted symbols itself
    channel_1_noise = np.sqrt(noise_var/2)*(np.random.randn(len(transmitted_symbols))+1j*np.random.randn(len(transmitted_symbols)))
    channel_1_output = np.squeeze(transmitted_symbols) + channel_1_noise
    #print(channel_1_output.shape)
    channel_1_corr=reciever.schmidl_cox_correlation(channel_1_output)

    #Channel 2 is the given channel in Q1
    channel_2_output = channel.act(transmitted_symbols,0,SNR)
    channel_2_corr=reciever.schmidl_cox_correlation(channel_2_output)

    channel_1_corr_normalised = channel_1_corr/np.max(np.abs(channel_1_corr))
    channel_2_corr_normalised = channel_2_corr/np.max(np.abs(channel_2_corr))

    reciever.plot_correlation(channel_1_corr_normalised,label="Channel h=1")
    reciever.plot_correlation(channel_2_corr_normalised,label="Channel h in Q1")
    plt.show()



def main():
    #channel = OFDM_Channel()
    transmitter = Transmitter()
    channel = OFDM_Channel()
    freq_offset = 0.5*channel.subcarrier_bandwidth
    #print(freq_offset)
    #transmit_block1=transmitter.create_transmit_block()
    preamble_block,preamble_symbols = transmitter.create_preamble_block()
    #transmitter.plot_preamble(preamble_block,preamble_symbols)
    random_block = transmitter.create_random_transmit_block()
    transmitted_symbols = transmitter.create_transmit_frame()
    channel_output_symbols = channel.act(transmitted_symbols,freq_offset,SNR=0)
    #print(transmitted_symbols.shape)
    reciever = Receiver()
    corr_output = reciever.schmidl_cox_correlation(channel_output_symbols)
    #reciever.plot_correlation(corr_output)
    freq_offset_est = reciever.get_frequency_offset_est(corr_output)
    print(freq_offset)
    print(freq_offset_est)

    print("MSE is",(freq_offset-freq_offset_est)**2)
    #print(channel.subcarrier_bandwidth/3)

if __name__ == "__main__":
    plot_channel_estimation_performance()
    #plot_sc_time_sync()
    #plot_mse_vs_SNR_freq_offset()
    #main()