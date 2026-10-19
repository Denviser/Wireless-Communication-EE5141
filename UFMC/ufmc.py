#Charan Srikkanth - EE23B127
"""This code contains my workings for assignment 2 for the course EE5141-Fundamentals of Wireless and Cellular Communication"""
"""For Q1 uncomment the function plot_mse_vs_SNR_freq_offset()"""
"""For Q2 uncomment the function plot_sc_time_sync()"""
"""For Q3 uncomment the function plot_channel_estimation_performance()"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy.signal import find_peaks
#np.random.seed(42) # For reproducibility


class Parameters:
    def __init__(self,Nrb = 16):
        self.N= 512
        self.subcarrier_bandwidth = 10000 # 10 Khz
        self.entire_bandwidth = 5.12e6 # 5.12 Mhz
        self.sampling_rate = self.entire_bandwidth
        self.cyclic_prefix_duration = 6.25e-6 # 6.25 microseconds
        self.N_cp = int(self.cyclic_prefix_duration*self.sampling_rate)
        #print("Ncp is ",self.N_cp)
        #self.N_cp = 10
        self.num_blocks_per_frame = 100
        self.blk_duration_w_o_cyclic_prefix = 1/self.subcarrier_bandwidth
        self.block_duration = self.blk_duration_w_o_cyclic_prefix + self.cyclic_prefix_duration
        self.frame_duration = self.num_blocks_per_frame * self.block_duration
        #0th index represents -255 and 512 is 255 subcarrier
        self.valid_frequency_indexes = np.array([i for i in range(15,255)]+[i for i in range(256,496)])
        self.Nrb = Nrb
        self.valid_freq_per_rb = {}

        for i in range(self.Nrb):
            freq_per_rb = int(len(self.valid_frequency_indexes)/self.Nrb)
            self.valid_freq_per_rb[i] = self.valid_frequency_indexes[i*freq_per_rb:(i+1)*freq_per_rb]

        #print(self.valid_freq_per_rb)
        #self.all_frequency_indexes = np.array([i for i in range(-255,257)])

class OFDM_Channel(Parameters):
    def __init__(self):
        super().__init__()
        self.channel_response = np.zeros(27,dtype=complex)
        tap_delays = np.array([0,7,13,18,21,26])
        tap_gains_dB = np.array([0,-3,-3,-4,-9,-17])
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

    def create_random_transmit_block(self)->tuple[np.ndarray, np.ndarray]:
        transmit_symbols = np.zeros((self.N),dtype=complex)
        #print(transmit_symbols.shape)
        for index in self.valid_frequency_indexes:
            transmit_symbols[index] = 2*np.random.randint(2)-1 + 1j*(2*np.random.randint(2)-1) # Generate QPSK symbols
            transmit_symbols[index] *= 1/np.sqrt(2) #Normalise to unit energy 
        #print(transmit_symbols)
        
        #Shifting so that index 0 is 0 freq
        transmit_block_shifted = np.fft.ifftshift(transmit_symbols)

        time_domain_N = np.fft.ifft(transmit_block_shifted, self.N, norm='ortho')
        
        # 4. Extract and attach the Cyclic Prefix (CP)
        # Grab the last N_cp samples from the time-domain signal
        cyclic_prefix = time_domain_N[-self.N_cp:]
        
        # Concatenate the CP to the front of the signal
        transmit_block = np.concatenate((cyclic_prefix, time_domain_N))
        #transmit_block = np.matmul(self.IDFT_matrix,transmit_symbols)
        return transmit_block , transmit_symbols

    # def create_pilot_block(self)->tuple[np.ndarray,np.ndarray]:
    #     pilot_symbols = np.zeros((self.N,1),dtype=complex)
    #     only_pilots_arr = []
    #     for freq_index in range(15,496):
    #         if freq_index%8 == 7: # Pilot is placed every 8 subcarriers
    #             if freq_index == 255: # Pilot is placed every 8 subcarriers
    #                 continue
    #             pilot = 2*np.random.randint(2)-1 + 1j*(2*np.random.randint(2)-1) # Generate QPSK symbols
    #             pilot *= 1/np.sqrt(2) #Normalise to unit energy 
    #             only_pilots_arr.append(pilot)
    #             pilot_symbols[freq_index,0] = pilot
    #         else:
    #             pilot_symbols[freq_index,0] = 2*np.random.randint(2)-1 + 1j*(2*np.random.randint(2)-1) # Generate QPSK symbols
    #             pilot_symbols[freq_index,0] *= 1/np.sqrt(2) #Normalise to unit energy

    #     only_pilots_arr = np.array(only_pilots_arr)
    #     pilot_block = np.matmul(self.IDFT_matrix,pilot_symbols)
    #     return pilot_block,only_pilots_arr
    
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
        preamble_symbols = np.zeros((self.N),dtype=complex)
        #For the schmidl cox we need the time domain to be symmetric for this the frequency domain should have alternate symbols.
        for index in range(0,self.N,2):
            preamble_symbols[index] = 2*np.random.randint(2)-1 + 1j*(2*np.random.randint(2)-1) # Generate QPSK symbols
            preamble_symbols[index] *= 1/np.sqrt(2) #Normalise to unit energy 
        
        preamble_block_shifted = np.fft.ifftshift(preamble_symbols)
        time_domain_N = np.fft.ifft(preamble_block_shifted, self.N, norm='ortho')
        preamble_block = np.concatenate((time_domain_N[-self.N_cp:],time_domain_N))
        return preamble_block,preamble_symbols
    
    def create_transmit_frame(self):
        transmit_frame = np.zeros((self.num_blocks_per_frame*(self.N+self.N_cp)),dtype=complex)
        transmit_freq_domain = np.zeros((self.num_blocks_per_frame*self.N),dtype=complex)
        first_preamble_frame,first_preamble_freq_symbols = self.create_preamble_block()
        transmit_frame[0:self.N+self.N_cp] = first_preamble_frame
        transmit_freq_domain[0:self.N] = first_preamble_freq_symbols
        for i in range(1,self.num_blocks_per_frame):
            random_block,random_freq_symbols = self.create_random_transmit_block()
            transmit_frame[i*(self.N+self.N_cp):(i+1)*(self.N+self.N_cp)] = random_block
            transmit_freq_domain[i*self.N:(i+1)*self.N] = random_freq_symbols

        return transmit_frame,transmit_freq_domain


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

    def calculate_ser_ofdm(self, transmitted_symbols, received_symbols):
        """Calculates the Symbol Error Rate (SER) for QPSK."""
        errors = 0
        for i in range(len(transmitted_symbols)):
            tx_real = np.sign(np.real(transmitted_symbols[i]))
            tx_imag = np.sign(np.imag(transmitted_symbols[i]))
            rx_real = np.sign(np.real(received_symbols[i]))
            rx_imag = np.sign(np.imag(received_symbols[i]))
            
            if (tx_real != rx_real) or (tx_imag != rx_imag):
                errors += 1

        return errors / len(transmitted_symbols)

    def demodulate_and_calculate_ser_for_frame(self, all_transmitted_true_symbols, received_frame):
        """
        Calculates the SER for an entire OFDM frame.
        """
        block_length = self.N + self.N_cp
        total_ser = 0
        
        # Start loop at 1 because index 0 is the Schmidl-Cox preamble!
        for i in range(1,self.num_blocks_per_frame):
            start_idx = i * block_length
            end_idx = start_idx + block_length
            
            # 1. Extract the time-domain block
            received_time_block = received_frame[start_idx:end_idx]
            
            # 2. Strip the Cyclic Prefix (CP)
            # The CP is at the beginning, so we keep only the last N samples
            received_time_no_cp = received_time_block[self.N_cp:]
            
            # 3. Take the FFT
            # Because you used ifftshift at the transmitter, you must fftshift here!
            rec_freq_block = np.fft.fftshift(np.fft.fft(received_time_no_cp, self.N, norm='ortho'))
            
            # 4. Extract active subcarriers
            # true_symbols list is 0-indexed for the data blocks, so we use i-1
            true_symbols_block = all_transmitted_true_symbols[i*self.N:(i+1)*self.N] 
            
            rec_symbols_active = rec_freq_block[self.valid_frequency_indexes]
            true_symbols_active = true_symbols_block[self.valid_frequency_indexes]
            
            # 5. Calculate SER
            block_ser = self.calculate_ser_ofdm(true_symbols_active, rec_symbols_active)
            total_ser += block_ser
            
        overall_frame_ser = total_ser / (self.num_blocks_per_frame - 1)  # Subtract 1 for the preamble block
        
        return overall_frame_ser
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

class UFMCTransmitter(Parameters):
    def __init__(self,Nrb = 16):
        super().__init__(Nrb = Nrb)


    def create_random_transmit_block_rb(self, rb_index):
        transmit_symbols = np.zeros(self.N, dtype=complex)
        
        for index in self.valid_freq_per_rb[rb_index]:
            # Generate QPSK symbols
            transmit_symbols[index] = 2*np.random.randint(2)-1 + 1j*(2*np.random.randint(2)-1) 
            # Normalize to unit energy
            transmit_symbols[index] *= 1/np.sqrt(2) 
            
        # 1. Perform IFFT here!
        # Flatten to 1D to ensure clean array operations with numpy
        #shifting as I have indexed 0 at 256
        transmit_symbols_freq_shifted = np.fft.ifftshift(transmit_symbols)
        transmit_symbols_time = np.fft.ifft(transmit_symbols_freq_shifted, self.N)
        
        # 2. Pass the TIME-DOMAIN symbols directly to the filter function
        filtered_symbols = self.convolve_filter_rb(rb_index, transmit_symbols_time)
        return filtered_symbols,transmit_symbols

    def convolve_filter_rb(self, rb_index, transmit_symbols_time):
        """
        Takes in the time-domain symbols of ONE resource block and 
        convolves it with a frequency-shifted Chebyshev filter.
        """
        # 1. Design the Base Prototype Filter
        L_filter = self.N_cp
        
        # Create a Chebyshev window with 120 dB stop-band attenuation
        base_filter = signal.windows.chebwin(L_filter, at=120)
        
        # Normalize the filter energy 
        base_filter = base_filter / np.linalg.norm(base_filter)
        
        # 2. Find the Center Frequency of this RB
        rb_bins = self.valid_freq_per_rb[rb_index]
        # As my freq array is from -fs/2 to fs/2 but numpy uses 0 to fs
        center_bin = np.mean(rb_bins) - self.N/2
            
        # 3. Frequency-Shift the Filter
        t = np.arange(L_filter)
        shifted_filter = base_filter * np.exp(1j * 2 * np.pi * center_bin * t / self.N)
        
        filter_freq_domain = np.fft.fft(shifted_filter, self.N)
        transmit_symbols_freq_domain = np.fft.fft(transmit_symbols_time, self.N)

        matched_filter = np.conj(shifted_filter[::-1])
        matched_filter_freq_domain = np.fft.fft(matched_filter, self.N)

        combined = np.convolve(shifted_filter,matched_filter,mode='full')
        combined_freq_domain = np.fft.fft(combined,self.N)

        plt.figure(figsize=(12, 6))
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.plot(np.abs(filter_freq_domain), label='Filter Frequency Domain')
        plt.plot(np.abs(transmit_symbols_freq_domain), label='Transmit Symbols Frequency Domain')
        plt.plot(np.abs(matched_filter_freq_domain), label='Matched Filter Frequency Domain')
        plt.plot(np.abs(combined_freq_domain), label='Combined Filter-Matched Frequency Domain', linestyle='--')
        plt.legend()
        plt.xlabel('Frequency Index')
        plt.ylabel('Magnitude')
        plt.title('Filter and Transmit Symbols Frequency Domains')
        plt.show()
        # 4. Perform the Linear Convolution
        # Mode='full' ensures we keep the entire smeared signal
        filtered_symbols_time = np.convolve(transmit_symbols_time, shifted_filter, mode='full')
        
        return filtered_symbols_time

    def create_entire_time_block(self):
        time_block = np.zeros(self.N + self.N_cp - 1, dtype=complex)
        freq_sent_block = np.zeros(self.N, dtype=complex)
        for i in range(self.Nrb):
            time_out, freq_out = self.create_random_transmit_block_rb(i)
            time_block += time_out
            freq_sent_block += freq_out
        return time_block,freq_sent_block

    def create_transmit_frame(self):
        """
        Creates a tightly-packed time-domain frame consisting of multiple UFMC blocks.
        """
        # Your logic: The exact length of a UFMC block
        block_length = self.N + self.N_cp - 1
        
        # Your exact total length calculation
        total_length = block_length * self.num_blocks_per_frame
        
        transmit_frame = np.zeros(total_length, dtype=complex)
        all_freq_symbols = []
        
        for i in range(self.num_blocks_per_frame):
            time_block, freq_block = self.create_entire_time_block()
            
            # The start index shifts by the exact block length each time
            start_idx = i * block_length
            end_idx = start_idx + block_length
            
            # Insert the time-domain block perfectly back-to-back
            transmit_frame[start_idx:end_idx] = time_block
            
            all_freq_symbols.append(freq_block)
            
        return transmit_frame, all_freq_symbols
    
class UFMCReciever(Parameters):
    def __init__(self,Nrb = 16):
        super().__init__(Nrb = Nrb)

    def receive_filter_and_fft_rb(self, rb_index, received_symbols_time_smeared):
        """
        Takes the incoming N+L-1 smeared time-domain signal,
        applies the matched filter to collapse it to N points,
        and takes the FFT.
        """
        # 1. Recreate the EXACT same shifted filter used at the Transmitter
        L_filter = self.N_cp 
        base_filter = signal.windows.chebwin(L_filter, at=120)
        base_filter = base_filter / np.linalg.norm(base_filter)
        
        rb_bins = self.valid_freq_per_rb[rb_index]
        center_bin = np.mean(rb_bins) - (self.N / 2) # Assuming your ifftshift fix is applied!
        
        t = np.arange(L_filter)
        shifted_filter = base_filter * np.exp(1j * 2 * np.pi * center_bin * t / self.N)
        
        # 2. Create the MATCHED Filter (Time-reversed and Complex Conjugated)
        matched_filter = np.conj(shifted_filter[::-1])
        
        # 3. Convolve with mode='valid'
        # This mathematically crushes the N+L-1 array down to exactly N points
        filtered_time_N_points = np.convolve(received_symbols_time_smeared.flatten(), matched_filter, mode='valid')
        
        # 4. Now that we have exactly N points, it is perfectly safe to take the FFT!
        # Remember to fftshift to get your centered DC back!
        rb_frequency_symbols = np.fft.fftshift(np.fft.fft(filtered_time_N_points, self.N))
        
        return rb_frequency_symbols

    def calculate_ser(self, transmitted_symbols, recieved_symbols, rb_index):
        """Calculates the Symbol Error Rate (SER) for QPSK."""
        assert len(transmitted_symbols) == len(recieved_symbols)

        errors = 0
        for i in range(len(transmitted_symbols)):
            # For QPSK, check which quadrant the symbol is in based on the signs.
            tx_real_sign = np.sign(np.real(transmitted_symbols[i]))
            tx_imag_sign = np.sign(np.imag(transmitted_symbols[i]))
            rx_real_sign = np.sign(np.real(recieved_symbols[i]))
            rx_imag_sign = np.sign(np.imag(recieved_symbols[i]))
            
            # If EITHER the real or imaginary part flipped, the entire symbol is wrong.
            if (tx_real_sign != rx_real_sign) or (tx_imag_sign != rx_imag_sign):
                errors += 1

        # Calculate SER: Total errors divided by total SYMBOLS (not bits)
        return errors / len(transmitted_symbols)

    def demodulate_and_calculate_ser_for_all_rb(self, transmitted_true_symbols, received_time_smeared):
        """
        Loops through all Resource Blocks, isolates them using the matched filter bank,
        extracts the active subcarriers, and calculates the overall SER.
        """
        total_ser = 0
        
        for rb_index in range(self.Nrb):
            # 1. Run the receive filter and FFT for this specific RB
            rec_freq_block = self.receive_filter_and_fft_rb(rb_index, received_time_smeared)
            
            # 2. Extract ONLY the active subcarriers for this RB
            active_bins = self.valid_freq_per_rb[rb_index]
            
            true_symbols_rb = transmitted_true_symbols[active_bins]
            rec_symbols_rb = rec_freq_block[active_bins]
            
            # 3. Calculate SER for this specific RB
            ser_rb = self.calculate_ser(true_symbols_rb, rec_symbols_rb, rb_index)
            total_ser += ser_rb
            
        # 4. Average the SER across all resource blocks
        overall_ser = total_ser / self.Nrb
        return overall_ser
    
    def demodulate_and_calculate_ser_for_frame(self, all_transmitted_true_symbols, received_frame):
        """
        Partitions a continuous tightly-packed time-domain frame into individual UFMC blocks,
        demodulates each block, and calculates the overall average SER for the frame.
        """
        # The exact length of a single UFMC block used in tight-packing
        block_length = self.N + self.N_cp - 1
        total_ser = 0
        
        for i in range(self.num_blocks_per_frame):
            # 1. Slice out the specific time-domain block for this iteration
            start_idx = i * block_length
            end_idx = start_idx + block_length
            received_time_block = received_frame[start_idx:end_idx]
            
            # 2. Grab the corresponding true frequency symbols for this block
            # (Remember, the transmitter returns this as a list of arrays!)
            true_symbols_block = all_transmitted_true_symbols[i]
            
            # 3. Use your existing function to calculate the SER across all RBs for this block
            block_ser = self.demodulate_and_calculate_ser_for_all_rb(true_symbols_block, received_time_block)
            
            total_ser += block_ser
            
        # 4. Average the SER across all the blocks in the frame
        overall_frame_ser = total_ser / self.num_blocks_per_frame
        
        return overall_frame_ser
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

def plot_ser_vs_snr():
    p = Parameters(Nrb=32)
    trans_ufmc = UFMCTransmitter(Nrb=32)
    rec_ufmc = UFMCReciever(Nrb=32)
    ofdm_trans = Transmitter()
    ofdm_rec = Receiver()
    
    print("Generating Transmission Frames...")
    # 1. Generate Transmit Frames ONCE to save simulation time
    time_domain_transmit_frame_ufmc, freq_domain_transmit_frame_ufmc = trans_ufmc.create_transmit_frame()
    time_domain_transmit_frame_ofdm, freq_domain_transmit_frame_ofdm = ofdm_trans.create_transmit_frame()
    
    # 2. Calculate true Signal Power ONCE
    signal_power_ufmc = np.mean(np.abs(time_domain_transmit_frame_ufmc)**2)
    signal_power_ofdm = np.mean(np.abs(time_domain_transmit_frame_ofdm)**2)
    
    # 3. Setup SNR array and tracking lists
    snr_db_range = np.arange(-2, 10, 2) # From 0 to 30 (inclusive) in steps of 2
    ser_ufmc_list = []
    ser_ofdm_list = []
    
    print(f"Simulating SNR from {snr_db_range[0]} dB to {snr_db_range[-1]} dB...")
    
    for snr in snr_db_range:
        # --- UFMC Processing ---
        noise_var_ufmc = signal_power_ufmc / (10**(snr/10))
        noise_ufmc = np.sqrt(noise_var_ufmc/2) * (np.random.randn(len(time_domain_transmit_frame_ufmc)) + 1j*np.random.randn(len(time_domain_transmit_frame_ufmc)))
        channel_output_ufmc = time_domain_transmit_frame_ufmc + noise_ufmc
        ser_ufmc = rec_ufmc.demodulate_and_calculate_ser_for_frame(freq_domain_transmit_frame_ufmc, channel_output_ufmc)
        ser_ufmc_list.append(ser_ufmc)
        
        # --- OFDM Processing ---
        noise_var_ofdm = signal_power_ofdm / (10**(snr/10))
        noise_ofdm = np.sqrt(noise_var_ofdm/2) * (np.random.randn(len(time_domain_transmit_frame_ofdm)) + 1j*np.random.randn(len(time_domain_transmit_frame_ofdm)))
        channel_output_ofdm = time_domain_transmit_frame_ofdm + noise_ofdm
        ser_ofdm = ofdm_rec.demodulate_and_calculate_ser_for_frame(freq_domain_transmit_frame_ofdm, channel_output_ofdm)
        ser_ofdm_list.append(ser_ofdm)
        
        print(f"SNR: {snr:2d} dB | OFDM SER: {ser_ofdm:.5f} | UFMC SER: {ser_ufmc:.5f}")

    # 4. Plot the Results
    plt.figure(figsize=(10, 6))
    
    # Use semilogy for standard telecom error rate curves
    plt.semilogy(snr_db_range, ser_ofdm_list, 'b.-', label='OFDM', markersize=12, linewidth=2)
    plt.semilogy(snr_db_range, ser_ufmc_list, 'r.-', label='UFMC', markersize=12, linewidth=2)
    
    plt.title('Symbol Error Rate (SER) vs SNR (AWGN Channel)', fontsize=14, fontweight='bold')
    plt.xlabel('SNR (dB)', fontsize=12)
    plt.ylabel('Symbol Error Rate (SER)', fontsize=12)
    
    # Add a nice grid that handles the log scale well
    plt.grid(True, which="both", ls="--", alpha=0.6)
    plt.legend(fontsize=12)
    
    # Save or show
    plt.tight_layout()
    plt.savefig("ser_vs_snr.png")

def plot_freq_response_of_one_rb(Nrb=32):
    p = Parameters(Nrb=Nrb)
    trans_ufmc = UFMCTransmitter(Nrb=Nrb)
    rb_index = 12
    # 1. Get UFMC time block and the raw frequency symbols
    time_block_ufmc, freq_block = trans_ufmc.create_random_transmit_block_rb(rb_index)
    
    # 2. Manually generate the OFDM time block from the exact same symbols
    # Shift to standard order, take IFFT, and attach Cyclic Prefix
    freq_block_shifted = np.fft.ifftshift(freq_block.flatten())
    time_block_ofdm_raw = np.fft.ifft(freq_block_shifted, p.N, norm='ortho')
    time_block_ofdm = np.concatenate((time_block_ofdm_raw[-p.N_cp:], time_block_ofdm_raw))
    
    # 3. HIGH-RESOLUTION FFT (Zero-Padding)
    # This is required to see the leakage between the discrete subcarriers!
    N_fft = 1024
    
    # Flatten arrays, pad to N_fft, FFT, and shift DC to center
    ufmc_fft = np.fft.fftshift(np.fft.fft(time_block_ufmc.flatten(), N_fft))
    ofdm_fft = np.fft.fftshift(np.fft.fft(time_block_ofdm.flatten(), N_fft))
    
    # 4. Normalize and convert to dB (with log(0) protection!)
    epsilon = 1e-15
    
    # Normalize the arrays first
    ufmc_norm = np.abs(ufmc_fft) / np.max(np.abs(ufmc_fft))
    ofdm_norm = np.abs(ofdm_fft) / np.max(np.abs(ofdm_fft))
    
    # Use np.maximum to floor the values at epsilon before taking the log
    ufmc_psd = 20 * np.log10(np.maximum(ufmc_norm, epsilon))
    ofdm_psd = 20 * np.log10(np.maximum(ofdm_norm, epsilon))
    # Create an X-axis that maps back to standard subcarrier indices (-256 to 256)
    freq_axis = np.linspace(-p.N/2, p.N/2, N_fft)
    
    # 5. Plotting
    plt.figure(figsize=(10, 6))
    
    # Plot OFDM first so UFMC sits on top of it
    plt.plot(freq_axis, ofdm_psd, 'b-', label='OFDM (Standard CP)', alpha=0.7)
    plt.plot(freq_axis, ufmc_psd, 'r-', label='UFMC (Chebyshev Filter)', linewidth=1.5)
    
    plt.title(f'Spectrum Comparison: Out-of-Band Leakage (Nrb = {Nrb}) (RB {rb_index})')
    plt.xlabel('Subcarrier Index')
    plt.ylabel('Normalized Magnitude of FFT (dB)')
    
    # Set the Y-axis to cut off the deep noise floor so the slopes are obvious
    plt.ylim(-100, 5)
    
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.show()

def plot_ser_vs_snr_multi_nrb():
    # 1. Define the Nrb values you want to test
    nrb_values_to_test = [16, 32, 64 , 128] 
    snr_db_range = np.arange(0, 10, 2)
    
    # 2. Run the OFDM Baseline (Only need to do this once!)
    print("Generating OFDM Baseline...")
    ofdm_trans = Transmitter()
    ofdm_rec = Receiver()
    time_ofdm, freq_ofdm = ofdm_trans.create_transmit_frame()
    power_ofdm = np.mean(np.abs(time_ofdm)**2)
    
    ser_ofdm_list = []
    for snr in snr_db_range:
        noise_var = power_ofdm / (10**(snr/10))
        noise = np.sqrt(noise_var/2) * (np.random.randn(len(time_ofdm)) + 1j*np.random.randn(len(time_ofdm)))
        rx_ofdm = time_ofdm + noise
        ser = ofdm_rec.demodulate_and_calculate_ser_for_frame(freq_ofdm, rx_ofdm)
        ser_ofdm_list.append(ser)

    # 3. Setup the plot
    plt.figure(figsize=(10, 6))
    plt.semilogy(snr_db_range, ser_ofdm_list, 'k--', label='OFDM Baseline', linewidth=2.5)

    # 4. Loop through the different Nrb values for UFMC
    colors = ['r', 'b', 'g', 'm', 'c'] # Colors for different Nrb lines
    
    for idx, current_nrb in enumerate(nrb_values_to_test):
        print(f"\nSimulating UFMC with Nrb = {current_nrb}...")
        
        # Initialize UFMC blocks with the specific Nrb
        ufmc_trans = UFMCTransmitter(Nrb=current_nrb)
        ufmc_rec = UFMCReciever(Nrb=current_nrb)
        
        time_ufmc, freq_ufmc = ufmc_trans.create_transmit_frame()
        power_ufmc = np.mean(np.abs(time_ufmc)**2)
        
        ser_ufmc_list = []
        for snr in snr_db_range:
            noise_var = power_ufmc / (10**(snr/10))
            noise = np.sqrt(noise_var/2) * (np.random.randn(len(time_ufmc)) + 1j*np.random.randn(len(time_ufmc)))
            rx_ufmc = time_ufmc + noise
            ser = ufmc_rec.demodulate_and_calculate_ser_for_frame(freq_ufmc, rx_ufmc)
            ser_ufmc_list.append(ser)
            
            print(f"  SNR: {snr:2d} dB | UFMC (Nrb={current_nrb}) SER: {ser:.5f}")
            
        # Plot this specific Nrb curve
        plt.semilogy(snr_db_range, ser_ufmc_list, marker='o', color=colors[idx % len(colors)], 
                     label=f'UFMC (Nrb = {current_nrb})', linewidth=2)

    # 5. Finalize Plotting
    plt.title('SER vs SNR for OFDM and UFMC (Varying $N_{RB}$)', fontsize=14, fontweight='bold')
    plt.xlabel('SNR (dB)', fontsize=12)
    plt.ylabel('Symbol Error Rate (SER)', fontsize=12)
    plt.grid(True, which="both", ls="--", alpha=0.6)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig("ser_vs_snr_multi_nrb.png")
    print("\nSimulation complete! Plot saved as 'ser_vs_snr_multi_nrb.png'")

def plot_ser_vs_snr_multi_nrb_with_cfo(cfo_value=0.1):
    # 1. Define the Nrb values you want to test
    nrb_values_to_test = [16, 32, 64 , 128] 
    snr_db_range = np.arange(0, 22, 2)
    # 2. Run the OFDM Baseline (Only need to do this once!)
    print("Generating OFDM Baseline...")
    ofdm_trans = Transmitter()
    ofdm_rec = Receiver()
    time_ofdm, freq_ofdm = ofdm_trans.create_transmit_frame()
    power_ofdm = np.mean(np.abs(time_ofdm)**2)
    #time_ofdm = time_ofdm * cfo

    # Calculate exactly one block length
    block_len_ofdm = ofdm_trans.N + ofdm_trans.N_cp
    
    # Generate CFO for just ONE block
    cfo_block_ofdm = np.exp(1j * 2 * np.pi * (cfo_value / ofdm_trans.N) * np.arange(block_len_ofdm))
    
    # Tile (repeat) it across the entire frame
    cfo_ofdm = np.tile(cfo_block_ofdm, ofdm_trans.num_blocks_per_frame)
    time_ofdm = time_ofdm * cfo_ofdm

    ser_ofdm_list = []
    for snr in snr_db_range:
        noise_var = power_ofdm / (10**(snr/10))
        noise = np.sqrt(noise_var/2) * (np.random.randn(len(time_ofdm)) + 1j*np.random.randn(len(time_ofdm)))
        rx_ofdm = time_ofdm + noise
        ser = ofdm_rec.demodulate_and_calculate_ser_for_frame(freq_ofdm, rx_ofdm)
        ser_ofdm_list.append(ser)

    # 3. Setup the plot
    plt.figure(figsize=(10, 6))
    plt.semilogy(snr_db_range, ser_ofdm_list, 'k--', label='OFDM Baseline', linewidth=2.5)

    # 4. Loop through the different Nrb values for UFMC
    colors = ['r', 'b', 'g', 'm', 'c'] # Colors for different Nrb lines
    
    for idx, current_nrb in enumerate(nrb_values_to_test):
        print(f"\nSimulating UFMC with Nrb = {current_nrb}...")
        
        # Initialize UFMC blocks with the specific Nrb
        ufmc_trans = UFMCTransmitter(Nrb=current_nrb)
        ufmc_rec = UFMCReciever(Nrb=current_nrb)
        
        time_ufmc, freq_ufmc = ufmc_trans.create_transmit_frame()
        power_ufmc = np.mean(np.abs(time_ufmc)**2)
        #cfo = np.exp(1j*2*np.pi*(cfo_value/ufmc_trans.N)*np.arange(len(time_ufmc)))
        #time_ufmc = time_ufmc * cfo
        # Calculate exactly one block length
        block_len_ofdm = ofdm_trans.N + ofdm_trans.N_cp - 1
        
        # Generate CFO for just ONE block
        cfo_block_ofdm = np.exp(1j * 2 * np.pi * (cfo_value / ofdm_trans.N) * np.arange(block_len_ofdm))
        
        # Tile (repeat) it across the entire frame
        cfo_ofdm = np.tile(cfo_block_ofdm, ofdm_trans.num_blocks_per_frame)
        time_ufmc = time_ufmc * cfo_ofdm
            
        
        ser_ufmc_list = []
        for snr in snr_db_range:
            noise_var = power_ufmc / (10**(snr/10))
            noise = np.sqrt(noise_var/2) * (np.random.randn(len(time_ufmc)) + 1j*np.random.randn(len(time_ufmc)))
            rx_ufmc = time_ufmc + noise
            ser = ufmc_rec.demodulate_and_calculate_ser_for_frame(freq_ufmc, rx_ufmc)
            ser_ufmc_list.append(ser)
            
            print(f"  SNR: {snr:2d} dB | UFMC (Nrb={current_nrb}) SER: {ser:.5f}")
            
        # Plot this specific Nrb curve
        plt.semilogy(snr_db_range, ser_ufmc_list, marker='o', color=colors[idx % len(colors)], 
                     label=f'UFMC (Nrb = {current_nrb})', linewidth=2)

    # 5. Finalize Plotting
    plt.title(f'SER vs SNR for OFDM and UFMC (Varying $N_{{RB}}$)  with CFO = {cfo_value*100:.0f}%', fontsize=14, fontweight='bold')
    plt.xlabel('SNR (dB)', fontsize=12)
    plt.ylabel('Symbol Error Rate (SER)', fontsize=12)
    plt.grid(True, which="both", ls="--", alpha=0.6)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig("ser_vs_snr_multi_nrb_with_cfo.png")
    print("\nSimulation complete! Plot saved as 'ser_vs_snr_multi_nrb.png'")




def plot_window_comparison_for_ppt():
    # --- 1. System Parameters ---
    N = 256  # Length of the Rectangular window (e.g., OFDM symbol length)
    L = 101   # Length of the Chebyshev filter (e.g., Cyclic Prefix length)
    attenuation_dB = 120 # Chebyshev stop-band attenuation
    
    # --- 2. Generate Time-Domain Signals ---
    # The Rectangular Window
    rect_window = np.ones(N)
    
    # The Chebyshev Filter
    cheb_filter = signal.windows.chebwin(L, at=attenuation_dB)
    # Normalize the filter so it doesn't amplify the signal's overall energy
    cheb_filter /= np.sum(cheb_filter) 
    
    # The Convolved Signal (Rect * Chebyshev)
    # mode='full' gives us the smeared N + L - 1 length array
    convolved_time = np.convolve(rect_window, cheb_filter, mode='full')
    
    # --- 3. Generate Frequency Responses (Zero-Padded) ---
    N_fft = 8192 # Massive zero-padding for extremely smooth high-res curves
    
    rect_fft = np.fft.fftshift(np.fft.fft(rect_window, N_fft))
    convolved_fft = np.fft.fftshift(np.fft.fft(convolved_time, N_fft))
    
    # Normalize and convert to dB (using the 1e-15 epsilon trick to prevent log(0) warnings)
    epsilon = 1e-15
    rect_psd = 20 * np.log10(np.maximum(np.abs(rect_fft) / np.max(np.abs(rect_fft)), epsilon))
    convolved_psd = 20 * np.log10(np.maximum(np.abs(convolved_fft) / np.max(np.abs(convolved_fft)), epsilon))
    
    # Create a normalized frequency axis from -0.5 to 0.5
    freq_axis = np.linspace(-0.5, 0.5, N_fft)
    
    # --- 4. Plotting for Presentation ---
    # Create a large figure to ensure it looks crisp on a slide
    fig = plt.figure(figsize=(14, 9))
    
    # Top Left: Rectangular Time Domain
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(rect_window, 'b-', linewidth=3)
    ax1.set_title("Time Domain: Rectangular Window", fontsize=16, fontweight='bold')
    ax1.set_xlabel("Sample Index", fontsize=14)
    ax1.set_ylabel("Amplitude", fontsize=14)
    ax1.set_ylim(0, 1.2)
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Top Right: Convolved Time Domain
    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(convolved_time, 'r-', linewidth=3)
    ax2.set_title("Time Domain: Rect convolved with Chebyshev", fontsize=16, fontweight='bold')
    ax2.set_xlabel("Sample Index", fontsize=14)
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Bottom: Frequency Response Comparison
    ax3 = plt.subplot(2, 1, 2)
    ax3.plot(freq_axis, rect_psd, 'b-', label='Rectangular (Sinc Response)', alpha=0.7, linewidth=2.5)
    ax3.plot(freq_axis, convolved_psd, 'r-', label='Rect * Chebyshev', linewidth=2.5)
    
    ax3.set_title("Frequency Response Comparison", fontsize=16, fontweight='bold')
    ax3.set_xlabel("Normalized Frequency", fontsize=14)
    ax3.set_ylabel("Magnitude (dB)", fontsize=14)
    
    # Zoom in to see the main lobe and the first few side lobes clearly
    ax3.set_xlim(-0.05, 0.05) 
    ax3.set_ylim(-140, 5)
    
    ax3.grid(True, which="both", linestyle='--', alpha=0.7)
    ax3.legend(fontsize=14, loc='upper right')
    
    plt.tight_layout()
    
    # Save a high-resolution version for the PowerPoint
    plt.savefig("window_comparison_slide.png", dpi=300, bbox_inches='tight')
    plt.show()

def test_chebhshev_filter():
    N_rb = 4
    p = Parameters(Nrb=N_rb)
    print(p.valid_freq_per_rb)
    ufmc_trans = UFMCTransmitter(Nrb=N_rb)
    rb_index = 3
    time_block_ufmc, freq_block = ufmc_trans.create_random_transmit_block_rb(rb_index)
    return
# Run the function
#if __name__ == "__main__":
    #plot_window_comparison_for_ppt()
    #test_chebhshev_filter()
if __name__ == "__main__":
    #plot_ser_vs_snr_multi_nrb()
    #plot_ser_vs_snr()
    #plot_freq_response_of_one_rb(16)
    #plot_ser_vs_snr_multi_nrb_with_cfo(cfo_value=0.15)
    #test_chebhshev_filter()
    p = Parameters(Nrb=32)
    print(len(p.valid_frequency_indexes)/128)