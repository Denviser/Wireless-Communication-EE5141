%% Charan Srikkanth - EE23B127 question 1
Ts = 1e-5;
fs=1/Ts;
fc = 2e9;
fd = (30/ 3e8)* fc;  % velocity is 30 m/s
N = 8000;
time = (0:N-1)*Ts;
%% Part a
order = 100; % number of coefficients of the iir filter
k = 0:order;
R = besselj(0,2*pi*fd*k*Ts); %Autocorrelation vector
R(1)=R(1)+1e-6; %Add a small amount in order to invert matrix easily
[filter_coeffs,noise_var] = levinson(R,order-1);

noise_real = randn(1, N);
noise_imag = randn(1, N);
noise = sqrt(noise_var/2) * (noise_real + 1i * noise_imag);

channel_output_levinson = filter(1,filter_coeffs,noise);
[auto_corr_lev,lags] = xcorr((channel_output_levinson));
[cross_corr_lev,lags] = xcorr(real(channel_output_levinson),imag(channel_output_levinson));

lag_times = lags*Ts;
theory_acf = besselj(0, 2*pi*fd*lag_times); % J0(2*pi*fd*n*Ts)


signal_power = auto_corr_lev(lags == 0); 
% Normalise the correlation by power
auto_corr_lev = auto_corr_lev./signal_power;
cross_corr_lev = cross_corr_lev./signal_power;

figure;
plot(time, abs(channel_output_levinson), 'b', 'LineWidth', 1.2);
title('Time-Selective Fading Envelope (Levinson Algorithm)');
xlabel('Time (seconds)');
ylabel('Channel Magnitude |h(t)|');
grid on;
%exportgraphics(gcf, 'fading_lev.pdf', 'ContentType', 'vector');

figure;
plot(lags, real(auto_corr_lev), 'r', 'LineWidth', 1.2);
hold on;
plot(lags, theory_acf, 'k--', 'LineWidth', 1.2);
hold off;
title('real part of auto correlation(Levinson)');
xlabel('Lag');
ylabel('auto correlation');
legend('Estimated', 'Theoretical J_0(2\pi f_d \tau)');
grid on;
%exportgraphics(gcf, 'auto_real_lev.pdf', 'ContentType', 'vector');


figure;
plot(lags, imag(auto_corr_lev), 'r', 'LineWidth', 1.2);
title('imaginary part of Auto correlation(Levinson)');
xlabel('Lag');
ylabel('auto correlation');
grid on;
%exportgraphics(gcf, 'auto_imag_lev.pdf', 'ContentType', 'vector');

figure;
plot(lags, cross_corr_lev, 'r', 'LineWidth', 1.2);
title('Cross correlation between real and imaginary(Levinson)');
xlabel('Lag');
ylabel('cross correlation');
grid on;
%exportgraphics(gcf, 'cross_lev.pdf', 'ContentType', 'vector');


%% Smiths Model
N=8192; %These samples go from -fs/2 to fs/2-fs/N
time = (0:N-1)*Ts;
noise_var_time_domain = 1 ;
freq_domain_noise_var = N*noise_var_time_domain ;

% Note: We DO NOT divide the variance by 2 here, because these are 1D real numbers 
% and don't share their power with an imaginary component.
DC_bin_I = sqrt(freq_domain_noise_var) * randn(1, 1);
Nyquist_bin_I = sqrt(freq_domain_noise_var) * randn(1, 1);

positive_frequencies_I = sqrt(freq_domain_noise_var/2)*(randn(1,N/2-1)+1i*randn(1,N/2-1));
negative_frequencies_I = fliplr(conj(positive_frequencies_I));
% 4. Assemble the Final Array for IFFT
% Layout: [ 0 Hz,  +df to +(fs/2-df),  +fs/2 (Nyquist),  -(fs/2-df) to -df ]
X_array_I = [DC_bin_I, positive_frequencies_I, Nyquist_bin_I, negative_frequencies_I];

DC_bin_Q = sqrt(freq_domain_noise_var) * randn(1, 1);
Nyquist_bin_Q = sqrt(freq_domain_noise_var) * randn(1, 1);

positive_frequencies_Q = sqrt(freq_domain_noise_var/2)*(randn(1,N/2-1)+1i*randn(1,N/2-1));
negative_frequencies_Q = fliplr(conj(positive_frequencies_Q));
% 4. Assemble the Final Array for IFFT
% Layout: [ 0 Hz,  +df to +(fs/2-df),  +fs/2 (Nyquist),  -(fs/2-df) to -df ]
X_array_Q = [DC_bin_Q, positive_frequencies_Q, Nyquist_bin_Q, negative_frequencies_Q];


N_samples_fd = floor(N/(fs/fd));
samples_doppler = 0:N_samples_fd-1;
Jakes_psd_pos = (1.5/ (pi*fd)) * 1./sqrt( 1 - (samples_doppler./N_samples_fd).^2);
jakes_filter = zeros(1,N);

effective_psd_for_filter = sqrt(Jakes_psd_pos./(2*noise_var_time_domain));
jakes_filter(1:N_samples_fd) = effective_psd_for_filter;
jakes_filter(end-(N_samples_fd-2):end) = fliplr(effective_psd_for_filter(2:end));

freq_resp_channel_I = X_array_I.*jakes_filter;
channel_I = ifft(freq_resp_channel_I);

freq_resp_channel_Q = X_array_Q.*jakes_filter;
channel_Q = ifft(freq_resp_channel_Q);

channel_resp_smith = channel_I + 1i*channel_Q;

[auto_corr_smith,lags] = xcorr(channel_resp_smith);
[cross_corr_smith,lags] = xcorr(real(channel_resp_smith),imag(channel_resp_smith));

lag_times = lags*Ts;
theory_acf = besselj(0, 2*pi*fd*lag_times); % J0(2*pi*fd*n*Ts)

signal_power = auto_corr_smith(lags==0); 
% Normalise the correlation by power
auto_corr_smith = auto_corr_smith./signal_power;
cross_corr_smith = cross_corr_smith./signal_power;


figure;
plot(time, abs(channel_resp_smith), 'b', 'LineWidth', 1.2);
title('Time-Selective Fading Envelope (Smiths Model)');
xlabel('Time (seconds)');
ylabel('Channel Magnitude |h(t)|');
grid on;
%exportgraphics(gcf, 'fading_smith.pdf', 'ContentType', 'vector');


figure;
plot(lags, real(auto_corr_smith), 'r', 'LineWidth', 1.2);
hold on;
plot(lags, theory_acf, 'k--', 'LineWidth', 1.2);
hold off;
title('Real part of Auto correlation(Smith)');
xlabel('Lag');
ylabel('auto correlation');
legend('Estimated', 'Theoretical J_0(2\pi f_d \tau)');% PSD via FFT of autocorrelation (Levinson)
%exportgraphics(gcf, 'auto_real_smith.pdf', 'ContentType', 'vector');

figure;
plot(lags, imag(auto_corr_smith), 'r', 'LineWidth', 1.2);
title('Imaginary part of autocorrelation(Smith)');
xlabel('Lag');
ylabel('auto correlation');
grid on;
%exportgraphics(gcf, 'auto_img_smith.pdf', 'ContentType', 'vector');


figure;
plot(lags, cross_corr_smith, 'r', 'LineWidth', 1.2);
title('Cross correlation between real and imaginary(Smith)');
xlabel('Lag');
ylabel('cross correlation');
grid on;
%exportgraphics(gcf, 'cross_smith.pdf', 'ContentType', 'vector');

%% Sum of Sinusoid Model
N = 8000;
time = (0:N-1)*Ts;
M = 100; %Number of sinusoids added

n = 1:M;
Z_c = zeros(1,N);
Z_s = zeros(1,N);
for i=1:M
    theta = -pi + 2*pi*rand(1,1);
    phi_i = -pi + 2*pi*rand(1,1);
    psi_i = -pi + 2*pi*rand(1,1);
    alpha_i = (2*pi*i - pi +  theta)./(4*M);
    Z_c = Z_c + sqrt(2/M)*cos(2*pi*fd.*time.*cos(alpha_i)+phi_i);
    Z_s = Z_s + sqrt(2/M)*cos(2*pi*fd.*time.*sin(alpha_i)+psi_i);
end

channel_output_sos = Z_c + 1j*Z_s;

[auto_corr_sos,lags] = xcorr(channel_output_sos);
[cross_corr_sos,lags] = xcorr(real(channel_output_sos),imag(channel_output_sos));

signal_power = auto_corr_sos(lags==0); 
% Normalise the correlation by power
auto_corr_sos = auto_corr_sos./signal_power;
cross_corr_sos = cross_corr_sos./signal_power;

lag_times = lags*Ts;
theory_acf = besselj(0, 2*pi*fd*lag_times); % J0(2*pi*fd*n*Ts)

figure;
plot(time, abs(channel_output_sos), 'b', 'LineWidth', 1.2);
title('Time-Selective Fading Envelope (Sum of sinusoid Model)');
xlabel('Time (seconds)');
ylabel('Channel Magnitude |h(t)|');
grid on;
%exportgraphics(gcf, 'fading_sos_3ms.pdf', 'ContentType', 'vector');


figure;
plot(lags, real(auto_corr_sos), 'r', 'LineWidth', 1.2);
hold on;
plot(lags, theory_acf, 'k--', 'LineWidth', 1.2);
hold off;

title('Real part of Auto correlation(SOS)');
xlabel('Lag');
ylabel('auto correlation');
legend('Estimated', 'Theoretical J_0(2\pi f_d \tau)');
grid on;
%exportgraphics(gcf, 'auto_real_sos.pdf', 'ContentType', 'vector');

figure;
plot(lags, imag(auto_corr_sos), 'r', 'LineWidth', 1.2);
title('Imaginary part of autocorrelation(SOS)');
xlabel('Lag');
ylabel('auto correlation');
grid on;
%exportgraphics(gcf, 'auto_img_sos.pdf', 'ContentType', 'vector');


figure;
plot(lags, cross_corr_sos, 'r', 'LineWidth', 1.2);
title('Cross correlation between real and imaginary(SOS)');
xlabel('Lag');
ylabel('cross correlation');
grid on;
%exportgraphics(gcf, 'cross_sos.pdf', 'ContentType', 'vector');


