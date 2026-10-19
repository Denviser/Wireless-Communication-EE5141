gain_db = [-2,0,-1,-6,-9,-14];
delay_us = [0,1.8,3.5,5.7,8.1,12.3];

fs = 1e7;
Ts_us = 0.1;

lin_gain = 10.^(gain_db./10);
lin_gain_norm = lin_gain / sum(lin_gain);

delay_samples = round(delay_us./Ts_us);

channel_length = max(delay_samples)+1;
channel_resp_1 = zeros(1,channel_length);
channel_resp_2 = zeros(1,channel_length);
channel_resp_3 = zeros(1,channel_length);

for i=1:6
channel_resp_1(delay_samples(i)+1) = sqrt(lin_gain_norm(i)/2) * (randn(1,1) + 1j*randn(1,1));
channel_resp_2(delay_samples(i)+1) = sqrt(lin_gain_norm(i)/2) * (randn(1,1) + 1j*randn(1,1));
channel_resp_3(delay_samples(i)+1) = sqrt(lin_gain_norm(i)/2) * (randn(1,1) + 1j*randn(1,1));
end

N=2048;

freq_resp_channel_1 = fft(channel_resp_1,N);
freq_resp_channel_2 = fft(channel_resp_2,N);
freq_resp_channel_3 = fft(channel_resp_3,N);
freq_arr = (-N/2:N/2-1)./N .*fs ;  

figure;
% Plot the first channel and tell MATLAB to hold the axes
plot(freq_arr, fftshift(10*log10(abs(freq_resp_channel_1).^2)), 'b', 'LineWidth', 1.2);
hold on; 

% Plot the second and third channels (they will now overlay)
plot(freq_arr, fftshift(10*log10(abs(freq_resp_channel_2).^2)), 'r', 'LineWidth', 1.2);
plot(freq_arr, fftshift(10*log10(abs(freq_resp_channel_3).^2)), 'g', 'LineWidth', 1.2);

% Add a legend to identify the realizations
legend('Realization 1', 'Realization 2', 'Realization 3');

title('Channel Frequency Response');
xlabel('Frequency (Hz)');
ylabel('|H(k)|^2 (dB)');
grid on;
%exportgraphics(gcf, 'frequency_selective.pdf', 'ContentType', 'vector');

% Optional: Release the hold so future plots don't accidentally draw here
hold off;

%% Question 3
function sos_channel = sos_model(M,N,Ts,fd)
    time = (0:N-1)*Ts;
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
    sos_channel = Z_c + 1j*Z_s;
end

N_time=7; % Time axis and delay axis will be channel length
channel_resp = zeros(channel_length,N_time);
Ts_for_plot = 5e-3;
fd = 90;

for delay_index=1:6
channel_for_specific_delay = sos_model(100,N_time,Ts_for_plot,fd);
channel_resp(delay_samples(delay_index)+1,:) = sqrt(lin_gain_norm(delay_index)/2) .* channel_for_specific_delay;
end

freq_response_channel = fft(channel_resp,2048,1);
H_mag_dB = 10*log10(abs(fftshift(freq_response_channel, 1)).^2);

time_arr = (0:N_time-1)*Ts_for_plot;

figure;
% Create the 3D surface
% 'EdgeColor', 'none' is crucial! Without it, the black grid lines 
% will completely color over the dense 2048-point plot.
surf(time_arr, freq_arr, H_mag_dB, 'EdgeColor', 'none'); 

% --- Professional Formatting ---
colormap(parula); % Adds a clean color gradient based on height
colorbar;         % Adds a scale on the side

title('Time-Varying Frequency-Selective Fading Channel for 90Hz Doppler');
xlabel('Time (seconds)');
ylabel('Frequency (Hz)');
zlabel('|H(f, t)|^2 (dB)');

% Adjust the default camera viewing angle for the best perspective
% view(Azimuth, Elevation)
view(-45, 45); 
grid on;
%exportgraphics(gcf, '3d_freq_time_varying.png', 'Resolution', 300);