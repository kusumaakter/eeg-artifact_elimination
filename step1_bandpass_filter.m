%% STEP 1: BANDPASS FILTER (1-40 Hz)
% Purpose: Remove noise, keep motor imagery frequencies
% Input: EEG_raw_data.mat (raw_left, raw_right, fs)
% Output: EEG_raw_clean_noise.mat (clean_left, clean_right, fs)

clear; clc;

fprintf('╔════════════════════════════════════════╗\n');
fprintf('║  STEP 1: BANDPASS FILTER (1-40 Hz)    ║\n');
fprintf('╚════════════════════════════════════════╝\n\n');

% ===== Load raw data =====
load('EEG_raw_data.mat'); % raw_left, raw_right, fs

fprintf('✓ Raw data loaded\n');
fprintf('  Sampling Frequency: %d Hz\n', fs);
fprintf('  Left data size: %d × %d\n', size(raw_left,1), size(raw_left,2));
fprintf('  Right data size: %d × %d\n\n', size(raw_right,1), size(raw_right,2));

% ===== Design Bandpass Filter =====
filter_order = 4;  % Butterworth filter order
f_low = 1;         % Lower cutoff (Hz)
f_high = 40;       % Upper cutoff (Hz)

% Normalize frequencies
nyquist = fs / 2;
normalized_low = f_low / nyquist;
normalized_high = f_high / nyquist;

fprintf('Filter Settings:\n');
fprintf('  Type: Butterworth Bandpass\n');
fprintf('  Order: %d\n', filter_order);
fprintf('  Frequency Range: %d–%d Hz\n', f_low, f_high);
fprintf('  Nyquist Frequency: %.0f Hz\n\n', nyquist);

% Create filter
[b, a] = butter(filter_order, [normalized_low, normalized_high], 'bandpass');

fprintf('✓ Filter designed\n\n');

% ===== Apply Filter (LEFT) =====
fprintf('Filtering LEFT channel...\n');
clean_left = zeros(size(raw_left));
for ch = 1:size(raw_left,1)
    clean_left(ch,:) = filtfilt(b, a, raw_left(ch,:));
    if mod(ch, 4) == 0
        fprintf('  Channel %d/%d\n', ch, size(raw_left,1));
    end
end

% ===== Apply Filter (RIGHT) =====
fprintf('Filtering RIGHT channel...\n');
clean_right = zeros(size(raw_right));
for ch = 1:size(raw_right,1)
    clean_right(ch,:) = filtfilt(b, a, raw_right(ch,:));
    if mod(ch, 4) == 0
        fprintf('  Channel %d/%d\n', ch, size(raw_right,1));
    end
end

% ===== Save =====
save('EEG_raw_clean_noise.mat', 'clean_left', 'clean_right', 'fs');
fprintf('\n✓ Filtered data saved: EEG_raw_clean_noise.mat\n\n');

% ===== Visualization =====
fprintf('Creating visualization...\n');

figure('Position',[100 100 1400 700]);

% Raw vs Filtered comparison
ch = 1;
t_sec = (1:5*fs) / fs;

subplot(2,3,1);
plot(t_sec, raw_left(ch,1:5*fs), 'Color', [0.7 0.7 0.7], 'LineWidth', 1);
title('Raw LEFT (Channel 1)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Amplitude (µV)');
xlabel('Time (s)');
grid on;

subplot(2,3,2);
plot(t_sec, clean_left(ch,1:5*fs), 'Color', [0 0.5 1], 'LineWidth', 1.5);
title('Filtered LEFT (1–40 Hz)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Amplitude (µV)');
xlabel('Time (s)');
grid on;

subplot(2,3,3);
plot(t_sec, raw_right(ch,1:5*fs), 'Color', [0.7 0.7 0.7], 'LineWidth', 1);
hold on;
plot(t_sec, clean_right(ch,1:5*fs), 'Color', [1 0.5 0], 'LineWidth', 1.5);
title('Raw vs Filtered RIGHT', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Amplitude (µV)');
xlabel('Time (s)');
legend('Raw', 'Filtered');
grid on;

% Frequency response
subplot(2,3,4:6);
freqz(b, a, 1024, fs);
grid on;
title('Filter Frequency Response', 'FontSize', 12, 'FontWeight', 'bold');

savefig('step1_bandpass_visualization.fig');
fprintf('✓ Visualization saved: step1_bandpass_visualization.fig\n\n');

fprintf('═════════════════════════════════════════\n');
fprintf('STEP 1 COMPLETE ✓\n');
fprintf('═════════════════════════════════════════\n');
