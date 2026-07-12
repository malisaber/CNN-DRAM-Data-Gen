function W = generate_edge_weights_K16C3()
    % Generates a [16 x 3 x 3 x 3] weight tensor for a Conv2D layer
    % Layout: [outChannels=16, inChannels=3, kH=3, kW=3]  (PyTorch/LibTorch convention)
    %
    % Each of the 16 filters is a classic edge/line detector, applied
    % identically to each of the 3 input channels (and divided by 3 so
    % the combined 3-channel response has similar magnitude to a
    % single-channel edge filter).

    kernels = cell(1, 16);

    % --- Sobel ---
    kernels{1}  = [-1 0 1; -2 0 2; -1 0 1];        % Sobel X
    kernels{2}  = [-1 -2 -1; 0 0 0; 1 2 1];        % Sobel Y
    kernels{3}  = [0 1 2; -1 0 1; -2 -1 0];        % Sobel diag 45
    kernels{4}  = [-2 -1 0; -1 0 1; 0 1 2];        % Sobel diag 135

    % --- Prewitt ---
    kernels{5}  = [-1 0 1; -1 0 1; -1 0 1];        % Prewitt X
    kernels{6}  = [-1 -1 -1; 0 0 0; 1 1 1];        % Prewitt Y
    kernels{7}  = [0 1 1; -1 0 1; -1 -1 0];        % Prewitt diag 45
    kernels{8}  = [-1 -1 0; -1 0 1; 0 1 1];        % Prewitt diag 135

    % --- Scharr (more rotationally accurate than Sobel) ---
    kernels{9}  = [-3 0 3; -10 0 10; -3 0 3];      % Scharr X
    kernels{10} = [-3 -10 -3; 0 0 0; 3 10 3];      % Scharr Y

    % --- Laplacian (isotropic, edge magnitude regardless of direction) ---
    kernels{11} = [0 1 0; 1 -4 1; 0 1 0];          % 4-connected Laplacian
    kernels{12} = [1 1 1; 1 -8 1; 1 1 1];          % 8-connected Laplacian

    % --- Roberts cross (padded to 3x3, centered) ---
    kernels{13} = [1 0 0; 0 -1 0; 0 0 0];          % Roberts Gx
    kernels{14} = [0 1 0; -1 0 0; 0 0 0];          % Roberts Gy

    % --- Line detectors ---
    kernels{15} = [-1 -1 -1; 2 2 2; -1 -1 -1];     % Horizontal line
    kernels{16} = [-1 2 -1; -1 2 -1; -1 2 -1];     % Vertical line

    % Build the [16, 3, 3, 3] tensor
    numFilters = 16;
    numChannels = 3;
    W = zeros(numFilters, numChannels, 3, 3);

    for f = 1:numFilters
        k = kernels{f} / numChannels;   % normalize across channels
        for c = 1:numChannels
            W(f, c, :, :) = k;
        end
	end
	
	W = permute(W,[3, 4, 2, 1]);
end