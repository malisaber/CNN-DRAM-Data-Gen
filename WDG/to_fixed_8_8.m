function W_fixed = to_fixed_8_8(W)
    % Converts a double-precision weight tensor to Q8.8 fixed-point,
    % represented as int16 values.
    %
    % Q8.8 format: value_fixed = round(value_float * 256)
    % Range: [-128, 127.99609375], resolution: 1/256

    scale = 256;              % 2^8 fractional bits
    minVal = -128;
    maxVal = 127.99609375;    % 32767/256

    % Clip to representable range before scaling (avoids overflow)
    W_clipped = max(min(W, maxVal), minVal);

    % Scale and round to nearest integer
    W_fixed = int16(round(W_clipped * scale));
end