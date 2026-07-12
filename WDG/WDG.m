clear
close all
clc

% original B, K, C, W, H, F    F



name = "image_";
type = ".jpg";

layer = 1;
K = 16;
C = 3;
FH = 3;
FW = 3;


% W(K, C, FH, FW)
W = generate_edge_weights_K16C3();
W_fixed = to_fixed_8_8(W);
fname = "Weight_" + layer + ".bin";
write_file(fname, W, K, C, FW, FH);

disp("done!")