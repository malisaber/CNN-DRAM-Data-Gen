clear
close all
clc

% original B, K, C, W, H, F    F

inp_path = 1;

name = "image_";
type = ".jpg";
B = 1;
C = 3;
H = 224;
W = 224;


% img = imread("image_1.jpg");
% img = imresize(img, [224, 224]);
% img(:,1:10,:) = 0;
% imshow(img)




fid = fopen("Input_" + inp_path + ".bin", 'w');
for b = 1:B
	img = imread(name + b + type);
	img = imresize(img, [224, 224]);
	for c = 1:C
		for w = 1:W
			for h = 1:H
				% data conversion:
				fwrite(fid,256 * uint16(img(h, w, c)),'uint16');
			end
		end
	end
end
fclose(fid);
disp("done!")



% fid = fopen("Input_1.bin");
% img = fread(fid, 'uint16');
% fclose(fid);
% img = reshape(img, [224, 224, 3]);
% imshow(uint8(img/256))







% A = imresize(A, [224, 224]);
% imshow(A)
% % H W C


%fid = fopen("Inputs2.bin", 'w');
%fprintf(fid, "%04X\r\n", B);
%fclose(fid);
%
%fid = fopen("Inputs2.bin", 'r');
%D = fscanf(fid,"%04X\r\n");
%fclose(fid);
%disp(D);
