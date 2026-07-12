function write_file(name, W, K, C, FW, FH)


fid = fopen(name, 'w');
for k = 1:K
	for c = 1:C
		for fw = 1:FW
			for fh = 1:FH
				fwrite(fid, W(fh, fw, c, k),'uint16');
			end
		end
	end
end
fclose(fid);


end