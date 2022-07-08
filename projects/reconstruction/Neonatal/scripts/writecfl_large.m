function writecfl_large(filenameBase,data)
% writecfl(filenameBase, data)
%    Writes recon data to filenameBase.cfl (complex float)
%    and write the dimensions to filenameBase.hdr.
%
%    Written to edit data for the Berkeley recon.
%
% 2012 Joseph Y Cheng (jycheng@mrsrl.stanford.edu).

    dims = size(data);
    writeReconHeader(filenameBase,dims);

    filename = strcat(filenameBase,'.cfl');
    fid = fopen(filename,'w');

    if numel(dims)~=4
        error('not supported')
    end
    d=dims(1:end-1);
    for ii=1:dims(end)
        data_o = zeros(prod(d)*2,1,'single');
        data_o(1:2:end) = real(data(:,:,:,ii));
        data_o(2:2:end) = imag(data(:,:,:,ii));
        fwrite(fid,data_o,'float32');
    end

    fclose(fid);
end

function writeReconHeader(filenameBase,dims)
    filename = strcat(filenameBase,'.hdr');
    fid = fopen(filename,'w');
    fprintf(fid,'# Dimensions\n');
    for N=1:length(dims)
        fprintf(fid,'%d ',dims(N));
    end
    if length(dims) < 5
        for N=1:(5-length(dims))
            fprintf(fid,'1 ');
        end
    end
    fprintf(fid,'\n');

    fclose(fid);
end
