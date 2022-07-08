clear all;
close all;
clc;

disp('setting paths')

setenv('TOOLBOX_PATH', '/opt/amc/bart-0.5.00/bin')
addpath(getenv('/opt/amc/bart-0.5.00/matlab'));
addpath(('/opt/amc/matlab/toolbox/spm12-r7771/'));

parentraw='/data/projects/recon/data/private/neonatal/utrecht/raw/umcu_220222a/2022_02_21/';
parentproc='/data/projects/recon/data/private/neonatal/utrecht/proc/umcu_220222a/';
subdirs={'GR_262175'};

doPreproc= 1; % should preproc be performed again? time-consuming

%%
iDir=1;

rawdir=fullfile(parentraw,subdirs{iDir});
odir=fullfile(parentproc,subdirs{iDir});
if ~isfolder(odir); mkdir(odir); end
disp(['subject ',num2str(iDir),', folder: ',rawdir]);

ref= dir([rawdir,'/*senserefscanV4.cpx']);
ref=fullfile(rawdir,ref(1).name);
refsin=dir([rawdir,'/*senserefscanV4.sin']);
refsin=fullfile(rawdir,refsin(1).name);
t1sin=dir([rawdir,'/*_t1_*.sin']);
% if isempty(t1sin); disp('no t1 for this subject, skipping'); continue; end

if length(t1sin)==1
    t1sin=fullfile(rawdir,t1sin(1).name);
else
    t1sin=fullfile(rawdir,t1sin(2).name);
end

t1=dir([rawdir,'/*_t1_*.raw']);
t1=fullfile(t1.folder, t1.name);

disp(['loaded files are: ', ref,', ', refsin,', ', t1sin,', ', t1])

rfile=fullfile(odir,'senserefall_real.nii');
ifile=fullfile(odir,'senserefall_imag.nii');
rrfile=fullfile(odir,'rsenserefall_real.nii');
rifile=fullfile(odir,'rsenserefall_imag.nii');

%% load data
disp('loading t1 list-data and applying preprocessing steps');
r = MRecon(t1);
r.Parameter.Parameter2Read.typ = 1;
r.Parameter.Parameter2Read.Update;
r.ReadData;
r.Parameter.Encoding.XRes=r.Parameter.Encoding.XReconRes;

% added
% r.Parameter.Encoding.KyRange=[-76 75];
% r.Parameter.Encoding.YRange=[-76 75];
% r.Parameter.Encoding.KzRange=[-32 31];
% r.Parameter.Encoding.ZRange=[-25 24.2188];
% r.Parameter.Encoding.YRes=193;
% r.Parameter.Encoding.YReconRes=r.Parameter.Encoding.YRes;
% r.Parameter.Encoding.ZRes=50;
% r.Parameter.Encoding.ZReconRes=r.Parameter.Encoding.ZRes;
% r.Parameter.Encoding.KzOversampling=1.28;

r.RandomPhaseCorrection;
r.RemoveOversampling;
r.PDACorrection;
r.DcOffsetCorrection;
r.MeasPhaseCorrection;
r.SortData;
r.RingingFilter;

% try cropping in imspace - is it needed ? doesn't seem to solve the
% scaling issue
% crop_rData=bart('fftshift 4', bart('fftshift 2', bart('fft -i 7', r.Data)));
% crop_rData=crop_rData(:, :, 7:56, :);
% crop_rData=bart('fft 7', bart('fftshift 2', bart('fftshift 4', crop_rData)));
% r.Data=crop_rData;

sx=size(r.Data,1); sy=size(r.Data,2); sz=size(r.Data,3); nrcoils=size(r.Data,4);

rrz_os=size(r.Data, 3);

[~, rrx]=matrix_from_sin(t1sin, rrz_os);
rry=rrx(2); rrz=rrx(3); rrx=rrx(1);

% try Y Recon resolution
% rry=404;

vol3=zeros(rrx, rry, rrz_os, nrcoils); % try padded
vol3(rrx/2 - sx/2 +1:rrx/2 +sx/2 , rry/2 - sy/2 +1:rry/2 +sy/2, round(rrz_os/2 - sz/2) +1:round(rrz_os/2 +sz/2), :) = r.Data;
% imspace_padded=bart('fftshift 4', bart('fftshift 2', bart('fft -i 7',vol3)));
clear r;

%% start on senseref.cpx
disp('loading senseref.cpx and applying transforms');

C=MRecon(ref);
C.ReadData; % read
vol=C.Data; clear C;

tmp1=squeeze(vol(:,:,:,:,1,1,1,1));
tmp=permute(tmp1,[2 1 3 4]); % from Z X Y C --> X Z Y C
tmp=flip(tmp, 1);

%%%% write senseref as nifti
disp('bringing senseref to nifti for checks');
sz=size(tmp);
dim=sz(1:3); %currently AP-FH-LR axes

[~,offcentr1]=unix(['cat ',refsin,' | grep ''loc_ap_rl_fh_offcentres'' | awk ''{print$6}'' | awk ''NR==1'' ']);
[~,offcentr2]=unix(['cat ',refsin,' | grep ''loc_ap_rl_fh_offcentres'' | awk ''{print$7}'' | awk ''NR==1'' ']);
[~,offcentr3]=unix(['cat ',refsin,' | grep ''loc_ap_rl_fh_offcentres'' | awk ''{print$8}'' | awk ''NR==1'' ']);
sref_off= [str2double(strip(offcentr1)) str2double(strip(offcentr2)) str2double(strip(offcentr3))] ; clear offcentr1 offcentr2 offcentr3

[~,vox1]=unix(['cat ',refsin,' | grep ''voxel_sizes'' | awk ''{print$6}'' ']);
[~,vox2]=unix(['cat ',refsin,' | grep ''voxel_sizes'' | awk ''{print$7}'' ']);
[~,vox3]=unix(['cat ',refsin,' | grep ''voxel_sizes'' | awk ''{print$8}'' ']);
voxelref= [str2double(strip(vox1)) str2double(strip(vox2)) str2double(strip(vox3))] ; clear vox1 vox2 vox3

% NII-matlab convention mismatch: LR - PA - FH
% currently: data in tmp are stored in
% NB: we flipped the RL-dimension but not the FH direction, so flip the FH
% offset
offset = -[dim(3)*voxelref(3)/2+sref_off(2) dim(1)*voxelref(1)/2+sref_off(1)  dim(2)*voxelref(2)/2-sref_off(3)]; % sagittal

% offcentre: half the number of voxels * voxel spacing;
% TODO: spacing when even number of voxels
a = [offset 0 pi/2 -pi/2 voxelref(1) voxelref(2) voxelref(3) 0 0 0];
A = spm_matrix(a);

% save real and imaginary components separately in 4D
n=nifti('/data/projects/recon/data/private/STAIRS/raw/stairs_pilot_20210603/stairs_pilot_20210606_stairs_pilot_20210606/20210603_1.3.46.670589.11.42151.5.0.3288.2021060316570383000/MR/00201_CS_3D_FLAIR_1.15_noreconframe/__CS_3D_FLAIR_1.15_noreconframe_20210603165704_201.nii');% load unrelated file as a baseline nifti structure
n.dat.fname=rfile;
n.dat.scl_slope=max(abs(tmp(:)))/1e4;
n.mat=A;
n.mat0=n.mat;
n.dat.dim=size(tmp);
create(n);
n.dat(:,:,:,:)=real(tmp);

n.dat.fname=ifile;
create(n);
n.dat(:,:,:,:)=imag(tmp);

%% create empty nifti to coregister t1
n.dat.fname = fullfile(odir,'dummy_t1.nii');
n.dat.scl_slope = max(abs(tmp(:)))/1e4;
[n.mat, resolution] = matrix_from_sin(t1sin, rrz_os);
% FIXME: quick hack
%resolution(2)=404
n.dat.dim = [rrz_os resolution(1) resolution(2) size(tmp,4)];
n.mat0=n.mat;
create(n);
n.dat(:,:,:,:)=zeros([rrz_os resolution(1) resolution(2) size(tmp,4)]);

%% reslice to "t1"
disp('reslicing senseref to t1');
flags.mean=0;
flags.which=1;
spm_reslice({fullfile(odir,'dummy_t1.nii'),rfile,ifile},flags)

%% bring sense back to raw data convention
disp('load resliced nifti to continue processing')

n=nifti(rrfile);
rout=n.dat(:,:,:,:);
n=nifti(rifile);
iout=n.dat(:,:,:,:);

% it follows from below that the senseref data need to be permuted here
rout=permute(rout,[3 2 1 4]);
rout=flip(rout,1);
iout=permute(iout,[3 2 1 4]);
iout=flip(iout,1);

s_resliced=rout+1i*iout; % make complex-valued output

%%
chalf=(512-404)/2; % half size
vol3_crop=vol3(:,(chalf+1):(end-chalf),:,:);

%%
imspace = bart('fftshift 4', bart('fftshift 2', bart('fft -i 7', vol3_crop)));
sensemap = s_resliced;
sensemap(:,:,:,8) = [];

% crop y and flip z
sensemap=sensemap(:,(chalf+1):(end-chalf),end:-1:1,:);

target = sum(imspace.*conj(sensemap), 4);


%% helper functions
function [matrix_perm,resolution]=matrix_from_sin(t1sin, rrz_os)

matrix_loca=zeros(3,3);

% [~,offcentr_incrs]=unix(['cat ',t1sin,' | grep ''loc_ap_rl_fh_offcentr_incrs'' | awk ''{print$6,$7,$8}'' '])
[~,offcentr1]=unix(['cat ',t1sin,' | grep ''loc_ap_rl_fh_offcentr_incrs'' | awk ''{print$6}'' ']);
[~,offcentr2]=unix(['cat ',t1sin,' | grep ''loc_ap_rl_fh_offcentr_incrs'' | awk ''{print$7}'' ']);
[~,offcentr3]=unix(['cat ',t1sin,' | grep ''loc_ap_rl_fh_offcentr_incrs'' | awk ''{print$8}'' ']);
matrix_loca(3,:)= [str2double(strip(offcentr1)) str2double(strip(offcentr2)) str2double(strip(offcentr3))] ; clear offcentr1 offcentr2 offcentr3

[~,row1]=unix(['cat ',t1sin,' | grep ''loc_ap_rl_fh_row_image_oris'' | awk ''{print$6}'' ']);
[~,row2]=unix(['cat ',t1sin,' | grep ''loc_ap_rl_fh_row_image_oris'' | awk ''{print$7}'' ']);
[~,row3]=unix(['cat ',t1sin,' | grep ''loc_ap_rl_fh_row_image_oris'' | awk ''{print$8}'' ']);
matrix_loca(1,:)= [str2double(strip(row1)) str2double(strip(row2)) str2double(strip(row3))] ; clear row1 row2 row3

[~,col1]=unix(['cat ',t1sin,' | grep ''loc_ap_rl_fh_col_image_oris'' | awk ''{print$6}'' ']);
[~,col2]=unix(['cat ',t1sin,' | grep ''loc_ap_rl_fh_col_image_oris'' | awk ''{print$7}'' ']);
[~,col3]=unix(['cat ',t1sin,' | grep ''loc_ap_rl_fh_col_image_oris'' | awk ''{print$8}'' ']);
matrix_loca(2,:)= [str2double(strip(col1)) str2double(strip(col2)) str2double(strip(col3))] ; clear col1 col2 col3

[~,vox1]=unix(['cat ',t1sin,' | grep ''voxel_sizes'' | awk ''{print$6}'' ']);
[~,vox2]=unix(['cat ',t1sin,' | grep ''voxel_sizes'' | awk ''{print$7}'' ']);
[~,vox3]=unix(['cat ',t1sin,' | grep ''voxel_sizes'' | awk ''{print$8}'' ']);
voxel_sizes= [str2double(strip(vox1)) str2double(strip(vox2)) str2double(strip(vox3))] ; clear vox1 vox2 vox3

[~,res1]=unix(['cat ',t1sin,' | grep ''output_resolutions'' | awk ''{print$6}'' ']);
[~,res2]=unix(['cat ',t1sin,' | grep ''output_resolutions'' | awk ''{print$7}'' ']);
% [~,res3]=unix(['cat ',t1sin,' | grep ''output_resolutions'' | awk ''{print$8}'' ']);
% resolution= [str2double(strip(res1)) str2double(strip(res2)) str2double(strip(res3))] ; clear res1 res2 res3
resolution= [str2double(strip(res1)) str2double(strip(res2)) rrz_os] ; clear res1 res2 res3

% [~,res1]=unix(['cat ',t1sin,' | grep ''recon_resolutions'' | awk ''{print$6}'' ']);
% [~,res2]=unix(['cat ',t1sin,' | grep ''recon_resolutions'' | awk ''{print$7}'' ']);
% resolution= [str2double(strip(res1)) str2double(strip(res2)) rrz_os] ; clear res1 res2 res3

[~,cc1]=unix(['cat ',t1sin,' | grep ''loc_ap_rl_fh_offcentres'' | awk ''{print$6}'' ']);
[~,cc2]=unix(['cat ',t1sin,' | grep ''loc_ap_rl_fh_offcentres'' | awk ''{print$7}'' ']);
[~,cc3]=unix(['cat ',t1sin,' | grep ''loc_ap_rl_fh_offcentres'' | awk ''{print$8}'' ']);
centre_coords= [-str2double(strip(cc1)) -str2double(strip(cc2)) str2double(strip(cc3))] ; clear cc1 cc2 cc3

matrix_loca=matrix_loca';

matrix_loca(1,3)=-matrix_loca(1,3);
matrix_loca(2,3)=-matrix_loca(2,3);
matrix_loca(3,1)=-matrix_loca(3,1);
matrix_loca(3,2)=-matrix_loca(3,2);

matrix(:,1)= matrix_loca(:,1)*voxel_sizes(1);
matrix(:,2)= matrix_loca(:,2)*voxel_sizes(2);
matrix(:,3)= matrix_loca(:,3);

offset1 =  -resolution(1)/2*matrix(1,1) + ...
            -resolution(2)/2*matrix(1,2) + ...
            -resolution(3)/2*matrix(1,3) + ...
            +centre_coords(1);
offset2 =   -resolution(1)/2*matrix(2,1) + ...
            -resolution(2)/2*matrix(2,2) + ...
            -resolution(3)/2*matrix(2,3) + ...
            +centre_coords(2);
offset3 =   -resolution(1)/2*matrix(3,1) + ...
            -resolution(2)/2*matrix(3,2) + ...
            -resolution(3)/2*matrix(3,3) + ...
            +centre_coords(3);

matrix(:,4)= [offset1; offset2; offset3];

matrix_perm = zeros(size(matrix));
 matrix_perm(:,1)= matrix(:,3);
 matrix_perm(:,2)= matrix(:,1);
 matrix_perm(:,3)= matrix(:,2);
 matrix_perm(:,4)= matrix(:,4);

 matrix_perm = matrix_perm([2 1 3],:);
 matrix_perm(4,:)= [0 0 0 1];
end
