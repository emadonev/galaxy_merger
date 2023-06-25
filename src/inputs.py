import torch

version_pristine = "pristine"
version_noisy = "noisy"
input_shape = (3, 75, 75)
file_url_pristine = 'https://archive.stsci.edu/hlsps/deepmerge/hlsp_deepmerge_hst-jwst_acs-wfc3-nircam_illustris-z2_f814w-f160w-f356w_v1_sim-'+version_pristine+'.fits'
file_url_noisy = 'https://archive.stsci.edu/hlsps/deepmerge/hlsp_deepmerge_hst-jwst_acs-wfc3-nircam_illustris-z2_f814w-f160w-f356w_v1_sim-'+version_noisy+'.fits'
path = "/Users/Ita/Library/Mobile Documents/com~apple~CloudDocs/2_PERSONAL/Ema/PROJECTS"
device = 'cuda' if torch.cuda.is_available() else 'cpu'