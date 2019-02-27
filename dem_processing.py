#!/usr/bin/env python
import numpy as np
from llc import jit_filter_function
from skimage.morphology import disk
from scipy.ndimage.filters import generic_filter
from pybob.GeoImg import GeoImg
from pybob import image_tools as it
from pybob import ddem_tools as dt


def neighborhood_filter(img, radius):
    @jit_filter_function
    def nanmean(a):
        return np.nanmean(a)
    
    return generic_filter(img, nanmean, footprint=disk(radius))


# load the full mask, which we'll re-project later
mask_full = GeoImg('../southeast_average_corr.tif')

for yr in [2012, 2013]:
    print("Loading {} data files.".format(yr))
    ifsar = GeoImg('{}/seak.ifsar.{}.dem.30m_adj.tif'.format(yr, yr))
    ifsar_srtm = GeoImg('{}/ifsar_srtm_{}_dh.tif'.format(yr, yr))
    srtm = GeoImg('{}/SRTM_SE_Alaska_30m_{}IfSAR_adj.tif'.format(yr, yr))

    valid_area = np.isfinite(ifsar.img)

    glac_shp = '../outlines/01_rgi60_Alaska_GlacierBay_02km_UTM_{}.shp'.format(yr)
    
    glacier_mask = it.create_mask_from_shapefile(ifsar, glac_shp)
    mask_geo = mask_full.reproject(ifsar_srtm)

    corrs = [35, 50, 70, 80, 90, 95]
    for i, corr in enumerate(corrs):
        corr_mask = mask_geo.img < corr
        masked_ifsar = ifsar.copy()
        masked_ifsar.img[corr_mask] = np.nan
        
        masked_ifsar_srtm = ifsar_srtm.copy()
        masked_ifsar_srtm.img[corr_mask] = np.nan
        
        print("Linear interpolation of dH")
        ifsar_srtm_lin_interp = dt.fill_holes(masked_ifsar_srtm, dt.linear, valid_area=valid_area)
        ifsar_srtm_lin_interp.write('ifsar_srtm_{}_dHinterp.tif'.format(yr),
                                    out_folder='filled_ddems/{}/void{}'.format(yr, corr))        
        
        print("Linear interpolation of Z")
        ifsar_lin_interp = dt.fill_holes(masked_ifsar, dt.linear, valid_area=valid_area)
        ifsar_lin_interp.write('ifsar_zinterp.tif',
                               out_folder='filled_ddems/{}/void{}'.format(yr, corr))

        ifsar_srtm_zinterp = ifsar_srtm.copy()
        ifsar_srtm_zinterp.img = ifsar_lin_interp.img - srtm.img
        ifsar_srtm_zinterp.write('ifsar_srtm_{}_zinterp.tif'.format(yr),
                                 out_folder='filled_ddems/{}/void{}'.format(yr, corr))
        
        print("1km neighborhood")
        nradius = int(1000 / masked_ifsar_srtm.dx)
        tmp_ddem = masked_ifsar_srtm.img
        tmp_ddem[~np.logical_and(glacier_mask, valid_area)] = np.nan
        filt_ddem = neighborhood_filter(tmp_ddem, nradius)
        ifsar_srtm_neighborhood = masked_ifsar_srtm.copy()
        ifsar_srtm_neighborhood.img = filt_ddem
        ifsar_srtm_neighborhood.write('ifsar_srtm_{}_nhood.tif'.format(yr),
                                      out_folder='filled_ddems/{}/void{}'.format(yr, corr))
