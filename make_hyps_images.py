from __future__ import division
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.interpolate import interp1d
from numpy.polynomial.polynomial import polyval, polyfit
from pybob.GeoImg import GeoImg
from pybob.bob_tools import bin_data


def curve_fit(glacier_mask, dDEM, DEM, valid_area):
        ddem_data = dDEM.img[np.logical_and(glacier_mask, valid_area)]
        dem_data = DEM.img[np.logical_and(glacier_mask, valid_area)]

        valid = np.logical_and(np.isfinite(dem_data), np.isfinite(ddem_data))

        dem_data = dem_data[valid]
        ddem_data = ddem_data[valid]

        missing = np.logical_and(np.isnan(dDEM.img), glacier_mask)
        missing_els = DEM.img[missing]

        if missing_els.size > 0:
            new_max = max(np.nanmax(missing_els), np.nanmax(dem_data))
            new_min = min(np.nanmin(missing_els), np.nanmin(dem_data))
        else:
            new_max = np.nanmax(dem_data)
            new_min = np.nanmin(dem_data)

        z_range = new_max - new_min
        if z_range > 500:
            bin_width = 50
        else:
            bin_width = int(z_range / 10)

        min_el = new_min - (new_min % bin_width)
        max_el = new_max + (bin_width - (new_max % bin_width))
        bins = np.arange(min_el, max_el+1, bin_width)

        mean_dH = bin_data(bins, ddem_data, dem_data, mode='mean')
        median_dH = bin_data(bins, ddem_data, dem_data, mode='median')

        if np.isnan(mean_dH[0]):
            mean_dH[0] = 0

        if np.isnan(mean_dH[-1]):
            mean_dH[-1] = 0

        if np.isnan(median_dH[0]):
            median_dH[0] = 0

        if np.isnan(median_dH[-1]):
            median_dH[-1] = 0

        valid_mean = np.isfinite(mean_dH)
        valid_median = np.isfinite(median_dH)

        tmp_mean = interp1d(bins[valid_mean], mean_dH[valid_mean])
        tmp_median = interp1d(bins[valid_median], median_dH[valid_median])

        mean_dH = tmp_mean(bins)
        median_dH = tmp_median(bins)

        pfit = polyfit(bins, mean_dH, 3)
        poly_fit = polyval(bins, pfit)

        return bins, mean_dH, median_dH, poly_fit


def get_bins(DEM, glacier_mask):
    dem_data = DEM.img[np.logical_and(glacier_mask, np.isfinite(DEM.img))]
    if dem_data.size == 0:
        return np.nan
    zmax = np.nanmax(dem_data)
    zmin = np.nanmin(dem_data)
    zrange = zmax - zmin
    bin_width = min(50, int(zrange / 10))
    min_el = zmin - (zmin % bin_width)
    max_el = zmax + (bin_width - (zmax % bin_width))
    bins = np.arange(min_el, max_el+1, bin_width)
    return bins


mask_full = GeoImg('../southeast_average_corr.tif')

for yr in [2012, 2014]:
    ifsar = GeoImg('{}/seak.ifsar.{}.dem.30m_adj.tif'.format(yr, yr))
    ifsar_srtm = GeoImg('{}/ifsar_srtm_{}_dh.tif'.format(yr, yr))
    srtm = GeoImg('{}/SRTM_SE_Alaska_30m_{}IfSAR_adj.tif'.format(yr, yr))
    aad_srtm = GeoImg('hypsometries/coreg_{}/srtm_seak_cgiar_30m_{}_adj.tif'.format(yr, yr))

    valid_area = np.isfinite(ifsar.img)
    orig_valid = np.logical_and(np.isfinite(srtm.img),
                                np.isfinite(ifsar.img))
    
    glac_shp = gpd.read_file('../outlines/01_rgi60_Alaska_GlacierBay_02km_UTM_{}.shp'.format(yr))
    
    gmask_tif = GeoImg('GlacierBay_Mask_{}.tif'.format(yr))
    glacier_mask = gmask_tif.img
    gmask = np.isfinite(glacier_mask)
    glacs = np.unique(gmask_tif.img[np.isfinite(gmask_tif.img)])

    mask_geo = mask_full.reproject(ifsar_srtm)

    # get the bin values for the SRTM dem
    all_srtm_binned = np.floor(aad_srtm.img / 50) * 50
    all_srtm_binned[~gmask] = np.nan
    ind_srtm_binned = aad_srtm.img
    
    curve_list = [pd.read_csv('hypsometries/{}/{}_curves.csv'.format(yr, gid)) for gid in glac_shp.RGIId]
    for i, g in enumerate(glacs):
        rgiid = glac_shp['RGIId'][glac_shp['fid'] == g].values[0]
        imask = glacier_mask == g
        aad_df = curve_list[i]
        #aad_df = pd.read_csv('hypsometries/{}/AADs/{}_curves.csv'.format(yr, rgiid))
        els = aad_df['elevation'].values
        bin_width = np.diff(els)[0]
        ind_srtm_binned[imask] = np.floor(ind_srtm_binned[imask] / bin_width) * bin_width

    #corrs = range(35, 100, 5)
    corrs = [35, 50, 70, 80, 90, 95]
    for i, corr in enumerate(corrs):
        corr_mask = mask_geo.img < corr
        masked_ifsar_srtm = ifsar_srtm.copy()
        masked_ifsar_srtm.img[corr_mask] = np.nan
        
        srtm_bins, meanfit, medfit, poly_fit = curve_fit(gmask, masked_ifsar_srtm, srtm, valid_area)
        glob_mean = masked_ifsar_srtm.copy()
        glob_med = masked_ifsar_srtm.copy()
        glob_poly = masked_ifsar_srtm.copy()

        loc_mean = masked_ifsar_srtm.copy()
        loc_med = masked_ifsar_srtm.copy()
        loc_poly = masked_ifsar_srtm.copy()        
        for i, z in enumerate(srtm_bins):
            el_mask = np.logical_and(gmask, all_srtm_binned == z)            
            glob_mean.img[el_mask] = meanfit[i]
            glob_med.img[el_mask] = medfit[i]
            glob_poly.img[el_mask] = poly_fit[i]

        # now, read the fitted elevation curves for each glacier, and populate the map
        for i, g in enumerate(glacs):
            rgiid = glac_shp['RGIId'][glac_shp['fid'] == g].values[0]
            imask = glacier_mask == g
            df = curve_list[i]
            try:
                mnstr = 'mean_dh_{}'.format(corr)
                mdstr = 'median_dh_{}'.format(corr)
                
                df[mnstr][df['void_{}'.format(corr)] < 0.01] = np.nan
                df[mdstr][df['void_{}'.format(corr)] < 0.01] = np.nan
                #pstr = 'poly_dh_{}'.format(corr)
                
                pfit = polyfit(df['elevation'][np.isfinite(df[mdstr])], df[mdstr][np.isfinite(df[mdstr])], 3)
                poly_fit = polyval(df.elevation.values, pfit)
                df[mdstr][np.isnan(df[mdstr])] = poly_fit[np.isnan(df[mdstr])]
                
                pfit = polyfit(df['elevation'][np.isfinite(df[mnstr])], df[mnstr][np.isfinite(df[mnstr])], 3)
                poly_fit = polyval(df.elevation.values, pfit)
                df[mnstr][np.isnan(df[mnstr])] = poly_fit[np.isnan(df[mnstr])]
            except:
                loc_mean.img[imask] = np.nan
                loc_med.img[imask] = np.nan
                loc_poly.img[imask] = np.nan
                continue
            for j, el in enumerate(df.elevation):
                el_mask = np.logical_and(imask, ind_srtm_binned == el)
                loc_mean.img[el_mask] = df[mnstr][j]
                loc_med.img[el_mask] = df[mdstr][j]
                loc_poly.img[el_mask] = poly_fit[j]
        glob_mean.write('ifsar_srtm_{}_globmean.tif'.format(yr), out_folder='filled_ddems/{}/void{}'.format(yr, corr))
        glob_med.write('ifsar_srtm_{}_globmed.tif'.format(yr), out_folder='filled_ddems/{}/void{}'.format(yr, corr))
        glob_poly.write('ifsar_srtm_{}_globpoly.tif'.format(yr), out_folder='filled_ddems/{}/void{}'.format(yr, corr))
        
        loc_mean.write('ifsar_srtm_{}_meanhyps.tif'.format(yr), out_folder='filled_ddems/{}/void{}'.format(yr, corr))
        loc_med.write('ifsar_srtm_{}_medhyps.tif'.format(yr), out_folder='filled_ddems/{}/void{}'.format(yr, corr))
        loc_poly.write('ifsar_srtm_{}_polyhyps.tif'.format(yr), out_folder='filled_ddems/{}/void{}'.format(yr, corr))