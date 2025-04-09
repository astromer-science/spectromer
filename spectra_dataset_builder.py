#!/usr/bin/env python

# -----------------------------------------------------------------------------
# MIT License
# Copyright (c) 2025 Luis Felipe Strano Moraes
#
# This file is part of Spectromer
# For license terms, see the LICENSE file in the root of this repository.
# -----------------------------------------------------------------------------

import argparse
import math
import matplotlib
import multiprocessing
import numpy as np
import os
import pandas as pd
import polars as pl
import sys
import urllib.request
import toml
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import fits
from functools import partial
from scipy.signal import savgol_filter
from scipy.integrate import simpson
import matplotlib.cm as cm
from scipy.interpolate import interp1d
from scipy.ndimage import convolve1d
from PIL import Image
from spectra_plotter import SpectraPlotter


# Spectra Dataset Import script
#
# This script is meant to perform multiple functions in order to both obtain and prepare a spectra
# dataset to use as part of Astromer.
# The initial dataset which we are calling 'sdss-small' here was obtained by leveraging the SQL interface
# made available by the SDSS project, more information can be obtained at: https://skyserver.sdss.org/dr18
# Database was queried via CasJobs, using DR18 as the context for the queries.
# The list of SQL commands used to generate the initial version of the dataset used were:
#
# SELECT specObjID,firstRelease,class,plateID,mjd,plate,fiberID,run2d,z,zErr,survey,ra,dec,subClass,waveMin,waveMax,wCoverage,snMedian_u,snMedian_g,snMedian_r,snMedian_i,snMedian_z,snMedian,elodieBV,elodieTEff,elodieLogG,elodieFeH,elodieZ into mydb.MyTable from SpecObj WHERE class='GALAXY' AND zWarning=0 AND instrument='BOSS' AND targetType='SCIENCE';
# INSERT INTO mydb.MyTable SELECT specObjID,firstRelease,class,plateID,mjd,plate,fiberID,run2d,z,zErr,survey,ra,dec,subClass,waveMin,waveMax,wCoverage,snMedian_u,snMedian_g,snMedian_r,snMedian_i,snMedian_z,snMedian,elodieBV,elodieTEff,elodieLogG,elodieFeH,elodieZ from SpecObj WHERE class='QSO' AND zWarning=0 AND instrument='BOSS' AND targetType='SCIENCE';
# INSERT INTO mydb.MyTable SELECT specObjID,firstRelease,class,plateID,mjd,plate,fiberID,run2d,z,zErr,survey,ra,dec,subClass,waveMin,waveMax,wCoverage,snMedian_u,snMedian_g,snMedian_r,snMedian_i,snMedian_z,snMedian,elodieBV,elodieTEff,elodieLogG,elodieFeH,elodieZ from SpecObj WHERE class='STAR' AND zWarning=0 AND instrument='BOSS' AND targetType='SCIENCE';
#
# The table created was later downloaded as a csv file (inputfile below) and used to obtain the FITS files for each entry.
#
# In order to create new variations, we can use the following command:
# >>> import pandas as pd
# >>> meta_lamost = pd.read_csv('meta-lamost-big.csv')
# >>> meta_sdss = pd.read_csv('meta-sdss-big.csv')
# >>> meta_sdss_small = meta_sdss.groupby('class').apply(lambda x: x.sample(n=30000, random_state=42)).reset_index(drop=True)
# >>> meta_lamost_medium = meta_lamost.groupby('class').apply(lambda x: x.sample(n=min(len(x), 200000), random_state=42)).reset_index(drop=True)
# >>> meta_sdss_medium = meta_sdss.groupby('class').apply(lambda x: x.sample(n=min(len(x), 200000), random_state=42)).reset_index(drop=True)
# >>> meta_lamost_small = meta_lamost_medium.groupby('class').apply(lambda x: x.sample(n=min(len(x), 30000), random_state=42)).reset_index(drop=True)
# >>> meta_sdss_small = meta_sdss_medium.groupby('class').apply(lambda x: x.sample(n=min(len(x), 30000), random_state=42)).reset_index(drop=True)
# >>> meta_lamost_medium.to_csv("meta-lamost-medium.csv", index=False)
# >>> meta_lamost_small.to_csv("meta-lamost-small.csv", index=False)
# >>> meta_sdss_medium.to_csv("meta-sdss-medium.csv", index=False)
# >>> meta_sdss_small.to_csv("meta-sdss-small.csv", index=False)
# 
# For LAMOST, we leveraged the available catalogue csv file distributed directly from their website:
# http://www.lamost.org/dr10/v2.0/catdl?name=dr10_v2.0_LRS_catalogue.csv.gz 
# http://www.lamost.org/dr10/v2.0/catdl?name=dr10_v2.0_LRS_stellar.csv.gz
# >>> meta_lamost = pd.read_csv("dr10_v2.0_LRS_catalogue.csv")
# >>> dr10_stellar = pd.read_csv("dr10_v2.0_LRS_stellar.csv")
# >>> additional_columns = [
# ...     'teff', 'teff_err', 'logg', 'logg_err', 'feh', 'feh_err',
# ...     'rv', 'rv_err', 'alpha_m', 'alpha_m_err', 'vsini_lasp'
# ... ]
# >>> merged_df = pd.merge(
# ...     meta_lamost,
# ...     dr10_stellar[['obsid', 'uid', 'gp_id', 'designation', 'obsdate', 'lmjd', 'mjd'] + additional_columns],
# ...     on=['obsid', 'uid', 'gp_id', 'designation', 'obsdate', 'lmjd', 'mjd'],
# ...     how='left'
# ... )
# >>> merged_df.to_csv("meta-lamost-big.csv", index=False)


# Parse the fits files and save the flux and wavelengths in a separate csv file
def generate_csv(opt, dataframe):
    datasource = opt.dataset.split("-")[0]
    fitsdir = './data/' + datasource + '-fits/'
    csvdir = './data/' + datasource + '-csv/'
    if not os.path.exists(csvdir):
        os.makedirs(csvdir, exist_ok=True)

    if datasource == "lamost":
        for index, row in dataframe.iterrows():
            specdir=row["obsdate"].replace("-","") + '/' + row["planid"] + '/'
            outputfilename=f'spec-lamost-{row["class"]}-{row["subclass"]}-{row["obsdate"]}-{row["planid"]}-{row["lmjd"]}-{row["spid"]}-{row["fiberid"]}.csv.gz'
            if not os.path.exists(csvdir + specdir):
                os.makedirs(csvdir + specdir, exist_ok=True)
            if os.path.isfile(csvdir + specdir + outputfilename):
                continue

            specfilename=f'spec-{row["lmjd"]}-{row["planid"]}_sp{str(row["spid"]).zfill(2)}-{str(row["fiberid"]).zfill(3)}.fits.gz'
            specfits=fits.open(fitsdir + specdir + specfilename)
            
            # https://github.com/fandongwei/pylamost/blob/master/pylamost.py
            len_list=len(specfits)
            if 1==len_list:
                head = specfits[0].header
                scidata = specfits[0].data
                coeff0 = head['COEFF0']
                coeff1 = head['COEFF1']
                pixel_num = head['NAXIS1']
                specflux = scidata[0,]
                spec_noconti = scidata[2,]
                wavelength=numpy.linspace(0,pixel_num-1,pixel_num)
                wavelength=numpy.power(10,(coeff0+wavelength*coeff1))
            elif 2==len_list:
                scidata = specfits[1].data
                wavelength = scidata[0][2]
                specflux = scidata[0][0]

            wavelength = wavelength.astype(np.float64)
            flux = specflux.astype(np.float64)
            df = pd.DataFrame({ 'wavelength': wavelength, 'flux': flux })

            # Applying red shift correction
            # https://voyages.sdss.org/preflight/light/redshift/
            # https://skyserver.sdss.org/dr1/en/proj/basic/universe/redshifts.asp
            if (not opt.donotapplyredshift):
                df['flux'] = df['flux']/(1 + row["z"])

            df.to_csv(csvdir + specdir + outputfilename, index=False)
            specfits.close()
    else: # datasource == "sdss"
        for index, row in dataframe.iterrows():
            # specObjID is enough for us to get any extra data from object, and class will be used for classification
            outputfilename=f'spec-sdss-{row["class"]}-{row["subClass"]}-{row["specObjID"]}.csv.gz'
            if os.path.isfile(csvdir + outputfilename):
                continue

            specfilename=f'spec-{str(row["plate"]).zfill(4)}-{row["mjd"]}-{str(row["fiberID"]).zfill(4)}.fits'
            if os.path.exists(fitsdir + specfilename):
                specfits=fits.open(fitsdir + specfilename)
            elif os.path.exists(fitsdir + specfilename + ".gz"):
                specfits=fits.open(fitsdir + specfilename + ".gz")
            else:
                debug(opt, "File not found: " + fitsdir + specfilename)
                continue

            # https://data.sdss.org/datamodel/files/BOSS_SPECTRO_REDUX/RUN2D/spectra/PLATE4/spec.html
            data=np.array(specfits[1].data.tolist())[:,[0,1]]
            # Flux is store as log so restoring the absolute value
            data[:,1]=np.power(10,data[:,1])

            # Applying red shift correction
            # https://voyages.sdss.org/preflight/light/redshift/
            # https://skyserver.sdss.org/dr1/en/proj/basic/universe/redshifts.asp
            if (not opt.donotapplyredshift):
                data[:,0]=data[:,0]/(1 + row["z"])

            data = np.fliplr(data)

            np.savetxt(csvdir + outputfilename, data, header="wavelength,flux", delimiter=",", comments="")
            specfits.close()


def plot_csv(opt, dataframe):
    datasource = opt.dataset.split("-")[0]
    csvdir = './data/' + datasource + '-csv/'
    pngdir = './data/' + opt.imagedir + '/'

    if not os.path.exists(pngdir):
        os.makedirs(pngdir, exist_ok=True)

    plotter = SpectraPlotter(opt)

    for index, row in dataframe.iterrows():
        if datasource == "lamost":
            specdir=row["obsdate"].replace("-","") + '/' + row["planid"] + '/'
            if (opt.suffix):
                outputimagename=f'spec-lamost-{row["class"]}-{row["subclass"]}-{row["obsdate"]}-{row["planid"]}-{row["lmjd"]}-{row["spid"]}-{row["fiberid"]}-{opt.suffix}.png'
            else:
                outputimagename=f'spec-lamost-{row["class"]}-{row["subclass"]}-{row["obsdate"]}-{row["planid"]}-{row["lmjd"]}-{row["spid"]}-{row["fiberid"]}.png'
            csv=csvdir + specdir + f'spec-lamost-{row["class"]}-{row["subclass"]}-{row["obsdate"]}-{row["planid"]}-{row["lmjd"]}-{row["spid"]}-{row["fiberid"]}.csv.gz'
            if not os.path.exists(pngdir + specdir):
                os.makedirs(pngdir + specdir, exist_ok=True)
            if os.path.isfile(pngdir + specdir + outputimagename):
                continue
        elif datasource == "sdss":
            if (opt.suffix):
                outputimagename=f'spec-sdss-{row["class"]}-{row["subClass"]}-{row["specObjID"]}-{opt.suffix}.png'
            else:
                outputimagename=f'spec-sdss-{row["class"]}-{row["subClass"]}-{row["specObjID"]}.png'
            csv=csvdir + f'spec-sdss-{row["class"]}-{row["subClass"]}-{row["specObjID"]}.csv.gz'
            if os.path.isfile(pngdir + outputimagename):
                continue
        else: # datasource == "desi"
            if (opt.suffix):
                outputimagename=f'spec-desi-GALAXY-UNKNOWN-{row["targetid"]}-{opt.suffix}.png'
            else:
                outputimagename=f'spec-desi-GALAXY-UNKNOWN-{row["targetid"]}.png'
            csv=csvdir + f'{row["targetid"]}.csv.gz'
            if os.path.isfile(pngdir + outputimagename):
                continue

        if not os.path.isfile(csv):
            print(f"Couldn't find file {csv}")
            continue
        spectra=pd.read_csv(csv)
        # Normalizing spectra using first and last quartile of the entire sdss-big spread of wavelengths
        lowest_wavelength = 3536.714
        highest_wavelength = 10413.580
        spectra = spectra[spectra['wavelength'] > lowest_wavelength]
        spectra = spectra[spectra['wavelength'] < highest_wavelength]
        # spectra['flux'] = spectra['flux'] - spectra['flux'].mean()

        img = plotter.plot_spectra(spectra)

        if datasource == "lamost":
            img.save(pngdir + specdir + outputimagename)
        else:
            img.save(pngdir + outputimagename)

def plot_datasetinfo(name, dataframe):
    outputimage = f'spectrer_wavelength_limits_{name}.png'
    plt.figure(figsize=(16,9))
    subset = ['waveMin', 'waveMax']
    subset_dataframe = dataframe[subset]
    subset_dataframe.boxplot()
    plt.xlabel('Columns')
    plt.ylabel('Values')
    plt.title(f'Dataset spread of wavelengths limits - {name}')
    table=[]
    for col in subset_dataframe.columns:
        q1 = subset_dataframe[col].quantile(0.25)
        q3 = subset_dataframe[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5*iqr
        upper_bound = q3 + 1.5*iqr
        outliers = subset_dataframe[(subset_dataframe[col] < lower_bound) | (subset_dataframe[col] > upper_bound)][col]
        table.append(["Outliers for " + col,len(outliers)])
    plt.table(cellText=table, loc='bottom', bbox=[0.0,-0.5,1,0.3])
    plt.tight_layout()
    plt.savefig(outputimage)
    plt.close()

    outputimage = f'spectrer_wavelength_coverage_{name}.png'
    plt.figure(figsize=(16,9))
    subset = ['wCoverage']
    subset_dataframe = dataframe[subset]
    subset_dataframe.boxplot()
    plt.xlabel('Columns')
    plt.ylabel('Values')
    plt.title(f'Dataset spread of wavelengths coverage - {name}')
    table=[]
    for col in subset_dataframe.columns:
        q1 = subset_dataframe[col].quantile(0.25)
        q3 = subset_dataframe[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5*iqr
        upper_bound = q3 + 1.5*iqr
        outliers = subset_dataframe[(subset_dataframe[col] < lower_bound) | (subset_dataframe[col] > upper_bound)][col]
        table.append(["Outliers for " + col,len(outliers)])
    plt.table(cellText=table, loc='bottom', bbox=[0.0,-0.5,1,0.3])
    plt.tight_layout()
    plt.savefig(outputimage)
    plt.close()

    outputimage = f'spectrer_signaltonoise_{name}.png'
    plt.figure(figsize=(16,9))
    subset = ['snMedian_u', 'snMedian_g', 'snMedian_r', 'snMedian_i', 'snMedian_z', 'snMedian']
    subset_dataframe = dataframe[subset]
    subset_dataframe.boxplot()
    plt.xlabel('Columns')
    plt.ylabel('Values')
    plt.title(f'Dataset spread of signal to noise - {name}')
    table=[]
    for col in subset_dataframe.columns:
        q1 = subset_dataframe[col].quantile(0.25)
        q3 = subset_dataframe[col].quantile(0.75)
        iqr = q3 - q1
        # FIXME: if we handle lower_bound like this, it goes < 0 and is useless
        lower_bound = q1 - 1.5*iqr
        upper_bound = q3 + 1.5*iqr
        outliers = subset_dataframe[(subset_dataframe[col] < lower_bound) | (subset_dataframe[col] > upper_bound)][col]
        table.append(["Outliers for " + col,len(outliers)])
    plt.table(cellText=table, loc='bottom', bbox=[0.0,-0.5,1,0.3])
    plt.tight_layout()
    plt.savefig(outputimage)
    plt.close()

def debug(opt, message):
    if opt.debug:
        print(message)

def initialize_pool(_dataframe):
    global dataframe_full
    dataframe_full = _dataframe

def run(opt):
    if opt.fillorder not in ["standard", "boustrophedon"]:
        debug(opt, "Wrong value given for --fillorder")
        return
    inputfile= './data/meta-' + opt.dataset + '.csv'
    spectradb = pd.read_csv(inputfile)
    debug(opt, f'Input spectra file contains {spectradb.shape[0]} entries')
    datasource = opt.dataset.split("-")[0]
    if (datasource == "lamost"):
        # https://www.lamost.org/dr10/v2.0/doc/lr-data-production-description
        spectradb = spectradb[spectradb['z'] != -9999]
        spectradb = spectradb[spectradb['z_err'] != -9999]
        debug(opt, f'Filtered spectra file of problematic redshift entries contains {spectradb.shape[0]} entries')
        snrcols = ['snru', 'snrg', 'snrr', 'snri', 'snrz']
        spectradb = spectradb[~spectradb[snrcols].isin([-9999]).any(axis=1)]
        debug(opt, f'Filtered spectra file of problematic snr entries contains {spectradb.shape[0]} entries')
        spectradb['snMedian'] = np.sqrt((spectradb[snrcols] ** 2).sum(axis=1))
        debug(opt, f'Calculated median SNR for remaining entries')
    elif (datasource =="sdss"):
        if (opt.filtersubclass):
            spectradb = spectradb[spectradb['subClass'].notna()]
            debug(opt, f'Filtered subclass spectra file contains {spectradb.shape[0]} entries')
        if (opt.nostars):
            spectradb = spectradb[spectradb['class'] != "STAR"]
            debug(opt, f'Filtered with no stars spectra file contains {spectradb.shape[0]} entries')
        if (opt.noquasars):
            spectradb = spectradb[spectradb['class'] != "QSO"]
            debug(opt, f'Filtered with no quasars spectra file contains {spectradb.shape[0]} entries')
        if (opt.nogalaxies):
            spectradb = spectradb[spectradb['class'] != "GALAXY"]
            debug(opt, f'Filtered with no galaxies spectra file contains {spectradb.shape[0]} entries')
        spectradb['wCoverage'] = np.power(10,spectradb['wCoverage'])
        spectradb['subClass'] = spectradb['subClass'].str.replace(r'[^A-Za-z0-9]', '', regex=True)
    elif (datasource == "desi"):
        spectradb['targetid'] = spectradb['targetid'].astype(str)
        pass
    else:
        debug(opt, "Inexistent datasource provided as input to dataset")
        return

    if (opt.filterlowsnr):
        spectradb = spectradb[spectradb['snMedian'] > opt.lowsnrthreshold]
        debug(opt, f'Filtered lowsnr spectra file contains {spectradb.shape[0]} entries')

    for column in spectradb.columns:
        if pd.api.types.is_numeric_dtype(spectradb[column]):
            spectradb[column] = spectradb[column].fillna(0)
        else:
            spectradb[column] = spectradb[column].fillna('null')

    if (opt.njobs != -1):
        num_cores = opt.njobs
    else:
        num_cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(num_cores, initializer = initialize_pool, initargs = (spectradb,))
    debug(opt, f'Starting execution with {num_cores} processes')

    df_split = np.array_split(spectradb, num_cores)

    if (opt.plotdataset):
        if (datasource == "lamost"):
            debug(opt, "FIXME: NOT SUPPORTED YET")
            return
        debug(opt, "Starting dataset plot process")
        plot_datasetinfo("full", spectradb)
        plot_datasetinfo("star", spectradb[spectradb["class"] == "STAR"])
        plot_datasetinfo("qso", spectradb[spectradb["class"] == "QSO"])
        plot_datasetinfo("galaxy", spectradb[spectradb["class"] == "GALAXY"])
        return

    if (datasource != "desi"):
        debug(opt, "Starting CSV generation process")
        partial_generate_csv = partial(generate_csv, opt)
        pool.map(partial_generate_csv, df_split)
    debug(opt, "Starting CSV plotting process")
    partial_plot_csv = partial(plot_csv, opt)
    pool.map(partial_plot_csv, df_split)

    pool.close()
    pool.join()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='sdss-small', type=str,
                    help='Metadata of dataset to use, will look for corresponding ./data/meta-<EXPERIMENT>.csv file')
    parser.add_argument('--imagedir', default='png-sdss-small', type=str,
                        help='directory to store png files: ./data/<IMAGES>')
    parser.add_argument('--debug', action='store_true', help='a debugging flag to be used when testing.')
    parser.add_argument('--plotdataset', action='store_true', help='plot multiple graphs showing information about the dataset')
    parser.add_argument('--filtersubclass', action='store_true', help='filter generated CSV/PNG files to not include entries with missing subclass')
    parser.add_argument('--filterlowsnr', action='store_true', help='filter generated CSV/PNG files to not include entries with low snr')
    parser.add_argument('--donotapplyredshift', action='store_true', help='Avoid applying redshift correction to CSV/plots')
    parser.add_argument('--lowsnrthreshold', default=5.0, type=float, help='set threshold for minimum snr if filtering')
    parser.add_argument('--overlapsplit', action='store_true', help='divides fluxes in 3 regions and plots them in overlapping fashion')
    parser.add_argument('--map2d', action='store_true', help='plots each flux as a 3x3 square in a 224x224 image')
    parser.add_argument('--map2dnormal', action='store_true', help='normalizes and plots each flux as a 3x3 square in a 224x224 image')
    parser.add_argument('--map2dnormaldev', action='store_true', help='normalizes and plots each flux, derivative and second derivative as a 3x3 square in a 224x224 image')
    parser.add_argument('--map2droi', action='store_true', help='plots each flux as a 3x3 square in a 224x224 image and marks regions of interest in it as well')
    parser.add_argument('--fillorder', default='standard', type=str,
                        help='fill order to use when generating map2d: standard, boustrophedon. Default is standard')
    parser.add_argument('--blocksize', default=3, type=int, help='set block size for map2d (default = 3)')
    parser.add_argument('--interpolate', action='store_true', help='when doing map2d, interpolate missing flux values when normalizing')
    parser.add_argument('--nostars', action='store_true', help='remove stars from the generated dataset')
    parser.add_argument('--noquasars', action='store_true', help='remove quasars from the generated dataset')
    parser.add_argument('--nogalaxies', action='store_true', help='remove galaxies from the generated dataset')
    parser.add_argument('--suffix', default='', type=str, help='append string the generated image name')

    parser.add_argument('--njobs', default=-1, type=int,
                    help='Number of cores to use')
    opt = parser.parse_args()
    run(opt)

