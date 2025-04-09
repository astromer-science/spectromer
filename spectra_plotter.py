# -----------------------------------------------------------------------------
# MIT License
# Copyright (c) 2025 Luis Felipe Strano Moraes
#
# This file is part of Spectromer
# For license terms, see the LICENSE file in the root of this repository.
# -----------------------------------------------------------------------------

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from PIL import Image
from scipy.signal import savgol_filter
from scipy.integrate import trapezoid
import scipy.integrate
import io

class SpectraPlotter:
    def __init__(self, options):
        self.opt = options

    def plot_spectra(self, spectra):
        opt = self.opt
        min_flux = -1.13  # Adjust these values if necessary
        max_flux = 144.97
        if opt.overlapsplit:
            return self.plot_overlapsplit(spectra)
        elif opt.map2d:
            return self.plot_map2d(spectra, min_flux, max_flux)
        elif opt.map2dnormal or opt.map2dnormaldev:
            return self.plot_map2dnormal(spectra)
        elif opt.map2droi:
            return self.plot_map2d_roi(spectra)
        else:
            return self.plot_default(spectra)

    def plot_overlapsplit(self, spectra):
        opt = self.opt
        groups = np.array_split(spectra['flux'], 3)
        wavelength_groups = np.array_split(spectra['wavelength'], 3)
        colors = ['red', 'green', 'blue']
        plt.figure(figsize=(2.24, 2.24))
        for i, group in enumerate(groups):
            min_wavelength = np.min(wavelength_groups[i])
            max_wavelength = np.max(wavelength_groups[i])
            normalized_wavelength = (wavelength_groups[i] - min_wavelength) / (max_wavelength - min_wavelength)
            plt.plot(normalized_wavelength, group, color=colors[i], linewidth=0.5, alpha=0.5)
        plt.axis('off')
        plt.tight_layout(pad=0)
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', pad_inches=0)
        plt.close()
        buf.seek(0)
        img = Image.open(buf)
        return img

    def plot_map2d(self, spectra, min_flux, max_flux):
        opt = self.opt
        flux = spectra['flux'].values
        output_size = 224
        block_size = opt.blocksize
        blocks_per_side = output_size // block_size
        image = np.zeros((output_size, output_size))
        for idx, value in enumerate(flux):
            row = (idx // blocks_per_side) * block_size
            col = (idx % blocks_per_side) * block_size
            image[row:row + block_size, col:col + block_size] = value
        plt.figure(figsize=(2.24, 2.24))
        plt.imshow(image, cmap='viridis', vmin=min_flux, vmax=max_flux)
        plt.axis('off')
        plt.gca().set_position([0, 0, 1, 1])
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', pad_inches=0)
        plt.close()
        buf.seek(0)
        img = Image.open(buf)
        return img

    def plot_map2dnormal(self, spectra):
        opt = self.opt
        wavelength = spectra['wavelength'].values
        flux = spectra['flux'].values
        img_size = 224
        block_size = opt.blocksize
        blocks_per_row = img_size // block_size
        blocks_per_column = img_size // block_size
        total_blocks = blocks_per_row * blocks_per_column
        if opt.interpolate and len(flux) < total_blocks:
            interp_func = interp1d(wavelength, flux, kind='linear', fill_value="extrapolate")
            new_wavelengths = np.linspace(min(wavelength), max(wavelength), total_blocks)
            flux = interp_func(new_wavelengths)
        min_flux = np.min(flux)
        max_flux = np.max(flux)
        normalized_flux = 255 * (flux - min_flux) / (max_flux - min_flux)
        if opt.map2dnormaldev:
            first_derivative = np.gradient(flux)
            second_derivative = np.gradient(first_derivative)
            first_derivative_norm = 255 * (first_derivative - np.min(first_derivative)) / (np.max(first_derivative) - np.min(first_derivative))
            second_derivative_norm = 255 * (second_derivative - np.min(second_derivative)) / (np.max(second_derivative) - np.min(second_derivative))
        image = np.zeros((img_size, img_size, 3), dtype=np.uint8)
        flux_idx = 0
        if opt.fillorder == "standard":
            for i in range(blocks_per_column):
                for j in range(blocks_per_row):
                    if flux_idx < len(flux):
                        image[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size, 0] = int(normalized_flux[flux_idx])
                        if opt.map2dnormaldev:
                            image[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size, 1] = int(first_derivative_norm[flux_idx])
                            image[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size, 2] = int(second_derivative_norm[flux_idx])
                        flux_idx += 1
        elif opt.fillorder == "boustrophedon":
            for i in range(blocks_per_column):
                if i % 2 == 0:
                    for j in range(blocks_per_row):
                        if flux_idx < len(flux):
                            image[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size, 0] = int(normalized_flux[flux_idx])
                            if opt.map2dnormaldev:
                                image[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size, 1] = int(first_derivative_norm[flux_idx])
                                image[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size, 2] = int(second_derivative_norm[flux_idx])
                            flux_idx += 1
                else:
                    for j in range(blocks_per_row-1, -1, -1):
                        if flux_idx < len(flux):
                            image[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size, 0] = int(normalized_flux[flux_idx])
                            if opt.map2dnormaldev:
                                image[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size, 1] = int(first_derivative_norm[flux_idx])
                                image[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size, 2] = int(second_derivative_norm[flux_idx])
                            flux_idx += 1
        img = Image.fromarray(image)
        return img

    def plot_map2d_roi(self, spectra):
        opt = self.opt
        flux = spectra['flux'].values
        wavelength = spectra['wavelength'].values
        output_size = 224
        block_size = opt.blocksize
        blocks_per_side = output_size // block_size

        # Create a copy of the spectra DataFrame
        spectra_df = spectra.copy().reset_index(drop=True)

        # Apply Savitzky-Golay filter for smoothing
        window_length = 25
        polyorder = 3
        spectra_df['smoothed_flux'] = savgol_filter(spectra_df['flux'], window_length, polyorder)

        # Calculate the rolling median on the smoothed data
        rolling_window_size = 50
        spectra_df['rolling_median'] = spectra_df['smoothed_flux'].rolling(window=rolling_window_size, center=True).median()

        # Define a threshold for significant deviation
        deviation_threshold = 1.5

        # Identify regions where flux deviates significantly from rolling median
        spectra_df['deviation'] = spectra_df['smoothed_flux'] - spectra_df['rolling_median']
        spectra_df['significant'] = np.abs(spectra_df['deviation']) > deviation_threshold
        # Handle NaN values resulting from rolling median
        spectra_df['significant'].fillna(False, inplace=True)

        # Find the start and end points of regions of interest
        regions_of_interest = []
        in_region = False
        start_index = None

        for i in range(len(spectra_df)):
            if spectra_df.loc[i, 'significant'] and not in_region:
                in_region = True
                start_index = i
            elif not spectra_df.loc[i, 'significant'] and in_region:
                in_region = False
                end_index = i - 1  # Subtract 1 because current point is not significant
                region_df = spectra_df.iloc[start_index:end_index + 1]
                normalized_flux = (region_df['smoothed_flux'] - region_df['rolling_median']) / region_df['rolling_median']
                # Calculate equivalent width
                equivalent_width = trapezoid(normalized_flux, region_df['wavelength'])
                # Store the region information
                regions_of_interest.append((start_index, end_index, equivalent_width))
        # Check if we're still in a region at the end
        if in_region:
            end_index = len(spectra_df) - 1
            region_df = spectra_df.iloc[start_index:end_index + 1]
            normalized_flux = (region_df['smoothed_flux'] - region_df['rolling_median']) / region_df['rolling_median']
            equivalent_width = trapezoid(normalized_flux, region_df['wavelength'])
            regions_of_interest.append((start_index, end_index, equivalent_width))

        # Sort the regions by equivalent width in descending order and select the top 50
        regions_of_interest = sorted(regions_of_interest, key=lambda x: abs(x[2]), reverse=True)[:50]

        # Initialize the image array with zeros
        image = np.zeros((output_size, output_size, 3), dtype=np.uint8)

        # Normalize flux to the range 0-255 for the red channel
        min_flux = np.min(flux)
        max_flux = np.max(flux)
        normalized_flux = 255 * (flux - min_flux) / (max_flux - min_flux)

        # Prepare an array indicating whether each flux point is in ROI
        roi_flags = np.zeros_like(flux, dtype=bool)
        for start_idx, end_idx, eq_width in regions_of_interest:
            roi_flags[start_idx:end_idx + 1] = True
        # Fill the image array
        flux_idx = 0
        for i in range(blocks_per_side):
            for j in range(blocks_per_side):
                if flux_idx < len(flux):
                    row = i * block_size
                    col = j * block_size
                    # Set the red channel based on normalized flux
                    image[row:row + block_size, col:col + block_size, 0] = int(normalized_flux[flux_idx])
                    # Set the green channel to a light green if ROI condition is met
                    if roi_flags[flux_idx]:
                        image[row:row + block_size, col:col + block_size, 1] = 50  # Light green
                    flux_idx += 1

        # Create an image from the array
        img = Image.fromarray(image)
        return img

    def plot_default(self, spectra):
        plt.figure(figsize=(2.24, 2.24))
        plt.plot(spectra['wavelength'], spectra['flux'], linewidth=0.5)
        plt.axis('off')
        plt.tight_layout(pad=0)
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', pad_inches=0)
        plt.close()
        buf.seek(0)
        img = Image.open(buf)
        return img

