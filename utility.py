import re
import pandas as pd
import cdsapi
from scipy.interpolate import interp1d
from datetime import datetime
import os
import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset
from scipy.constants import N_A
import unicodedata
import copy
from scipy.stats import pearsonr


# Proportion of total solar radiation that is PAR (400-700 nm)
PAR_PROPORTION = 0.45

# Broadband conversion factor from Watts (J/s) to quanta/s
# From Morel and Smith (1974), cited in Kirk (2011), p. 19.
# Units: photons s⁻¹ W⁻¹ or photons J⁻¹
Q_PER_JOULE = 2.77e18

# Derived constant for converting energy (Joules) to moles of photons
MOLES_PER_JOULE = Q_PER_JOULE / N_A  # Units: moles J⁻¹

def save_plot(fig, lake_name, title, subfolder=None):
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'plots')
    
    # Clean up the title to make it filename-safe
    filename = title.replace(' ', '_').replace('-', '_').replace(':', '').replace(',', '') + '.png'
    
    # Create the full path: save_dir/lake_name/subfolder (if provided)
    lake_dir = os.path.join(save_dir, lake_name)
    if subfolder:
        lake_dir = os.path.join(lake_dir, subfolder)
    os.makedirs(lake_dir, exist_ok=True)
    
    # Save the plot
    fig.savefig(os.path.join(lake_dir, filename), bbox_inches='tight')
    plt.close(fig)  # Close the figure after saving to free up memory

def clean_gps_format(gps):
    if isinstance(gps, str) and '°' in gps:
        parts = gps.split(', ')
        latitude = parts[0]
        longitude = parts[1]

        latitude_parts = latitude.split('°')
        latitude_parts[1] = latitude_parts[1].replace('.', '', 1)
        latitude = '.'.join(latitude_parts)

        longitude_parts = longitude.split('°')
        longitude_parts[1] = longitude_parts[1].replace('.', '', 1)
        longitude = '.'.join(longitude_parts)

        gps = f"{latitude}, {longitude}"
        gps = gps.replace(' N', '')
        gps = gps.replace(' E', '')
    return gps

def parse_lat_lon(gps):
    if isinstance(gps, str):
        gps = re.sub(r'[^\d.,-]', '', gps)  # Remove unwanted characters
        lat, lon = gps.split(',')
        return float(lat), float(lon)
    else:
        return None, None


def download_era5_data(date, latitude, longitude, filepath):

    # Sanitize file path
    filepath = filepath.strip()
    directory = os.path.dirname(filepath)

    # Ensure the directory exists
    if not os.path.exists(directory):
        print(f"Creating directory: {directory}")
        os.makedirs(directory, exist_ok=True)
    else:
        print(f"Directory exists: {directory}")

    # Initialize CDS API client
    client = cdsapi.Client()

    # Define request parameters
    dataset = "reanalysis-era5-single-levels"
    request = {
        "product_type": "reanalysis",
        "variable": "surface_solar_radiation_downwards",
        "year": date.year,
        "month": f"{date.month:02}",
        "day": f"{date.day:02}",
        "time": [f"{hour:02}:00" for hour in range(24)],
        "area": [
            latitude + 0.25, longitude - 0.25,  # North, West
            latitude - 0.25, longitude + 0.25,  # South, East
        ],
        "format": "netcdf",
    }

    print(f"Downloading ERA5 data to: {filepath}")
    print(f"Request parameters: {request}")

    try:
        # Download the data
        client.retrieve(dataset, request).download(filepath)
        print(f"Download successful: {filepath}")
    except Exception as e:
        print(f"Error during download: {e}")

def load_irr_data(era5_filepath):
    try:
        print(f"Loading ERA5 data from: {era5_filepath}")
        with Dataset(era5_filepath, "r") as era5_data:
            swdown_vector = era5_data.variables['ssrd'][:]
            
            # ERA5 SSRD is in Joules per hour, averaged over spatial dimensions
            swdown_joules_per_hour = np.mean(swdown_vector, axis=(1, 2))
            
            # --- Convert to daily and maximum PAR values ---
            
            # Calculate total daily energy (sum of hourly values)
            daily_irradiance = swdown_joules_per_hour.sum().item()
            
            # Calculate max hourly energy
            max_irradiance = swdown_joules_per_hour.max().item()

            # Convert total solar energy (Joules) to PAR energy (Joules)
            daily_par_joules = daily_irradiance * PAR_PROPORTION
            max_par_joules = max_irradiance * PAR_PROPORTION
            hourly_par_joules = swdown_joules_per_hour * PAR_PROPORTION
            
            # --- Convert PAR in Joules to PAR in moles of photons ---
            daily_par_moles = daily_par_joules * MOLES_PER_JOULE
            max_par_moles = max_par_joules * MOLES_PER_JOULE
            hourly_par_moles = hourly_par_joules * MOLES_PER_JOULE

            return {
                'daily_par_moles': daily_par_moles,
                'max_par_moles': max_par_moles,
                'hourly_par_moles': hourly_par_moles,
            }
    except Exception as e:
        print(f"Error loading ERA5 data: {e}")
        raise


def integrate_hours(x):
    time = np.arange(24)
    
    # Perform integration using np.trapz
    return np.trapz(x, x=time)


def integrate_wavelengths(x):
    wavelengths = np.arange(400, 710, 10)
    
    # Perform integration using np.trapz
    return np.trapz(x, x=wavelengths)

def integrate_depth(x, depths, z_lim):
    mask = depths <= z_lim
    x_limited = x[mask]
    depths_limited = depths[mask]

    # Perform the integration using np.trapz
    integrated_value = np.trapz(x_limited, x=depths_limited)

    return integrated_value


def calculate_pb_opt(sst):
    if sst < -10.0:
        return 0.00
    elif sst < -1.0:
        return 1.13
    elif sst > 28.5:
        return 4.00
    else:
        return (
            1.2956 + 2.749e-1 * sst + 6.17e-2 * sst ** 2 - 2.05e-2 * sst ** 3
            + 2.462e-3 * sst ** 4 - 1.348e-4 * sst ** 5 + 3.4132e-6 * sst ** 6
            - 3.27e-8 * sst ** 7
        )
    
def linear_interpolate(depths, values):
    values = values.astype(np.float64, copy=False)

    # Identify non-NaN points
    valid_mask = ~np.isnan(values)
    valid_depths = depths[valid_mask]
    valid_values = values[valid_mask]

    # If fewer than 2 valid points, can't interpolate => return original array
    if len(valid_depths) < 2:
        return values

    # Set up interpolation
    f = interp1d(valid_depths, valid_values, kind='linear', 
                  bounds_error=False, fill_value="extrapolate")

    out = f(depths)
    # Clamp negative results to zero
    out[out < 0] = 0
    return out
    
def extract_unique_string(df, column_name):
    # Filter out NaN values
    non_nan_values = df[column_name].dropna().unique()
    
    # Check if there's exactly one unique string value
    if len(non_nan_values) == 1:
        return non_nan_values[0]
    elif len(non_nan_values) > 1:
        raise ValueError("Multiple unique non-NaN strings found.")
    else:
        return None 
    
def extract_incubation_time(df, column_name):
    # Filter out invalid values
    valid_times = df[column_name].dropna().apply(lambda x: x if 0 <= x <= 24 else None).dropna().unique()
    
    # Check if there's exactly one unique valid time
    if len(valid_times) == 1:
        return valid_times[0]
    else:
        return None

def time_to_hours(time_str):
    # Check if the time string is NaN
    if pd.isna(time_str):
        return np.nan
    # Split the time string into hours and minutes
    hours, minutes = map(int, time_str.split(':'))
    # Convert to float hours
    return hours + minutes / 60.0


def sanitize_string(input_string):
    invalid_chars = [":", "*", "?", '"', "<", ">", "|", "\\", "/"]
    for char in invalid_chars:
        input_string = input_string.replace(char, "_")
    
    # Normalize special characters (e.g., ä -> a, ö -> o)
    input_string = unicodedata.normalize('NFKD', input_string).encode('ascii', 'ignore').decode('ascii')
    
    return input_string



def separate_outliers(all_lakes):
    all_lakes_no_outliers = {}
    all_lakes_only_outliers = {}

    for lake_name, lake_obj in all_lakes.items():
        # Make copies so modifications do not affect the original data
        lake_no_outliers = copy.deepcopy(lake_obj)
        lake_only_outliers = copy.deepcopy(lake_obj)

        # Filter stations in each copied lake
        lake_no_outliers.stations = [
            stn for stn in lake_no_outliers.stations if not stn.is_outlier
        ]
        lake_only_outliers.stations = [
            stn for stn in lake_only_outliers.stations if stn.is_outlier
        ]

        # Store in the new dictionaries
        all_lakes_no_outliers[lake_name] = lake_no_outliers
        all_lakes_only_outliers[lake_name] = lake_only_outliers

    return all_lakes_no_outliers, all_lakes_only_outliers



def create_station_level_tidy_df(all_lakes, groups=None, min_obs_pp=0.0):
    for lake in all_lakes.values():
        lake.classify_stations()

    records = []

    stations_processed = 0
    stations_removed_low_pp = 0

    for lake_name, lake_obj in all_lakes.items():
        for station in lake_obj.stations:
            stations_processed += 1

            obs_integrated_pp = station.variables.get('pp_hourly_depth_integrated', {}).get('value')

            if obs_integrated_pp is None or obs_integrated_pp < min_obs_pp:
                stations_removed_low_pp += 1
                continue

            if not hasattr(station, 'models'):
                continue

            for model in station.models:
                mae = getattr(model, 'mae', np.nan)
                mape = getattr(model, 'mape', np.nan)
                pcorr = getattr(model, 'pattern_correlation', np.nan)
                case = getattr(model, 'case', np.nan) 
                
                if np.isnan(mape) and np.isnan(pcorr):
                    continue

                try:
                    base_model = model.name[2]       # P-I function: A/B/C/D
                    corr_method = model.name[1]      # Upwelling correction: A/B
                    vert_uni = model.name[0] == 'A'  # Vertical resolution: A=uniform, B=depth-resolved
                except IndexError:
                    continue

                record = {
                    'lake': lake_name,
                    'station_date': station.date,
                    'model_full_name': model.name,
                    'base_model': base_model,
                    'correction': corr_method,
                    'uniform_chl': vert_uni,
                    'MAE': mae,
                    'MAPE': mape,
                    'p_corr': pcorr,
                    'case': case
                }
                records.append(record)

    if stations_processed > 0:
        removal_percentage = (stations_removed_low_pp / stations_processed) * 100
        print("\n--- Data Filtering Summary ---")
        print(f"  - Filtered out {stations_removed_low_pp} of {stations_processed} total stations ({removal_percentage:.1f}%) due to low observed PP (< {min_obs_pp}).")
        print("------------------------------\n")

    station_df = pd.DataFrame(records)

    if groups and not station_df.empty:
        group_map = {}
        for group_name, lake_dict in groups.items():
            for lake_name_in_group in lake_dict.keys():
                group_map[lake_name_in_group] = group_name
        station_df['group'] = station_df['lake'].map(group_map)
        
    return station_df


def generate_summary_tables(station_df):
    if station_df.empty:
        return {}

    q25 = lambda x: x.quantile(0.25)
    q75 = lambda x: x.quantile(0.75)

    detailed_summary = station_df.groupby(['group', 'base_model', 'uniform_chl', 'correction']).agg(
        median_r=('p_corr', 'median'), q1_r=('p_corr', q25), q3_r=('p_corr', q75),
        median_mape=('MAPE', 'median'), q1_mape=('MAPE', q25), q3_mape=('MAPE', q75),
        count=('p_corr', 'count')
    ).reset_index()

    paired_chl = pd.DataFrame()
    chl_effect_df = pd.DataFrame()
    if station_df['uniform_chl'].nunique() == 2:
        df_uniform = station_df[station_df['uniform_chl'] == True]; df_resolved = station_df[station_df['uniform_chl'] == False]
        paired_chl = pd.merge(df_uniform, df_resolved, on=['station_date', 'base_model', 'correction'], suffixes=('_uniform', '_resolved'))
        if not paired_chl.empty:
            paired_chl['delta_r'] = paired_chl['p_corr_resolved'] - paired_chl['p_corr_uniform']
            paired_chl['delta_mape'] = paired_chl['MAPE_resolved'] - paired_chl['MAPE_uniform']
            chl_effect_df = pd.DataFrame({'Metric': ['Pearson r change', 'MAPE change (%)'], 'Median Delta': [paired_chl['delta_r'].median(), paired_chl['delta_mape'].median()], 'IQR': [f"{paired_chl['delta_r'].quantile(0.25):.2f} to {paired_chl['delta_r'].quantile(0.75):.2f}", f"{paired_chl['delta_mape'].quantile(0.25):.2f} to {paired_chl['delta_mape'].quantile(0.75):.2f}"]})
    paired_upwelling = pd.DataFrame()
    upwelling_effect_df = pd.DataFrame()
    if station_df['correction'].nunique() == 2:
        df_A = station_df[station_df['correction'] == 'A']; df_B = station_df[station_df['correction'] == 'B']
        paired_upwelling = pd.merge(df_A, df_B, on=['station_date', 'base_model', 'uniform_chl'], suffixes=('_A', '_B'))
        if not paired_upwelling.empty:
            paired_upwelling['delta_r'] = paired_upwelling['p_corr_B'] - paired_upwelling['p_corr_A']
            paired_upwelling['delta_mape'] = paired_upwelling['MAPE_B'] - paired_upwelling['MAPE_A']
            upwelling_effect_df = pd.DataFrame({'Metric': ['Pearson r change', 'MAPE change (%)'], 'Median Delta': [paired_upwelling['delta_r'].median(), paired_upwelling['delta_mape'].median()], 'IQR': [f"{paired_upwelling['delta_r'].quantile(0.25):.2f} to {paired_upwelling['delta_r'].quantile(0.75):.2f}", f"{paired_upwelling['delta_mape'].quantile(0.25):.2f} to {paired_upwelling['delta_mape'].quantile(0.75):.2f}"]})

    best_idx = detailed_summary.loc[detailed_summary.groupby('group')['median_r'].idxmax()]
    regional_winners = best_idx.rename(columns={'group': 'Region', 'base_model': 'P-I Function', 'uniform_chl': 'Uniform Chl-a', 'correction': 'Upwelling Correction', 'median_r': 'Median r', 'median_mape': 'Median MAPE', 'count': 'No. Runs'})
    regional_winners = regional_winners[['Region', 'P-I Function', 'Uniform Chl-a', 'Upwelling Correction', 'Median r', 'Median MAPE', 'No. Runs']]

    def format_table(df, main_cols):
        df['r (Median, IQR)'] = df.apply(lambda r: f"{r['median_r']:.2f} ({r['q1_r']:.2f}-{r['q3_r']:.2f})", axis=1)
        df['MAPE (%) (Median, IQR)'] = df.apply(lambda r: f"{r['median_mape']:.1f} ({r['q1_mape']:.1f}-{r['q3_mape']:.1f})", axis=1)
        df_formatted = df[main_cols + ['r (Median, IQR)', 'MAPE (%) (Median, IQR)', 'count']]
        df_formatted = df_formatted.rename(columns={'count': 'No. Runs'})
        return df_formatted
    
    agg_dict = {
        'median_r': ('p_corr', 'median'), 'q1_r': ('p_corr', q25), 'q3_r': ('p_corr', q75),
        'median_mape': ('MAPE', 'median'), 'q1_mape': ('MAPE', q25), 'q3_mape': ('MAPE', q75),
        'count': ('p_corr', 'count')
    }

    # b) Table 1: P-I Model Comparison
    regional_pi = station_df.groupby(['group', 'base_model']).agg(**agg_dict).reset_index()
    global_pi = station_df.groupby('base_model').agg(**agg_dict).reset_index()
    global_pi['group'] = 'All Sites'
    table1_data = pd.concat([global_pi, regional_pi], ignore_index=True)
    table1_data.rename(columns={'group': 'Region', 'base_model': 'P-I Model'}, inplace=True)
    table1_pi_model = format_table(table1_data, ['Region', 'P-I Model'])

    # c) Table 2: Chl-a Resolution Comparison
    regional_chl = station_df.groupby(['group', 'uniform_chl']).agg(**agg_dict).reset_index()
    global_chl = station_df.groupby('uniform_chl').agg(**agg_dict).reset_index()
    global_chl['group'] = 'All Sites'
    table2_data = pd.concat([global_chl, regional_chl], ignore_index=True)
    table2_data.rename(columns={'group': 'Region', 'uniform_chl': 'Chl-a Profile'}, inplace=True)
    table2_data['Chl-a Profile'] = table2_data['Chl-a Profile'].map({True: 'Uniform', False: 'Resolved'})
    table2_chla_effect = format_table(table2_data, ['Region', 'Chl-a Profile'])

    # d) Table 3: Upwelling Correction Comparison
    regional_up = station_df.groupby(['group', 'correction']).agg(**agg_dict).reset_index()
    global_up = station_df.groupby('correction').agg(**agg_dict).reset_index()
    global_up['group'] = 'All Sites'
    table3_data = pd.concat([global_up, regional_up], ignore_index=True)
    table3_data.rename(columns={'group': 'Region', 'correction': 'Upwelling Correction'}, inplace=True)
    table3_upwelling_effect = format_table(table3_data, ['Region', 'Upwelling Correction'])
    
    for tbl in [table1_pi_model, table2_chla_effect, table3_upwelling_effect]:
        tbl['Region'] = pd.Categorical(tbl['Region'], categories=['All Sites', 'Baltic Sea', 'Estonian Lakes', 'Lake Geneva'], ordered=True)
        tbl.sort_values(by='Region', inplace=True)

    return {
        'detailed_breakdown': detailed_summary,
        'chl_a_effect_size': chl_effect_df,
        'upwelling_effect_size': upwelling_effect_df,
        'regional_winners': regional_winners,
        'Table2_PI_Model': table1_pi_model,
        'Table3_ChlA_Effect': table2_chla_effect,
        'Table4_Upwelling_Effect': table3_upwelling_effect,
        '_internal_paired_chl_df': paired_chl,
        '_internal_paired_upwelling_df': paired_upwelling
    }

def generate_summary_statistics_for_timeseries(lake_objects_dict, model_name, save_path=None):
    results = []
    analysis_cases = {
        'PRICKEN': [
            {'name': '2000-2012', 'start': '2000-01-01', 'end': '2012-12-31'},
            {'name': '2013-2020', 'start': '2013-01-01', 'end': '2020-12-31'}
        ],
        'Leman(Geneva Lake)': [
            {'name': '2000-2020', 'start': '2000-01-01', 'end': '2020-12-31'},
            {'name': '2019 Productive Season', 'start': '2019-04-01', 'end': '2019-10-31'}
        ]
    }

    for lake_name, lake_obj in lake_objects_dict.items():
        if lake_name not in analysis_cases:
            continue
        for period in analysis_cases[lake_name]:
            start_date = datetime.strptime(period['start'], '%Y-%m-%d')
            end_date = datetime.strptime(period['end'], '%Y-%m-%d')
            num_days_in_period = (end_date - start_date).days + 1

            for data_type in ['daily', 'hourly']:
                if data_type == 'daily':
                    pp_variable = 'pp_daily_depth_integrated'
                else:
                    pp_variable = 'pp_hourly_depth_integrated'

                records = []
                for station in sorted(lake_obj.stations, key=lambda s: s.date):
                    measured_val = station.variables.get(pp_variable, {}).get('value')
                    target_model = next((m for m in station.models if m.name == model_name), None)

                    if data_type == 'daily':
                        model_val = target_model.model_pp if target_model and hasattr(target_model, 'model_pp') else np.nan
                    else:
                        model_val = np.nan
                        if target_model and hasattr(target_model, 'model_pp_t'):
                            inc_hours = station.variables.get('incubation_hours', {}).get('value')
                            if inc_hours is not None and len(inc_hours) > 0:
                                model_val = np.mean(target_model.model_pp_t[inc_hours])

                    records.append({'date': station.date, 'measured': measured_val, 'modeled': model_val})

                df = pd.DataFrame(records).set_index(pd.to_datetime([r['date'] for r in records])).sort_index()
                df_period = df.loc[period['start']:period['end']]
                df_paired = df_period.dropna(subset=['measured', 'modeled'])

                # Initialize all variables
                n_points, mean_measured, mean_modeled, mape, r = len(df_paired), np.nan, np.nan, np.nan, np.nan
                total_measured_g = np.nan
                total_modeled_g = np.nan

                if len(df_paired) >= 2:
                    n_points = len(df_paired)
                    mean_measured = df_paired['measured'].mean()
                    mean_modeled = df_paired['modeled'].mean()
                    mape_df = df_paired[df_paired['measured'] > 0.01]
                    mape = (np.abs(mape_df['modeled'] - mape_df['measured']) / mape_df['measured']).mean() * 100 if not mape_df.empty else np.nan
                    r, _ = pearsonr(df_paired['measured'], df_paired['modeled']) if df_paired['measured'].nunique() > 1 and df_paired['modeled'].nunique() > 1 else (np.nan, np.nan)

                    if data_type == 'daily':
                        # Convert mean daily mg to total g for the period
                        if not np.isnan(mean_measured):
                            total_measured_g = mean_measured * num_days_in_period / 1000
                        if not np.isnan(mean_modeled):
                            total_modeled_g = mean_modeled * num_days_in_period / 1000

                results.append({
                    'Location': lake_name,
                    'Period': period['name'],
                    'Data Type': data_type,
                    'N': n_points,
                    'Mean Measured': mean_measured,
                    'Mean Modeled': mean_modeled,
                    'MAPE (%)': mape,
                    "Pearson's r": r,
                    'Total Measured (g C m-2)': total_measured_g,
                    'Total Modeled (g C m-2)': total_modeled_g
                })

    final_table = pd.DataFrame(results)

    if save_path:
        output_dir = os.path.dirname(save_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        final_table.round(4).to_csv(save_path, index=False)
        print(f"Final statistics table saved to: {save_path}")

    return final_table


def generate_site_characteristics_table(all_lakes, save_path=None, min_obs_pp=0.5):
    print(f"\n>>> Generating Updated Table 1 (Filter: Hourly PP < {min_obs_pp})...")

    static_meta = {
        'Leman(Geneva Lake)':  {'Lat': '46.45 N', 'Max_Depth': '310 m',   'Print_Name': 'Lake Geneva'},
        'Peipsi':             {'Lat': '58.84 N', 'Max_Depth': '12.9 m',  'Print_Name': 'Peipsi'},
        'Harku':              {'Lat': '59.41 N', 'Max_Depth': '2.5 m',   'Print_Name': 'Harku'},
        'Vortsjarv':          {'Lat': '58.12 N', 'Max_Depth': '6 m',     'Print_Name': 'Vortsjärv'},
        'PRICKEN':            {'Lat': '58.26 N', 'Max_Depth': '> 20 m',  'Print_Name': 'Baltic Sea, Pricken'},
        'SLAGGO':             {'Lat': '58.26 N', 'Max_Depth': '> 20 m',  'Print_Name': 'Baltic Sea, Släggö'},
        'BROA E':             {'Lat': '58.25 N', 'Max_Depth': '> 20 m',  'Print_Name': 'Baltic Sea, Broa E'}
    }

    table_rows = []

    for lake_name, lake_obj in all_lakes.items():
        if not lake_obj.stations:
            continue
            
        # --- 1. Filter Stations (Match Logic from create_station_level_tidy_df) ---
        valid_stations = []
        for s in lake_obj.stations:
            # CHECK 'pp_hourly_depth_integrated' NOT 'daily'
            pp_val = s.variables.get('pp_hourly_depth_integrated', {}).get('value', np.nan)
            
            # If PP is missing or below threshold, SKIP this station
            if pd.isna(pp_val) or pp_val < min_obs_pp:
                continue
                
            valid_stations.append(s)

        if not valid_stations:
            continue

        # --- 2. Calculate Stats on Valid Stations ---
        count = len(valid_stations)
        
        # Time Period
        years = [s.date.year for s in valid_stations]
        period = f"{min(years)} - {max(years)}"
        
        # Averages
        chla_values = []
        secchi_values = []
        
        for s in valid_stations:
            # Chl-a
            val_chl = getattr(s, 'chl_a_surface', np.nan)
            if np.isnan(val_chl) and 'chl_a_surface' in s.variables:
                val_chl = s.variables['chl_a_surface']['value']
            if not np.isnan(val_chl):
                chla_values.append(val_chl)
                
            # Secchi
            val_sec = np.nan
            for key in ['disk_depth', 'secchi', 'secchi_depth']:
                if key in s.variables:
                    val_sec = s.variables[key]['value']
                    break
            if np.isnan(val_sec) and hasattr(s, 'kd_par_value'):
                 if s.kd_par_value and s.kd_par_value > 0:
                     val_sec = 1.7 / s.kd_par_value
            if not np.isnan(val_sec):
                secchi_values.append(val_sec)

        avg_chla = np.nanmean(chla_values) if chla_values else np.nan
        avg_secchi = np.nanmean(secchi_values) if secchi_values else np.nan

        # Combine
        meta = static_meta.get(lake_name, {'Lat': '?', 'Max_Depth': '?', 'Print_Name': lake_name})
        table_rows.append({
            'Sites': meta['Print_Name'],
            'Latitude': meta['Lat'],
            'Time period': period,
            'Avg. Chl-a': avg_chla,
            'Avg. Secchi': avg_secchi,
            'Max depth': meta['Max_Depth'],
            'Sample count': count
        })

    df_table1 = pd.DataFrame(table_rows)
    cols = ['Sites', 'Latitude', 'Time period', 'Avg. Chl-a', 'Avg. Secchi', 'Max depth', 'Sample count']
    df_table1 = df_table1[cols]
    
    # Calculate Total Count
    total_n = df_table1['Sample count'].sum()
    
    print("\n" + "="*80)
    print(f"UPDATED TABLE 1 (Matched to Analysis n={total_n})")
    print("="*80)
    print(df_table1.round(2).to_string(index=False))
    print("="*80)

    if save_path:
        df_table1.round(2).to_csv(save_path, index=False)
        print(f"...Table 1 saved to: {save_path}")

    return df_table1