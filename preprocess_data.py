from station import Station
from lake import depth_resolved_lake
import pandas as pd
from utility import clean_gps_format, parse_lat_lon, load_irr_data, download_era5_data, extract_unique_string, extract_incubation_time, time_to_hours, sanitize_string
import os
import numpy as np
 


def preprocess_depth_resolved(file_path_chla, file_path_disk, file_path_pp, file_path_temp, save_directory, separator=';'):
    
    chla_data = pd.read_csv(file_path_chla, sep=separator)
    disk_data = pd.read_csv(file_path_disk, sep=separator)
    pp_data = pd.read_csv(file_path_pp, sep=separator)
    temp_data = pd.read_csv(file_path_temp, sep=separator)

    chla_data['date'] = pd.to_datetime(chla_data['date'], format='%d.%m.%Y', errors='coerce')
    disk_data['date'] = pd.to_datetime(disk_data['date'], format='%d.%m.%Y', errors='coerce')
    pp_data['date'] = pd.to_datetime(pp_data['date'], format='%d.%m.%Y', errors='coerce')
    temp_data['date'] = pd.to_datetime(temp_data['date'], format='%d.%m.%Y', errors='coerce')

    pp_data[['latitude', 'longitude']] = pp_data['gps'].apply(lambda x: pd.Series(parse_lat_lon(x)))

    lakes = {}
    grouped = pp_data.groupby(['lake_name', 'station_name', 'date', 'latitude', 'longitude'])

    for (lake_name, station_name, date, latitude, longitude), group in grouped:
        pp_hourly_depth_resolved = group.set_index('depth')['pp_hourly_depth_resolved']

        matching_disk_data = disk_data[(disk_data['lake_name'] == lake_name) & 
                                       (disk_data['station_name'] == station_name) & 
                                       (disk_data['date'] == date)]
        secchi_value = matching_disk_data['secchi'].values[0] if not matching_disk_data.empty else np.nan

        chla_depth_resolved = chla_data[(chla_data['lake_name'] == lake_name) & 
                                        (chla_data['station_name'] == station_name) & 
                                        (chla_data['date'] == date)].groupby('depth')['chl_a_depth_resolved'].mean()
        chla_depth_resolved = chla_depth_resolved.reindex(pp_hourly_depth_resolved.index).fillna(np.nan)

        temperature_depth_resolved = temp_data[(temp_data['lake_name'] == lake_name) & 
                                               (temp_data['station_name'] == station_name) & 
                                               (temp_data['date'] == date)].groupby('depth')['temperature_depth_resolved'].mean()
        temperature_depth_resolved = temperature_depth_resolved.reindex(pp_hourly_depth_resolved.index).fillna(np.nan)
        
        incubation_start = extract_unique_string(group, 'incubation_start')
        incubation_end = extract_unique_string(group, 'incubation_end')
        group.loc[:, 'incubation_time'] = group['incubation_time'].apply(time_to_hours)
        incubation_time = extract_incubation_time(group, 'incubation_time')

        lake_directory = os.path.join(save_directory, lake_name)
        os.makedirs(lake_directory, exist_ok=True)

        era5_filename = f"Era5_data_{lake_name}_{station_name}_{date.strftime('%Y%m%d')}.nc"
        era5_filepath = os.path.join(lake_directory, era5_filename)
        if not os.path.exists(era5_filepath):
            download_era5_data(date, latitude, longitude, era5_filepath)
        else:
            print(f"File already exists: {era5_filepath}")
        
        irradiance_data = load_irr_data(era5_filepath)

        station = Station(
            station_id=station_name,
            lake=lake_name,
            latitude=latitude,
            longitude=longitude,
            date=date,
            pp_hourly_depth_resolved=pp_hourly_depth_resolved,
            pp_hourly_depth_integrated=None,
            chl_a_depth_resolved=chla_depth_resolved,
            chl_a_depth_integrated=None,  
            secchi=secchi_value,
            temperature_depth_resolved=temperature_depth_resolved,
            par = irradiance_data['daily_par_moles'],
            par_max = irradiance_data['max_par_moles'],
            par_hourly=irradiance_data['hourly_par_moles'],
            dl=np.sum(irradiance_data['hourly_par_moles'] > 0.05),
            incubation_start=incubation_start,
            incubation_end=incubation_end,
            incubation_time=incubation_time
        )

        if lake_name not in lakes:
            lakes[lake_name] = depth_resolved_lake(lake_name)
        lakes[lake_name].add_station(station)

    return lakes

def preprocess_depth_resolved_estonia(file_path, save_directory, separator=';'):
    dat_all = pd.read_csv(file_path, index_col=0, encoding='UTF-8', sep=separator)

    dat_all = dat_all.reset_index()
    dat_all['date'] = pd.to_datetime(dat_all['date'], dayfirst=True, errors='coerce')
    dat_all['gps'] = dat_all['gps'].apply(clean_gps_format)
    dat_all[['latitude', 'longitude']] = dat_all['gps'].apply(lambda x: pd.Series(parse_lat_lon(x)))

    grouped = dat_all.groupby(['lake_name', 'date', 'latitude', 'longitude', 'incubation_hours'])

    lakes = {}

    for (lake_name, date, latitude, longitude, incubation_hours), group in grouped:
        lake_directory = os.path.join(save_directory, lake_name)
        if lake_name not in lakes:
            os.makedirs(lake_directory, exist_ok=True)
            lakes[lake_name] = depth_resolved_lake(lake_name)

        sanitized_incubation_hours = sanitize_string(incubation_hours)
        era5_filename = f"Era5_data_{lake_name}_{sanitized_incubation_hours}_{date.strftime('%Y%m%d')}.nc"
        era5_filepath = os.path.join(lake_directory, era5_filename)

        if not os.path.exists(era5_filepath) or os.path.getsize(era5_filepath) == 0:
            download_era5_data(date, latitude, longitude, era5_filepath)
        irradiance_data = load_irr_data(era5_filepath)

        pp_hourly_depth_resolved = group.groupby('depth')['pp_hourly_depth_resolved'].mean().reset_index()
        pp_hourly_depth_resolved.set_index('depth', inplace=True)

        chl_a_depth_resolved = group.groupby('depth')['chl_a_depth_resolved'].mean().reset_index()
        chl_a_depth_resolved.set_index('depth', inplace=True)

        secchi_value = pd.to_numeric(group['secchi'].iloc[0], errors='coerce') if not group['secchi'].isnull().all() else None
        chl_a_surface = pd.to_numeric(group['chl_a_depth_resolved'].iloc[0], errors='coerce') if not group['chl_a_depth_resolved'].isnull().all() else None

        incubation_start = extract_unique_string(group, 'incubation_start')
        incubation_time = extract_incubation_time(group, 'incubation_time')

        an_values = group.groupby('depth')['an'].mean().reset_index()
        an_values.set_index('depth', inplace=True)
        surface_an = an_values.mean()

        temperature_values = group.groupby('depth')['surface_temperature'].mean().reset_index()
        temperature_values.set_index('depth', inplace=True)
        surface_temperature = temperature_values.mean()

        station_id = f"{lake_name}_{date.strftime('%Y-%m-%d')}_{sanitized_incubation_hours}"

        station = Station(
            station_id=station_id,
            lake=lake_name,
            latitude=latitude,
            longitude=longitude,
            date=date,
            pp_hourly_depth_resolved=pp_hourly_depth_resolved,
            pp_hourly_depth_integrated=None,
            chl_a_depth_resolved=chl_a_depth_resolved,
            chl_a_surface=chl_a_surface,
            secchi=secchi_value,
            par=irradiance_data['daily_par_moles'],
            par_max=irradiance_data['max_par_moles'],
            par_hourly=irradiance_data['hourly_par_moles'],
            dl=np.sum(irradiance_data['hourly_par_moles'] > 0.05),
            incubation_start=incubation_start,
            incubation_time=incubation_time,
            an_values=an_values,
            surface_an=surface_an[0],
            surface_temperature=surface_temperature[0]
        )

        lakes[lake_name].add_station(station)

    return lakes


def preprocess_depth_resolved_balticsea(file_path, save_directory, separator=';'):
    dat_all = pd.read_csv(file_path, index_col=0, encoding="ISO-8859-1", sep=';', low_memory=False)

    df_pivot = pd.pivot_table(dat_all, values='Value', index=['Sampling date (start)', 'Station name', 'Sampling depth (m)'], columns='Parameter', aggfunc='first')

    # Merge the Secchi depth column from the original dataframe
    df_all_wide = pd.merge(df_pivot, dat_all[['Sampling date (start)', 'Station name', 'Sample latitude (DD)', 'Sample longitude (DD)', 'Sampling depth (m)', 'Secchi depth', 'Incubation start time', 'Incubation end time', 'Incubation time (h)']], 
                           on=['Sampling date (start)', 'Station name', 'Sampling depth (m)'], how='left')

    # Ensure the date column is correctly parsed
    df_all_wide['Sampling date (start)'] = pd.to_datetime(df_all_wide['Sampling date (start)'], dayfirst=True, errors='coerce')

    # Rename columns to be compatible with Station class
    df_all_wide.rename(columns={
        'Sampling date (start)': 'date',
        'Station name': 'station_id',
        'Sample latitude (DD)': 'latitude',
        'Sample longitude (DD)': 'longitude',
        'Sampling depth (m)': 'depth',
        'Net carbon prod_prod-resp': 'pp_hourly_depth_resolved',
        'Chlorophyll-a': 'chl_a_depth_resolved',
        'Temperature': 'temperature_depth_resolved',
        'Secchi depth': 'secchi',
        'Incubation start time': 'incubation_start',
        'end of incubation (h)': 'incubation_end',
        'Incubation time (h)': 'incubation_time'
    }, inplace=True)
        
  
    
    lakes = {}
    lake_name = 'BalticSea'
    # Group data by station, date, latitude, and longitude to create Station objects
    grouped = df_all_wide.groupby(['station_id', 'date', 'latitude', 'longitude'])
    
    
    for (station_id, date, latitude, longitude), group in grouped:
        try:
            group.loc[:, 'pp_hourly_depth_resolved'] = pd.to_numeric(group['pp_hourly_depth_resolved'], errors='coerce')
            group.loc[:, 'chl_a_depth_resolved'] = pd.to_numeric(group['chl_a_depth_resolved'], errors='coerce')
            group.loc[:, 'temperature_depth_resolved'] = pd.to_numeric(group['temperature_depth_resolved'], errors='coerce')
            group.loc[:, 'secchi'] = pd.to_numeric(group['secchi'], errors='coerce')

            # Average the values for each depth
            pp_hourly_depth_resolved = group.groupby('depth')['pp_hourly_depth_resolved'].mean().reset_index()
            chl_a_depth_resolved = group.groupby('depth')['chl_a_depth_resolved'].mean().reset_index()
            temperature_depth_resolved = group.groupby('depth')['temperature_depth_resolved'].mean().reset_index()

            # Set index to 'depth' to align with the expected format
            pp_hourly_depth_resolved.set_index('depth', inplace=True)
            chl_a_depth_resolved.set_index('depth', inplace=True)
            temperature_depth_resolved.set_index('depth', inplace=True)

            # Extract secchi value (not depth-resolved)
            secchi_value = group['secchi'].iloc[0] if not group['secchi'].isnull().all() else None
            if secchi_value is not None:
                secchi_value = pd.to_numeric(secchi_value, errors='coerce')

            # Create directory for lake if it doesn't exist
            lake_directory = os.path.join(save_directory, station_id)
            os.makedirs(lake_directory, exist_ok=True)

          
            #  Download Era5 data
            era5_filename = f"Era5_data_{lake_name}_{station_id}_{date.strftime('%Y%m%d')}.nc"
            era5_filepath = os.path.join(lake_directory, era5_filename)
            # Check if file already exists to avoid re-downloading
            if not os.path.exists(era5_filepath):
                download_era5_data(date, latitude, longitude, era5_filepath)
            else:
                print(f"File already exists: {era5_filepath}")

            irradiance_data = load_irr_data(era5_filepath)


            # Extract incubation_start time (not depth-resolved)
            incubation_start = extract_unique_string(group, 'incubation_start')

            # Extract incubation_time (not depth-resolved)
            incubation_time = extract_incubation_time(group, 'incubation_time')

            # Create Station object with depth-resolved attributes
            station = Station(
                station_id=station_id,
                lake=station_id,  # Use station_id as the lake name
                latitude=latitude,
                longitude=longitude,
                date=date,
                pp_hourly_depth_resolved=pp_hourly_depth_resolved,
                pp_hourly_depth_integrated=None,  # Assuming depth-integrated data is not present
                chl_a_depth_resolved=chl_a_depth_resolved,
                chl_a_depth_integrated=None,  # Assuming depth-integrated data is not present
                secchi=secchi_value,
                temperature_depth_resolved=temperature_depth_resolved,
                par=irradiance_data['daily_par_moles'],
                par_max=irradiance_data['max_par_moles'],
                par_hourly=irradiance_data['hourly_par_moles'],
                dl=np.sum(irradiance_data['hourly_par_moles'] > 0.05),            
                incubation_start=incubation_start,
                incubation_time=incubation_time
            )

            # Add Station to the lakes dictionary using station_id as the key
            if station_id not in lakes:
                lakes[station_id] = depth_resolved_lake(station_id)
            lakes[station_id].add_station(station)

        except Exception as e:
            print(f"Error processing group for station {station_id}, date {date}: {e}")

    return lakes
