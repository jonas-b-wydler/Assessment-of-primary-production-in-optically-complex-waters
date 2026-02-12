"""
Assessment of primary production in optically complex waters:
Towards a generalized bio-optical modelling approach

Jonas Wydler
2026_02_11
"""

from preprocess_data import preprocess_depth_resolved, preprocess_depth_resolved_balticsea, preprocess_depth_resolved_estonia
from model import lee_like_model, vgpm_model, arst_like_model, cafe_like_model, vgpm_depth_resolved
import numpy as np
from plots import Plots
import copy
from utility import separate_outliers, create_station_level_tidy_df, generate_summary_tables, generate_summary_statistics_for_timeseries, generate_site_characteristics_table
import os
import pandas as pd
import matplotlib.pyplot as plt
import pickle

plt.rcParams['svg.fonttype'] = 'none'

# Configuration
USE_CACHE = True
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
ALL_LAKES_CACHE_PATH = os.path.join(RESULTS_DIR, 'data_cache', 'processed_all_lakes_og.pkl')

def main():
    # File paths
    era5_save_dir = os.path.join(DATA_DIR, 'era5_data')

    # Lake Geneva (SLH2) data
    file_path_chla = os.path.join(DATA_DIR, 'original_data', 'slh2', 'chla_slh2_2000_2023.csv')
    file_path_disk = os.path.join(DATA_DIR, 'original_data', 'slh2', 'disk_slh2_2000_2023.csv')
    file_path_pp = os.path.join(DATA_DIR, 'original_data', 'slh2', 'pp_slh2_2000_2023.csv')
    file_path_temp = os.path.join(DATA_DIR, 'original_data', 'slh2', 'temp_slh2_2000_2023.csv')
    era5_save_dir_baltic_estonian = os.path.join(DATA_DIR, 'preprocessed_data_resolved')

    # Baltic Sea data
    file_path_balticsea = os.path.join(DATA_DIR, 'original_data', 'baltic', 'PP_Baltic.csv')

    # Estonian lakes data
    file_path_estonian_depth_resolved = os.path.join(DATA_DIR, 'original_data', 'otherlakes', 'estonian_lakes_depth_resolved.csv')

    # Data reading and pre-processing
    lakes_resolved_slh2 = preprocess_depth_resolved(file_path_chla, file_path_disk, file_path_pp, file_path_temp, era5_save_dir)

    # Process depth-resolved data for baltic sea
    lakes_resolved_baltic = preprocess_depth_resolved_balticsea(file_path_balticsea, era5_save_dir_baltic_estonian)

    # Process depth-resolved data for estonian lakes
    lakes_resolved_estonia = preprocess_depth_resolved_estonia(file_path_estonian_depth_resolved, era5_save_dir_baltic_estonian)
   
    # Gather lakes together
    lakes_depth_resolved = lakes_resolved_slh2 | lakes_resolved_estonia | lakes_resolved_baltic

    # Data processing
    for lake_name, lake in lakes_depth_resolved.items():
        for station in lake.stations:
            station.trapezoidal_integration_PP()  
    

    for lake_name, lake in lakes_depth_resolved.items():
        for station in lake.stations:
            station.interpolate_chla_pp()
            station.set_surface_chl_a()
            station.a_phy()  
            station.par_wavelengths()
            station.kd_from_secchi()
            station.zeu()
            station.set_PbOpt()
            station.set_depths()
            station.set_incubation_hours()  
            station.set_daily_pp()
            station.set_surface_temp()
  

    for lake_name, lake in lakes_depth_resolved.items():
        lake.filter_stations()
    
    all_lakes = lakes_depth_resolved 
    all_lakes = {lake_name: lake for lake_name, lake in all_lakes.items() if len(lake.stations) > 10}
    
    baltics = {name: lake for name, lake in all_lakes.items() if name in ['SLAGGO', 'PRICKEN', 'BROA E']}
    estonian_lakes = {name: lake for name, lake in all_lakes.items() if name in ['Harku', 'Peipsi', 'Vortsjarv']}
    slh2 = {name: lake for name, lake in all_lakes.items() if name in ['Leman(Geneva Lake)']}

    # Outlier detection
    p = Plots()                     
    
    for lake_name, lake_obj in all_lakes.items():
        # --- reset any earlier flags -----------------------------------
        lake_obj.reset_outliers()
        
        # --- absolute rule-based filters -------------------------------
        lake_obj.check_low_chl(
            variable_pp='pp_hourly_depth_resolved',
            variable_chla='chl_a_depth_resolved'
        )
        lake_obj.check_max_pp_depth(
            variable_pp='pp_hourly_depth_resolved',
            variable_kd='kd_par'
        )
        
        # -- PCA shape filter + same-depth spike rescue ------------------
        # percentile mode: keep 97 %, flag top 3 %
        outliers, pca = lake_obj.derivative_pca_outlier_check(
            distance_cut = None,
            distance_q   = 0.97,
            window_r     = 2,
            r_thresh     = 0.5
        )
    
    # --- split into sets with / without outliers -----------------------
    all_lakes_no_outliers, all_lakes_only_outliers = separate_outliers(all_lakes)
    
    # Supplementary Figures 6, 7, 8
    # --- visualise the flagged profiles -------------------------------
    p.plot_all_profiles_with_chla_and_zeu(
        all_lakes_only_outliers,
        variable_pp  ='pp_hourly_depth_resolved',
        variable_chla='chl_a_depth_resolved',
        variable_kd  ='kd_par'
    )
    
    
    all_lakes = all_lakes_no_outliers
    
    for lake_name, lake in all_lakes.items():
        for station in lake.stations:
            station.compute_pp_peak_depth()
 
    
    # Main model runs
    # Check if cached results exist
    if USE_CACHE and os.path.exists(ALL_LAKES_CACHE_PATH):
        # If it exists, load it and skip the slow part completely.
        print(f"\n>>> Found cached model results. Loading from:\n    {ALL_LAKES_CACHE_PATH}")
        with open(ALL_LAKES_CACHE_PATH, 'rb') as f:
            all_lakes = pickle.load(f)
        print("    ...loading complete.")
    
    else:
        print("\n>>> No cache found or USE_CACHE=False. Running all model configurations...")

        model_classes = {
            'A': lee_like_model,
            'B': arst_like_model,
            'C': cafe_like_model,
            'D': vgpm_depth_resolved
        }
        correction_methods = ['A', 'B']
        vertical_uniformities = [True, False]

        for lake_name, lake in all_lakes.items():
            for station in lake.stations:
                for model_letter, model_class in model_classes.items():
                    for corr in correction_methods:
                        for vert in vertical_uniformities:
                            vert_label = 'A' if vert else 'B'
                            instance_name = f"{vert_label}{corr}{model_letter}"
                            model_instance = model_class(
                                name=instance_name,
                                zlim=np.max(station.variables['pp_measurement_depths']['value']),
                                correction_method=corr,
                                vert_uni=vert
                            )
                            station.apply_model(model_instance)
                station.calculate_model_error()

  
        print("    ...model runs complete.")
        
        # --- After running, save the result to the cache for next time ---
        print(f"\n>>> Saving processed 'all_lakes' object to cache:\n    {ALL_LAKES_CACHE_PATH}")
        
        # Ensure the directory exists before trying to save
        cache_directory = os.path.dirname(ALL_LAKES_CACHE_PATH)
        os.makedirs(cache_directory, exist_ok=True)
        
        # Save the file
        with open(ALL_LAKES_CACHE_PATH, 'wb') as f:
            pickle.dump(all_lakes, f)
        print("    ...caching complete.")
     


    # Final analysis and manuscript outputs
    print("\n>>> STARTING FINAL ANALYSIS AND OUTPUT GENERATION...")

    TABLES_DIR = os.path.join(RESULTS_DIR, 'tables')
    FIGURES_DIR = os.path.join(RESULTS_DIR, 'figures')
    os.makedirs(TABLES_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    
    p = Plots()
    groups = {
        'Baltic Sea': {name: lake for name, lake in all_lakes.items() if name in ['SLAGGO', 'PRICKEN', 'BROA E']},
        'Estonian Lakes': {name: lake for name, lake in all_lakes.items() if name in ['Harku', 'Peipsi', 'Vortsjarv']},
        'Lake Geneva': {name: lake for name, lake in all_lakes.items() if name == 'Leman(Geneva Lake)'}
    }

    master_df = create_station_level_tidy_df(all_lakes, groups=groups, min_obs_pp=0.5)
    # Table 1
    print("\n--- Generating and printing all manuscript tables ---")
    generate_site_characteristics_table(
        all_lakes, 
        save_path=os.path.join(TABLES_DIR, 'Table1_Site_Characteristics_Updated.csv')
    )
    
    summary_tables = generate_summary_tables(master_df)
    table2 = summary_tables['Table2_PI_Model']
    table3 = summary_tables['Table3_ChlA_Effect']
    table4 = summary_tables['Table4_Upwelling_Effect']
    
    # Generate Table 2 in the Manuscript
    print("\n" + "-"*70 + "\nTable 2: Effect of P-I Model (by Region)\n" + "-"*70)
    print(table2.to_string())
    table2.to_csv(os.path.join(TABLES_DIR, 'Table1_PI_Model_Summary_og.csv'), index=False)
    
    # Generate Table 3 in the Manuscript
    print("\n" + "-"*70 + "\nTable 3: Effect of Chl-a Profile (by Region)\n" + "-"*70)
    print(table3.to_string())
    table3.to_csv(os.path.join(TABLES_DIR, 'Table3_ChlA_Effect_Summary_og.csv'), index=False)
    
    # Generate Table 4 in the Manuscript
    print("\n" + "-"*70 + "\nTable 4: Effect of Upwelling Correction (by Region)\n" + "-"*70)
    print(table4.to_string())
    table4.to_csv(os.path.join(TABLES_DIR, 'Table4_Upwelling_Effect_Summary_og.csv'), index=False)

    print(f"\n--- Saving all manuscript figures to '{FIGURES_DIR}' ---")
    p.plot_sites_map(all_lakes, groups, save_path=os.path.join(FIGURES_DIR, 'Fig1_Study_Sites_Map.svg'))
    p.plot_group_summary_bars(all_lakes, groups, key_variables=['zeu', 'chl_a_surface', 'pp_daily_depth_integrated'], save_path_prefix=os.path.join(FIGURES_DIR, 'Fig2_Env_Summary'))

    # Generate Figure 4 in the Manuscript
    p.plot_final_quadrant(master_df, group_name='All Data', save_path=os.path.join(FIGURES_DIR, 'Fig3_Performance_All_Data_og.svg'))
    for region_name in [r for r in master_df['group'].unique() if pd.notna(r)]:
        regional_df = master_df[master_df['group'] == region_name]
        p.plot_final_quadrant(regional_df, group_name=region_name, save_path=os.path.join(FIGURES_DIR, f'Fig3_Performance_og_{region_name.replace(" ", "_")}.svg'))

    lakes_for_timeseries = ['Leman(Geneva Lake)', 'PRICKEN', 'BROA E', 'SLAGGO']
    timeseries_dir = os.path.join(FIGURES_DIR, 'time_series')
    os.makedirs(timeseries_dir, exist_ok=True)
    model_to_plot = 'BAA'

    for lake_name in lakes_for_timeseries:
        if lake_name in all_lakes:
            lake_to_plot = all_lakes[lake_name]
            clean_name = lake_name.replace('(', '_').replace(')', '_').replace(' ', '_')
            save_name = f"timeseries_{clean_name}_{model_to_plot}.svg"
            p.plot_single_model_timeseries(
                lake_object=lake_to_plot,
                model_name_to_plot=model_to_plot,
                save_path=os.path.join(timeseries_dir, save_name)
            )
            
    # Generate Figure 5 in the Manuscript
    pricken_obj = all_lakes.get('PRICKEN')
    geneva_obj = all_lakes.get('Leman(Geneva Lake)')

    if pricken_obj and geneva_obj:
        model_to_plot = 'BAA'

        daily_save_path = os.path.join(FIGURES_DIR, 'Fig_LongTerm_Daily_og.svg')
        p.plot_longterm_detailed_figure(
            lake1_obj=pricken_obj,
            lake2_obj=geneva_obj,
            model_name=model_to_plot,
            data_type='daily',
            save_path=daily_save_path
        )
        # Generate Figure 11 in the Manuscript

        hourly_save_path = os.path.join(FIGURES_DIR, 'Fig_LongTerm_Hourly_og.svg')
        p.plot_longterm_detailed_figure(
            lake1_obj=pricken_obj,
            lake2_obj=geneva_obj,
            model_name=model_to_plot,
            data_type='hourly',
            save_path=hourly_save_path
        )
    # Figure 3
    p.plot_modeled_curve(all_lakes, variable_name='pp_hourly_depth_resolved')
    
    
    # Generate Table 5 in the Manuscript
    final_model_name = 'BAA'
    lakes_to_analyze = {
        'PRICKEN': all_lakes.get('PRICKEN'),
        'Leman(Geneva Lake)': all_lakes.get('Leman(Geneva Lake)')
    }

    stats_csv_path = os.path.join(TABLES_DIR, 'final_timeseries_stats.csv')
    stats_df = generate_summary_statistics_for_timeseries(
        lake_objects_dict=lakes_to_analyze,
        model_name=final_model_name,
        save_path=stats_csv_path
    )
    

    # Sup. Results, Figure 10
    print("\n>>> STARTING VGPM FORMULATION COMPARISON FOR SUPPLEMENTARY FIGURE...")
    
    all_lakes_for_vgpm_comparison = copy.deepcopy(all_lakes)
    for lake_name, lake in all_lakes_for_vgpm_comparison.items():
        for station in lake.stations:
            model1 = vgpm_model()
            model2 = vgpm_depth_resolved(vert_uni=True)
            station.apply_model(model1)
            station.apply_model(model2)
    print("    ...ran the two VGPM models for comparison.")
    p = Plots()
    p.plot_vgpm_comparison_scatter(
        all_lakes_comparison=all_lakes_for_vgpm_comparison,
        model_x_name='vgpm_model',          
        model_y_name='vgpm_depth_resolved'
    )
    print("    ...VGPM comparison plot generation complete.")
