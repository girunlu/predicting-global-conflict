import re
import numpy as np
import pandas as pd
import geopandas as gpd
import unicodedata
from shapely.errors import TopologicalError
from collections import defaultdict
from fuzzywuzzy import fuzz, process
from tqdm import tqdm


def normalize(text, strip_punctuation=False):
    if pd.isna(text):
        return ''
    text = str(text)
    text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')
    text = re.sub(r'\s+', ' ', text)  # collapse multiple spaces
    if strip_punctuation:
        text = re.sub(r'[^\w\s]', '', text)  # remove punctuation
    return text.strip().title()


def fix_france(gdf):
    """
    Spatially dissolves French departments into regions, labels them by region name,
    and appends them to the original GeoDataFrame.
    
    Parameters:
        gdf (GeoDataFrame): GeoDataFrame with French departments. Must include 'adm0_a3' == 'FRA' and 'name_en'.
    
    Returns:
        GeoDataFrame: Original gdf with regional geometries for France appended (name_en = region name).
    """

    # Mapping of departments to regions (simplified but covers all)
    department_to_region = {
        # Auvergne-Rhône-Alpes
        'Ain': 'Auvergne-Rhone-Alpes', 'Allier': 'Auvergne-Rhone-Alpes', 'Ardèche': 'Auvergne-Rhone-Alpes',
        'Cantal': 'Auvergne-Rhone-Alpes', 'Drôme': 'Auvergne-Rhone-Alpes', 'Isère': 'Auvergne-Rhone-Alpes',
        'Haute-Loire': 'Auvergne-Rhone-Alpes', 'Loire': 'Auvergne-Rhone-Alpes', 'Puy-de-Dôme': 'Auvergne-Rhone-Alpes',
        'Rhône': 'Auvergne-Rhone-Alpes', 'Savoie': 'Auvergne-Rhone-Alpes', 'Haute-Savoie': 'Auvergne-Rhone-Alpes',

        # Bourgogne-Franche-Comté
        'Côte-d’Or': 'Bourgogne-Franche-Comte', 'Doubs': 'Bourgogne-Franche-Comte', 'Jura': 'Bourgogne-Franche-Comte',
        'Nièvre': 'Bourgogne-Franche-Comte', 'Haute-Saône': 'Bourgogne-Franche-Comte',
        'Saône-et-Loire': 'Bourgogne-Franche-Comte', 'Yonne': 'Bourgogne-Franche-Comte',
        'Territoire de Belfort': 'Bourgogne-Franche-Comte',

        # Bretagne
        'Côtes-d\'Armor': 'Bretagne', 'Finistère': 'Bretagne', 'Ille-et-Vilaine': 'Bretagne', 'Morbihan': 'Bretagne',

        # Centre-Val de Loire
        'Cher': 'Centre-Val De Loire', 'Eure-et-Loir': 'Centre-Val De Loire', 'Indre': 'Centre-Val De Loire',
        'Indre-et-Loire': 'Centre-Val De Loire', 'Loir-et-Cher': 'Centre-Val De Loire',
        'Loiret': 'Centre-Val De Loire',

        # Corse
        'Corse-du-Sud': 'Corse', 'Haute-Corse': 'Corse',

        # Grand Est
        'Ardennes': 'Grand Est', 'Aube': 'Grand Est', 'Bas-Rhin': 'Grand Est', 'Haut-Rhin': 'Grand Est',
        'Haute-Marne': 'Grand Est', 'Marne': 'Grand Est', 'Meurthe-et-Moselle': 'Grand Est',
        'Meuse': 'Grand Est', 'Moselle': 'Grand Est', 'Vosges': 'Grand Est',

        # Hauts-de-France
        'Aisne': 'Hauts-De-France', 'Nord': 'Hauts-De-France', 'Oise': 'Hauts-De-France',
        'Pas-de-Calais': 'Hauts-De-France', 'Somme': 'Hauts-De-France',

        # Île-de-France
        'Paris': 'Ile-De-France', 'Seine-et-Marne': 'Ile-De-France', 'Yvelines': 'Ile-De-France',
        'Essonne': 'Ile-De-France', 'Hauts-de-Seine': 'Ile-De-France', 'Seine-Saint-Denis': 'Ile-De-France',
        'Val-de-Marne': 'Ile-De-France', "Val-d'Oise": 'Ile-De-France',

        # Normandie
        'Calvados': 'Normandie', 'Eure': 'Normandie', 'Manche': 'Normandie', 'Orne': 'Normandie',
        'Seine-Maritime': 'Normandie',

        # Nouvelle-Aquitaine
        'Charente': 'Nouvelle-Aquitaine', 'Charente-Maritime': 'Nouvelle-Aquitaine', 'Corrèze': 'Nouvelle-Aquitaine',
        'Creuse': 'Nouvelle-Aquitaine', 'Dordogne': 'Nouvelle-Aquitaine', 'Gironde': 'Nouvelle-Aquitaine',
        'Landes': 'Nouvelle-Aquitaine', 'Lot-et-Garonne': 'Nouvelle-Aquitaine', 'Pyrénées-Atlantics': 'Nouvelle-Aquitaine',
        'Deux-Sèvres': 'Nouvelle-Aquitaine', 'Vienne': 'Nouvelle-Aquitaine', 'Haute-Vienne': 'Nouvelle-Aquitaine',

        # Occitanie
        'Ariège': 'Occitanie', 'Aude': 'Occitanie', 'Aveyron': 'Occitanie', 'Gard': 'Occitanie',
        'Haute-Garonne': 'Occitanie', 'Gers': 'Occitanie', 'Hérault': 'Occitanie', 'Lot': 'Occitanie',
        'Lozère': 'Occitanie', 'Hautes-Pyrénées': 'Occitanie', 'Pyrénées-Orientales': 'Occitanie',
        'Tarn': 'Occitanie', 'Tarn-et-Garonne': 'Occitanie',

        # Pays de la Loire
        'Loire-Atlantique': 'Pays De La Loire', 'Maine-et-Loire': 'Pays De La Loire',
        'Mayenne': 'Pays De La Loire', 'Sarthe': 'Pays De La Loire', 'Vendée': 'Pays De La Loire',

        # Provence-Alpes-Côte d'Azur
        'Alpes-de-Haute-Provence': "Provence-Alpes-Cote D'Azur", 'Hautes-Alpes': "Provence-Alpes-Cote D'Azur",
        'Alpes-Maritimes': "Provence-Alpes-Cote D'Azur", 'Bouches-du-Rhône': "Provence-Alpes-Cote D'Azur",
        'Var': "Provence-Alpes-Cote D'Azur", 'Vaucluse': "Provence-Alpes-Cote D'Azur",
    }

    # Filter for France departments
    france_depts = gdf[(gdf['adm0_a3'] == 'FRA') & (gdf['name_en'].isin(department_to_region.keys()))].copy()

    # Assign region name
    france_depts['region_name'] = france_depts['name_en'].map(department_to_region)

    # Dissolve by region name
    regions = france_depts.dissolve(by='region_name').reset_index()
    regions['adm0_a3'] = 'FRA'
    regions['name_en'] = regions['region_name']
    regions = regions.drop(columns=['region_name'])

    # Append regions to original gdf
    gdf = pd.concat([gdf, regions], ignore_index=True)

    return gdf


def fix_libya(gdf):
    """
    Replace Libya's Admin-1 regions with grouped 'West', 'East', and 'South' regions.

    Parameters:
        gdf (GeoDataFrame): Original GeoDataFrame with 'adm0_a3' and 'name_en' columns.

    Returns:
        GeoDataFrame: Modified GeoDataFrame with grouped Libya regions.
    """

    # Define regional groupings
    west = [
        'Tripoli District', 'Zawiya', 'Misrata', 'Murqub',
        'Nuqat al Khams', 'Jafara', 'Mizdah'
    ]

    east = [
        'Butnan', 'Derna', 'Jabal al Akhdar', 'Marj',
        'Benghazi', 'Al Qubbah', 'Ajdabiya'
    ]

    south = [
        'Ghadames', 'Ghat', 'Murzuq', 'Wadi al Hayaa',
        'Sabha', 'Ash Shati', 'Jufra', 'Kufra'
    ]

    # Some names in the list of options are missing or alternate
    # Adjust east to what's present in the gdf
    east = [r for r in east if r in gdf['name_en'].values]
    south = [r for r in south if r in gdf['name_en'].values]

    libya_mask = gdf['adm0_a3'] == 'LBY'

    def dissolve_region(region_list, new_name):
        region_gdf = gdf[libya_mask & gdf['name_en'].isin(region_list)]
        if region_gdf.empty:
            return None
        dissolved = region_gdf.dissolve().copy()
        dissolved['adm0_a3'] = 'LBY'
        dissolved['name_en'] = new_name
        return dissolved.reset_index(drop=True)

    # Create dissolved regions
    west_d = dissolve_region(west, 'West')
    east_d = dissolve_region(east, 'East')
    south_d = dissolve_region(south, 'South')

    # Remove all original Libya regions that were grouped
    grouped = set(west + east + south)
    gdf = gdf[~(libya_mask & gdf['name_en'].isin(grouped))]

    # Concatenate updated Libya rows back
    new_regions = [r for r in [west_d, east_d, south_d] if r is not None]
    if new_regions:
        gdf = pd.concat([gdf] + new_regions, ignore_index=True)

    return gdf


def update_boundaries(gdf, countries, wb_file="data/raw/boundaries/World Bank Official Boundaries - Admin 1/WB_GAD_ADM1.shp"):
    """
    Replaces rows for specified countries (by adm0_a3 code) in a GeoDataFrame with 
    official Admin 1 boundaries from the World Bank shapefile.

    Parameters:
        gdf (GeoDataFrame): Original GeoDataFrame with an 'adm0_a3' and 'name_en' column.
        countries (list of str): List of adm0_a3 ISO codes (e.g., ['NPL', 'FRA']).
        wb_file (str): Path to World Bank Admin-1 shapefile.

    Returns:
        GeoDataFrame: Updated GeoDataFrame with specified countries replaced.
    """

    # Load World Bank shapefile
    gdf_wb = gpd.read_file(wb_file)

    # Ensure CRS compatibility
    if gdf.crs != gdf_wb.crs:
        gdf_wb = gdf_wb.to_crs(gdf.crs)

    # Create new rows from World Bank for all specified countries
    replacements = []
    for iso3 in countries:
        wb_subset = gdf_wb[gdf_wb['ISO_A3'] == iso3].copy()
        if wb_subset.empty:
            continue  # skip if country not in WB file

        # Create a blank data structure with expected columns
        new_cols = gdf.columns.drop('geometry')
        blank = pd.DataFrame(columns=new_cols, index=wb_subset.index)
        blank['adm0_a3'] = iso3
        blank['name_en'] = wb_subset['NAM_1'].values  # update name
        country_fixed = gpd.GeoDataFrame(blank, geometry=wb_subset.geometry, crs=gdf_wb.crs)
        replacements.append(country_fixed)

    # Concatenate all replacement regions
    if replacements:
        new_admin1 = pd.concat(replacements, ignore_index=True)
    else:
        new_admin1 = gpd.GeoDataFrame(columns=gdf.columns, geometry=[], crs=gdf.crs)

    # Remove original rows for those countries
    gdf_filtered = gdf[~gdf['adm0_a3'].isin(countries)]

    # Combine and return
    gdf_out = pd.concat([gdf_filtered, new_admin1], ignore_index=True)

    return gdf_out


def match_admin1_to_gdf(df, gdf):
    # Preprocess gdf
    gdf = gdf.copy()
    gdf = fix_france(gdf)
    gdf = fix_libya(gdf)
    gdf = update_boundaries(gdf, [
        'NPL', 'ESP', 'BFA', 'LKA', 'PHL', 'LBN', 'MAR', 'BEL', 'BGD',
        'AFG', 'MAR', 'KEN', 'ISL', 'COD', 'XKX', 'SOM', 'MTQ', 'MNE', 'CIV'])
    gdf['name_en_norm'] = gdf['name_en'].apply(normalize)
    gdf['name_norm'] = gdf['name'].apply(normalize)
    gdf['name_alt'] = gdf['name_alt'].fillna('')
    gdf['name_alt_list'] = gdf['name_alt'].apply(lambda x: [normalize(n) for n in x.split('|') if n.strip()])
    gdf['admin1_id'] = gdf['adm0_a3'] + ' - ' + gdf['name_en']

    # Build lookup maps
    name_en_map = {
        (row['adm0_a3'], row['name_en_norm']): row['admin1_id']
        for _, row in gdf.iterrows()
    }

    name_map = {
        (row['adm0_a3'], row['name_norm']): row['admin1_id']
        for _, row in gdf.iterrows()
    }

    altname_map = defaultdict(dict)
    fuzzy_match_pool = defaultdict(dict)

    for _, row in gdf.iterrows():
        country = row['adm0_a3']
        adm1_id = row['admin1_id']
        # Add name_en and name
        fuzzy_match_pool[country][row['name_en_norm']] = adm1_id
        fuzzy_match_pool[country][row['name_norm']] = adm1_id
        # Add alts
        for alt in row['name_alt_list']:
            altname_map[country][alt] = adm1_id
            fuzzy_match_pool[country][alt] = adm1_id

    # Build word overlap match pool
    tokenized_name_words = defaultdict(dict)
    for country, names in fuzzy_match_pool.items():
        for name, adm1_id in names.items():
            tokens = set(normalize(w, strip_punctuation=True) for w in name.split() if len(w) > 3)
            tokenized_name_words[country][name] = tokens

    # Prepare df country codes and normalized admin1
    df = df.copy()
    df['country_code'] = df['event_id_cnty'].str[:3]

    country_code_fixes = {
        'Chile': 'CHL', 'Burkina Faso': 'BFA', 'Morocco': 'MAR', 'Nigeria': 'NGA',
        'Paraguay': 'PRY', 'Palestine': 'PSX', 'Chad': 'TCD', 'Central African Republic': 'CAF',
        'Democratic Republic of Congo': 'COD', 'Niger': 'NER', 'South Africa': 'ZAF',
        'Sudan': 'SDN', 'South Sudan': 'SDS', 'Cameroon': 'CMR', 'Nepal': 'NPL', 'Mozambique': 'MOZ',
        'Sri Lanka': 'LKA', 'Madagascar': 'MDG', 'Portugal': 'PRT', 'Belgium': 'BEL', 'Mauritania': 'MRT',
        'Algeria': 'DZA', 'Burundi': 'BDI', 'Guinea': 'GIN', 'Mali': 'MLI', 'Equatorial Guinea': 'GNQ',
        'Republic of the Congo': 'CON', 'Tanzania': 'TZA', 'Uruguay': 'URY', 'Angola': 'AGO', 'Mayotte': 'FRA',
        'Trinidad and Tobago': 'TTO', 'Ivory Coast': 'CIV', 'Zimbabwe': 'ZWE', 'Lesotho': 'LSO', 'Zambia': 'ZMB',
        'Botswana': 'BWA', 'Puerto Rico': 'PRI', 'Sierra Leone': 'SLE', 'Gambia': 'GMB', 'Togo': 'TGO',
        'eSwatini': 'SWZ'
    }
    df['country_code'] = df['country_code'].mask(df['country'].isin(country_code_fixes), df['country'].map(country_code_fixes))
    df['country_code'] = np.where(df['admin1'] == 'Crimea', 'RUS', df['country_code'])
    df['admin1'] = np.where(df['country'] == 'Mayotte', 'Mayotte', df['admin1'])
    df['admin1'] = np.where(df['country'] == 'Martinique', 'Martinique', df['admin1'])
    df['admin1'] = np.where(df['country'] == 'Guam', 'Guam', df['admin1'])
    df['admin1'] = np.where(df['country_code'] == 'PRI', 'Puerto Rico', df['admin1'])

    df['admin1_norm'] = df['admin1'].apply(normalize)
    admin_name_fixes = {
        'Hadarom': 'Southern', 'Hazafon': 'Northern', 'Hamerkaz': 'Central', 'Ituri': 'Orientale', 'Province 1': 'Koshi',
        'Vlaanderen': 'Vlaams Gewest', 'Menaka': 'Gao'
    }
    df['admin1_norm'] = df['admin1_norm'].mask(df['admin1_norm'].isin(admin_name_fixes), df['admin1_norm'].map(admin_name_fixes))

    unique_keys = df[['country_code', 'admin1_norm']].dropna().drop_duplicates()

    match_cache = {}

    for _, row in tqdm(unique_keys.iterrows(), total=len(unique_keys), desc="Matching"):
        country_code = row['country_code']
        admin1 = row['admin1_norm']
        key = (country_code, admin1)

        if not admin1:
            match_cache[key] = None
            continue

        # 1. Exact match with name_en
        if key in name_en_map:
            match_cache[key] = name_en_map[key]
            continue

        # 2. Exact match with name
        if key in name_map:
            match_cache[key] = name_map[key]
            continue

        # 3. Exact match with alt name
        if admin1 in altname_map.get(country_code, {}):
            match_cache[key] = altname_map[country_code][admin1]
            continue

        # 4. Fuzzy match with name_en + name + alt
        choices_dict = fuzzy_match_pool.get(country_code, {})
        if choices_dict:
            result = process.extractOne(admin1, list(choices_dict.keys()), scorer=fuzz.ratio)
            if result:
                match, score = result
                if score >= 60:
                    match_cache[key] = choices_dict[match]
                    continue

        # 5. Word overlap match
        admin1_words = set(normalize(w, strip_punctuation=True) for w in admin1.split() if len(w) > 3)
        for name_candidate, adm1_id in choices_dict.items():
            candidate_words = tokenized_name_words[country_code][name_candidate]
            if admin1_words & candidate_words:
                match_cache[key] = adm1_id
                break

    # Map match results back to full df
    df['matched_admin1_id'] = df.apply(
        lambda row: match_cache.get((row['country_code'], row['admin1_norm']), None),
        axis=1
    )
    
    # 6. Spatial coordinate fallback for anything dropped completely
    missing_mask = df['matched_admin1_id'].isnull() & df['latitude'].notnull() & df['longitude'].notnull()
    
    if missing_mask.any():
        print(f"Applying spatial geocoordinate fallback for {missing_mask.sum()} unmapped events...")
        missing_df = df[missing_mask].copy()

        # Convert to Geopandas using event points
        missing_points = gpd.GeoDataFrame(
            missing_df, 
            geometry=gpd.points_from_xy(missing_df.longitude, missing_df.latitude),
            crs=gdf.crs  # Same CRS as polygons
        )
        
        # Spatial join against the region polygons
        joined = gpd.sjoin(missing_points, gdf[['admin1_id', 'geometry']], how="left", predicate="intersects")
        
        # Deduplicate index: if a point intersects multiple borders, keep the first to avoid reassignment errors
        joined = joined[~joined.index.duplicated(keep='first')]
        
        # Bring the matched names back into the original dataframe
        df.loc[missing_mask, 'matched_admin1_id'] = joined['admin1_id'].str.strip()
        
        # Update our match cache so future runs of these names know what spatial mapping hit
        for _, row in joined[joined['admin1_id'].notnull()].iterrows():
            key = (row['country_code'], row['admin1_norm'])
            match_cache[key] = row['admin1_id']

    return df, gdf
    

def add_admin1_neighbors(df, gdf):
    """
    Adds a 'admin1_neighbors' column to df, listing neighbors for each matched admin1 polygon.

    Parameters:
        df (pd.DataFrame): Input dataframe with country/admin1 info.
        gdf (GeoDataFrame): GeoDataFrame with admin1 geometries and 'adm0_a3', 'name_en', etc.

    Returns:
        DataFrame: Updated df with a new 'admin1_neighbors' column (list of neighbor admin1_ids).
    """
    # Step 1: Match df rows to gdf admin1 features
    df_matched, gdf_matched = match_admin1_to_gdf(df, gdf)

    # Step 2: Fix geometry issues
    gdf_matched = gdf_matched.copy()
    gdf_matched['geometry'] = gdf_matched['geometry'].buffer(0)

    # Step 3: Add admin1_id if not already there
    if 'admin1_id' not in gdf_matched.columns:
        gdf_matched['admin1_id'] = gdf_matched['adm0_a3'] + ' - ' + gdf['name_en']

    # Step 4: Spatial join to find touching geometries (neighbors)
    try:
        neighbors = gpd.sjoin(
            gdf_matched[['admin1_id', 'geometry']],
            gdf_matched[['admin1_id', 'geometry']],
            how="left",
            predicate='touches'
        )
    except TopologicalError:
        raise ValueError("Invalid geometry encountered; consider cleaning geometries with buffer(0) or fix tools.")

    # Remove self matches
    neighbors = neighbors[neighbors['admin1_id_left'] != neighbors['admin1_id_right']]

    # Step 5: Build neighbor lookup
    neighbor_dict = neighbors.groupby('admin1_id_left')['admin1_id_right'].apply(list).to_dict()

    # Step 6: Add neighbor info to df
    df_matched['admin1_neighbors'] = df_matched['matched_admin1_id'].map(neighbor_dict)

    return df_matched