import requests
import pandas as pd
import json
import time
from typing import List, Dict, Optional
import os

class WorldBankDataFetcher:
    """
    A class to fetch data from the World Bank API and format it for analysis.
    """
    
    def __init__(self):
        self.base_url = "https://api.worldbank.org/v2"
        self.format_param = "format=json"
        
        # World Bank indicators from your CSV
        self.indicators = {
            'income_inequality': 'SI.POV.GINI',
            'youth_unemployment': 'SL.UEM.NEET.ZS', 
            'inflation': 'FP.CPI.TOTL.ZG'
        }
    
    def get_countries(self) -> pd.DataFrame:
        """
        Fetch all countries and their metadata from World Bank API.
        """
        url = f"{self.base_url}/country?{self.format_param}&per_page=500"
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            
            if len(data) > 1:
                countries_data = data[1]  # Skip pagination info
                countries_df = pd.DataFrame(countries_data)
                
                # Clean and select relevant columns
                countries_df = countries_df[[
                    'id', 'name', 'iso2Code', 'capitalCity', 
                    'longitude', 'latitude', 'incomeLevel', 'region'
                ]]
                
                # Expand nested dictionaries
                countries_df['income_level'] = countries_df['incomeLevel'].apply(
                    lambda x: x['value'] if isinstance(x, dict) else x
                )
                countries_df['region_name'] = countries_df['region'].apply(
                    lambda x: x['value'] if isinstance(x, dict) else x
                )
                
                countries_df = countries_df.drop(['incomeLevel', 'region'], axis=1)
                
                return countries_df
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching countries: {e}")
            return pd.DataFrame()
    
    def get_indicator_data(self, indicator_code: str, countries: List[str] = None, 
                          start_year: int = 2010, end_year: int = 2027) -> pd.DataFrame:
        """
        Fetch data for a specific indicator from World Bank API.
        
        Args:
            indicator_code: World Bank indicator code (e.g., 'SI.POV.GINI')
            countries: List of country codes (if None, gets all countries)
            start_year: Starting year for data
            end_year: Ending year for data
        """
        
        # If no specific countries provided, use 'all'
        country_string = ';'.join(countries) if countries else 'all'
        
        url = (f"{self.base_url}/country/{country_string}/indicator/{indicator_code}"
               f"?{self.format_param}&date={start_year}:{end_year}&per_page=10000")
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            
            if len(data) > 1 and data[1]:
                indicator_data = data[1]
                df = pd.DataFrame(indicator_data)
                
                # Clean the data
                df = df[['country', 'countryiso3code', 'date', 'value', 'indicator']]
                df['country_name'] = df['country'].apply(
                    lambda x: x['value'] if isinstance(x, dict) else x
                )
                df['indicator_name'] = df['indicator'].apply(
                    lambda x: x['value'] if isinstance(x, dict) else x
                )
                df['year'] = pd.to_numeric(df['date'])
                df['indicator_code'] = indicator_code
                
                # Select final columns
                df = df[['countryiso3code', 'country_name', 'year', 'value', 
                        'indicator_code', 'indicator_name']]
                
                # Remove rows with null values
                df = df.dropna(subset=['value'])
                df['value'] = pd.to_numeric(df['value'], errors='coerce')
                
                return df
            else:
                print(f"No data found for indicator {indicator_code}")
                return pd.DataFrame()
                
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data for {indicator_code}: {e}")
            return pd.DataFrame()
    
    def get_all_indicators(self, countries: List[str] = None, 
                          start_year: int = 2010, end_year: int = 2023) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for all indicators defined in the class.
        """
        all_data = {}
        
        for indicator_name, indicator_code in self.indicators.items():
            print(f"Fetching data for {indicator_name} ({indicator_code})...")
            
            df = self.get_indicator_data(indicator_code, countries, start_year, end_year)
            
            if not df.empty:
                all_data[indicator_name] = df
                print(f"✓ Successfully fetched {len(df)} records for {indicator_name}")
            else:
                print(f"✗ No data retrieved for {indicator_name}")
            
            # Be respectful to the API
            time.sleep(0.5)
        
        return all_data
    
    def combine_indicators(self, data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Combine all indicator datasets into a single wide-format DataFrame.
        """
        if not data_dict:
            return pd.DataFrame()
        
        combined_df = None
        
        for indicator_name, df in data_dict.items():
            # Pivot to get years as columns
            pivot_df = df.pivot_table(
                index=['countryiso3code', 'country_name'], 
                columns='year', 
                values='value', 
                aggfunc='first'
            )
            
            # Flatten column names
            pivot_df.columns = [f"{indicator_name}_{col}" for col in pivot_df.columns]
            pivot_df = pivot_df.reset_index()
            
            if combined_df is None:
                combined_df = pivot_df
            else:
                combined_df = pd.merge(
                    combined_df, pivot_df, 
                    on=['countryiso3code', 'country_name'], 
                    how='outer'
                )
        
        return combined_df

    def save_data(self, data_dict: Dict[str, pd.DataFrame], countries_df: pd.DataFrame = None,
                  output_dir: str = "data/raw/world_bank"):

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Save metadata if provided
        if countries_df is not None and not countries_df.empty:
            metadata_path = os.path.join(output_dir, "country_metadata.csv")
            countries_df.to_csv(metadata_path, index=False)
            print(f"Saved country metadata to {metadata_path}")

        for indicator_name, df in data_dict.items():
            filename = f"{output_dir}/{indicator_name}_worldbank.csv"
            df.to_csv(filename, index=False)
            print(f"Saved {indicator_name} data to {filename}")

        if len(data_dict) > 1:
            combined_df = self.combine_indicators(data_dict)
            combined_filename = f"{output_dir}/combined_indicators.csv"
            combined_df.to_csv(combined_filename, index=False)
            print(f"Saved combined dataset to {combined_filename}")

# Example usage
if __name__ == "__main__":
    # Initialize the fetcher
    wb_fetcher = WorldBankDataFetcher()
    
    # Get countries information
    print("Fetching country information...")
    countries_df = wb_fetcher.get_countries()
    if not countries_df.empty:
        print(f"Found {len(countries_df)} countries/regions")
        
        # Filter for actual countries (exclude aggregates)
        actual_countries = countries_df[
            (countries_df['longitude'].notna()) & 
            (countries_df['latitude'].notna())
        ]

        if not actual_countries.empty:
            os.makedirs("data/raw/world_bank", exist_ok=True)

            actual_countries.to_csv(
                "data/raw/world_bank/country_metadata.csv",
                index=False
            )

        print(f"Filtered to {len(actual_countries)} actual countries")
    
    # Fetch all indicator data
    print("\nFetching indicator data...")
    data_dict = wb_fetcher.get_all_indicators(
        countries=None,  # Get all countries
        start_year=2010, 
        end_year=2027
    )
    
    # Save the data
    print("\nSaving data...")
    wb_fetcher.save_data(data_dict)
    
    # Display summary statistics
    print("\n" + "="*50)
    print("DATA SUMMARY")
    print("="*50)
    
    for indicator_name, df in data_dict.items():
        if not df.empty:
            print(f"\n{indicator_name.upper()}:")
            print(f"  Countries: {df['countryiso3code'].nunique()}")
            print(f"  Years: {df['year'].min()} - {df['year'].max()}")
            print(f"  Records: {len(df)}")
            print(f"  Latest value (sample): {df[df['year'] == df['year'].max()]['value'].mean():.2f}")
    
    print(f"\nAll data saved to 'worldbank_data/' directory")