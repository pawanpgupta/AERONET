import os
import numpy as np
import glob
import xarray
import random
import csv
import math
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import warnings 
warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None

# "quality" is a global variable set in final block
def extract_data(file_id, quality):
    # Read in  data
    data = C['AOD'].data
    dqf = C['DQF'].data
    
    # Select quality of AOD data by masking pixels we don't want to display using DQF variable
    # high quality: DQF = 0, medium quality: DQF = 1, low quality: DQF = 2, not retrieved (NR): DQF = 3
    # Use high + medium qualities ("top 2 qualities") for operational applications (e.g., mask low quality and NR pixels)
    
    if quality == 'high':
        new_dqf = np.where(dqf < 2,0,dqf)
    if quality == 'top2':
         new_dqf = np.where(dqf < 4,0,dqf)
    if quality == 'all':
         new_dqf = dqf
         
    masked_data = np.ma.masked_array(data,new_dqf)
    
    return masked_data

def extract_time(file_id):
    # Read in time data
    time = file_id.variables['t'].data
    return time

#converts tuple to string for directories
def convertTuple(tup):
    st = ''.join(map(str, tup))
    return st

# Define a custom function to find the closest timestamp and return merged values
def find_closest_timestamp(timestamp, df, tolerance):
    closest_timestamp = (df['Timestamp'] - timestamp).abs().idxmin()
    if abs(df.loc[closest_timestamp, 'Timestamp'] - timestamp) <= pd.Timedelta(minutes=tolerance):
        return df.loc[closest_timestamp].drop('Timestamp')  # Exclude the Timestamp column
    else:
        return pd.Series(index=df.columns)
    
df = pd.read_csv("LatLon_Collocation_G16.csv")
#df = pd.read_csv("LatLon_Collocation_G18.csv")
GOES_names = df.iloc[:, 0].tolist()

# Specify the directory path where .txt files are located
AERONET_directory_path = '/home/pgrigoro/GOES_satellite_processing/dosadka/files'
#AERONET_directory_path = 'C:/Users/pgrigoro/Documents/dosadka/files'

# Initialize an empty list to store the filenames without the extension
AERONET_file_names_without_extension = []

# Use os.listdir() to get a list of files in the directory
AERONET_file_list = os.listdir(AERONET_directory_path)

# Iterate through the files and extract the filenames without the .txt extension
for AERONET_file_name in AERONET_file_list:
    if AERONET_file_name.endswith('.txt'):
        # Remove the ".txt" extension and append the name to the list
        AERONET_name_without_extension = os.path.splitext(AERONET_file_name)[0]
        AERONET_file_names_without_extension.append(AERONET_name_without_extension)

# Convert the lists to sets
AERONET_list = set(AERONET_file_names_without_extension)
GOES_list = set(GOES_names)

location_list = list(AERONET_list.intersection(GOES_list))
df = df[df['Name'].isin(location_list)]

# Define the directory where the netCDF files are located
base_directory = "GOES_DATA"

# Define the output directory
output_directory = "GOES_OUTPUT"

# Define the start and end dates (NEED TO ADJUST AS NEEDED)
start_date = datetime(2023, 6, 1)
end_date = datetime(2023, 6, 30)

# Define the time step for iteration (1 day)
time_step = timedelta(days=1)

# Iterate through each day from start_date to end_date
current_date = start_date

print("Task 1: Extracting AOD550 metrics from GOES-R netcdf files.")

while current_date <= end_date:
    print(current_date)
    year = current_date.year
    month = current_date.strftime("%b").upper()
    day = current_date.day

    # Construct the file path based on the current date
    file_path = os.path.join(base_directory, str(year), month, f"{day:02d}", "*.nc")

    # Get a list of matching files
    matching_files = glob.glob(file_path)
    
    # Process each matching file
    for file in matching_files:
        C = xarray.open_dataset(file)

        data = extract_data(C,'high') #this is where you specify DQF quality; 'high' is 0, 'top2' is 0 or 1, 'all is everything
        
        timestamp = extract_time(C).astype(str) #extracts ISO format timestamp from NC file and converts to string
        timestamp = str(timestamp)
        timestamp = timestamp.split('.')[0] #removes fraction of a second
        datetime_object = datetime.strptime(timestamp, '%Y-%m-%dT%H:%M:%S') #Parses the timestamp as a string
        date = datetime_object.date() #extracts the date from the timestamp
        time = datetime_object.strftime('%H:%M:%S') #extracts the time from the timestamp; converts to 24-hour format
        
        site_arr = []
        aod_closest_arr = []
        aod_avg_arr = []
        aod_std_arr = []
        aod_num_arr = []
        dqf_avg_arr = []
        dqf_std_arr = []
        dqf_num_arr = []
        lat = []
        lon = []
        dist = []
        vza = []

        for i, row in df.iterrows():
            
            iy = row['Gidx']
            ix = row['Gjdx']
            
            iy_arr = [iy-4, iy-3, iy-2, iy-1, iy, iy+1, iy+2, iy+3, iy+4]
            ix_arr = [ix-4, ix-3, ix-2, ix-1, ix, ix+1, ix+2, ix+3, ix+4]
            
            if 0 <= ix < data.shape[0] and 0 <= iy < data.shape[1]:
                aod_n = []
                dqf_n = []
                
                for j in range(len(iy_arr)):
                    for k in range(len(ix_arr)):
                        
                        if data[ix_arr[k], iy_arr[j]] >= -1:
                            aod = data[ix_arr[k], iy_arr[j]]
                            dqf = int(data.mask[ix_arr[k], iy_arr[j]])
                            aod_n.append(aod)
                            dqf_n.append(dqf)
                            
                        else:            
                            aod = -999.
                            dqf = -999.
                            aod_n.append(aod)
                            dqf_n.append(dqf)      
                            
                aod_closest = aod_n[12] #index 12 contains ix,iy - the center pixel
                aod_avg = np.average(np.array(aod_n)[(np.array(aod_n) != -999.)])
                aod_std = np.std(np.array(aod_n)[(np.array(aod_n) != -999.)])
                dqf_avg = np.average(np.array(dqf_n)[(np.array(dqf_n) != -999.)])
                dqf_std = np.std(np.array(dqf_n)[(np.array(dqf_n) != -999.)])
                aod_num = np.count_nonzero(np.array(aod_n) != -999)
                dqf_num = np.count_nonzero(np.array(dqf_n) != -999) 
 
            site_arr.append(row['Name'])
            lat.append(row['Lat'])
            lon.append(row['Lon'])
            dist.append(row['Dist(km)'])
            vza.append(row['VZA'])
            aod_closest_arr.append(aod_closest)
            aod_avg_arr.append(aod_avg)
            dqf_avg_arr.append(dqf_avg)
            aod_std_arr.append(aod_std)
            dqf_std_arr.append(dqf_std)
            aod_num_arr.append(aod_num)
            dqf_num_arr.append(dqf_num)

        #creates arrays with timestamp on file, with size that matches other arrays
        date_arr = [date]*len(df)
        time_arr = [time]*len(df)
        
        df_output = pd.DataFrame(
            {'Name': site_arr,
             'Latitude': lat,
             'Longitude': lon,
             'Distance (km)': dist,
             'VZA': vza,
             'GOES_AOD550_avg': aod_avg_arr,
             'GOES_AOD550_std': aod_std_arr,
             'GOES_AOD550_Nearest_Pixel': aod_closest_arr, 
             'GOES_AOD550_num_valid': aod_num_arr,
             'Date': date_arr,
             'Time': time_arr
            })
        
        df_output['Timestamp'] = pd.to_datetime(df_output['Date'].astype(str) + ' ' + df_output['Time'].astype(str))
        df_output = df_output.drop(['Date','Time'], axis=1)
        df_output = df_output[df_output['GOES_AOD550_avg'] > -999.]
        
        # Extract the netCDF file name (excluding the extension)
        netcdf_file_name = os.path.splitext(os.path.basename(file))[0]
        
        # Create the output subdirectories if they don't exist
        os.makedirs(output_directory, exist_ok=True)

        # Define the output file name for the CSV, including the netCDF file name
        output_file = os.path.join(output_directory, f"{netcdf_file_name}.txt")
        
        # Save the DataFrame as a CSV
        df_output.to_csv(output_file, index=False)

        # Close the netCDF dataset
        C.close()

    # Move to the next day
    current_date += time_step
    
print("All data from netcdf files was extracted!")

print("\nTask 2: Converting GOES-R time files to site files.")

input_folder = "GOES_OUTPUT"
output_folder = "GOES_SITE"

# List all .txt files in the directory
txt_files = [f for f in os.listdir(input_folder) if f.endswith('.txt')]

# Randomly select a .txt file from the list
random_txt_file = random.choice(txt_files)

# Construct the full path to the randomly selected .txt file
txt_file_path = os.path.join(input_folder, random_txt_file)

# Now you can proceed to read the headers from this randomly selected .txt file
with open(txt_file_path, 'r', newline='') as file:
    reader = csv.reader(file)
    headers = next(reader)

# Create the output directory if it doesn't exist
os.makedirs(output_folder, exist_ok=True)
for location_name in location_list:  
    # Iterate through all .txt files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.txt'):
            input_filepath = os.path.join(input_folder, filename)
            # Open the input .txt file
            with open(input_filepath, mode='r') as input_csv:
                csv_reader = csv.reader(input_csv)
                # Iterate through rows in the input .txt file
                for row in csv_reader:
                    # Check if the first column matches the location name
                    if row[0] == location_name:
                        # Create an output .txt file for the current location
                        output_filename = os.path.join(output_folder, f'{location_name}.txt')
                        with open(output_filename, mode='a', newline='') as output_csv:
                            output_writer = csv.writer(output_csv)
                            # Write the data rows
                            output_writer.writerow(row)
                            
# Add headers to all files in the output directory
for filename in os.listdir(output_folder):
    if filename.endswith('.txt'):
        output_filepath = os.path.join(output_folder, filename)
        with open(output_filepath, mode='r') as output_csv:
            data = output_csv.read()
        with open(output_filepath, mode='w', newline='') as output_csv:
            output_csv.write(",".join(headers) + "\n" + data)
            
print("All time data converted to site data!")

print("\nTask 3: Crossreferencing AERONET and GOES-R files.")

for i in range(len(location_list)):
    try:
        print('Site ',location_list[i])
        
        dir_AERONET = r'/home/pgrigoro/GOES_satellite_processing/dosadka/files/',location_list[i],'.txt'
        dir_GOES = r'/home/pgrigoro/GOES_satellite_processing/GOES_SITE/',location_list[i],'.txt'
        #dir_AERONET = r'C:/Users/pgrigoro/Documents/dosadka/files/',location_list[i],'.txt'
        #dir_GOES = r'C:/Users/pgrigoro/Documents/GOES_SITE/',location_list[i],'.txt'  
        
        path_AERONET = convertTuple(dir_AERONET)
        path_GOES = convertTuple(dir_GOES)
        
        df_AERONET = pd.read_csv(path_AERONET,sep=',',engine='python')
        df_AERONET = df_AERONET[(df_AERONET['Level'] == "L1.5V") | (df_AERONET['Level'] == "L2")].reset_index(drop=True)
        if len(df_AERONET) > 0:
            df_AERONET[['Day','Month','Year']] = df_AERONET['Date'].str.split(':',expand=True)
            df_AERONET['Date'] = df_AERONET[['Year','Month','Day']].apply(lambda x: '-'.join(x.values.astype(str)), axis="columns")
            df_AERONET = df_AERONET.drop(columns=['Year','Month','Day'])
            df_AERONET['Date']= pd.to_datetime(df_AERONET['Date'])
            df_AERONET['Timestamp'] = pd.to_datetime(df_AERONET['Date'].astype(str) + ' ' + df_AERONET['Time'].astype(str))
            df_AERONET = df_AERONET.drop(['Date','Time'], axis=1)
            df_AERONET['Timestamp'] = pd.to_datetime(df_AERONET['Timestamp'])
            df_AERONET.sort_values('Timestamp', inplace=True)
            
            x = np.array([np.log(440), np.log(500), np.log(675)]) #may eventually include 870
            df_AERONET['AOT550'] = np.nan
            
            for j in range(len(df_AERONET)):
                aod = np.array([df_AERONET['AOT440'][j], df_AERONET['AOT500'][j], df_AERONET['AOT675'][j]])
                if (aod >= 0).all():
                    y = np.array([np.log(df_AERONET['AOT440'][j]), np.log(df_AERONET['AOT500'][j]), np.log(df_AERONET['AOT675'][j])])
                    a, b, c = np.polyfit(x, y, 2)
                    df_AERONET['AOT550'][j] = math.exp((a*pow(np.log(550),2)) + (b*np.log(550)) + c)
                else:
                    df_AERONET['AOT550'][j] = np.nan
            
            df_GOES = pd.read_csv(path_GOES)
            df_GOES['Timestamp'] = pd.to_datetime(df_GOES['Timestamp'])
            df_GOES['Timestamp_GOES'] = df_GOES['Timestamp']
            df_GOES.sort_values('Timestamp', inplace=True)
            
            merged_values = df_AERONET.apply(lambda row: find_closest_timestamp(row['Timestamp'], df_GOES, 10), axis=1)
            df = pd.concat([df_AERONET, merged_values], axis=1)
            df.rename(columns={'Timestamp': 'Timestamp_AERONET'}, inplace=True)
            df = df.dropna(axis=1, how='all')
            df = df.dropna().reset_index(drop=True)
            
            if 'AOT550' in df.columns:
                df = df[['Latitude','Longitude','SZA','Alt(m)','AOT340','AOT380','AOT440','AOT500','AOT675','AOT870',
                         'AOT1020','AOT1020In','AOT1640','AE','WV','AOT550','GOES_AOD550_avg','GOES_AOD550_std',
                'GOES_AOD550_num_valid','GOES_AOD550_Nearest_Pixel','Distance (km)','VZA','Timestamp_AERONET','Timestamp_GOES']]
                df.reset_index(drop=True, inplace=True)
                
                df['Date'] = df['Timestamp_AERONET'].dt.date
                df['Hour'] = np.nan
                
                for k in range(len(df)):
                    df['Hour'][k] = df['Timestamp_AERONET'][k].strftime('%H:%M:%S')   #isolates the hours, minutes and seconds from the solar timestamp
                    df['Hour'][k] = df['Hour'][k][:-6]
                
                df['Date'] = pd.to_datetime(df['Date'])
                df['Daily_Occurence'] = df.groupby('Date')['Date'].transform('size') 
                df = df[df['Daily_Occurence'] >= 10]
                df = df.drop(columns=['Daily_Occurence','Timestamp_AERONET','Timestamp_GOES'])
                
                df = df.groupby(['Date','Hour']).mean().reset_index()
                os.makedirs("AERONET_GOES", exist_ok=True)
                outdir = '/home/pgrigoro/GOES_satellite_processing/AERONET_GOES/',location_list[i],'.csv'
                #outdir = 'C:/Users/pgrigoro/Documents/AERONET_GOES/',location_list[i],'.csv'
                outpath = convertTuple(outdir)
                df.to_csv(outpath,index=False)
    except:
        print("Site not found: ", location_list[i])
        
print("All files successfully cross-referenced!")

print("\nTask 4: Preparing Statistics File")

input_folder = "AERONET_GOES"
csv_files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]
name_list = [filename.split('.')[0] for filename in csv_files]

names_arr = []
lat_arr = []
lon_arr = []
num_points = []
dist_closest = []
corr_arr = []
rmse_arr = []
mbe_arr = []
vza_arr = []

for i in range(len(csv_files)):
    df = pd.DataFrame()
    directory = '/home/pgrigoro/GOES_satellite_processing/AERONET_GOES/',csv_files[i]
    #directory = 'C:/Users/pgrigoro/Documents/AERONET_GOES/',csv_files[i]
    path = convertTuple(directory)
    df = pd.read_csv(path)
    df.replace(-999.0, np.nan, inplace=True)
    
    if len(df) > 0:
        corr = df['AOT550'].corr(df['GOES_AOD550_avg'])
        rmse = math.sqrt(np.square(np.subtract(df['AOT550'],df['GOES_AOD550_avg'])).mean())
        mbe = np.sum(np.subtract(df['AOT550'],df['GOES_AOD550_avg']))/len(df)
        
        corr_arr.append(corr)
        rmse_arr.append(rmse)
        mbe_arr.append(mbe)
        
        names_arr.append(name_list[i])
        lat_arr.append(df['Latitude'][0])
        lon_arr.append(df['Longitude'][0])
        dist_closest.append(df['Distance (km)'][0])
        vza_arr.append(df['VZA'][0])
        num_points.append(len(df))

df_corr = pd.DataFrame(
    {'Name': names_arr,
     'Latitude': lat_arr,
     'Longitude': lon_arr,
     'Num Collocations': num_points,
     'Distance (km)': dist_closest,
     'VZA': vza_arr,
     'Correlation': corr_arr,
     'RMSE': rmse_arr,
     'Mean Bias': mbe_arr
    })    

df_corr = df_corr.sort_values(by='Name')
df_corr.to_csv("GOES_AERONET_AOD_Statistics.csv", index=False)

print("Statistics file produced!")

print("\nTask 5: Preparing Correlation Plots")

output_folder = "CORRELATION_PLOTS"
os.makedirs(output_folder, exist_ok=True)

for i in range(len(name_list)):
    df = pd.DataFrame()
    directory = '/home/pgrigoro/GOES_satellite_processing/AERONET_GOES/',name_list[i],'.csv'
    #directory = 'C:/Users/pgrigoro/Documents/AERONET_GOES/',name_list[i],'.csv'
    path = convertTuple(directory)
    df = pd.read_csv(path)
    
    if len(df) > 0:
        corr = df['AOT550'].corr(df['GOES_AOD550_avg'])
        df1 = df[['AOT550','GOES_AOD550_avg']].dropna()
        max_value = round(df1.max().max(),1)
        plt.scatter(df1['AOT550'], df1['GOES_AOD550_avg'], s=10)
        m, b = np.polyfit(df1['AOT550'], df1['GOES_AOD550_avg'], 1)
        plt.plot(df1['AOT550'], m*df1['AOT550']+b, color='red')
        plt.xlim(0,df1.max().max())
        plt.ylim(0,df1.max().max())
        trendline = 'y = '+str(round(m,3))+'x + '+str(round(b,3))
        plt.xlabel('AERONET AOT550')
        plt.ylabel('GOES16 AOT550')
        plt.title('AERONET Site: '+str(name_list[i])+'\nTrendline: '+str(trendline)+'         Coefficient: '+str(round(corr,4)))
        plt.savefig('/home/pgrigoro/GOES_satellite_processing/'+str(output_folder)+'/'+str(name_list[i])+'.png', dpi=300)
        #plt.savefig('C:/Users/pgrigoro/Documents/'+str(output_folder)+'/'+str(name_list[i])+'.png', dpi=300)
        plt.clf()
    print("Site "+str(name_list[i]))

print("Correlation plots produced!")