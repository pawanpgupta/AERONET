import pandas as pd
import numpy as np
import xarray

df = pd.read_csv(r'/home/pgrigoro/GOES_satellite_processing/Site_Locations/locations_v4000.fil', header=None)
df = df[[0,1,2]] #filters dataset to site name, latitude and longitude columns
df = df.rename(columns={0: "Name", 1: "Lon", 2: "Lat"})

def degrees(file_id):
    # Ignore numpy error for sqrt of negative number ('x'); occurs for GOES-16 ABI CONUS sector data
    np.seterr(invalid='ignore')
    
    # Read in GOES Imager Projection data
    lat_rad_1d = file_id.variables['x'][:]
    lon_rad_1d = file_id.variables['y'][:]
    projection_info = file_id.variables['goes_imager_projection']
    lon_origin = projection_info.attrs['longitude_of_projection_origin']
    H = projection_info.attrs['perspective_point_height']+projection_info.attrs['semi_major_axis']
    r_eq = projection_info.attrs['semi_major_axis']
    r_pol = projection_info.attrs['semi_minor_axis']
    
    # Create meshgrid filled with radian angles
    lat_rad,lon_rad = np.meshgrid(lat_rad_1d,lon_rad_1d)
    
    # lat/lon calculus routine from satellite radian angle vectors
    lambda_0 = (lon_origin*np.pi)/180.0
    
    a_var = np.power(np.sin(lat_rad),2.0) + (np.power(np.cos(lat_rad),2.0)*(np.power(np.cos(lon_rad),2.0)+(((r_eq*r_eq)/(r_pol*r_pol))*np.power(np.sin(lon_rad),2.0))))
    b_var = -2.0*H*np.cos(lat_rad)*np.cos(lon_rad)
    c_var = (H**2.0)-(r_eq**2.0)
    r_s = (-1.0*b_var - np.sqrt((b_var**2)-(4.0*a_var*c_var)))/(2.0*a_var)
    s_x = r_s*np.cos(lat_rad)*np.cos(lon_rad)
    s_y = - r_s*np.sin(lat_rad)
    s_z = r_s*np.cos(lat_rad)*np.sin(lon_rad)
    
    lat = (180.0/np.pi)*(np.arctan(((r_eq*r_eq)/(r_pol*r_pol))*((s_z/np.sqrt(((H-s_x)*(H-s_x))+(s_y*s_y))))))
    lon = (lambda_0 - np.arctan(s_y/(H-s_x)))*(180.0/np.pi)
    
    return lat, lon

def haversine(lat1, lon1, lat2, lon2):
    # Calculate the great-circle distance between two points on the Earth's surface
    # using their latitudes and longitudes in radians
    R = 6371  # Radius of the Earth in kilometers
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c
    return distance

def extract_vza(file_id):
    # Read in the cloud mask data
    vza = file_id.variables['Sensor_Zenith_Angle'].data
    return vza

#opens a sample NC file and calculates LAT/LON just once. It doesn't matter which file we choose because
#coordinates will never change between files due to satellite's geostationary orbit
LATS,LONS = degrees(xarray.open_dataset("sample_g16.nc"))
#LATS,LONS = degrees(xarray.open_dataset("sample_g18.nc"))

pixel_array = np.empty((0,5)) #initializes empty array with 3 columns that store distance, i and j indices.

for k in range(len(df)):
    print(df['Name'][k], k)
    
    site_lat = df['Lat'][k]  #latitude of site
    site_lon = df['Lon'][k]  #longitude of site
    
    # Calculate the latitude and longitude range for the site
    lat_min = site_lat - 0.1
    lat_max = site_lat + 0.1
    lon_min = site_lon - 0.1
    lon_max = site_lon + 0.1

    distances = [] #initializes empty 1D array where distances for each site is stored
    x = [] #initializes 1D array where 'j' pixel index is stored for each corresponding distance
    y = [] #initializes 1D array where 'i' pixel index is stored for each corresponding distance
    lat = []
    lon = []
    
    #subsetting is done here
    for i in range(len(LATS)): #iterates over the column (latitude) index
        for j in range(len(LONS)): #iterates over the row (longitude) index
            pixel_lat = LATS[i,j]
            pixel_lon = LONS[i,j]
            # Check if the pixel is within the specified latitude and longitude range
            if (lat_min <= pixel_lat <= lat_max) and (lon_min <= pixel_lon <= lon_max):
                #computes distance between site and each of the pixels
                distance = haversine(site_lat, site_lon, pixel_lat, pixel_lon) 
                distances.append(distance) #appends the computed distance to the 'distances' array
                x.append(j) #appends the row and column index of the pixel that belongs to that distance
                y.append(i)
                lat.append(pixel_lat) #appends the latitude and longitude of the corresponding pixel
                lon.append(pixel_lon)
            
    #converts the distances and indices to a pandas dataframe
    df_min = pd.DataFrame({'Glon':lon,'Glat':lat,'Gidx':x,'Gjdx':y,'Dist(km)':distances})
    #takes the row with the smallest distance, and converts to a numpy array
    df_min = df_min[df_min['Dist(km)'] == df_min['Dist(km)'].min()].to_numpy()
    
    if df_min.shape == (0, 5):   #if there is no data for a particular site (GOES-R out of range)
        df_min = np.zeros((1, 5))   #it will populate the values with zeroes instead
        
    #appends the minimum distance row of each site to the combined pixel array
    pixel_array = np.vstack([pixel_array,df_min])
        
df_new = pd.DataFrame(pixel_array, columns = ['Glon','Glat','Gidx','Gjdx','Dist(km)']) 
df_combined = pd.concat([df,df_new], axis=1) #combines original dataframe with distance, i and j pixel indices
df_combined['Gidx'] = df_combined['Gidx'].astype('int')
df_combined['Gjdx'] = df_combined['Gjdx'].astype('int')
df_combined = df_combined.loc[df_combined['Dist(km)'] > 0]

C = xarray.open_dataset("/home/pgrigoro/GOES_satellite_processing/g16_east_vza.nc")

vza_data = extract_vza(C)
vza_arr = []

for i, row in df_combined.iterrows():
    iy = row['Gidx']
    ix = row['Gjdx']
    
    vza = vza_data[ix*2, iy*2]
    vza_arr.append(vza)
    
df_combined['VZA'] = vza_arr
df_combined = df_combined.loc[df_combined['VZA'] >= 0]

df_combined.to_csv('LatLon_Collocation_G16.csv',index=False) #saves the combined dataframe as a csv file
#df_combined.to_csv('LatLon_Collocation_G18.csv',index=False) #saves the combined dataframe as a csv file
