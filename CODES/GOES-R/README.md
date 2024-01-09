The codes should be ran in the following order:

1. get_goes_data_from_amazon_bucket_2018.py
This code downloads the GOES-R data. Within the code, user specifies range of dates, satellite, product type, etc.

2. latlon_collocation.py
This code creates an Excel spreadsheet, which collocates the coordinates of AERONET sites with the corresponding GOES pixel index. The table contains site name,
latitude and longitude, i and j index of the pixel, the coordinates of the pixel center, distance between site and closest pixel center, and Viewing Zenith Angle.
User is responsible for selecting the proper satellite sample file, as well as list of AERONET sites. For ease, the collocation file has been provided so that
the user would not need to run the collocation script, which may take days to complete. Note that the contents of the file are not expected to change, since
the GOES satellites are in geostationary orbit.

3. GOES_AEROSOLS.py
This comprehensive code consists of 5 major tasks. If the user is only interested in extracting AOD values and nothing more, then Task 1 is sufficient.

Task 1: Reads the pixels of the collocation file produced by 2., and creates a grid of 25x25 closest pixels to the AERONET site. The AOD data are filtered by a
data quality mask, and extracted. A spreadsheet is generated that includes site name, coordinates, distance to closest pixel, VZA, average AOD value of the 25-pixel
pixel grid, the AOD standard deviation, the AOD of the closet pixel, the number of pixels with valid data, date of measurement, and time of measurement. One file is
produced for each timestamp.

Task 2: The files created by Task 1 are parsed, and indexed by the site name. Separate files, each corresponding to an AERONET site is created. In other words, the
"one timestamp, all sites" files are converted to "one site, all timestamps" files. That way there is one file per AERONET site, rather than one file per timestamp.

Task 3 (optional): GOES-R AOD data is crossreferenced with AERONET AOD data, and matched within a 10-minute time tolerance.
Task 4 (optional): A statistics file is prepared that shows correlation, RMSE, and Mean Bias between GOES-R and AERONET AOD data for each site.
Task 5 (optional): Correlation plots are produced for each site, with trendline equation and best-fit line.
