import numpy as np
import s3fs
from datetime import datetime,timedelta

beg = "202315200"
end = "202318123"

beg_time = datetime.strptime(beg,'%Y%j%H')
end_time = datetime.strptime(end,'%Y%j%H')

delta = timedelta(hours=1) #increment one hour at a time

while beg_time <= end_time:
    print(beg_time.strftime("%Y-%j-%H"))
    hour = datetime.strftime(beg_time,"%H")
    year = datetime.strftime(beg_time,"%Y")
    jday = datetime.strftime(beg_time,"%j")
    day =  datetime.strftime(beg_time,"%d")
    month = datetime.strftime(beg_time,"%b").upper()
    
    satellite = "noaa-goes16"
    #satellite = "noaa-goes18"
    
    prod_level = "ABI-L2-"
    
    #products = ['ACM','ACTP','AOD','COD','CPS']
    products = ['AOD']
    
	#set to CONUS domain  (use 'F' for full disk)
    domain = 'F'

	#Setup system to connect to Amazon GOES Bucket
    fs = s3fs.S3FileSystem(anon=True)
	# List contents of GOES-16 bucket.
    for prod in products:
        prod_path = satellite + '/' + prod_level  + prod + domain +'/'+year+'/'+jday+'/'+hour+'/'
        files = np.array(fs.ls(prod_path))
        for f in files:
            filename = "GOES_DATA/" + year + '/' + month + '/' + day + '/' + f.split('/')[-1]
            fs.get(f, filename)  
    beg_time += delta