Variables stored in separate files (Header+values)

Filename

	Data_separate_files_header_startdate(YYYYMMDD)_enddate(YYYYMMDD)_userid_randomstring_currrentdate(YYYYMMDD).zip
	
	e.g., Data_separate_files_header_20050316_20050601.zip

	
Folder structure

	Networkname
		Stationname

		
Dataset Filename

	CSE_Network_Station_Variablename_depthfrom_depthto_startdate_enddate.ext

	CSE	- Continental Scale Experiment (CSE) acronym, if not applicable use Networkname
	Network	- Network abbreviation (e.g., OZNET)
	Station	- Station name (e.g., Widgiewa)
	Variablename - Name of the variable in the file (e.g., Soil-Moisture)
	depthfrom - Depth in the ground in which the variable was observed (upper boundary)
	depthto	- Depth in the ground in which the variable was observed (lower boundary)
	startdate -	Date of the first dataset in the file (format YYYYMMDD)
	enddate	- Date of the last dataset in the file (format YYYYMMDD)
	ext	- Extension .stm (Soil Temperature and Soil Moisture Data Set see CEOP standard)
	
	e.g., OZNET_OZNET_Widgiewa_Soil-Temperature_0.150000_0.150000_20010103_20090812.stm

	
File Content Sample
	
	REMEDHUS   REMEDHUS        Zamarron          41.24100    -5.54300  855.00    0.05    0.05  (Header)
	2005/03/16 00:00    10.30 U	M	(Records)
	2005/03/16 01:00     9.80 U M

	
Header

	CSE Identifier - Continental Scale Experiment (CSE) acronym, if not applicable use Networkname
	Network	- Network abbreviation (e.g., OZNET)
	Station	- Station name (e.g., Widgiewa)
	Latitude - Decimal degrees. South is negative.
	Longitude - Decimal degrees. West is negative.
	Elevation - Meters above sea level
	Depth from - Depth in the ground in which the variable was observed (upper boundary)
	Depth to - Depth in the ground in which the variable was observed (lower boundary)

	
Record

	UTC Actual Date and Time
	yyyy/mm/dd HH:MM
	Variable Value
	ISMN Quality Flag
	Data Provider Quality Flag, if existing


Network Information

	ARM
		Abstract: The soil moisture datasets collected at ARM facilities originates from two different instruments. SWATS instrument measure soil moisture in two different profiles at 8 depths from 0,05 to 1,75m, the SEBS instrument measure three profiles in depth of 0,025m. The site is managed by U.S. Department of Energy as part of the Atmospheric Radiation Measurement Climate Research Facility.
		Continent: Americas
		Country: USA
		Stations: 35
		Status: running
		Data Range: from 1996-02-05 
		Type: project
		Url: http://www.arm.gov/
		Reference: Cook, D. R. (2016a), Soil temperature and moisture proﬁle (stamp) system handbook, Technical report, DOE Oﬃce of Science Atmospheric Radiation Measurement (ARM) Program. https://www.osti.gov/biblio/1332724;

Cook, D. R. (2016b), Soil water and temperature system (swats) instrument handbook, Technical report, DOE Oﬃce of Science Atmospheric Radiation Measurement (ARM) Program. https://www.osti.gov/biblio/1004944;

Cook, D. R. & Sullivan, R. C. (2018), Surface energy balance system (sebs) instrument handbook, Technical report, DOE Office of Science Atmospheric Radiation Measurement (ARM) Program. https://www.arm.gov/publications/tech_reports/handbooks/sebs_handbook.pdf;
		Variables: precipitation, soil temperature, air temperature, soil moisture, 
		Soil Moisture Depths: 0.02 - 0.02 m, 0.05 - 0.05 m, 0.10 - 0.10 m, 0.15 - 0.15 m, 0.20 - 0.20 m, 0.25 - 0.25 m, 0.35 - 0.35 m, 0.50 - 0.50 m, 0.60 - 0.60 m, 0.75 - 0.75 m, 0.80 - 0.80 m, 0.85 - 0.85 m, 1.25 - 1.25 m, 1.75 - 1.75 m
		Soil Moisture Sensors: Hydraprobe II Sdi-12 E, Hydraprobe II Sdi-12 S, Hydraprobe II Sdi-12 W, SMP1, Water Matric Potential Sensor 229L, 

	SCAN
		Abstract: Soil Climate Analysis Network contains 239 stations all over the USA including stations in Alaska, Hawaii, Puerto Rico or even one in Antarctica. Apart from soil moisture and soil temperature, also precipitation and air temperature are measured. Some stations have also additional measurements of snow depth and snow water equivalent. Almost 150 stations are updated on daily basis. The network is operated by the USDA NRCS National Water and Climate Center with assistance from the USDA NRCS National Soil Survey Center.
		Continent: Americas
		Country: USA
		Stations: 222
		Status: running
Data Range: 

		Type: project
		Url: http://www.wcc.nrcs.usda.gov/
		Reference: Schaefer, G., Cosh, M. & Jackson, T. (2007), ‘The usda natural resources conservation service soil climate analysis network (scan)’, Journal of Atmospheric and Oceanic Technology - J ATMOS OCEAN TECHNOL 24, https://doi.org/10.1175/2007JTECHA930.1;
		Variables: snow water equivalent, precipitation, soil temperature, air temperature, soil moisture, snow depth, 
		Soil Moisture Depths: 0.05 - 0.05 m, 0.10 - 0.10 m, 0.15 - 0.15 m, 0.20 - 0.20 m, 0.25 - 0.25 m, 0.30 - 0.30 m, 0.38 - 0.38 m, 0.51 - 0.51 m, 0.61 - 0.61 m, 0.69 - 0.69 m, 0.76 - 0.76 m, 0.84 - 0.84 m, 0.89 - 0.89 m, 1.02 - 1.02 m, 1.09 - 1.09 m, 1.30 - 1.30 m, 1.42 - 1.42 m
		Soil Moisture Sensors: Hydraprobe Sdi-12, Hydraprobe Analog, n.s., 

	USCRN
		Abstract: Soil moisture NRT network USCRN (Climate Reference Network) in United States;the  datasets of 114 stations were collected and processed by the National Oceanicand Atmospheric Administration"s National Climatic Data Center (NOAA"s NCDC)
		Continent: Americas
		Country: USA
		Stations: 115
		Status: running
		Data Range: from 2009-06-09 
		Type: meteo
		Url: https://www.ncei.noaa.gov/access/crn/
		Reference: Bell, J. E., M. A. Palecki, C. B. Baker, W. G. Collins, J. H. Lawrimore, R. D. Leeper, M. E. Hall, J. Kochendorfer, T. P. Meyers, T. Wilson, and H. J. Diamond. 2013: U.S. Climate Reference Network soil moisture and temperature observations. J. Hydrometeorol., 14, 977-988, https://doi.org/10.1175/JHM-D-12-0146.1;
		Variables: surface temperature, precipitation, soil temperature, air temperature, soil moisture, 
		Soil Moisture Depths: 0.05 - 0.05 m, 0.10 - 0.10 m, 0.20 - 0.20 m, 0.50 - 0.50 m, 1.00 - 1.00 m
		Soil Moisture Sensors: Stevens Hydraprobe II Sdi-12, 

