# Keep only observations at monitor locations 
# Functions
# Bayesian Nonparametric Ensemble 
# Sebastian T. Rowland and Lawrence G. Chilrud

####***********************
#### Table of Contents ####
####***********************

# I: Ideas 
# 0: Preparation 
# 1: Set Function Parameters
# 2: Get Distinct Input Model Locations
# 3: Convert Input Model Data to Spatial Polygon
# 4: Convert Monitor Locations to Spatial Point
# 5: Identify Locations via Join 
# 6: Keep Relevant Observations
# 7: Save Results

####**************
#### I: Ideas ####
####**************

# I1: The set of monitors could be somehow automaticlaly selected based on 
# a characteristic of input model 

# I2: currently assume that latitude and longitude columns are named 
# "lat" and "long"

####********************
#### 0: Preparation ####
####********************

# 0a Whose Computer? 
user <- "STR" # LGC

# 0b Load packages
pacman::p_load(tidyverse, sf, rgdal)

# 0c Readin data
# this section where go into a separate script - where you first read in the data 
# make sure it is appropriately processed
# and then you run the function 
if(user == "LGC"){
  # set parameters
  InputDataPath <- here::here("CMAQ/outputs/2011_pm25_daily_average.csv")
  MonitorLocationsPath <- here::here("EPA_data/latest_version_clean_annual_data/annual_data_2000-2016.csv")
  
  # import data:
  InputData <- read_csv(InputDataPath, col_types = "Dcdddd") %>% 
    rename(lat = Latitude, long = Longitude)
  # note: we should do this sort of cleaning in a different script so that we have one file we use for everything 
  MonitorLocations.Raw <- read_csv(MonitorLocationsPath) 
  MonitorLocations.Int <- MonitorLocations.Raw %>% 
    select(!Arithmetic.Mean) %>% na.omit() %>% 
    rename(lat = Latitude, long = Longitude) %>%
    filter(Year == 2011)
  MonitorsList <- unique(MonitorLocations.Int$Monitor.ID)
  
  # split up different datum types
  MonitorLocations.Int.WGS84 <- MonitorLocations.Int %>% 
    filter(Datum == "WGS84") %>% 
    select(!Datum)
  MonitorLocations.Int.NAD83 <- MonitorLocations.Int %>% 
    filter(Datum == "NAD83") %>% 
    select(!Datum)
  
  # standardize Lon Lat coords:
  MonitorLocations.sf.NAD83 <- SpatialPointsDataFrame(
    coords = MonitorLocations.Int.NAD83[, c("long", "lat")], 
    data = MonitorLocations.Int.NAD83, proj4string = CRS("+init=epsg:4269"))
  MonitorLocations.sf.NAD83.as.WGS84 <- spTransform(MonitorLocations.sf.NAD83, 
                                                    CRS("+init=epsg:4326"))
  MonitorLocations.sf.WGS84 <- SpatialPointsDataFrame(
    coords = MonitorLocations.Int.WGS84[, c("long", "lat")], 
    data = MonitorLocations.Int.WGS84, proj4string = CRS("+init=epsg:4326"))
  MonitorLocations.sf <- rbind(MonitorLocations.sf.NAD83.as.WGS84, 
                               MonitorLocations.sf.WGS84)
  
  #MonitorLocations.Int.sf <- st_as_sf(MonitorLocations.Int, 
                                      #coords = c("long", "lat"), 
                                      #crs=st_crs("+init=epsg:4326"))
}

if(user == "STR"){
  # set parameters
  InputDataPath <- "data/predictions_MERRA.csv"
  MonitorLocationsPath <- "data/annual_75.csv"
  
  # readin 
  InputData <- read_csv(here::here(InputDataPath))
  MonitorLocations <- read_csv(here::here(MonitorLocationsPath))
  
  # rename
  MonitorLocations <- MonitorLocations %>%
    rename(lat = Latitude, long = Longitude)
  
}  


####**********************
#### BEGIN FUNCTION ####
####**********************

####********************************
#### 1: Set Function Parameters ####
####********************************

# 1a Set parameters
if(user == "LGC"){
  InputmodelShape <- "CensusTracts"
  LocKeyVar <- c('lat', 'long', 'fips') 
  # Lawrence, I think we can actually just use fips here 
  # LocKeyVar would have to include year if year is relevant 
}
if(user == "STR"){
  InputmodelShape <- "impliedGrid"
  LocKeyVar <- c('lat', 'long')
}

# Check for potential issues 
# Note: if the input model uses fits or some other character name for location 
# then we do not need lat long for them. 
if(str_detect(toString(names(MonitorLocations)), "lat") == FALSE | 
   str_detect(toString(names(MonitorLocations)), "long") == FALSE |
   InputmodelShape == "impliedGrid" & (str_detect(toString(names(InputData)), "lat") == FALSE) | 
   InputmodelShape == "impliedGrid" & (str_detect(toString(names(InputData)), "long") == FALSE)){
  print("one of the datasets does not have columns named lat or long, 
        please check the names of the columns containing latitude and 
        longitude")
}

####*******************************************
#### 2: Get Distinct Input Model Locations ####
####*******************************************

# 2a Keep only the unique locations
# can add year here if loc might change year to year 
InputLocations <- InputData %>%
  distinct(across(!!LocKeyVar)) %>% 
  mutate(Index = row_number()) %>%
  unite("locKey", !!LocKeyVar, sep = "_", remove = FALSE)

####****************************************************
#### 3: Convert Input Model Data to Spatial Polygon ####
####****************************************************

# 3a Define the projection string 
projString <- "+proj=laea +lat_0=45 +lon_0=-100 +x_0=0 
                            +y_0=0 +a=6370997 +b=6370997 +units=m +no_defs"

# 3a Convert implied grids to polygons 
if(InputmodelShape == "impliedGrid"){
  # 3b.i Convert to sf 
  InputLocations <- st_as_sf(InputLocations, coords = c("long", "lat"), 
                      crs=st_crs("epsg:4326"))
  # 3b.ii Transform geographical coordinates to Lambert Azimuth Equal Area Projection
  InputLocations <- st_transform(InputLocations, crs=st_crs(projString))
  # convert projection
  # 3b.ii Create voronoi polygons 
  InputLocations.vor1 <- st_voronoi(st_union(InputLocations)) 
  InputLocations.vor2 <- st_collection_extract(InputLocations.vor1)
  #dta.vor3 <- st_intersection(dta.vor2, st_union(usa))
  InputLocations.vor3 <- st_sf(InputLocations.vor2)
  InputLocations.vor4 <- st_join(InputLocations.vor3, InputLocations)
  InputLocations.poly <- InputLocations.vor4
}

# 3c Convert Census tract id's to polygons 
if(InputmodelShape == "CensusTracts"){
  # 3c.i Readin Census data 
  Censustract.loc <- read_sf("tl_2015_us_ttract")
  # maybe we need to change a variable name here to get fips 
  # 3c.ii Join with Input model 
  Inputlocations.tract <- Censustract.loc %>% 
    inner_join(InputLocations, by = "fips")
  # 3c.iii Transform geographical coordinates to Lambert Azimuth Equal Area Projection
  InputLocations.tract <- st_transform(InputLocations.tract, crs=st_crs(projString))
  # 3c.iv Store 
  InputLocations.poly <- InputLocations.tract
}

####***************************************************
#### 4: Convert Monitor Locations to Spatial Point ####
####***************************************************

# 4a Convert Monitor locations to simple feature
MonitorLocations <- st_as_sf(MonitorLocations, coords = c("long", "lat"), 
                            crs=st_crs("epsg:4326"))

# 4b Transform geographical coordinates to Lambert Azimuth Equal Area Projection
MonitorLocations <- st_transform(MonitorLocations, crs=st_crs(projString))

####************************************
#### 5: Identify Locations via Join ####
####************************************

# 5a Spatial join 
monitors_in_input <- st_join(MonitorLocations, InputLocations.poly, join = st_within)

# 5b Identify the active locations
activeLocKey <- monitors_in_input$locKey

####***********************************
#### 6: Keep Relevant Observations ####
####***********************************

# 6b Create the locKey variable for the full dataset 
InputData <- InputData %>% 
  unite("locKey", !!LocKeyVar, sep = "_", remove = FALSE)

# 6b Filter
InputData.wMonitor <- InputData %>% 
  filter(locKey %in% activeLocKey)

####*********************
#### 7: Save Results ####
####*********************

# 7a Get name 
InputName0 <- stringr::str_split(InputDataPath, "/")
InputName1 <- InputName0[[1]][length(InputName0[[1]])]
InputName2 <- stringr::str_split(InputName1, ".")
InputName <- InputName2[[1]][1]

# 7b Save 
InputData.wMonitor %>% 
  fst::write_fst(here::here(paste0("data/", InputName, "_atMonitors.fst")))
