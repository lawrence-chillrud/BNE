# Keep only observations at monitor locations 
# Functions
# Bayesian Nonparametric Ensemble 
# Sebastian T. Rowland and Lawrence G. Chilrud

####***********************
#### Table of Contents ####
####***********************

# I: Ideas 
# 0: Preparation 
# 1: Readin Data 
# 2: Simplify Input Model Data
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

# 0a Load packages
pacman::p_load(tidyverse, sf)

# 0b Whose Computer? 
user <- "STR" # LGC

####********************
#### 1: Readin Data ####
####********************

# parameters
# file paths for data:
if(user == "LGC"){
  # set parameters
  InputDataPath <- here::here("CMAQ/outputs/2011_pm25_daily_average.csv")
  MonitorLocationsPath <- here::here("EPA_data/latest_version_clean_annual_data/annual_data_2000-2016.csv")
  InputmodelShape <- "CensusTracts"
  
  # import data:
  InputData <- read_csv(InputDataPath, col_types = "Dcdddd")
  # note: we should do this sort of cleaning in a different script so that we have one file we use for everything 
  MonitorLocations.Raw <- read_csv(MonitorLocationsPath) 
  MonitorLocations.Int <- MonitorLocations.Raw %>% 
    select(!Arithmetic.Mean) %>% na.omit() %>% 
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
    coords = MonitorLocations.Int.NAD83[, c("Longitude", "Latitude")], 
    data = MonitorLocations.Int.NAD83, proj4string = CRS("+init=epsg:4269"))
  MonitorLocations.sf.NAD83.as.WGS84 <- spTransform(MonitorLocations.sf.NAD83, 
                                                    CRS("+init=epsg:4326"))
  MonitorLocations.sf.WGS84 <- SpatialPointsDataFrame(
    coords = MonitorLocations.Int.WGS84[, c("Longitude", "Latitude")], 
    data = MonitorLocations.Int.WGS84, proj4string = CRS("+init=epsg:4326"))
  MonitorLocations.sf <- rbind(MonitorLocations.sf.NAD83.as.WGS84, 
                               MonitorLocations.sf.WGS84)
  
  MonitorLocations.Int.sf <- st_as_sf(MonitorLocations.Int, 
                                      coords = c("Longitude", "Latitude"), 
                                      crs=st_crs("+init=epsg:4326"))
}

if(user == "STR"){
  # set parameters
  InputDataPath <- "data/predictions_MERRA.csv"
  MonitorLocationsPath <- "data/annual_75.csv"
  InputmodelShape <- "impliedGrid"
  
  # readin 
  InputData <- read_csv(here::here(InputDataPath))
  MonitorLocations <- read_csv(here::here(MonitorLocationsPath))
  
  # rename
  MonitorLocations <- MonitorLocations %>%
    rename(lat = Latitude, long = Longitude)
  
}  

####**********************
#### 2: Simplify Data ####
####**********************

# 2a Create location variable 
# can add year here if loc might change year to year 
InputData <- InputData %>% 
  mutate(loc = paste0(lat, "_", long)) 

# 2b Keep only the unique locations
InputLocations <- InputData %>%
  distinct(loc, lat, long) %>% 
  mutate(Index = row_number()) 

####****************************************************
#### 3: Convert Input Model Data to Spatial Polygon ####
####****************************************************

# 3a Convert implied grids to polygons 
if(InputmodelShape == "impliedGrid"){
  # 3a.i Convert to sf 
  InputLocations <- st_as_sf(InputLocations, coords = c("long", "lat"), 
                      crs=st_crs("epsg:4326"))
  # 1a set the projection string 
  projString <- "+proj=laea +lat_0=45 +lon_0=-100 +x_0=0 
                            +y_0=0 +a=6370997 +b=6370997 +units=m +no_defs"
  # 3e Transform geographical coordinates to Lambert Azimuth Equal Area Projection
  InputLocations <- st_transform(InputLocations, crs=st_crs(projString))
  # convert projection
  # 3a.ii Create voronoi polygons 
  InputLocations.vor1 <- st_voronoi(st_union(InputLocations)) 
  InputLocations.vor2 <- st_collection_extract(InputLocations.vor1)
  #dta.vor3 <- st_intersection(dta.vor2, st_union(usa))
  InputLocations.vor3 <- st_sf(InputLocations.vor2)
  InputLocations.vor4 <- st_join(InputLocations.vor3, InputLocations)
  InputLocations.poly <- InputLocations.vor4
}

# 3b Convert Census tract id's to polygons 
if(InputmodelShape == "CensusTracts"){
  # 3b.i Readin Census data 
  Censustract.loc <- read_sf(X)
  # maybe we need to change a variable name here to get fips 
  # 3b.ii Join with Input model 
  Inputlocations.tract <- Censustract.loc %>% 
    inner_join(Inputlocations, by = "fips")
  # 3b.iii Transform geographical coordinates to Lambert Azimuth Equal Area Projection
  InputLocations.tract <- st_transform(InputLocations.tract, crs=st_crs(projString))
  # 3b.iv Store 
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

# do a spatial join 
# if the spatial join takes a long time, then just drop it and do FNN with 
# centroids of nearest monitor


# 5b Keep those indices. as a vector

# if it is one file 
activeLoc <- monitors_in_input$loc

####***********************************
#### 6: Keep Relevant Observations ####
####***********************************

# 6a Filter
InputData.wMonitor <- InputData %>% 
  filter(loc %in% activeLoc)

####*********************
#### 7: Save Results ####
####*********************

# 7a Get name 
a <- stringr::str_split(InputDataPath, "/")

InputName <- a[[1]][length(a[[1]])]
InputName <- str_sub(InputName, 0, -5)
# 7b Save 
InputData.wMonitor %>% 
  fst::write_fst(here:here(paste0("data/", InputName, "_atMonitors.fst")))
