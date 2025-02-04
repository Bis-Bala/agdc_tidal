from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import click
import functools
import sys
import os
import csv 
import numpy as np
import rasterio
import datetime as DT
import xarray as xr
from datetime import datetime
from itertools import product
import logging
import logging.handlers as lh
import dask.array as da
import datacube.api
import fnmatch
import copy
from collections import defaultdict
from dateutil.relativedelta import relativedelta
from dateutil.rrule import rrule, YEARLY
from datacube.ui import click as ui
# from datacube.ui.expression import parse_expressions
from enum import Enum
from pathlib import Path
from datacube.storage.storage import write_dataset_to_netcdf
from rasterio.enums import ColorInterp
from gqa_filter import list_gqa_filtered_cells, get_gqa
from dateutil.rrule import rrule, YEARLY
from otps import TimePoint
from otps.predict_wrapper import predict_tide
from datacube.api.grid_workflow import Tile
from datacube_stats import statistics
from math import cos, asin, sqrt
import hdmedians as hd
from datacube.utils import geometry

logging.basicConfig()
_log = logging.getLogger('agdc-temporal-geomedian-test')
_log.setLevel(logging.INFO)
#: pylint: disable=invalid-name
required_option = functools.partial(click.option, required=True)
MY_GEO = {}
MY_EPOCH = {}
DEFAULT_PROFILE = {
    'blockxsize': 256,
    'blockysize': 256,
    'compress': 'lzw',
    'driver': 'GTiff',
    'interleave': 'band',
    'nodata': -999,
    'tiled': True}

#: pylint: disable=too-many-arguments
@ui.cli.command(name='tidal-temporal-range')
@ui.global_cli_options
# @ui.executor_cli_options
# @click.command()
@required_option('--epoch', 'epoch', default=2, type=int, help='epoch like 2 5 10')
@required_option('--lon_range', 'lon_range', type=str, default='', help='like (130.01, 130.052) under quote')
@required_option('--lat_range', 'lat_range', type=str, default='', help='like (-13.023,-13.052) under quote')
@required_option('--year_range', 'year_range', type=str, required=True, help='2010-2017 i.e 2010-01-01 to 2017-01-01')
@click.option('--date_all_1', 'date_all_1', type=str, default='', 
                 help='Run first extract_tidal_datelist.py to get low date list')
@required_option('--date_all_2', 'date_all_2', type=str, required=True, 
                 help='Run first extract_tidal_datelist.py to get high date list')
@click.option('--tide_post', 'tide_post', type=str, default='',
              help='pick up tide post from epoch_tide_post_model.csv in current directory using Haversin algorithm for a closest cluster or provide from google map like (130.0123, -11.01)')
@click.option('--per', 'per', default=10, type=int, help='10 25 50 for low tide/high tide 10/10 25/25 50/50' )
@click.option('--season', 'season', default='dummy', type=str, help='summer winter autumn spring')
@click.option('--poly', 'poly', default='', type=str, help='allow polygon coordinates')
@click.option('--crs', 'crs', default='', type=str, help='crs from the polygon file')
@click.option('--ebb', 'ebb', default='', type=str, help='extracting composite while tide drains away from shore')
@click.option('--flow', 'flow', default='', type=str, help='extracting composite while tide rises')
@click.option('--ls7fl', default=True, is_flag=True, help='To include all LS7 data set it to False')
@click.option('--debug', default=False, is_flag=True, help='Build in debug mode to get details of tide height within time range')
# @ui.parsed_search_expressions
# @ui.pass_index(app_name='agdc-tidal-analysis-app')

def main(epoch, lon_range, lat_range, year_range, date_all_1, date_all_2, tide_post, per, season, poly, crs, 
         ebb, flow, ls7fl, debug):
    # dc = datacube.Datacube(app="tidal-range")
    products = ['ls5_nbar_albers', 'ls7_nbar_albers', 'ls8_nbar_albers']
    if per == "50" and len(ebb) != 0:
        print ("not supported for 50 percent ebb images")
        return
    if per == "50" and len(flow) != 0: 
        print ("not supported for 50 percent flow images ")
        return
    if len(ebb) != 0 and len(flow) != 0:
        print ("not supported for both ebb flow together")
        return
  
    dc=datacube.Datacube(app='tidal_temporal_test')

    td_info = MyTide(dc, lon_range, lat_range, products, epoch, year_range, date_all_1, date_all_2, tide_post,
                     per, season, poly, crs, ebb, flow, ls7fl, debug)
    print ("Input date range " + year_range )
    for (acq_min, acq_max) in td_info.get_epochs():
        if season == "dummy":
            print ("running task for epoch " + str(acq_min) + " TO " + str(acq_max) + " on percentile " + str(per
               ) + " tide post " + tide_post + " for lon/lat range " + lon_range + lat_range + " epoch " + str(epoch))
        else:
            print ("running task for epoch " + str(acq_min) + " TO " + str(acq_max) + " on percentile " + str(per
                   ) + " tide post " + tide_post + " for lon/lat range " + lon_range + lat_range + " epoch " + str(epoch
                   ) + " for season " + season)
        
        td_info.tidal_task(acq_min, acq_max)


def pq_fuser(dest, src):
    valid_bit = 8
    valid_val = (1 << valid_bit)
    no_data_dest_mask = ~(dest & valid_val).astype(bool)
    np.copyto(dest, src, where=no_data_dest_mask)
    both_data_mask = (valid_val & dest & src).astype(bool)
    np.copyto(dest, src & dest, where=both_data_mask)


class MyTide():
    def __init__(self, dc, lon_range, lat_range, products, epoch, year_range, date_all_1, date_all_2, tide_post,
                 per, season, poly, crs, ebb, flow, ls7fl, debug):
        self.dc = dc
        if len(poly) > 0:
            self.lon = ''
            self.lat = ''
            if len(tide_post) == 0:
                print ("Please provide tide post parameter")
                sys.exit()
        else: 
            self.lon = eval(lon_range)
            self.lat = eval(lat_range)
        self.products = products
        self.epoch = epoch
        #self.start_epoch = datetime.strptime(year_range.split('-')[0] +"-01-01", "%Y-%m-%d").date()
        #self.end_epoch = datetime.strptime(year_range.split('-')[1]+"-01-01", "%Y-%m-%d").date()
        self.start_epoch = datetime.strptime(year_range.split('_')[0], "%Y-%m-%d").date()
        self.end_epoch = datetime.strptime(year_range.split('_')[1], "%Y-%m-%d").date()
        if len(date_all_1) > 0:
            self.date_all_1 = list(eval(date_all_1))
        self.date_all_2 = list(eval(date_all_2))
        self.tide_post = tide_post
        self.per = per
        self.season = season
        self.poly = poly
        self.crs = crs
        self.ebb = ebb
        self.flow = flow
        self.ls7fl = ls7fl
        self.debug = debug

    def get_epochs(self):
        for dt in rrule(YEARLY, interval=self.epoch, dtstart=self.start_epoch, until=self.end_epoch):

            if dt.date() >= self.end_epoch:
               print ("CALCULATION finished and data available in MY_GEO and MY_EPOCH dictionary ")
               return 
            acq_min = dt.date()
            acq_max = acq_min + relativedelta(years=self.epoch, days=-1)
            acq_min = max(self.start_epoch, acq_min)
            acq_max = min(self.end_epoch, acq_max)
            yield acq_min, acq_max

    def distance(self, lat1, lon1, lat2, lon2):
        p = 0.017453292519943295
        a = 0.5 - cos((lat2 - lat1) * p)/2 + cos(lat1 * p) * cos(lat2 * p) * (1 - cos((lon2 - lon1) * p)) / 2
        return 12742 * asin(sqrt(a))

    def init_polygon(self):
        crs = self.crs
        crs = geometry.CRS(crs)
        first_geometry = {'type': 'Polygon', 'coordinates': eval(self.poly)}
        geom = geometry.Geometry(first_geometry, crs=crs)
        return geom

    def extract_ebb_flow_tides(self, date_list):
        tp = list()
        tide_dict = dict()
        ln = eval(self.tide_post)[0]
        la = eval(self.tide_post)[1]
        ndate_list=list()
        mnt=timedelta(minutes=15)
        for dt in date_list:
            ndate_list.append(dt-mnt)
            ndate_list.append(dt)
            ndate_list.append(dt+mnt)
        for dt in ndate_list:
            tp.append(TimePoint(ln, la, dt)) 
        tides = predict_tide(tp)
        if len(tides) == 0:
           print ("No tide height observed from OTPS model within lat/lon range")
           sys.exit()
        print ("received from predict tides ", str(datetime.now()))
        # collect in ebb/flow list
        for tt in tides:
            tide_dict[datetime.strptime(tt.timepoint.timestamp.isoformat()[0:19], "%Y-%m-%dT%H:%M:%S")] = tt.tide_m
        tide_dict = sorted(tide_dict.items(), key=lambda x: x[1])
        tmp_lt = list()
        for k, v in tide_dict.items():
            tmp_lt.append([k, v])
        if self.debug:
            print (str(tmp_lt))
        tmp_lt = [[tmp_lt[i+1][0].strftime("%Y-%m-%d"), 'ph'] \
                 if tmp_lt[i][1] < tmp_lt[i+1][1] and tmp_lt[i+2][1] <  tmp_lt[i+1][1]  else \
                 [tmp_lt[i+1][0].strftime("%Y-%m-%d"), 'pl'] if tmp_lt[i][1] > tmp_lt[i+1][1] and \
                 tmp_lt[i+2][1] >  tmp_lt[i+1][1]  else [tmp_lt[i+1][0].strftime("%Y-%m-%d"),'f'] \
                 if tmp_lt[i][1] < tmp_lt[i+2][1] else [tmp_lt[i+1][0].strftime("%Y-%m-%d"),'e'] \
                 for i in range(0, len(tmp_lt), 3)]    
        return tmp_lt     
        

    def build_my_dataset(self, acq_min, acq_max):
        
        nbar_data = None 
        dt5 = "2011-12-01"
        dtt7 = "1999-07-01"
        dt7 = "2003-05-01"
        dt8 = "2013-04-01"
        geom = None
        ed = acq_max
        sd = acq_min
        if len(self.poly) > 0:
            geom = self.init_polygon()
        for i, st in enumerate(self.products):
            prod = None
            acq_max = ed
            acq_min = sd        
            print (" doing for sensor",  st )
            if st == 'ls5_nbar_albers' and acq_max > datetime.strptime(dt5, "%Y-%m-%d").date() and  \
                  acq_min > datetime.strptime(dt5, "%Y-%m-%d").date():
                print ("LS5 post 2011 Dec data is not exist")
                continue
            elif st == 'ls5_nbar_albers' and acq_max > datetime.strptime(dt5, "%Y-%m-%d").date() and \
                  acq_min < datetime.strptime(dt5, "%Y-%m-%d").date():
                acq_max = datetime.strptime(dt5, "%Y-%m-%d").date()
                print (" epoch end date is reset for LS5 2011/12/01")
            if st == 'ls7_nbar_albers' and self.ls7fl and acq_max > datetime.strptime(dt7, "%Y-%m-%d").date() and  \
                  acq_min > datetime.strptime(dt7, "%Y-%m-%d").date():
                print ("LS7 post 2003 May data is not included")
                continue
            elif st == 'ls7_nbar_albers' and self.ls7fl and acq_max > datetime.strptime(dt7, "%Y-%m-%d").date() and \
                  acq_min < datetime.strptime(dt7, "%Y-%m-%d").date():
                acq_max = datetime.strptime(dt7, "%Y-%m-%d").date()
                print (" epoch end date is reset for LS7 2003/05/01")
            if st == 'ls7_nbar_albers' and acq_max < datetime.strptime(dtt7, "%Y-%m-%d").date() and \
                  acq_min < datetime.strptime(dtt7, "%Y-%m-%d").date():
                continue
            if st == 'ls8_nbar_albers' and acq_max < datetime.strptime(dt8, "%Y-%m-%d").date() and \
                  acq_min < datetime.strptime(dt8, "%Y-%m-%d").date():
                continue
            elif st == 'ls8_nbar_albers' and acq_max > datetime.strptime(dt8, "%Y-%m-%d").date() and \
                acq_min < datetime.strptime(dt8, "%Y-%m-%d").date():
                acq_min = datetime.strptime(dt8, "%Y-%m-%d").date()
            if st == 'ls5_nbar_albers':
                prod = 'ls5_pq_albers'
            elif st == 'ls7_nbar_albers':
                prod = 'ls7_pq_albers'
            else:
                prod = 'ls8_pq_albers'
            # add extra day to the maximum range to include the last day in the search
            # end_ep = acq_max + relativedelta(days=1)
            if geom:
                indexers = {'time':(acq_min, acq_max), 'geopolygon': geom, 'group_by':'solar_day'}
            else:
                indexers = {'time':(acq_min, acq_max), 'x':(str(self.lon[0]), str(self.lon[1])), 
                            'y':(str(self.lat[0]), str(self.lat[1])), 'group_by':'solar_day'}
            pq = self.dc.load(product=prod, fuse_func=pq_fuser, **indexers)   
            if st == 'ls5_nbar_albers' and len(pq) == 0:
                print ("No LS5 data found")
                continue
            if nbar_data is not None and st == 'ls7_nbar_albers' and len(pq) == 0:
                print ("No LS7 data found")
                continue
            if geom:
                indexers = {'time':(acq_min, acq_max), 'measurements':['blue', 'green', 'red', 'nir', 'swir1', 'swir2'],
                            'geopolygon': geom, 'group_by':'solar_day'}
            else:
                indexers = {'time':(acq_min, acq_max), 'x':(str(self.lon[0]), str(self.lon[1])), 'y':(str(self.lat[0]), str(self.lat[1])),
                        'measurements':['blue', 'green', 'red', 'nir', 'swir1', 'swir2'], 'group_by':'solar_day'}
            mask_clear = pq['pixelquality'] & 15871 == 15871
            if nbar_data is not None:
                new_data = self.dc.load(product=st, **indexers)
                new_data = new_data.where(mask_clear)
                nbar_data = xr.concat([nbar_data, new_data], dim='time')
            else:
                nbar_data = self.dc.load(product=st, **indexers)
                nbar_data = nbar_data.where(mask_clear)
        # if season then filtered only season data
        if self.season.upper() == 'WINTER':
            nbar_data = nbar_data.isel(time=nbar_data.groupby('time.season').groups['JJA'])
        elif self.season.upper() == 'SUMMER':
            nbar_data = nbar_data.isel(time=nbar_data.groupby('time.season').groups['DJF'])
        elif self.season.upper() == 'SPRING':
            nbar_data = nbar_data.isel(time=nbar_data.groupby('time.season').groups['SON'])
        elif self.season.upper() == 'AUTUMN':
            nbar_data = nbar_data.isel(time=nbar_data.groupby('time.season').groups['MAM'])
        # filtered out lowest and highest date range
        date_list = nbar_data.time.values.astype('M8[s]').astype('O').tolist() 
        date_all = [s.strftime("%Y-%m-%d") for s in date_list]
        nbar_high = None
        high_dict = dict()
        date_h = ""
        high_match = [i for i, item in enumerate(date_all) for x in self.date_all_2 if item in x]
        if len(high_match) == 0:
            print (" No dataset matches with this epoch ")
            return None, None
        high_all = [item for item in self.date_all_2 if item.split(',')[0] in date_all]
        for i in high_all: 
            high_dict[i.split(',')[0]] = i.split(',')[1]
        high_dict = sorted(high_dict.items(), key=lambda x: x[1])
        date_high = [x for x  in high_dict] 
        nbar_high = nbar_data.isel(time=high_match)
        if self.debug:
            #dt_lst=nbar_high.time.values.astype('M8[D]').astype('O').tolist()  
            print ("tide list ")
            # print ( [datetime.strftime(date, "%Y-%m-%d")  for date in dt_lst])
            print (date_high)
            print ("")
        date_h = str(date_high[0][1]) + "," + str(date_high[len(date_high)-1][1]) + "," + str(len(date_high))
        print (" loaded nbar data " +  str(datetime.now().time())) 
        
        return nbar_high, date_h

    def tidal_task(self, acq_min, acq_max):
        #  gather latest datasets as per product names 
           
        ds_high, date_h = self.build_my_dataset(acq_min, acq_max)
        if ds_high is None:
           return
        # calculate medoid
        # For a slice of 1000:1000 for entire time seried do like  
        # smallds = ds_high.isel(x=slice(None, None, 4), y=slice(None, None, 4)) 
        key = ''
        print ("creating GEOMEDIAN for epoch " + str(acq_min) + "_" + str(acq_max) + str(datetime.now().time()))      
        med_high = statistics.combined_var_reduction(ds_high, hd.nangeomedian)         
        key = str(acq_min) + "_" + str(acq_max) 
        
        MY_GEO[key] = copy.deepcopy(med_high)
        MY_EPOCH[key] = copy.deepcopy(date_h)
        if acq_max == self.end_epoch:
            print (" calculation finished for MY_GEO and MY_EPOCH dictionaries " +  
                   str(datetime.now().time()))
        return
            

if __name__ == '__main__':
    '''
    The program gets all LANDSAT datasets excluding post March 2003 LS7 datasets and applied cloud free pq data.
    It accepts lon and lat ranges and optional tide_post. If it is not available, it pulls tidal post model data and 
    extracts epoch/seasonal data and calculates geomedian and low high data dictionaries for six bands and output to a
    MY_GEO and MY_EPOCH.
    MY_GEO dictionary has LOW and HIGH xarray geomedian datasets and MY_EPOCH has low high tidal ranges and number of datasets.
    It needs small spatial range to cover wide time range.
    '''
    main()
