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

logging.basicConfig()
_log = logging.getLogger('agdc-temporal-geomedian-test')
_log.setLevel(logging.INFO)
#: pylint: disable=invalid-name
required_option = functools.partial(click.option, required=True)
MY_DATE = dict() 

#: pylint: disable=too-many-arguments
@ui.cli.command(name='tidal-temporal-range')
@ui.global_cli_options
# @ui.executor_cli_options
# @click.command()
@required_option('--epoch', 'epoch', default=40, type=int, help='epoch like 2 5 10')
@required_option('--lon_range', 'lon_range', type=str, required=True, help='like (130.01, 130.052) under quote')
@required_option('--lat_range', 'lat_range', type=str, required=True, help='like (-13.023,-13.052) under quote')
@click.option('--year_range', 'year_range', default='1986-2017', help='2010-2017 i.e 2010-01-01 to 2017-01-01')
@click.option('--tide_post', 'tide_post', type=str, default='',
              help='pick up tide post from epoch_tide_post_model.csv in current directory using Haversin algorithm for a closest cluster or provide from google map like (130.0123, -11.01)')
@click.option('--season', 'season', default='dummy', type=str, help='summer winter autumn spring')
@click.option('--ls7fl', default=True, is_flag=True, help='To include all LS7 data set it to False')
@click.option('--debug', default=False, is_flag=True, help='Build in debug mode to get details of tide height within time range')
# @ui.parsed_search_expressions
# @ui.pass_index(app_name='agdc-tidal-analysis-app')

def main(epoch, lon_range, lat_range, year_range, tide_post, season, ls7fl, debug):
    # dc = datacube.Datacube(app="tidal-range")
    products = ['ls5_nbar_albers', 'ls7_nbar_albers', 'ls8_nbar_albers']
    dc=datacube.Datacube(app='tidal_temporal_test')
    td_info = MyTide(dc, lon_range, lat_range, products, epoch, year_range, tide_post, season, ls7fl, debug)
    print ("Input date range " + year_range + " starting extracting datelist " + str(datetime.now().time()) )
         
    for (acq_min, acq_max) in td_info.get_epochs():
        if season == "dummy":
            print ("running task for epoch " + str(acq_min) + " TO " + str(acq_max) 
                + " tide post " + tide_post + " for lon/lat range " + lon_range + lat_range + " epoch " + str(epoch))
        else:
            print ("running task for epoch " + str(acq_min) + " TO " + str(acq_max) 
                   + " tide post " + tide_post + " for lon/lat range " + lon_range + lat_range + " epoch " + str(epoch
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
    def __init__(self, dc, lon_range, lat_range, products, epoch, year_range, tide_post, season, ls7fl, debug):
        self.dc = dc
        self.lon = eval(lon_range)
        self.lat = eval(lat_range)
        self.products = products
        self.epoch = epoch
        self.start_epoch = datetime.strptime(year_range.split('-')[0] +"-01-01", "%Y-%m-%d").date()
        self.end_epoch = datetime.strptime(year_range.split('-')[1]+"-01-01", "%Y-%m-%d").date()
        self.tide_post = tide_post
        self.season = season
        self.ls7fl = ls7fl
        self.debug = debug

    def get_epochs(self):
        for dt in rrule(YEARLY, interval=self.epoch, dtstart=self.start_epoch, until=self.end_epoch):

            acq_min = dt.date()
            acq_max = acq_min + relativedelta(years=self.epoch, days=-1)
            acq_min = max(self.start_epoch, acq_min)
            acq_max = min(self.end_epoch, acq_max)
            yield acq_min, acq_max

    def distance(self, lat1, lon1, lat2, lon2):
        p = 0.017453292519943295
        a = 0.5 - cos((lat2 - lat1) * p)/2 + cos(lat1 * p) * cos(lat2 * p) * (1 - cos((lon2 - lon1) * p)) / 2
        return 12742 * asin(sqrt(a))

    def extract_otps_range(self, date_list):
        # open the otps lon/lat file
        tp = list()
        ln = 0
        la = 0
        if self.tide_post:
            ln = eval(self.tide_post)[0]
            la = eval(self.tide_post)[1]
        else:
            print ("reading from tidal model file and using Haversine algorithm to extract shortest distance")
            from operator import itemgetter 
            # first find centroid of lat lon range
            la = (self.lat[0] + self.lat[1])/2
            ln = (self.lon[0] + self.lon[1])/2
            rdlist = list()
            fname = "./epoch_tide_post_model.csv"
            try:
                with open (fname, 'rt') as f:
                    reader = csv.reader(f, delimiter=',')
                    for rd in reader:
                        rdlist.append((rd[0], rd[1], rd[2],
                                      self.distance(la, ln, float(rd[1]), float(rd[0]))))
                rdlist = sorted(rdlist, key=itemgetter(3))
                print ( "Found tide post coordinates,depth and shortest distance", rdlist[0] )
                la = float(rdlist[0][1])
                ln = float(rdlist[0][0])
            except IOError as e:
                print ("Unable to open file: " +str(fname))
                sys.exit() 
        for dt in date_list:
            tp.append(TimePoint(ln, la, dt)) 
        tides = predict_tide(tp)
        if len(tides) == 0:
           print ("No tide height observed from OTPS model within lat/lon range")
           sys.exit()
        print ("received from predict tides ", str(datetime.now()))
        date_low = list()
        date_high = list()
        tide_dict = dict()
        for tt in tides:
            tide_dict[datetime.strptime(tt.timepoint.timestamp.isoformat()[0:19], "%Y-%m-%dT%H:%M:%S")] = tt.tide_m
        tide_dict = sorted(tide_dict.items(), key=lambda x: x[1])
        # lowest = round(float(self.per)*len(tide_dict)/100)
        if self.debug:
            print ("")
            print ("ALL TIDES LIST", [[datetime.strftime(date[0], "%Y-%m-%d"), date[1]] for date in tide_dict])
            print ("")
        MY_DATE['ALL_TIDES'] = [[x] for x  in tide_dict]
        return 

    def build_my_dataset(self, acq_min, acq_max):
        
        nbar_data = None 
        dt5 = "2011-12-01"
        dtt7 = "1999-07-01"
        dt7 = "2003-03-01"
        dt8 = "2013-04-01"
        ed = acq_max
        sd = acq_min
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
                print ("LS7 post 2003 March data is not included")
                continue
            elif st == 'ls7_nbar_albers' and self.ls7fl and acq_max > datetime.strptime(dt7, "%Y-%m-%d").date() and \
                  acq_min < datetime.strptime(dt7, "%Y-%m-%d").date():
                acq_max = datetime.strptime(dt7, "%Y-%m-%d").date()
                print (" epoch end date is reset for LS7 2003/03/01")
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
            indexers = {'time':(acq_min, acq_max), 'x':(str(self.lon[0]), str(self.lon[1])), 
                        'y':(str(self.lat[0]), str(self.lat[1])), 'group_by':'solar_day'}
            pq = self.dc.load(product=prod, fuse_func=pq_fuser, **indexers)   
            if st == 'ls5_nbar_albers' and len(pq) == 0:
                print ("No LS5 data found")
                continue
            if nbar_data is not None and st == 'ls7_nbar_albers' and len(pq) == 0:
                print ("No LS7 data found")
                continue
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
        return self.extract_otps_range(date_list) 

    def tidal_task(self, acq_min, acq_max):
        #  gather latest datasets as per product names 
           
        self.build_my_dataset(acq_min, acq_max)
        print (" calculation finished and date dictionary is available in MY_DATE ", 
                   str(datetime.now().time()))      
        return


if __name__ == '__main__':
    '''
    The program gets all LANDSAT datasets excluding post March 2003 LS7 datasets and applied cloud free pq data.
    It accepts lon and lat ranges and optional tide_post. If it is not available, it pulls tidal post model data and 
    extracts epoch/seasonal data and calculates geomedian and low high data dictionaries for six bands and output to a
    MY_DATE.
    It needs small spatial range to cover wide time range.
    '''
    main()
