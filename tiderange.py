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
from datacube.api.geo_xarray import append_solar_day
from datacube.storage.masking import make_mask
from dateutil.rrule import rrule, YEARLY
from datacube.ui import click as ui
from datacube.ui.expression import parse_expressions
from enum import Enum
from datacube.api import GridWorkflow
from pathlib import Path
from datacube.storage.storage import write_dataset_to_netcdf
from rasterio.enums import ColorInterp
from gqa_filter import list_gqa_filtered_cells, get_gqa
from dateutil.rrule import rrule, YEARLY
from otps import TimePoint
from otps.predict_wrapper import predict_tide
from datacube.api.grid_workflow import Tile
from datacube_stats import statistics

logging.basicConfig()
_log = logging.getLogger('agdc-tidal-range-test')
_log.setLevel(logging.INFO)
#: pylint: disable=invalid-name
required_option = functools.partial(click.option, required=True)
MY_DATA = {}
COUNT = 0
DEFAULT_PROFILE = {
    'blockxsize': 256,
    'blockysize': 256,
    'compress': 'lzw',
    'driver': 'GTiff',
    'interleave': 'band',
    'nodata': -999,
    'tiled': True}

#: pylint: disable=too-many-arguments
@ui.cli.command(name='tidal-range')
@ui.global_cli_options
# @ui.executor_cli_options
# @click.command()
@required_option('--epoch', 'epoch', default=2, type=int, help='epoch like 2 5 10')
@required_option('--cell', 'cell', type=str, required=True, help='cell like (-9,-18) under quotes')
@required_option('--year_range', 'year_range', type=str, required=True, help='2010-2017 i.e 2010-01-01 to 2017-01-01')
@click.option('--per', 'per', default=25, type=int, help='10 25 50 for low tide/high tide 10/90 25/75 50/50' )
@click.option('--season', 'season', default='dummy', type=str, help='summer winter autumn spring')
@click.option('--odir', 'odir', default='', type=str, help='if no output directory then data for each epoch will be stored in a dictionary MY_DATA with low and high tides xarray datasets')
@ui.parsed_search_expressions
@ui.pass_index(app_name='agdc-tidal-analysis-app')

def main(index, epoch, cell, year_range, per, season, odir, expressions):
    # dc = datacube.Datacube(app="tidal-range")
    products = ['ls5_nbar_albers', 'ls7_nbar_albers', 'ls8_nbar_albers']
    dc=datacube.Datacube(app='inter_tidal_test')
    td_info = MyTide(dc.index, cell, products, epoch, year_range, per, season, odir)
    for (acq_min, acq_max) in td_info.get_epochs():
        print ("running task for epoch " + str(acq_min) + " TO " + str(acq_max) + " on percentile " + str(per))
        td_info.tidal_task(acq_min, acq_max)


def pq_fuser(dest, src):
    valid_bit = 8
    valid_val = (1 << valid_bit)
    no_data_dest_mask = ~(dest & valid_val).astype(bool)
    np.copyto(dest, src, where=no_data_dest_mask)
    both_data_mask = (valid_val & dest & src).astype(bool)
    np.copyto(dest, src & dest, where=both_data_mask)


class MyTide():
    def __init__(self, index, cell, products, epoch, year_range, per, season, odir):
        self.index = index
        self.cell = cell
        self.products = products
        self.epoch = epoch
        self.start_epoch = datetime.strptime(year_range.split('-')[0] +"-01-01", "%Y-%m-%d").date()
        self.end_epoch = datetime.strptime(year_range.split('-')[1]+"-01-01", "%Y-%m-%d").date()
        self.per = per
        self.season = season
        self.odir = odir

    def get_epochs(self):
        for dt in rrule(YEARLY, interval=self.epoch, dtstart=self.start_epoch, until=self.end_epoch):
            acq_min = dt.date()
            acq_max = acq_min + relativedelta(years=self.epoch, days=-1)
            acq_min = max(self.start_epoch, acq_min)
            acq_max = min(self.end_epoch, acq_max)
            yield acq_min, acq_max

    def extract_pq_dataset(self, acq_min, acq_max):
        gw = GridWorkflow(index=self.index, product=self.products[0])
        ls_5 = defaultdict()
        ls_7 = defaultdict()
        ls_8 = defaultdict()
        pq = None

        for i, st in enumerate(self.products):
            pq_cell_info = defaultdict(dict)
            cell_info = defaultdict(dict)
            prod = None
            if st == 'ls5_nbar_albers':
                prod = 'ls5_pq_albers'
            elif st == 'ls7_nbar_albers':
                prod = 'ls7_pq_albers'
            else:
                prod = 'ls8_pq_albers'
            print ("my cell and sensor", self.cell, st )
            if len(self.odir) > 0:
                filepath = self.odir + '/' + 'TIDAL_' + ''.join(map(str, cell))  \
                       + "_MEDOID_" + str(self.per) + "_PERC_" + str(self.epoch) + "_EPOCH.nc"
                if os.path.isfile(filepath):
                   print ("file exists " + filepath)
                   continue
            if st == 'ls7_nbar_albers' and acq_max > datetime.strptime("2003-01-01", "%Y-%m-%d").date():
                print ("LS7 post 2003 Jan data is not included")
                continue
            if st == 'ls8_nbar_albers' and acq_max < datetime.strptime("2013-01-01", "%Y-%m-%d").date():
                print ("No data for LS8 and hence not searching")
                continue
            indexers = {'product':prod, 'time':(self.start_epoch, self.end_epoch)}
            if i != 0 and len(pq) > 0:
                import pdb; pdb.set_trace()
                pq[self.cell].sources = xr.concat([pq[self.cell].sources, 
                                                  list_gqa_filtered_cells(self.index, gw, pix_th=1, cell_index=eval(self.cell),
                                                                          **indexers)[self.cell].sources], dim='time')
            else:
                pq = list_gqa_filtered_cells(self.index, gw, pix_th=1, cell_index=eval(self.cell), **indexers)

        return pq, gw 

    def extract_otps_range(self, date_list):
        # open the otps lon/lat file
        my_file = "/g/data/u46/users/bxb547/otps/cell_map_lon_lat.csv"
        lon = ''
        lat = ''
        tp = list()
        with open(my_file, 'r') as f:
           reader = csv.reader(f, dialect='excel', delimiter='/')
           for row in reader:
               if self.cell in row:
                  lon = row[1]
                  lat = row[2]
                  break
        for dt in date_list:
            if len(lon) > 0 and len(lat) > 0:
                tp.append(TimePoint(float(lon), float(lat), dt)) 
            else:
                print ("Please provide longitude and latitude of tidal point through a file cell_map_lon_lat.csv")
                return
        tides = predict_tide(tp)
        print ("received from predict tides ", str(datetime.now()))
        date_low = list()
        date_high = list()
        tide_dict = dict()
        for tt in tides:
            # print (tt) 
            # print (datetime.strptime(tt.timepoint.timestamp.isoformat()[0:19], "%Y-%m-%dT%H:%M:%S"), tt.tide_m)
            tide_dict[datetime.strptime(tt.timepoint.timestamp.isoformat()[0:19], "%Y-%m-%dT%H:%M:%S")] = tt.tide_m
        tide_dict = sorted(tide_dict.items(), key=lambda x: x[1])
        lowest = round(float(self.per)*len(tide_dict)/100)
        date_low = tide_dict[:int(lowest)]
        print ("lowest tides list", date_low)
        date_high = tide_dict[-int(len(tide_dict)-lowest):]
        print ("highest tides list", date_high)
        date_low = [dd[0] for dd in date_low]
        date_high = [dd[0] for dd in date_high]
        return date_low, date_high 

    def build_my_dataset(self, gw, pq, date_low, date_high):
        ls_5 = defaultdict()
        ls_7 = defaultdict()
        ls_8 = defaultdict()
        date_list = list(set(date_low) | set(date_high))
        date_list.sort()
        acq_min = date_list[0].date()  
        acq_max = date_list[-1].date()  
        
        nbar_data = None 
        for i, st in enumerate(self.products):
            cell_info = defaultdict(dict)
            prod = None        
            print ("my cell and sensor", self.cell, st )
            if len(self.odir) > 0:
                filepath = self.odir + '/' + 'TIDAL_' + ''.join(map(str, cell))  \
                       + "_MEDOID_" + str(self.per) + "_PERC_" + str(self.epoch) + "_EPOCH.nc" 
                if os.path.isfile(filepath):
                   print ("file exists " + filepath)
                   continue
            if st == 'ls7_nbar_albers' and acq_max > datetime.strptime("2003-01-01", "%Y-%m-%d").date():
                print ("LS7 post 2003 Jan data is not included")
                continue
            if st == 'ls8_nbar_albers' and acq_max < datetime.strptime("2013-01-01", "%Y-%m-%d").date():
                print ("No data for LS8 and hence not searching")
                continue
            # add extra day to the maximum range to include the last day in the search
            end_ep = acq_max + relativedelta(days=1)
            indexers = {'product':st, 'time':(acq_min, end_ep)}
            if i != 0 and len(nbar_data) > 0:
                nbar_data[self.cell].sources = xr.concat([nbar_data[self.cell].sources,
                                                  list_gqa_filtered_cells(self.index, gw, pix_th=1, cell_index=eval(self.cell),
                                                                          **indexers)[self.cell].sources], dim='time')
            else:
                nbar_data = list_gqa_filtered_cells(self.index, gw, pix_th=1, cell_index=eval(self.cell), **indexers)

        # filteredt out lowest and highest date range
        pq_date = pq[eval(self.cell)].sources.time.values.astype('M8[s]').astype(datetime).tolist()
        pq_date = [s.strftime("%Y-%m-%d %H:%M:%S") for s in pq_date]
        date_low = [s.strftime("%Y-%m-%d %H:%M:%S") for s in date_low]
        date_high = [s.strftime("%Y-%m-%d %H:%M:%S") for s in date_high]
        low_match = [i for i, item in enumerate(pq_date) if item in date_low] 
        high_match = [i for i, item in enumerate(pq_date) if item in date_high] 
        pq_low = copy.deepcopy(pq)
        tl = Tile(pq_low[eval(self.cell)].sources.isel(time=low_match), pq_low[eval(self.cell)].geobox)
        pq_low = gw.load(tl, fuse_func=pq_fuser)
        print (" loading data for all sensors at lowest range ", str(datetime.now().time())) 
        nbar_low = copy.deepcopy(nbar_data)
        # create a new tile object
        tl = Tile(nbar_low[eval(self.cell)].sources.isel(time=low_match), nbar_low[eval(self.cell)].geobox)
        nbar_low = gw.load(tl, measurements=['blue', 'green', 'red', 'nir', 'swir1', 'swir2'])
        print (" loaded nbar data" , str(datetime.now().time())) 
        print (" loaded pq data for all sensors at lowest range ", str(datetime.now().time())) 
        mask_clear = pq_low['pixelquality'] & 15871 == 15871
        nbar_low = nbar_low.where(mask_clear)
        tl = Tile(pq[eval(self.cell)].sources.isel(time=high_match), pq[eval(self.cell)].geobox)
        pq_high = gw.load(tl, fuse_func=pq_fuser)
        print (" loaded pq data for all sensors at highest range ", str(datetime.now().time())) 
        print (" loading data for all nbar sensors at highest range ", str(datetime.now().time())) 
        tl = Tile(nbar_data[eval(self.cell)].sources.isel(time=high_match), nbar_data[eval(self.cell)].geobox)
        nbar_high = gw.load(tl, measurements=['blue', 'green', 'red', 'nir', 'swir1', 'swir2'])
        print (" loaded nbar data" , str(datetime.now().time())) 
        mask_clear = pq_high['pixelquality'] & 15871 == 15871
        nbar_high = nbar_high.where(mask_clear)
        return nbar_low, nbar_high 

    def tidal_task(self, acq_min, acq_max):
        print ("date range is FROM " + str(self.start_epoch) + " TO " + str(self.end_epoch))
        # check first whether file name exists
        if len(self.odir) > 0:
            for cl in  self.cell:
                filename = 'TIDAL_' + ''.join(map(str, self.cell))  \
                       + "_MEDOID_" + str(self.per) + "_PERC_" + str(self.epoch) + "_EPOCH.nc"
                print ("my cell and filename  ", cl, filename )
                for file in os.listdir(self.odir):
                    if fnmatch.fnmatch(file, filename):
                        print ("file exists " + filename)
                        return
        # gather latest datasets as per product names and build a dataset dictionary of 
        # unique cells/datasets against the latest dates
        pq, gw =  self.extract_pq_dataset(acq_min, acq_max)
        date_list = pq[eval(self.cell)].sources.time.values.astype('M8[s]').astype('O').tolist() 
        date_low, date_high = self.extract_otps_range(date_list) 
        ds_low, ds_high = self.build_my_dataset(gw, pq, date_low, date_high)
        import pdb; pdb.set_trace()
        # calculate medoid
        # For a slice of 1000:1000 for entire time seried do like  
        # smallds = ds_high.isel(x=slice(None, None, 4), y=slice(None, None, 4)) 
        print ("calculating medoid for lower range", str(datetime.now().time()))      
        med_low = statistics.combined_var_reduction(ds_low, statistics.nanmedoid)         
        print (" medoid for lower range done", str(datetime.now().time()))      
        import pdb; pdb.set_trace()
        print ("calculating medoid for higher range", str(datetime.now().time()))      
        med_high = statistics.combined_var_reduction(ds_high, statistics.nanmedoid)         
        print ("medoid finished for higher range", str(datetime.now().time()))      
        import pdb; pdb.set_trace()
        key=''
        key = str(acq_min) + "_" + str(acq_max) + "_LOW"
        MY_DATA[key] = copy.deepcopy(ds_low)
        key=''
        key = str(acq_min) + "_" + str(acq_max) + "_HIGH" 
        MY_DATA[key] = copy.deepcopy(ds_high)
        return
        # self.create_files(ds_low, ds_high)


if __name__ == '__main__':
    '''
    The program gets latest LANDSAT 7/8 dataset and creates netcdf file or output as in dictionary after 
    applying cloud free pq data. If no output directory is mentioned then it puts cell as key and seven bands
    dataset as value. The program strictly checks pq data. If it is not available then do not process further.
    '''
    main()
