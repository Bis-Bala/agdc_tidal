from __future__ import absolute_import, division, print_function

import math
from datacube.utils import check_intersect
from datacube.api.query import Query, query_group_by

BADCOUNT = 100

def get_gqa(index, id_):                                                              
    dataset = index.datasets.get(id_, include_sources=True)                    
    mean_sq_val = 0
    try:
        mean_dic = dataset.sources['0'].sources['level1'].metadata_doc['gqa']['residual']['iterative_mean']
        mean_sq_val = math.sqrt(mean_dic['x'] * mean_dic['x'] + mean_dic['y'] * mean_dic['y'])
    except KeyError:
        print ("gqa KeyError No reference image check but dataset is included", dataset)
        return 0
    except TypeError: # Mostlikely final_gcp_count is zero and NaN values in iterative_mean 
        print ("gqa TypeError bad dataset and is not including", dataset)
        return BADCOUNT
    return mean_sq_val

def list_gqa_filtered_cells(index, gw, pix_th=None, cell_index=None, **indexers):
    geobox = gw.grid_spec.tile_geobox(cell_index)
    query = Query(index=index, geopolygon=None, **indexers)
    observations = index.datasets.search_eager(**query.search_terms)
    # filter now with pixel threshold value
    datasets = {}
    if pix_th is None:
        pix_th = 1
    for dataset in observations:                                                          
        if check_intersect(geobox.extent, dataset.extent.to_crs(gw.grid_spec.crs)):
            if get_gqa(index, dataset.id) < pix_th:                                  
                datasets.setdefault(cell_index,{'datasets': [],
                                    'geobox': geobox})['datasets'].append(dataset)
    return gw.cell_sources(datasets, query_group_by(**indexers))
