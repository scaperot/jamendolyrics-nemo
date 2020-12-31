
from configparser import ConfigParser
import argparse

import sys
import numpy as np
sys.path.append('..')
import Evaluate


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='print parameters of jamendo-nemo config file')
    parser.add_argument('-c','--config', required=True,type=str,default='jamendo_for_nemo.cfg',
            help='config file for setting up evaluation of nemo model with jamendo')

    args = parser.parse_args()
    
    config = ConfigParser(inline_comment_prefixes=["#"])
    config.read(args.config)
    
    #delay = np.arange(0,25.0,1.0)
    delay = [0]
    ae_list = []
    results_list = []
    preds_list = []
    for i in delay:
        config['main']['DELAY'] = str(i)
        results, preds = Evaluate.compute_metrics(config)
        #Evaluate.print_results(results)
        ae_list.append(results['mean_AE'][0])
        results_list.append(results)
        preds_list.append(preds)

    ndx = np.argmin(ae_list)
    print('Lowest Error was:',delay[ndx])
    print( Evaluate.print_results( results_list[ndx]) )
    print(ae_list)
    #results = Evaluate.compute_metrics(config)
    #Evaluate.print_results(results)
