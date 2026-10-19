"""Publication-oriented Prange study; run: python -m experiments.prange_study ..."""
import argparse
import csv
import json
import random
import time
from pathlib import Path
from isd_hqc.algorithms.prange_experiment import prange_many
from isd_hqc.instances.random_code import generate_random_code_instance
from isd_hqc.instances.quasi_cyclic import generate_quasi_cyclic_code_instance
from isd_hqc.instances.doom import qc_syndrome_orbit
from isd_hqc.linear_algebra import gf2_rank

FIELDS = ['mode','block_size','n','r','w','targets','trial','seed','rank','success',
          'max_iterations','iterations','singular_systems','systems_solved','elapsed_seconds','target_index']

def run(mode, block_size, weight, targets, trials, max_iterations, seed, output):
    if mode not in ('random','qc','qc-doom'):
        raise ValueError('Unknown mode')
    if block_size < 2 or not 0 <= weight <= 2*block_size or trials < 1:
        raise ValueError('Invalid parameters')
    if mode != 'qc-doom' and targets != 1:
        raise ValueError('Multiple targets require qc-doom mode')
    if mode == 'qc-doom' and not 1 <= targets <= block_size:
        raise ValueError('Invalid orbit size')
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f'Output already exists: {output}')
    with output.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        for trial in range(trials):
            # Independent reproducible seeds per trial, identical across code families.
            instance_rng = random.Random(seed + 1000003*trial)
            decoder_rng = random.Random(seed + 1000003*trial + 7919)
            # Rejection sampling ensures full-row-rank H in both ensembles.
            for attempt in range(10000):
                instance = (generate_random_code_instance(block_size, 2*block_size, weight, instance_rng)
                    if mode == 'random' else generate_quasi_cyclic_code_instance(block_size, weight, instance_rng))
                if gf2_rank(instance.parity_check_matrix) == block_size:
                    break
            else:
                raise RuntimeError('Could not generate full-row-rank instance')
            syndromes = ([instance.syndrome] if mode != 'qc-doom' else
                         qc_syndrome_orbit(instance.parity_check_matrix, instance.error, block_size, targets))
            start = time.perf_counter()
            stats = prange_many(instance.parity_check_matrix, syndromes, weight, max_iterations, decoder_rng)
            elapsed = time.perf_counter() - start
            writer.writerow(dict(mode=mode,block_size=block_size,n=2*block_size,r=block_size,
                w=weight,targets=len(syndromes),trial=trial,seed=seed+1000003*trial,
                rank=block_size,success=int(stats.error is not None),max_iterations=max_iterations,iterations=stats.iterations,
                singular_systems=stats.singular_systems,systems_solved=stats.systems_solved,
                elapsed_seconds=f'{elapsed:.9f}',target_index=stats.target_index))
    print(json.dumps({'output':str(output),'mode':mode,'trials':trials}))

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=['random','qc','qc-doom'],required=True)
    parser.add_argument('--block-size',type=int,default=8)
    parser.add_argument('--weight',type=int,default=2)
    parser.add_argument('--targets',type=int,default=1)
    parser.add_argument('--trials',type=int,default=100)
    parser.add_argument('--max-iterations',type=int,default=1000)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    run(args.mode,args.block_size,args.weight,args.targets,args.trials,
        args.max_iterations,args.seed,args.output)
