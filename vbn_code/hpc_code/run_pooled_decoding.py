import argparse
import decoding_utils as du

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--label', type=str)
    parser.add_argument('--region', type=str)
    parser.add_argument('--cluster', type=str)
    parser.add_argument('--unitSampleSize', type=int)
    parser.add_argument('--nPseudoFlashes', type=int)
    parser.add_argument('--nUnitSamples', type=int)
    parser.add_argument('--condition', type=str)
    args = parser.parse_args()

    print(f'calling pooled decoding, {args.label}, {args.region} {args.cluster}')
    du.pooledDecoding(args.label, args.region, args.cluster, args.unitSampleSize, args.nPseudoFlashes, args.nUnitSamples, args.condition)
