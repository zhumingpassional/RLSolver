import importlib
import argparse
from absl import app


parser = argparse.ArgumentParser()
parser.add_argument('--sampler', type=str, default="dlmc")
parser.add_argument('--model', type=str, default="maxcut")
args = parser.parse_args()

def main(_):
    model_mod = importlib.import_module('models.%s' % args.model)
    model = model_mod.build_model()

    sampler_mod = importlib.import_module('%s' % args.sampler)
    sampler = sampler_mod.build_sampler()

    expirement_mod = importlib.import_module('expirement' )
    expirement = expirement_mod.build_expirement()

    expirement.get_results(model,sampler)

if __name__ == '__main__':
    app.run(main)

    
