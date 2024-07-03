from config import get_config
from until import get_data
from absl import app
from iSCO import iSCO_fast_vmap
def main(_):
    config = get_config()
    data = get_data(config['data_root'])
    sampler = iSCO_fast_vmap(config,data)
    sampler.get_result()

if __name__ == '__main__':
    app.run(main)

    
