from until import get_data
from absl import app
from iSCO import iSCO_fast
from config import *


def main(_):
    data = get_data(DATA_ROOT)
    sampler = iSCO_fast(data)
    sampler.get_result()

if __name__ == '__main__':
    app.run(main)
