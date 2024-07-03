def get_config():
    config = dict(data_root = r"D:\Document\zdhs\isco\gset\gset_14.txt",
                  sampler = 'iSCO_fast_vmap',
                  init_temperature = 1.0,
                  final_temperature = 0.0,
                  batch_size = 300,
                  device = 'cuda',
                  chain_length = 50,
                  )
    return config

